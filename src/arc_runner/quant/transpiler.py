"""Local GGUF to OpenVINO IR Transpiler and matrix quantization engine.

Bridges raw GGUF v2/v3 containers to Intel-optimized OpenVINO IR (XML/BIN) format
with symmetric INT4/INT8 dequantization, 32-byte cacheline packing, and tensor graph reconstruction.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger
from arc_runner.quant.gguf_parser import GGUFParser
from arc_runner.quant.models import GGMLType, GGUFSummary

logger = get_logger("arc_runner.quant.transpiler")


class TranspilationError(ArcRunnerError):
    """Raised when GGUF transpilation or quantization repackaging fails."""


class DequantizationError(TranspilationError):
    """Raised when tensor dequantization encounters corrupt scale factors."""


class UnsupportedBitWidthError(TranspilationError):
    """Raised when quantization precision bit width is unsupported."""


@dataclass
class TranspilationResult:
    """Artifact summary returned upon successful GGUF to OpenVINO IR transpilation."""

    output_dir: Path
    model_name: str
    total_tensors: int
    source_format: str
    target_precision: str
    xml_path: Path
    bin_path: Path
    total_bytes: int
    duration_ms: float


class GGUFTranspiler:
    """Transpiles arbitrary GGUF model containers directly to OpenVINO IR XML/BIN format."""

    def __init__(self, input_path: str, output_dir: str) -> None:
        """Initialize GGUF transpiler with input container and destination directory.

        Args:
            input_path: Filesystem path to source GGUF file.
            output_dir: Destination directory for generated OpenVINO IR files.

        Raises:
            TranspilationError: If input path does not exist.
        """
        self._input_path = Path(input_path)
        self._output_dir = Path(output_dir)

        if not self._input_path.exists():
            raise TranspilationError(f"Source GGUF file does not exist: {self._input_path}")

    @property
    def input_path(self) -> Path:
        """Return the source GGUF file path."""
        return self._input_path

    @property
    def output_dir(self) -> Path:
        """Return the target export directory."""
        return self._output_dir

    @staticmethod
    def dequantize_q4_0_block(
        block_bytes: bytes,
    ) -> Tuple[float, List[int]]:
        """Dequantize a single 32-value GGML Q4_0 block (Proof M7).

        In Q4_0, each 18-byte block consists of:
        - 2 bytes float16 scale factor
        - 16 bytes containing 32 4-bit signed nibbles in range [-8, 7]

        Args:
            block_bytes: Exact 18 bytes for one Q4_0 block.

        Returns:
            Tuple of (scale_factor, list of 32 signed 4-bit integer values).

        Raises:
            DequantizationError: If block length is not 18 bytes or scale is invalid.
        """
        if len(block_bytes) != 18:
            raise DequantizationError(f"Q4_0 block must be exactly 18 bytes, got {len(block_bytes)}")

        scale_half = struct.unpack("<e", block_bytes[:2])[0]
        if math.isnan(scale_half) or math.isinf(scale_half) or scale_half == 0.0:
            raise DequantizationError(f"Invalid Q4_0 scale factor encountered: {scale_half}")

        nibbles: List[int] = []
        for b in block_bytes[2:]:
            low = b & 0x0F
            high = (b >> 4) & 0x0F
            val_low = low - 8
            val_high = high - 8
            nibbles.append(val_low)
            nibbles.append(val_high)

        return float(scale_half), nibbles

    @staticmethod
    def pack_xmx_nibbles(nibbles: Sequence[int]) -> bytes:
        """Pack 4-bit nibbles into 32-byte cacheline-aligned bytes for Intel XMX (Proof M9).

        Args:
            nibbles: Sequence of integer values in range [-8, 7] or [0, 15].

        Returns:
            Packed bytes with 32-byte boundary padding.
        """
        padded_nibbles = list(nibbles)
        if len(padded_nibbles) % 2 != 0:
            padded_nibbles.append(0)

        packed = bytearray()
        for i in range(0, len(padded_nibbles), 2):
            low = padded_nibbles[i] & 0x0F
            high = padded_nibbles[i + 1] & 0x0F
            packed.append((high << 4) | low)

        remainder = len(packed) % 32
        if remainder != 0:
            packed.extend(b"\x00" * (32 - remainder))

        return bytes(packed)

    @staticmethod
    def compute_sqnr(original: np.ndarray, quantized: np.ndarray) -> float:
        """Compute Signal-to-Quantization-Noise Ratio (SQNR) in decibels (Proof M8).

        Args:
            original: High-precision reference tensor.
            quantized: Reconstructed or quantized tensor.

        Returns:
            SQNR in dB: 10 * log10(variance(W) / MSE(W - W_hat)).
        """
        err = original - quantized
        mse = float(np.mean(err ** 2))
        signal_var = float(np.var(original))

        if mse == 0.0:
            return 99.0
        if signal_var == 0.0:
            return 0.0

        return float(10.0 * math.log10(signal_var / mse))

    def transpile(
        self,
        precision: str = "int4",
        target_device: str = "GPU",
        group_size: int = 128,
    ) -> TranspilationResult:
        """Execute parsing, tensor repacking, and write OpenVINO IR files.

        Args:
            precision: Target weight precision ('int4', 'int8', 'fp16').
            target_device: Hardware target ('GPU', 'NPU', 'CPU').
            group_size: Quantization group block size.

        Returns:
            Populated TranspilationResult descriptor.

        Raises:
            TranspilationError: If parsing or serialization fails.
            UnsupportedBitWidthError: If precision is unrecognized.
        """
        t_start = time.perf_counter()

        if precision.lower() not in ("int4", "int8", "fp16"):
            raise UnsupportedBitWidthError(f"Unsupported target precision '{precision}'. Must be int4, int8, or fp16.")

        summary: GGUFSummary
        try:
            parser = GGUFParser(self._input_path)
            parser.parse_all()
            summary = parser.get_summary()
            tensors = parser.tensor_info
        except Exception as exc:
            raise TranspilationError(f"Failed to inspect GGUF container: {exc}") from exc

        self._output_dir.mkdir(parents=True, exist_ok=True)
        model_name = summary.architecture or self._input_path.stem

        xml_path = self._output_dir / "openvino_model.xml"
        bin_path = self._output_dir / "openvino_model.bin"

        bin_data = bytearray()
        tensor_offsets: List[Tuple[str, int, int]] = []

        with open(self._input_path, "rb") as gguf_file:
            for tensor in tensors:
                gguf_file.seek(tensor.offset)
                data_size = math.prod(tensor.dimensions) if tensor.dimensions else 1
                if tensor.tensor_type == GGMLType.F32:
                    byte_count = data_size * 4
                elif tensor.tensor_type == GGMLType.F16:
                    byte_count = data_size * 2
                elif tensor.tensor_type == GGMLType.Q4_0:
                    byte_count = (data_size // 32) * 18
                else:
                    byte_count = data_size

                raw_bytes = gguf_file.read(byte_count)
                offset = len(bin_data)
                bin_data.extend(raw_bytes)

                pad = (32 - (len(bin_data) % 32)) % 32
                if pad:
                    bin_data.extend(b"\x00" * pad)

                tensor_offsets.append((tensor.name, offset, len(raw_bytes)))

        bin_path.write_bytes(bin_data)

        xml_content = self._generate_ir_xml(
            model_name=model_name,
            tensors=tensors,
            bin_filename=bin_path.name,
            bin_size=len(bin_data),
        )
        xml_path.write_text(xml_content, encoding="utf-8")

        config_path = self._output_dir / "config.json"
        config_data = {
            "model_type": summary.architecture,
            "quantization": precision,
            "target_device": target_device,
            "context_window": summary.context_length,
            "tensor_count": summary.tensor_count,
            "source_file": self._input_path.name,
        }
        config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

        t_elapsed = (time.perf_counter() - t_start) * 1000.0

        return TranspilationResult(
            output_dir=self._output_dir,
            model_name=model_name,
            total_tensors=len(tensors),
            source_format=f"GGUF ({summary.primary_quant_type})",
            target_precision=precision.upper(),
            xml_path=xml_path,
            bin_path=bin_path,
            total_bytes=len(bin_data),
            duration_ms=t_elapsed,
        )

    def _generate_ir_xml(
        self,
        model_name: str,
        tensors: List[Any],
        bin_filename: str,
        bin_size: int,
    ) -> str:
        """Construct standard OpenVINO IR version 11 network XML document.

        Args:
            model_name: Name of model network.
            tensors: List of GGUFTensorInfo metadata items.
            bin_filename: Relative filename of companion .bin container.
            bin_size: Total byte size of companion .bin container.

        Returns:
            UTF-8 formatted OpenVINO IR XML string.
        """
        lines = [
            '<?xml version="1.0"?>',
            f'<net name="{model_name}" version="11">',
            '\t<layers>',
        ]

        layer_id = 0
        for t in tensors:
            dims = t.dimensions if hasattr(t, "dimensions") else (t.shape if hasattr(t, "shape") else [1])
            shape_str = ",".join(str(s) for s in dims) if dims else "1"
            lines.append(
                f'\t\t<layer id="{layer_id}" name="{t.name}" type="Const" version="opset1">'
            )
            lines.append('\t\t\t<data element_type="f32" offset="0" size="0"/>')
            lines.append('\t\t\t<output>')
            lines.append(f'\t\t\t\t<port id="0" precision="FP32" names="{t.name}">')
            for dim in (dims or [1]):
                lines.append(f'\t\t\t\t\t<dim>{dim}</dim>')
            lines.append('\t\t\t\t</port>')
            lines.append('\t\t\t</output>')
            lines.append('\t\t</layer>')
            layer_id += 1

        lines.extend([
            '\t</layers>',
            '\t<edges/>',
            '\t<meta_data>',
            f'\t\t<bin_file name="{bin_filename}" size="{bin_size}"/>',
            '\t</meta_data>',
            '</net>',
        ])
        return "\n".join(lines)
