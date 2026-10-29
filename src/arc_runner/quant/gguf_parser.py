"""Zero-dependency binary parser for GGUF v2 and v3 container formats.

Deserializes headers, metadata key-value tables, and tensor descriptor records
directly from binary streams without loading raw weight arrays into system RAM.
"""

from collections import Counter
from io import BufferedReader, SEEK_END
from pathlib import Path
import struct
from typing import Any, BinaryIO, Optional, Union

from arc_runner.core.constants import GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_SUPPORTED_VERSIONS
from arc_runner.core.exceptions import GGUFParseError, InvalidGGUFError
from arc_runner.core.logging import get_logger
from arc_runner.quant.models import GGUFHeader, GGUFSummary, GGUFTensorInfo, GGUFValueType, GGMLType

logger = get_logger(__name__)

GGUF_MAX_DIMENSIONS = 4
GGUF_MAX_ARRAY_DEPTH = 8
GGUF_MAX_ALIGNMENT = 1 << 20
GGUF_MAX_TOTAL_ARRAY_ELEMENTS = 1 << 24
GGUF_MIN_KV_BYTES = 12
GGUF_MIN_TENSOR_BYTES = 24

_SCALAR_STRUCT_FORMATS = {
    GGUFValueType.UINT8: "B",
    GGUFValueType.INT8: "b",
    GGUFValueType.UINT16: "H",
    GGUFValueType.INT16: "h",
    GGUFValueType.UINT32: "I",
    GGUFValueType.INT32: "i",
    GGUFValueType.FLOAT32: "f",
    GGUFValueType.UINT64: "Q",
    GGUFValueType.INT64: "q",
    GGUFValueType.FLOAT64: "d",
}
_MIN_ELEMENT_BYTES = {
    **{kind: struct.calcsize("<" + code) for kind, code in _SCALAR_STRUCT_FORMATS.items()},
    GGUFValueType.BOOL: 1,
    GGUFValueType.STRING: 8,
    GGUFValueType.ARRAY: 12,
}


def _enum_or_error(enum_cls: Any, raw: int, what: str, stream: BinaryIO) -> Any:
    """Convert a raw integer into an enum member or raise a typed parse error.

    Args:
        enum_cls: Enum class to convert into.
        raw: Integer read from the file.
        what: Description for the error message.
        stream: Stream whose position is reported.

    Returns:
        Any: The enum member.

    Raises:
        GGUFParseError: If raw is not a member of the enum.
    """
    try:
        return enum_cls(raw)
    except ValueError:
        raise GGUFParseError(f"Unsupported {what} id {raw}", stream.tell()) from None


def _bytes_remaining(stream: BinaryIO) -> int:
    """Return how many bytes lie between the stream position and its end.

    Args:
        stream: Open seekable binary stream.

    Returns:
        int: Remaining byte count; the stream position is unchanged.
    """
    position = stream.tell()
    end = stream.seek(0, SEEK_END)
    stream.seek(position)
    return end - position


def _read_declared(stream: BinaryIO, length: int, what: str) -> bytes:
    """Read a length declared inside the file only if that many bytes actually remain.

    The declared length is untrusted input; reading it blindly can request terabytes or
    overflow the platform index type.

    Args:
        stream: Open seekable binary stream.
        length: Byte count declared by the file.
        what: Description used in the error message.

    Returns:
        bytes: Exactly length bytes.

    Raises:
        GGUFParseError: If fewer than length bytes remain.
    """
    remaining = _bytes_remaining(stream)
    if length > remaining:
        raise GGUFParseError(f"{what}: declared {length} bytes but only {remaining} remain", stream.tell())
    return stream.read(length)


class GGUFParser:
    """High-performance binary decoder for GGUF model containers."""

    def __init__(self, file_path: Union[Path, str]) -> None:
        """Initialize GGUF parser targeting local container file.

        Args:
            file_path: Path to target .gguf file or content-addressable Ollama blob.
        """
        self.path = Path(file_path)
        self._header: Optional[GGUFHeader] = None
        self._metadata: Optional[dict[str, Any]] = None
        self._tensor_info: Optional[list[GGUFTensorInfo]] = None
        self._data_offset: int = 0
        self._array_elements_left: int = GGUF_MAX_TOTAL_ARRAY_ELEMENTS

    @property
    def header(self) -> GGUFHeader:
        """Return parsed header, parsing on first access if uninitialized."""
        if self._header is None:
            self.parse_all()
        assert self._header is not None
        return self._header

    @property
    def metadata(self) -> dict[str, Any]:
        """Return parsed metadata dictionary, parsing on first access."""
        if self._metadata is None:
            self.parse_all()
        assert self._metadata is not None
        return self._metadata

    @property
    def tensor_info(self) -> list[GGUFTensorInfo]:
        """Return parsed list of tensor descriptors, parsing on first access."""
        if self._tensor_info is None:
            self.parse_all()
        assert self._tensor_info is not None
        return self._tensor_info

    @property
    def data_offset(self) -> int:
        """Return byte offset where raw tensor payloads begin."""
        if self._header is None:
            self.parse_all()
        return self._data_offset

    def parse_header(self, stream: BinaryIO) -> GGUFHeader:
        """Read and validate the 24-byte GGUF container header.

        Args:
            stream: Open binary stream positioned at byte 0.

        Returns:
            Validated GGUFHeader instance.

        Raises:
            InvalidGGUFError: If magic bytes mismatch or version is unsupported.
            GGUFParseError: If file is truncated before completing the header.
        """
        raw_header = stream.read(4)
        if len(raw_header) < 4:
            raise GGUFParseError("Unexpected end of file while reading GGUF magic", offset=0)
        if raw_header != GGUF_MAGIC:
            raise InvalidGGUFError(f"Expected magic b'{GGUF_MAGIC.decode()}', got {raw_header!r}", file_path=str(self.path))

        payload = stream.read(20)
        if len(payload) < 20:
            raise GGUFParseError("Unexpected end of file while reading GGUF header fields", offset=4)

        version, tensor_count, metadata_kv_count = struct.unpack("<IQQ", payload)
        if version not in GGUF_SUPPORTED_VERSIONS:
            raise InvalidGGUFError(f"Unsupported GGUF version {version}; supported are {GGUF_SUPPORTED_VERSIONS}", file_path=str(self.path))

        return GGUFHeader(
            magic=raw_header,
            version=version,
            tensor_count=tensor_count,
            metadata_kv_count=metadata_kv_count,
        )

    def _read_string(self, stream: BinaryIO) -> str:
        """Read 64-bit length-prefixed UTF-8 string from stream.

        Args:
            stream: Open binary stream.

        Returns:
            Decoded Unicode string.
        """
        len_bytes = stream.read(8)
        if len(len_bytes) < 8:
            raise GGUFParseError("Unexpected EOF while reading string length prefix", stream.tell())
        length = struct.unpack("<Q", len_bytes)[0]
        string_bytes = _read_declared(stream, length, "Truncated string")
        return string_bytes.decode("utf-8", errors="replace")

    def _read_value(self, stream: BinaryIO, value_type: GGUFValueType, depth: int = 0) -> Any:
        """Deserialize typed scalar or composite value from binary stream.

        Args:
            stream: Open binary stream.
            value_type: Descriptor indicating binary wire format.

        Returns:
            Deserialized Python primitive, string, or list.
        """
        if value_type == GGUFValueType.UINT8:
            b = stream.read(1)
            if not b:
                raise GGUFParseError("EOF reading UINT8", stream.tell())
            return struct.unpack("<B", b)[0]
        if value_type == GGUFValueType.INT8:
            b = stream.read(1)
            if not b:
                raise GGUFParseError("EOF reading INT8", stream.tell())
            return struct.unpack("<b", b)[0]
        if value_type == GGUFValueType.UINT16:
            b = stream.read(2)
            if len(b) < 2:
                raise GGUFParseError("EOF reading UINT16", stream.tell())
            return struct.unpack("<H", b)[0]
        if value_type == GGUFValueType.INT16:
            b = stream.read(2)
            if len(b) < 2:
                raise GGUFParseError("EOF reading INT16", stream.tell())
            return struct.unpack("<h", b)[0]
        if value_type == GGUFValueType.UINT32:
            b = stream.read(4)
            if len(b) < 4:
                raise GGUFParseError("EOF reading UINT32", stream.tell())
            return struct.unpack("<I", b)[0]
        if value_type == GGUFValueType.INT32:
            b = stream.read(4)
            if len(b) < 4:
                raise GGUFParseError("EOF reading INT32", stream.tell())
            return struct.unpack("<i", b)[0]
        if value_type == GGUFValueType.FLOAT32:
            b = stream.read(4)
            if len(b) < 4:
                raise GGUFParseError("EOF reading FLOAT32", stream.tell())
            return struct.unpack("<f", b)[0]
        if value_type == GGUFValueType.BOOL:
            b = stream.read(1)
            if not b:
                raise GGUFParseError("EOF reading BOOL", stream.tell())
            return bool(struct.unpack("<B", b)[0])
        if value_type == GGUFValueType.STRING:
            return self._read_string(stream)
        if value_type == GGUFValueType.UINT64:
            b = stream.read(8)
            if len(b) < 8:
                raise GGUFParseError("EOF reading UINT64", stream.tell())
            return struct.unpack("<Q", b)[0]
        if value_type == GGUFValueType.INT64:
            b = stream.read(8)
            if len(b) < 8:
                raise GGUFParseError("EOF reading INT64", stream.tell())
            return struct.unpack("<q", b)[0]
        if value_type == GGUFValueType.FLOAT64:
            b = stream.read(8)
            if len(b) < 8:
                raise GGUFParseError("EOF reading FLOAT64", stream.tell())
            return struct.unpack("<d", b)[0]
        if value_type == GGUFValueType.ARRAY:
            type_bytes = stream.read(4)
            len_bytes = stream.read(8)
            if len(type_bytes) < 4 or len(len_bytes) < 8:
                raise GGUFParseError("EOF reading ARRAY metadata", stream.tell())
            elem_type_int = struct.unpack("<I", type_bytes)[0]
            if depth >= GGUF_MAX_ARRAY_DEPTH:
                raise GGUFParseError(f"Arrays nested deeper than {GGUF_MAX_ARRAY_DEPTH} levels", stream.tell())
            elem_type = _enum_or_error(GGUFValueType, elem_type_int, "array element type", stream)
            array_len = struct.unpack("<Q", len_bytes)[0]
            self._reserve_array_elements(stream, elem_type, array_len)
            fmt = _SCALAR_STRUCT_FORMATS.get(elem_type)
            if fmt is not None:
                raw = stream.read(array_len * struct.calcsize("<" + fmt))
                return list(struct.unpack(f"<{array_len}{fmt}", raw))
            return [self._read_value(stream, elem_type, depth + 1) for _ in range(array_len)]

        raise GGUFParseError(f"Unsupported GGUF value type {value_type}", stream.tell())

    def _reserve_array_elements(self, stream: BinaryIO, elem_type: GGUFValueType, array_len: int) -> None:
        """Charge an array's declared length against the file size and the per-file element budget.

        The length is untrusted. It must be satisfiable by the bytes that remain (at the minimum
        wire size of one element) and the running total over the whole file must stay under
        GGUF_MAX_TOTAL_ARRAY_ELEMENTS. Nothing is truncated: exceeding either bound is an error.

        Args:
            stream: Open seekable binary stream positioned after the array header.
            elem_type: Declared element type.
            array_len: Declared element count.

        Raises:
            GGUFParseError: If the array cannot fit the remaining bytes or exhausts the budget.
        """
        remaining = _bytes_remaining(stream)
        needed = array_len * _MIN_ELEMENT_BYTES[elem_type]
        if needed > remaining:
            raise GGUFParseError(f"Array declares {array_len} elements needing at least {needed} bytes but only {remaining} remain", stream.tell())
        if array_len > self._array_elements_left:
            raise GGUFParseError(f"Array elements exceed the per-file limit of {GGUF_MAX_TOTAL_ARRAY_ELEMENTS}", stream.tell())
        self._array_elements_left -= array_len

    def _check_record_count(self, stream: BinaryIO, count: int, min_bytes: int, what: str) -> None:
        """Reject a declared record count that cannot fit in the remaining bytes.

        Args:
            stream: Open seekable binary stream.
            count: Count declared by the header.
            min_bytes: Minimum wire size of one record.
            what: Record kind used in the error message.

        Raises:
            GGUFParseError: If count records of min_bytes each exceed the remaining bytes.
        """
        remaining = _bytes_remaining(stream)
        if count * min_bytes > remaining:
            raise GGUFParseError(f"Declared {what} count {count} cannot fit in the {remaining} remaining bytes", stream.tell())

    def parse_metadata(self, stream: BinaryIO, count: int) -> dict[str, Any]:
        """Deserialize all key-value pairs from the metadata header block.

        Args:
            stream: Open binary stream positioned at metadata block start.
            count: Number of key-value pairs defined by header.

        Returns:
            Dictionary mapping string keys to deserialized values.
        """
        metadata: dict[str, Any] = {}
        self._array_elements_left = GGUF_MAX_TOTAL_ARRAY_ELEMENTS
        self._check_record_count(stream, count, GGUF_MIN_KV_BYTES, "metadata key-value")
        for _ in range(count):
            key = self._read_string(stream)
            type_bytes = stream.read(4)
            if len(type_bytes) < 4:
                raise GGUFParseError(f"EOF reading value type for metadata key '{key}'", stream.tell())
            val_type_int = struct.unpack("<I", type_bytes)[0]
            val_type = _enum_or_error(GGUFValueType, val_type_int, "metadata value type", stream)
            val = self._read_value(stream, val_type)
            metadata[key] = val
        return metadata

    def parse_tensor_info(self, stream: BinaryIO, count: int) -> list[GGUFTensorInfo]:
        """Decode tensor descriptor table describing shapes and byte offsets.

        Args:
            stream: Open binary stream positioned at tensor info block start.
            count: Number of tensor records to parse.

        Returns:
            List of parsed GGUFTensorInfo descriptor dataclasses.
        """
        tensor_list: list[GGUFTensorInfo] = []
        self._check_record_count(stream, count, GGUF_MIN_TENSOR_BYTES, "tensor")
        for _ in range(count):
            name = self._read_string(stream)
            dims_bytes = stream.read(4)
            if len(dims_bytes) < 4:
                raise GGUFParseError(f"EOF reading dimensions count for tensor '{name}'", stream.tell())
            n_dims = struct.unpack("<I", dims_bytes)[0]
            if n_dims > GGUF_MAX_DIMENSIONS:
                raise GGUFParseError(f"Tensor '{name}' declares {n_dims} dimensions, maximum is {GGUF_MAX_DIMENSIONS}", stream.tell())
            dims_raw = _read_declared(stream, 8 * n_dims, f"EOF reading dimension array for tensor '{name}'")
            dimensions = struct.unpack(f"<{n_dims}Q", dims_raw)

            type_bytes = stream.read(4)
            offset_bytes = stream.read(8)
            if len(type_bytes) < 4 or len(offset_bytes) < 8:
                raise GGUFParseError(f"EOF reading type and offset for tensor '{name}'", stream.tell())
            tensor_type_int = struct.unpack("<I", type_bytes)[0]
            tensor_type = _enum_or_error(GGMLType, tensor_type_int, "tensor type", stream)
            offset = struct.unpack("<Q", offset_bytes)[0]

            tensor_list.append(
                GGUFTensorInfo(
                    name=name,
                    n_dimensions=n_dims,
                    dimensions=dimensions,
                    tensor_type=tensor_type,
                    offset=offset,
                )
            )
        return tensor_list

    def parse_all(self) -> None:
        """Perform full parse of container header, metadata, and tensor info table."""
        if not self.path.exists():
            raise InvalidGGUFError(f"File not found: {self.path}", file_path=str(self.path))
        if not self.path.is_file():
            raise InvalidGGUFError(f"Not a regular file: {self.path}", file_path=str(self.path))

        with open(self.path, "rb") as f:
            self._header = self.parse_header(f)
            self._metadata = self.parse_metadata(f, self._header.metadata_kv_count)
            self._tensor_info = self.parse_tensor_info(f, self._header.tensor_count)

            alignment = self._metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
            if isinstance(alignment, bool) or not isinstance(alignment, int) or not 1 <= alignment <= GGUF_MAX_ALIGNMENT or alignment & (alignment - 1):
                raise GGUFParseError(f"general.alignment must be a power of two in [1, {GGUF_MAX_ALIGNMENT}], got {alignment!r}", f.tell())
            current_pos = f.tell()
            remainder = current_pos % alignment
            self._data_offset = current_pos if remainder == 0 else current_pos + (alignment - remainder)

    def _int_metadata(self, key: str, default: Any) -> int:
        """Read a metadata value as an integer, raising a typed error for hostile values.

        Args:
            key: Metadata key.
            default: Value used when the key is absent.

        Returns:
            int: The integer value.

        Raises:
            GGUFParseError: If the value is a string, array, NaN, infinity or otherwise not integral.
        """
        assert self._metadata is not None
        value = self._metadata.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GGUFParseError(f"Metadata '{key}' must be a number, got {type(value).__name__}", 0)
        try:
            return int(value)
        except (ValueError, OverflowError):
            raise GGUFParseError(f"Metadata '{key}' is not a finite number: {value!r}", 0) from None

    def get_summary(self) -> GGUFSummary:
        """Construct comprehensive architectural and quantization summary.

        Returns:
            GGUFSummary populated with parameters, architecture, and quant distribution.
        """
        self.parse_all()
        assert self._header is not None
        assert self._metadata is not None
        assert self._tensor_info is not None

        architecture = str(self._metadata.get("general.architecture", "unknown"))
        context_length = self._int_metadata(f"{architecture}.context_length", self._metadata.get("general.context_length", 4096))
        embedding_length = self._int_metadata(f"{architecture}.embedding_length", 0)
        block_count = self._int_metadata(f"{architecture}.block_count", 0)
        head_count = self._int_metadata(f"{architecture}.attention.head_count", 0)

        total_parameters = 0
        type_counts: Counter[str] = Counter()

        for t in self._tensor_info:
            param_count = 1
            for d in t.dimensions:
                param_count *= d
            total_parameters += param_count
            type_counts[t.tensor_type.name] += 1

        total_tensors = len(self._tensor_info)
        distribution: dict[str, float] = {}
        primary_quant = "UNKNOWN"

        if total_tensors > 0:
            for type_name, count in type_counts.items():
                distribution[type_name] = round((count / total_tensors) * 100.0, 2)
            primary_quant = type_counts.most_common(1)[0][0]

        file_size = self.path.stat().st_size if self.path.exists() else 0

        return GGUFSummary(
            file_path=str(self.path),
            file_size_bytes=file_size,
            architecture=architecture,
            context_length=context_length,
            embedding_length=embedding_length,
            block_count=block_count,
            head_count=head_count,
            tensor_count=self._header.tensor_count,
            total_parameters=total_parameters,
            quant_distribution=distribution,
            primary_quant_type=primary_quant,
        )
