"""Independent GGUF container writer used by the test_ind_* files.

Written from the public GGUF layout described in the parser docstrings (little-endian, 64-bit length
prefixed strings, header = magic + u32 version + u64 tensor count + u64 metadata count).
"""

import struct
from typing import Any, List, Sequence, Tuple

U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)

_FMT = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", F32: "<f", U64: "<Q", I64: "<q", F64: "<d"}


def pack_string(text: str) -> bytes:
    """Return a GGUF string (u64 length + bytes)."""
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def pack_value(vtype: int, value: Any) -> bytes:
    """Return the wire bytes of one typed value; arrays are (elem_type, list)."""
    if vtype == STR:
        return pack_string(value)
    if vtype == BOOL:
        return struct.pack("<B", 1 if value else 0)
    if vtype == ARR:
        elem_type, items = value
        return struct.pack("<IQ", elem_type, len(items)) + b"".join(pack_value(elem_type, v) for v in items)
    return struct.pack(_FMT[vtype], value)


def build_gguf(
    metadata: Sequence[Tuple[str, int, Any]],
    tensors: Sequence[Tuple[str, Sequence[int], int, int]],
    payload: bytes = b"",
    alignment: int = 32,
    version: int = 3,
) -> Tuple[bytes, int]:
    """Build a GGUF file; returns (bytes, end_of_tensor_info_offset)."""
    buf = bytearray(b"GGUF")
    buf += struct.pack("<IQQ", version, len(tensors), len(metadata))
    for key, vtype, value in metadata:
        buf += pack_string(key) + struct.pack("<I", vtype) + pack_value(vtype, value)
    for name, dims, ggml_type, offset in tensors:
        buf += pack_string(name) + struct.pack("<I", len(dims))
        for d in dims:
            buf += struct.pack("<Q", d)
        buf += struct.pack("<IQ", ggml_type, offset)
    info_end = len(buf)
    pad = (-len(buf)) % alignment
    buf += b"\x00" * pad
    buf += payload
    return bytes(buf), info_end


def layered_f32_model(num_layers: int, hidden: int, seed: int = 0) -> Tuple[bytes, List[bytes]]:
    """Build a GGUF with `num_layers` layers of two F32 tensors each plus per-layer distinct bytes.

    Returns:
        (file bytes, per-layer payload bytes in tensor order concatenated)
    """
    import random

    rng = random.Random(seed)
    metadata = [
        ("general.architecture", STR, "llama"),
        ("general.alignment", U32, 32),
        ("llama.block_count", U32, num_layers),
    ]
    tensors = []
    payload = bytearray()
    per_layer: List[bytes] = []
    for layer in range(num_layers):
        layer_bytes = bytearray()
        for name in ("attn_q.weight", "ffn_up.weight"):
            n = hidden * 2
            raw = bytes(rng.randrange(256) for _ in range(n * 4))
            tensors.append((f"blk.{layer}.{name}", (hidden, 2), 0, len(payload)))
            payload += raw
            layer_bytes += raw
        per_layer.append(bytes(layer_bytes))
    blob, _ = build_gguf(metadata, tensors, bytes(payload))
    return blob, per_layer
