"""A tiny hand-built ONNX model with the same interface as the exported descriptor.

Input ``pixel_values`` [batch,3,S,S], output ``embedding`` [batch,D],
normalised to length 1. This lets us test the ONNX Runtime code without
torch, the ``onnx`` package or downloading a model. The model just does:
AveragePool to a G x G grid -> Flatten -> LpNormalization. It is NOT
DINOv2, so any accuracy you get with it means nothing; it only checks that
the pieces fit together.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path


def _varint(n: int) -> bytes:
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field: int, wt: int) -> bytes:
    return _varint((field << 3) | wt)


def _int(field: int, v: int) -> bytes:
    return _key(field, 0) + _varint(v)


def _bytes(field: int, b: bytes) -> bytes:
    return _key(field, 2) + _varint(len(b)) + b


def _str(field: int, s: str) -> bytes:
    return _bytes(field, s.encode())


def _attr_int(name: str, v: int) -> bytes:
    return _str(1, name) + _int(20, 2) + _int(3, v)  # attribute type INT


def _attr_ints(name: str, vs) -> bytes:
    return _str(1, name) + _int(20, 7) + b"".join(_int(8, v) for v in vs)  # attribute type INTS


def _node(op: str, inputs, outputs, attrs=()) -> bytes:
    b = b"".join(_str(1, i) for i in inputs) + b"".join(_str(2, o) for o in outputs)
    b += _str(4, op) + b"".join(_bytes(5, a) for a in attrs)
    return b


def _value_info(name: str, dims) -> bytes:
    shape = b"".join(_bytes(1, _str(2, d) if isinstance(d, str) else _int(1, d)) for d in dims)
    tensor = _int(1, 1) + _bytes(2, shape)  # data type FLOAT
    return _str(1, name) + _bytes(2, _bytes(1, tensor))


def toy_model_bytes(size: int = 224, grid: int = 7) -> bytes:
    k = size // grid
    nodes = [
        _node("AveragePool", ["pixel_values"], ["pooled"],
              [_attr_ints("kernel_shape", [k, k]), _attr_ints("strides", [k, k])]),
        _node("Flatten", ["pooled"], ["flat"], [_attr_int("axis", 1)]),
        _node("LpNormalization", ["flat"], ["embedding"],
              [_attr_int("axis", 1), _attr_int("p", 2)]),
    ]
    graph = b"".join(_bytes(1, n) for n in nodes) + _str(2, "toy_descriptor")
    graph += _bytes(11, _value_info("pixel_values", ["batch", 3, size, size]))
    graph += _bytes(12, _value_info("embedding", ["batch", 3 * grid * grid]))
    opset = _str(1, "") + _int(2, 17)
    return _int(1, 8) + _str(2, "vision-tests") + _bytes(7, graph) + _bytes(8, opset)


def write_toy_model(path: str | Path, size: int = 224, grid: int = 7) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(toy_model_bytes(size, grid))
    p.with_suffix(".json").write_text(json.dumps({
        "model_version": f"toy-grid{grid}:avgpool:{size}:fp32", "size": size,
        "pooling": "avgpool", "clahe": False}), encoding="utf-8")
    return p
