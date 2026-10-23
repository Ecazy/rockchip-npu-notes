"""rknn-toolkit2 2.3.2 calls ``onnx.mapping`` which onnx>=1.14 removed.

Installing this module recreates the legacy tables so the ONNX frontend can
resolve tensor element types.
"""
import sys
import types

import numpy as np
import onnx
from onnx import TensorProto

_MAP = {
    TensorProto.FLOAT: np.dtype("float32"),
    TensorProto.UINT8: np.dtype("uint8"),
    TensorProto.INT8: np.dtype("int8"),
    TensorProto.UINT16: np.dtype("uint16"),
    TensorProto.INT16: np.dtype("int16"),
    TensorProto.INT32: np.dtype("int32"),
    TensorProto.INT64: np.dtype("int64"),
    TensorProto.BOOL: np.dtype("bool"),
    TensorProto.FLOAT16: np.dtype("float16"),
    TensorProto.DOUBLE: np.dtype("float64"),
    TensorProto.UINT32: np.dtype("uint32"),
    TensorProto.UINT64: np.dtype("uint64"),
    TensorProto.COMPLEX64: np.dtype("complex64"),
    TensorProto.COMPLEX128: np.dtype("complex128"),
}


def install() -> None:
    if hasattr(onnx, "mapping"):
        return
    module = types.ModuleType("onnx.mapping")
    module.TENSOR_TYPE_TO_NP_TYPE = _MAP
    module.NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in _MAP.items()}
    module.TENSOR_TYPE_TO_ONNX_TYPE = _MAP
    module.ONNX_TYPE_TO_NP_TYPE = _MAP
    onnx.mapping = module
    sys.modules["onnx.mapping"] = module
