import os, sys
import numpy as np
from rknn.api import RKNN

def conv(onnx, name, dataset=None, quant=True):
    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk3588",
                quantized_method="channel",
                quantized_algorithm="normal",
                optimization_level=3)
    ret = rknn.load_onnx(model=onnx)
    assert ret == 0, f"load_onnx {onnx} -> {ret}"
    ret = rknn.build(do_quantization=quant, dataset=dataset)
    assert ret == 0, f"build {name} -> {ret}"
    out = f"{name}.rknn"
    ret = rknn.export_rknn(out)
    assert ret == 0, f"export {out} -> {ret}"
    rknn.release()
    print("=== built", out, "quant" if quant else "fp16")

conv("bn_only.onnx", "bn_only_i8", dataset="ds_bn.txt", quant=True)
conv("bn_only.onnx", "bn_only_fp16", quant=False)
conv("conv_bn_conv.onnx", "conv_bn_conv_i8", dataset="ds_cbc.txt", quant=True)
conv("conv_bn_conv.onnx", "conv_bn_conv_fp16", quant=False)
print("ALL DONE")
