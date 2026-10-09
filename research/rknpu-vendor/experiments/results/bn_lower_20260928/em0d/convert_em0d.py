import numpy as np
from rknn.api import RKNN

rng = np.random.default_rng(555)
for i in range(4):
    x = rng.integers(-40, 41, size=(1, 8, 8, 8)).astype(np.float32) / 8.0
    np.save(f"calem_{i}.npy", x)
with open("ds_em.txt", "w") as f:
    for i in range(4):
        f.write(f"calem_{i}.npy\n")

def conv(onnx, name, quant=True):
    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk3588", quantized_method="channel", quantized_algorithm="normal", optimization_level=3)
    assert rknn.load_onnx(model=onnx) == 0
    assert rknn.build(do_quantization=quant, dataset="ds_em.txt" if quant else None) == 0
    assert rknn.export_rknn(f"{name}.rknn") == 0
    rknn.release()
    print("built", name)

conv("mul_only.onnx", "mul_only_i8")
conv("mul_only.onnx", "mul_only_fp16", quant=False)
conv("transpose_only.onnx", "transpose_only_fp16", quant=False)
conv("layernorm_only.onnx", "layernorm_only_fp16", quant=False)
print("ALL DONE")
