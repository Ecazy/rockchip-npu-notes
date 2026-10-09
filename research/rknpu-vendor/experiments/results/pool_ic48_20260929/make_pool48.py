import numpy as np, onnx
import onnx.mapping
from onnx import helper, TensorProto

import os
os.makedirs("/tmp/opencode/pool_ic48", exist_ok=True)

def make_pool(name, c, hw=64):
    node = onnx.helper.make_node("MaxPool", ["in"], ["out"], name="p",
                                 kernel_shape=[2, 2], strides=[2, 2])
    g = onnx.helper.make_graph([node], name,
        [onnx.helper.make_tensor_value_info("in", TensorProto.FLOAT, [1, c, hw, hw])],
        [onnx.helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, c, hw // 2, hw // 2])])
    m = onnx.helper.make_model(g, opset_imports=[onnx.helper.make_opsetid("", 13)])
    onnx.save(m, f"/tmp/opencode/pool_ic48/{name}.onnx")
    print("saved", name)

make_pool("maxpool_ic48", 48, 64)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from rknn.api import RKNN
r = RKNN(verbose=False)
r.config(mean_values=[[0] * 48], std_values=[[255] * 48], target_platform="rk3588",
         quantized_dtype="w8a8", quantized_algorithm="normal")
if r.load_onnx("/tmp/opencode/pool_ic48/maxpool_ic48.onnx") != 0:
    raise SystemExit("load fail")
np.random.seed(3)
calib = np.random.randint(0, 256, (1, 48, 64, 64)).astype(np.float32)
np.save("/tmp/opencode/pool_ic48/cal.npy", calib)
with open("/tmp/opencode/pool_ic48/cal.txt", "w") as f:
    f.write("/tmp/opencode/pool_ic48/cal.npy\n")
if r.build(do_quantization=True, dataset="/tmp/opencode/pool_ic48/cal.txt") != 0:
    raise SystemExit("build fail")
r.export_rknn("/tmp/opencode/pool_ic48/maxpool_ic48_i8.rknn")
print("EXPORT OK")
