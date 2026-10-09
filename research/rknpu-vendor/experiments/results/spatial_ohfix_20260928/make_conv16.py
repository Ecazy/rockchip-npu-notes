import numpy as np, onnx
import onnx.mapping
from onnx import helper, TensorProto

def make_conv(name, iw, ih):
    np.random.seed(7)
    W = (np.random.randint(-8, 8, (4, 3, 3, 3))).astype(np.float32)
    B = np.arange(4, dtype=np.float32) * 0.5
    node = onnx.helper.make_node("Conv", ["in", "w", "b"], ["out"], name="c",
                                 kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
    g = onnx.helper.make_graph([node], name,
        [onnx.helper.make_tensor_value_info("in", TensorProto.FLOAT, [1, 3, ih, iw])],
        [onnx.helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, ih, iw])],
        initializer=[onnx.numpy_helper.from_array(W, "w"),
                     onnx.numpy_helper.from_array(B, "b")])
    m = onnx.helper.make_model(g, opset_imports=[onnx.helper.make_opsetid("", 13)])
    onnx.save(m, f"/tmp/opencode/bn_probe/{name}.onnx")
    print("saved", name)

make_conv("conv16", 16, 16)

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from rknn.api import RKNN
r = RKNN(verbose=False)
r.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588",
         quantized_dtype="w8a8", quantized_algorithm="normal")
if r.load_onnx("/tmp/opencode/bn_probe/conv16.onnx") != 0: raise SystemExit("load fail")
np.random.seed(3)
calib = np.random.randint(0, 256, (1, 3, 16, 16)).astype(np.float32)
r.build(do_quantization=True, dataset="/tmp/opencode/bn_probe/conv16_cal.txt") if False else None
# calibration via dataset file
with open("/tmp/opencode/bn_probe/conv16_cal.txt", "w") as f:
    f.write("/tmp/opencode/bn_probe/conv16_cal.npy\n")
np.save("/tmp/opencode/bn_probe/conv16_cal.npy", calib)
if r.build(do_quantization=True, dataset="/tmp/opencode/bn_probe/conv16_cal.txt") != 0:
    raise SystemExit("build fail")
r.export_rknn("/tmp/opencode/bn_probe/conv16_i8.rknn")
print("EXPORT OK")
