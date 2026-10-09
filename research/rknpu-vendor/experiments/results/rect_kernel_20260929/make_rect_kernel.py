import numpy as np, onnx
import onnx.mapping
from onnx import helper, TensorProto

def make_conv(name, kw, kh, iw=16, ih=16, ic=3, oc=16):
    np.random.seed(11)
    W = (np.random.randint(-8, 8, (oc, ic, kh, kw))).astype(np.float32)
    B = np.zeros(oc, dtype=np.float32)
    node = onnx.helper.make_node("Conv", ["in", "w", "b"], ["out"], name="c",
                                 kernel_shape=[kh, kw], strides=[1, 1], pads=[0, 0, 0, 0])
    g = onnx.helper.make_graph([node], name,
        [onnx.helper.make_tensor_value_info("in", TensorProto.FLOAT, [1, ic, ih, iw])],
        [onnx.helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, oc, ih - kh + 1, iw - kw + 1])],
        initializer=[onnx.numpy_helper.from_array(W, "w"),
                     onnx.numpy_helper.from_array(B, "b")])
    m = onnx.helper.make_model(g, opset_imports=[onnx.helper.make_opsetid("", 13)])
    onnx.save(m, f"/tmp/opencode/rect_kernel/{name}.onnx")
    print("saved", name)
    return name

os_dir = "/tmp/opencode/rect_kernel"
import os
os.makedirs(os_dir, exist_ok=True)

import sys
for name, kw, kh in [("rect21", 2, 1), ("rect12", 1, 2)]:
    n = make_conv(name, kw, kh)
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    from rknn.api import RKNN
    r = RKNN(verbose=False)
    r.config(mean_values=[[0] * 3], std_values=[[255] * 3], target_platform="rk3588",
             quantized_dtype="w8a8", quantized_algorithm="normal")
    if r.load_onnx(f"{os_dir}/{n}.onnx") != 0:
        raise SystemExit("load fail")
    np.random.seed(3)
    calib = np.random.randint(0, 256, (1, 3, 16, 16)).astype(np.float32)
    np.save(f"{os_dir}/{n}_cal.npy", calib)
    with open(f"{os_dir}/{n}_cal.txt", "w") as f:
        f.write(f"{os_dir}/{n}_cal.npy\n")
    if r.build(do_quantization=True, dataset=f"{os_dir}/{n}_cal.txt") != 0:
        raise SystemExit("build fail")
    r.export_rknn(f"{os_dir}/{n}_i8.rknn")
    print("EXPORT OK", n)

# float32 runner inputs (NCHW 0..255)
for n in ["rect21", "rect12"]:
    np.random.seed(5)
    x = np.random.randint(0, 256, (1, 3, 16, 16)).astype(np.float32)
    x.tofile(f"{os_dir}/{n}_in.bin")
    print("input", n, x.shape)
