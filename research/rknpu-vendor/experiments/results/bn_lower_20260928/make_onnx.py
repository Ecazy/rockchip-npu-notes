import numpy as np
import onnx
from onnx import helper, TensorProto

def save(path, model):
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print("saved", path)

C = 8
gamma = np.array([1.0, 2.0, 0.5, 1.25, -1.0, 0.75, 1.5, -0.25], dtype=np.float32)
beta = np.array([0.0, 1.5, -2.0, 0.25, 4.0, -1.5, 2.25, 3.0], dtype=np.float32)
mean = np.array([0.0, 0.5, -0.25, 1.0, 0.0, -0.5, 0.25, 1.5], dtype=np.float32)
var = np.array([1.0, 2.0, 4.0, 0.25, 1.0, 2.25, 0.5, 1.0], dtype=np.float32)
eps = 1e-5

def bn_model(cin):
    node = helper.make_node("BatchNormalization", ["x", "g", "b", "m", "v"], ["y"], epsilon=eps)
    graph = helper.make_graph(
        [node], "bn_only",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, cin, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, cin, 8, 8])],
        [
            helper.make_tensor("g", TensorProto.FLOAT, [cin], gamma[:cin].tolist()),
            helper.make_tensor("b", TensorProto.FLOAT, [cin], beta[:cin].tolist()),
            helper.make_tensor("m", TensorProto.FLOAT, [cin], mean[:cin].tolist()),
            helper.make_tensor("v", TensorProto.FLOAT, [cin], var[:cin].tolist()),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="bnprobe")

save("bn_only.onnx", bn_model(C))

# Conv1x1 -> BN -> Conv1x1 sandwich, C=8 hidden, distinct int weights
rng = np.random.default_rng(7)
w0 = rng.integers(-3, 4, size=(8, 3, 1, 1)).astype(np.float32)
b0 = rng.integers(-2, 3, size=(8,)).astype(np.float32)
w1 = rng.integers(-3, 4, size=(4, 8, 1, 1)).astype(np.float32)
b1 = rng.integers(-2, 3, size=(4,)).astype(np.float32)

def conv(name, w, b, ins, outs):
    return helper.make_node("Conv", ins, outs, name=name, kernel_shape=[1, 1], strides=[1, 1])

n1 = conv("c0", w0, b0, ["x", "w0", "b0"], ["t0"])
n2 = helper.make_node("BatchNormalization", ["t0", "g", "b", "m", "v"], ["t1"], name="bn0", epsilon=eps)
n3 = conv("c1", w1, b1, ["t1", "w1", "b1"], ["y"])

inits = [
    helper.make_tensor("g", TensorProto.FLOAT, [8], gamma.tolist()),
    helper.make_tensor("b", TensorProto.FLOAT, [8], beta.tolist()),
    helper.make_tensor("m", TensorProto.FLOAT, [8], mean.tolist()),
    helper.make_tensor("v", TensorProto.FLOAT, [8], var.tolist()),
    helper.make_tensor("w0", TensorProto.FLOAT, w0.shape, w0.flatten().tolist()),
    helper.make_tensor("b0", TensorProto.FLOAT, [8], b0.tolist()),
    helper.make_tensor("w1", TensorProto.FLOAT, w1.shape, w1.flatten().tolist()),
    helper.make_tensor("b1", TensorProto.FLOAT, [4], b1.tolist()),
]
graph = helper.make_graph(
    [n1, n2, n3], "conv_bn_conv",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])],
    inits,
)
save("conv_bn_conv.onnx", helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="bnprobe"))

np.savez("bn_params.npz", gamma=gamma, beta=beta, mean=mean, var=var, eps=np.float32(eps))

# calibration data
rng = np.random.default_rng(42)
for i, c in enumerate([3, 4, 8]):
    x = rng.integers(-40, 41, size=(1, c, 8, 8)).astype(np.float32) / 8.0
    np.save(f"cal_{c}ch_{i}.npy", x)
rng2 = np.random.default_rng(99)
for i, c in enumerate([3, 4, 8]):
    x = rng2.integers(-40, 41, size=(1, c, 8, 8)).astype(np.float32) / 8.0
    np.save(f"cal2_{c}ch_{i}.npy", x)
print("calibration data written")
