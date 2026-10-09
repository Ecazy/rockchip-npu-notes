import numpy as np
import onnx
from onnx import helper, TensorProto

def save(path, model):
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print("saved", path)

# 1) Mul: y = x * scalar(1.5)  (elementwise, same shape)
g = helper.make_graph(
    [helper.make_node("Mul", ["x", "s"], ["y"])],
    "mul_only",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 8, 8])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 8, 8])],
    [helper.make_tensor("s", TensorProto.FLOAT, [], [1.5])],
)
save("mul_only.onnx", helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]))

# 2) Transpose: (1,8,8,8) NCHW -> (1,8,8,8) with perm (0,2,3,1)
g = helper.make_graph(
    [helper.make_node("Transpose", ["x"], ["y"], perm=[0, 2, 3, 1])],
    "transpose_only",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 8, 8])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 8, 8])],
)
save("transpose_only.onnx", helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], producer_name="bnprobe"))

# 3) LayerNorm: axis=-1 over 8, gamma/beta distinctive
g = helper.make_graph(
    [helper.make_node("LayerNormalization", ["x", "g", "b"], ["y"], axis=-1, epsilon=1e-5)],
    "layernorm_only",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 8, 8])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 8, 8])],
    [
        helper.make_tensor("g", TensorProto.FLOAT, [8], [1.0, 2.0, 0.5, 1.25, -1.0, 0.75, 1.5, -0.25]),
        helper.make_tensor("b", TensorProto.FLOAT, [8], [0.0, 1.5, -2.0, 0.25, 4.0, -1.5, 2.25, 3.0]),
    ],
)
save("layernorm_only.onnx", helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], producer_name="bnprobe"))
print("done")
