import numpy as np

rng = np.random.default_rng(1234)

# bn_only: C=8
x8 = rng.integers(-40, 41, size=(1, 8, 8, 8)).astype(np.float32) / 8.0
x8.astype(np.float32).tofile("in_bn_only.bin")
# conv_bn_conv: C=3
x3 = rng.integers(-40, 41, size=(1, 3, 8, 8)).astype(np.float32) / 8.0
x3.astype(np.float32).tofile("in_cbc.bin")

p = np.load("bn_params.npz")
gamma, beta, mean, var, eps = p["gamma"], p["beta"], p["mean"], p["var"], float(p["eps"])

def bn(x):
    return gamma[None, :, None, None] * (x - mean[None, :, None, None]) / np.sqrt(var[None, :, None, None] + eps) + beta[None, :, None, None]

# references
y = bn(x8)
y.astype(np.float32).tofile("ref_bn_only.bin")
np.save("ref_bn_only.npy", y)

w0 = np.array([[[-2.]], [[1.]], [[3.]], [[-1.]], [[2.]], [[0.]], [[-3.]], [[1.]]])  # placeholder replaced below
# rebuild conv weights deterministically same as make_onnx.py (rng seeded 7 after bn_model save)
rng2 = np.random.default_rng(7)
w0 = rng2.integers(-3, 4, size=(8, 3, 1, 1)).astype(np.float32)
b0 = rng2.integers(-2, 3, size=(8,)).astype(np.float32)
w1 = rng2.integers(-3, 4, size=(4, 8, 1, 1)).astype(np.float32)
b1 = rng2.integers(-2, 3, size=(4,)).astype(np.float32)

t0 = w0[:, :, 0, 0][None] @ x3.reshape(1, 3, 64).transpose(0, 2, 1)[0].T  # awkward; do direct
t0 = np.zeros((1, 8, 8, 8), np.float32)
xh = x3.reshape(3, 64)  # CHW
t0f = (w0[:, :, 0, 0] @ xh) + b0[:, None]
t1 = gamma[:, None] * (t0f - mean[:, None]) / np.sqrt(var[:, None] + eps) + beta[:, None]
t2 = (w1[:, :, 0, 0] @ t1) + b1[:, None]
ref = t2.reshape(1, 4, 8, 8)
ref.astype(np.float32).tofile("ref_cbc.bin")
np.save("ref_cbc.npy", ref)
print("inputs + refs written")
print("bn_only ref[0,:4]:", y.flatten()[:4])
print("cbc ref[0,:4]:", ref.flatten()[:4])
