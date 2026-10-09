import numpy as np, struct, json, sys

s_in, zp_out, s_out = 0.0392156877, 11, 0.106179431
b = open("bos/bn_only_i8/01.bin", "rb").read()
mul = np.array(struct.unpack_from("<8h", b, 0), np.int64)
bias = np.array(struct.unpack_from("<8i", b, 64), np.int64)
xin = np.fromfile("in_bn_only.bin", np.float32).reshape(1, 8, 8, 8)
out = np.fromfile("out_bn_only_i8.bin", np.float32).reshape(1, 8, 8, 8)
total = ok = 0
for c in range(8):
    xq = np.round(xin[0, :, :, c] / s_in).astype(np.int64).ravel()
    yint = np.round(out[0, c, :, :].ravel() / s_out).astype(np.int64)
    acc = xq * mul[c] + bias[c]
    pred = (acc * 30256 + (1 << 29)) >> 30
    ok += int((pred == yint).sum()); total += 64
    print(f"ch{c}: mul={mul[c]:6d} bias={bias[c]:8d} exact={int((pred==yint).sum())}/64")
print(f"TOTAL {ok}/{total}")
sys.exit(0 if ok == total else 1)
