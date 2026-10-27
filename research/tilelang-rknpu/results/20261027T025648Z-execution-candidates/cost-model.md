# Execution cost model

The model uses the generic profile ledger: C workspace, pack/download, submit, guard, Host actions and executor. Prepared residency removes the measured Python plan-node surface (5% residual), adds one 8 GB/s resident write-back per Host output byte, and leaves C transfers/submits unchanged.

| profile | binding slots | auto choice | generic ms | prepared ms | measured saving ms | modeled saving ms | error ms | host-fusion model ms | device-QKV screen ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| mnist_mlp.b32 | 13 | generic | 2.152 | 2.187 | -0.036 | 0.123 | +0.158 | 0.000 | 0.000 |
| mnist_mlp.b8 | 13 | generic | 1.271 | 1.368 | -0.097 | 0.075 | +0.172 | 0.000 | 0.000 |
| mnist_mlp.b1 | 13 | generic | 1.233 | 1.293 | -0.060 | 0.077 | +0.137 | 0.000 | 0.000 |
| tiny_vit.b8 | 210 | prepared | 10.238 | 7.442 | 2.796 | 2.795 | -0.001 | 0.007 | 0.377 |
| tiny_vit.b1 | 42 | prepared | 3.075 | 2.499 | 0.577 | 0.601 | +0.024 | 0.002 | 0.049 |

Notes:

- Tiny ViT binding surfaces (42 and 210 slots) are above the 32-slot cut used by these samples; it is an empirical selector, not a proven general performance crossover. MLP has 13 slots, where the modeled prepared gain is at or below observed noise and the measured campaign reverses its sign; auto keeps it on the generic path.
- The Host-fusion control removes only a boundary, not a submit or transfer. Its modeled saving is a few microseconds and the campaign shows no stable additive gain.
- The device-QKV screen is a coefficient-based transfer estimate, not a strict upper bound. It excludes device reorder cost, does not reduce the current Host RKPlan submit calls and still leaves Host Softmax between QK and PV. Missing binding/layout templates are listed in each profile JSON.
- For Tiny ViT b8 the screen removes 49,152 bytes of QKV download and 163,840 bytes of QKV-attributable QK/PV upload, yet estimates only about 0.38 ms. This is not a strict upper bound and does not explain the vendor gap on its own.
