# Graph-execution prototype

- status: **PASS**
- full test images: 10000
- rounds: 3
- application timing: `runner.run(..., stats=None)`
- instrumentation: separate `RunStats` sample, not in app interval
- CPU governor: `performance`, pinned CPU 6 at 2.304 GHz
- NPU: one core at 1.0 GHz

## Same numeric policy — Tiny ViT

| profile | generic ms | prepared ms | auto graph ms | auto/generic | auto/prepared | tasks/submits | view/skip |
|---|---:|---:|---:|---:|---:|---|---|
| tiny_vit.b8 | 9.219 | 6.371 | 6.072 | 1.518x | 1.049x | 70/8 | 1/2 |
| tiny_vit.b1 | 2.875 | 2.232 | 2.088 | 1.377x | 1.069x | 14/8 | 1/2 |

Auto executes the compiled flat graph for the high-binding-fanout ViT
profiles. It removes the metadata reshape action, skips two production
writes of trace-only `gate/up` values, and preserves all 8 RKPlan submits,
transfers and device tasks. Generic/prepared are same-policy controls.

## Application and instrumentation

| profile | arm | app ms/call | instrumented ms/call | Host actions ms | C workspace ms | executor ms | setup ms | resident B |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tiny_vit.b8 | generic | 9.219 | 9.471 | 1.963 | 4.095 | 0.664 | 19.114 | 0 |
| tiny_vit.b8 | prepared | 6.371 | 6.617 | 1.876 | 3.985 | 0.593 | 21.204 | 505664 |
| tiny_vit.b8 | auto | 6.072 | 6.375 | 1.674 | 3.970 | 0.567 | 21.535 | 505664 |
| tiny_vit.b1 | generic | 2.875 | 3.242 | 0.696 | 1.367 | 0.511 | 10.377 | 0 |
| tiny_vit.b1 | prepared | 2.232 | 2.543 | 0.701 | 1.268 | 0.462 | 9.321 | 63208 |
| tiny_vit.b1 | auto | 2.088 | 2.425 | 0.586 | 1.273 | 0.454 | 9.642 | 63208 |

## External references (separate numeric policies)

| profile | auto graph ms | vendor ms | CPU FP32 ms | vendor/auto | CPU/auto | vendor max err | packed atol |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny_vit.b8 | 6.072 | 2.907 | 5.847 | 2.089x | 1.038x | 0.12277 | 0.0625 |
| tiny_vit.b1 | 2.088 | 0.795 | 0.869 | 2.627x | 2.402x | 0.10715 | 0.0625 |

Vendor uses non-quantized FP16 with graph fusion and does not satisfy the
packed per-logit tolerance on ViT. CPU is FP32 reference math. Their times
are references, not a same-policy ranking. The vendor one-call runtime API
does not reveal its internal device Submit count.

## Independent numeric-policy candidate — device FFN

| profile | same-policy auto ms | device FFN ms | device/policy | max logit delta | tasks/submits | upload/download B |
|---|---:|---:|---:|---:|---|---|
| tiny_vit.b8 | 5.962 | 5.709 | 0.958x | 0.03125 | 75/12 | 280064/164608 |
| tiny_vit.b1 | 1.923 | 2.059 | 1.071x | 0.03125 | 19/12 | 36352/21248 |

Device FFN uses the existing independent numeric policy; it is not a drop-in packed-logit path. It removes Host SwiGLU/residual work and bridge bytes but adds device tasks and submits. The full-test probe shows a b8 gain and a b1 regression, so it is not included in the default auto path. Adoption would require an explicit numeric-policy decision and a uniform per-profile acceptance story.

## Host boundary ledger

| profile | Host actions | Host total ms | provably removed now ms | view aliases | trace-only tensors |
|---|---:|---:|---:|---:|---:|
| mnist_mlp.b1 | 3 | 0.122 | 0.000 | 0 | 0 |
| mnist_mlp.b32 | 3 | 0.415 | 0.000 | 0 | 0 |
| mnist_mlp.b8 | 3 | 0.184 | 0.000 | 0 | 0 |
| tiny_vit.b1 | 13 | 0.701 | 0.048 | 1 | 5 |
| tiny_vit.b8 | 13 | 1.876 | 0.089 | 1 | 5 |

The Host total is observed action dwell, not a recoverable bound.
Numeric Host arithmetic is retained; only the metadata reshape is counted
as directly removable in the ledger. The flat executor's measured gain
also includes eliminating per-node Python/template work and trace-only
production writes, which the action sum does not represent.

## Problems

- none
