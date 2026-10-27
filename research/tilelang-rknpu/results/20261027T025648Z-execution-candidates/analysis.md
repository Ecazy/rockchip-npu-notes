# Execution-plan candidate comparison

- status: **PASS**
- rounds: up to 5 custom arms (minimum 3), 2 external
- full 10,000-image profiles: True
- same-policy reference arm: `generic`

Warm latency is per static batch call: raw preparation + execution + output materialisation, excluding argmax and references.  Ratio columns use the median of paired round ratios (`reference_ms / arm_ms`); >1 means the arm is faster.

## Same-policy compiler arms

| profile | generic ms | prepared ms | auto ms | hostfused ms | prepared/generic | auto/generic | hostfused/generic |
|---|---:|---:|---:|---:|---:|---:|---:|
| mnist_mlp.b32 | 2.152 | 2.187 | 2.163 | 2.127 | 0.970x | 1.035x | 1.037x |
| mnist_mlp.b8 | 1.271 | 1.368 | 1.304 | 1.330 | 0.948x | 0.905x | 0.945x |
| mnist_mlp.b1 | 1.233 | 1.293 | 1.206 | 1.234 | 0.947x | 1.040x | 0.977x |
| tiny_vit.b8 | 10.238 | 7.442 | 7.303 | 7.550 | 1.367x | 1.383x | 1.352x |
| tiny_vit.b1 | 3.075 | 2.499 | 2.358 | 2.283 | 1.230x | 1.311x | 1.380x |

## External baselines (separate numeric contracts)

| profile | auto ms | vendor ms | CPU ms | vendor/deploy | CPU/deploy | vendor max err | packed atol |
|---|---:|---:|---:|---:|---:|---:|---:|
| mnist_mlp.b32 | 2.163 | 0.834 | 6.861 | 2.594x | 0.315x | 0.03049 | 0.0625 |
| mnist_mlp.b8 | 1.304 | 0.527 | 1.751 | 2.473x | 0.745x | 0.03049 | 0.0625 |
| mnist_mlp.b1 | 1.206 | 0.409 | 0.255 | 2.949x | 4.722x | 0.03049 | 0.0625 |
| tiny_vit.b8 | 7.303 | 4.272 | 5.873 | 1.709x | 1.244x | 0.12277 | 0.0625 |
| tiny_vit.b1 | 2.358 | 1.022 | 0.874 | 2.308x | 2.697x | 0.10715 | 0.0625 |

Vendor is class-preserving but uses non-quantized FP16 graph fusion; its per-logit contract is not the custom packed policy.  CPU is FP32 reference math.  Neither is ranked against the custom arms.

## Setup and application end-to-end

| profile | arm | setup ms | preprocess ms | model-call ms | argmax-inclusive e2e ms |
|---|---|---:|---:|---:|---:|
| mnist_mlp.b32 | generic | 37.391 | 0.386 | 2.152 | 2.153 |
| mnist_mlp.b32 | prepared | 38.295 | 0.417 | 2.187 | 2.189 |
| mnist_mlp.b32 | auto | 37.083 | 0.382 | 2.163 | 2.165 |
| mnist_mlp.b32 | hostfused | 38.171 | 0.412 | 2.127 | 2.129 |
| mnist_mlp.b32 | vendor | 205.151 | - | 0.834 | 0.835 |
| mnist_mlp.b32 | cpu | 0.000 | - | 6.861 | 6.862 |
| mnist_mlp.b8 | generic | 17.861 | 0.155 | 1.271 | 1.272 |
| mnist_mlp.b8 | prepared | 18.185 | 0.177 | 1.368 | 1.368 |
| mnist_mlp.b8 | auto | 17.946 | 0.155 | 1.304 | 1.305 |
| mnist_mlp.b8 | hostfused | 18.281 | 0.174 | 1.330 | 1.330 |
| mnist_mlp.b8 | vendor | 61.466 | - | 0.527 | 0.528 |
| mnist_mlp.b8 | cpu | 0.000 | - | 1.751 | 1.751 |
| mnist_mlp.b1 | generic | 18.492 | 0.095 | 1.233 | 1.233 |
| mnist_mlp.b1 | prepared | 18.960 | 0.110 | 1.293 | 1.293 |
| mnist_mlp.b1 | auto | 18.458 | 0.094 | 1.206 | 1.206 |
| mnist_mlp.b1 | hostfused | 18.717 | 0.108 | 1.234 | 1.234 |
| mnist_mlp.b1 | vendor | 61.348 | - | 0.409 | 0.409 |
| mnist_mlp.b1 | cpu | 0.000 | - | 0.255 | 0.255 |
| tiny_vit.b8 | generic | 60.877 | 0.213 | 10.238 | 10.239 |
| tiny_vit.b8 | prepared | 63.285 | 0.227 | 7.442 | 7.443 |
| tiny_vit.b8 | auto | 64.301 | 0.224 | 7.303 | 7.303 |
| tiny_vit.b8 | hostfused | 64.642 | 0.232 | 7.550 | 7.550 |
| tiny_vit.b8 | vendor | 64.197 | - | 4.272 | 4.273 |
| tiny_vit.b8 | cpu | 0.000 | - | 5.873 | 5.873 |
| tiny_vit.b1 | generic | 43.977 | 0.095 | 3.075 | 3.075 |
| tiny_vit.b1 | prepared | 45.561 | 0.098 | 2.499 | 2.499 |
| tiny_vit.b1 | auto | 45.258 | 0.095 | 2.358 | 2.358 |
| tiny_vit.b1 | hostfused | 47.353 | 0.095 | 2.283 | 2.283 |
| tiny_vit.b1 | vendor | 61.819 | - | 1.022 | 1.022 |
| tiny_vit.b1 | cpu | 0.000 | - | 0.874 | 0.874 |

Vendor and CPU expose no separate per-call preprocessing bucket in their application harness; raw preparation is inside their model-call interval. Custom preprocessing and Host layout conversion are measured separately. Setup is one-time and excluded from warm latency; the MLP entry builds its declared b32/b8/b1 runners once and reports that aggregate.

## Auto selection by static binding fanout

| profile | selected path | binding slots | threshold | resident bytes |
|---|---|---:|---:|---:|
| mnist_mlp.b32 | generic | 13 | 32 | 0 |
| mnist_mlp.b8 | generic | 13 | 32 | 0 |
| mnist_mlp.b1 | generic | 13 | 32 | 0 |
| tiny_vit.b8 | prepared | 210 | 32 | 505664 |
| tiny_vit.b1 | prepared | 42 | 32 | 63208 |

MLP auto rows execute the generic path unchanged; their small round-to-round ratio spread is measurement variance, not a code-path change. Tiny ViT auto rows execute prepared residency.

## Prepared lowering cost decomposition

| profile | arm | warm ms | Python plan-node ms | Host actions ms | C workspace ms | executor ms | submit/call | tasks/call |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mnist_mlp.b32 | generic | 2.152 | 0.156 | 0.420 | 1.428 | 0.121 | 8.0 | 11.0 |
| mnist_mlp.b32 | prepared | 2.187 | 0.020 | 0.461 | 1.543 | 0.131 | 8.0 | 11.0 |
| mnist_mlp.b32 | auto | 2.163 | 0.156 | 0.415 | 1.448 | 0.121 | 8.0 | 11.0 |
| mnist_mlp.b32 | hostfused | 2.127 | 0.020 | 0.453 | 1.503 | 0.127 | 8.0 | 11.0 |
| mnist_mlp.b8 | generic | 1.271 | 0.080 | 0.184 | 0.898 | 0.104 | 8.0 | 11.0 |
| mnist_mlp.b8 | prepared | 1.368 | 0.018 | 0.214 | 0.998 | 0.116 | 8.0 | 11.0 |
| mnist_mlp.b8 | auto | 1.304 | 0.082 | 0.184 | 0.924 | 0.104 | 8.0 | 11.0 |
| mnist_mlp.b8 | hostfused | 1.330 | 0.017 | 0.210 | 0.974 | 0.111 | 8.0 | 11.0 |
| mnist_mlp.b1 | generic | 1.233 | 0.065 | 0.123 | 0.930 | 0.104 | 8.0 | 11.0 |
| mnist_mlp.b1 | prepared | 1.293 | 0.017 | 0.148 | 0.989 | 0.114 | 8.0 | 11.0 |
| mnist_mlp.b1 | auto | 1.206 | 0.067 | 0.122 | 0.895 | 0.104 | 8.0 | 11.0 |
| mnist_mlp.b1 | hostfused | 1.234 | 0.016 | 0.143 | 0.947 | 0.109 | 8.0 | 11.0 |
| tiny_vit.b8 | generic | 10.238 | 2.861 | 2.131 | 4.511 | 0.690 | 8.0 | 70.0 |
| tiny_vit.b8 | prepared | 7.442 | 0.135 | 2.219 | 4.333 | 0.677 | 8.0 | 70.0 |
| tiny_vit.b8 | auto | 7.303 | 0.141 | 2.176 | 4.268 | 0.669 | 8.0 | 70.0 |
| tiny_vit.b8 | hostfused | 7.550 | 0.155 | 2.246 | 4.398 | 0.697 | 8.0 | 70.0 |
| tiny_vit.b1 | generic | 3.075 | 0.637 | 0.718 | 1.209 | 0.493 | 8.0 | 14.0 |
| tiny_vit.b1 | prepared | 2.499 | 0.100 | 0.759 | 1.093 | 0.495 | 8.0 | 14.0 |
| tiny_vit.b1 | auto | 2.358 | 0.093 | 0.736 | 1.049 | 0.465 | 8.0 | 14.0 |
| tiny_vit.b1 | hostfused | 2.283 | 0.089 | 0.721 | 1.004 | 0.452 | 8.0 | 14.0 |

## Numeric equality against the same-policy reference

| arm | all profiles bitwise | max logit delta | all trace bitwise |
|---|---:|---:|---:|
| prepared | True | 0 | True |
| auto | True | 0 | True |
| hostfused | True | 0 | True |

## Problems

- none
