# Runtime layer isolation

Seven outer rounds with rotated path/case order; each process uses
changing input frames. Values below are **p10 of warm samples**, a
stable floor rather than a mean contaminated by device-clock/thermal
outliers; medians and full ranges are in `analysis.json`.

Paths: A0 = Python generic `run_into`/`run_group_into`; A1 = Python prepared binding + direct legacy C; B = native C RKPlan; C = direct custom-ops; D = resident execute only.

| case | A0 µs | A1 µs | B µs | C µs | D µs | A0/A1 Python | A1/B FFI | B/C runtime | C/D data+boundary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP b1 | 477.5 | 427.6 | 427.6 | 420.9 | 228.1 | +49.9 | +0.0 | +6.7 | +192.8 |
| MLP b32 | 767.1 | 655.7 | 650.1 | 687.5 | 240.3 | +111.4 | +5.5 | -37.3 | +447.1 |
| QKV b8 | 246.7 | 186.1 | 182.0 | 189.6 | 54.0 | +60.7 | +4.1 | -7.6 | +135.6 |
| QK group b8 | 2228.3 | 897.2 | 834.2 | 466.7 | 179.4 | +1331.2 | +63.0 | +367.5 | +287.3 |
| PV group b8 | 1861.4 | 714.9 | 660.6 | 454.7 | 184.6 | +1146.5 | +54.2 | +205.9 | +270.1 |

A positive B/C delta means direct custom-ops ran faster than the
generic RKPlan runtime. A negative A1/B value is FFI/measurement
noise; the prepared-binding path does not bypass the C runtime.

## Setup, first execution and warm

`warm_median` is the actual center of all changed-input samples;
`warm_p10` is the stable floor used in the table above.

| case | path | setup µs | first µs | warm median µs | warm p10 µs | warm range µs |
|---|---|---:|---:|---:|---:|---:|
| MLP b1 | A0 | 5329.3 | 1015.6 | 492.6 | 477.5 | 474.5–841.7 |
| MLP b1 | A1 | 5345.0 | 863.0 | 431.4 | 427.6 | 425.2–752.8 |
| MLP b1 | B | 4213.9 | 577.8 | 437.2 | 427.6 | 424.7–737.0 |
| MLP b1 | C | 4779.8 | 499.3 | 434.3 | 420.9 | 419.7–749.6 |
| MLP b1 | D | 4926.2 | 275.6 | 231.9 | 228.1 | 227.2–548.3 |
| MLP b32 | A0 | 5700.6 | 1304.0 | 773.5 | 767.1 | 762.1–1139.8 |
| MLP b32 | A1 | 5654.2 | 1104.2 | 661.2 | 655.7 | 653.0–996.0 |
| MLP b32 | B | 4264.7 | 817.2 | 652.4 | 650.1 | 649.2–962.5 |
| MLP b32 | C | 4915.4 | 762.4 | 690.1 | 687.5 | 685.1–761.5 |
| MLP b32 | D | 5178.5 | 327.5 | 241.8 | 240.3 | 239.5–548.0 |
| QKV b8 | A0 | 919.3 | 465.5 | 251.7 | 246.7 | 243.8–348.2 |
| QKV b8 | A1 | 909.1 | 354.4 | 189.0 | 186.1 | 184.6–242.4 |
| QKV b8 | B | 419.4 | 290.5 | 184.0 | 182.0 | 177.9–233.0 |
| QKV b8 | C | 385.3 | 242.7 | 193.7 | 189.6 | 188.1–268.0 |
| QKV b8 | D | 412.4 | 83.1 | 55.1 | 54.0 | 53.7–60.4 |
| QK group b8 | A0 | 4831.4 | 3617.2 | 2254.3 | 2228.3 | 2212.8–2409.7 |
| QK group b8 | A1 | 4841.3 | 1195.5 | 913.8 | 897.2 | 891.3–1001.6 |
| QK group b8 | B | 992.5 | 1091.1 | 837.1 | 834.2 | 828.9–944.4 |
| QK group b8 | C | 790.4 | 703.5 | 468.4 | 466.7 | 462.3–526.7 |
| QK group b8 | D | 1253.9 | 275.3 | 180.2 | 179.4 | 176.5–249.1 |
| PV group b8 | A0 | 4803.4 | 3195.8 | 1896.4 | 1861.4 | 1853.2–2004.0 |
| PV group b8 | A1 | 4804.3 | 1003.9 | 752.2 | 714.9 | 713.1–812.0 |
| PV group b8 | B | 1000.4 | 949.4 | 665.6 | 660.6 | 658.3–778.7 |
| PV group b8 | C | 787.8 | 680.4 | 456.5 | 454.7 | 449.7–506.6 |
| PV group b8 | D | 1253.9 | 271.2 | 185.2 | 184.6 | 184.3–246.7 |

## Separate phase samples

B is the runtime attribution sample (96 calls per case, medians).
C is direct custom-ops over 32 changing frames, divided per call; it
uses a separate sample, not a sum borrowed from B.

| case | B validate | B pack | B emit | B command | B submit/wait | B output-sync | B download | B guard | B publish | C finite | C upload | C emit | C execute+sync | C download | C guard | C publish |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP b1 | 2.3 | 11.4 | 14.3 | 30.3 | 175.3 | 27.7 | 2.6 | 163.3 | 0.0 | 0.4 | 14.5 | 14.2 | 237.7 | 2.5 | 163.8 | 0.1 |
| MLP b32 | 68.0 | 156.0 | 14.3 | 28.6 | 184.9 | 30.0 | 3.8 | 161.0 | 0.3 | 10.8 | 254.5 | 14.3 | 250.7 | 3.5 | 161.0 | 0.1 |
| QKV b8 | 17.2 | 43.5 | 1.5 | 2.0 | 48.7 | 5.0 | 35.6 | 26.5 | 1.8 | 2.7 | 67.5 | 1.4 | 56.7 | 25.6 | 26.2 | 1.8 |
| QK group b8 | 423.5 | 39.7 | 46.2 | 23.9 | 97.7 | 94.5 | 112.9 | 24.8 | 1.2 | 65.2 | 25.9 | 46.5 | 189.4 | 105.5 | 24.6 | 1.2 |
| PV group b8 | 236.8 | 45.2 | 45.6 | 24.8 | 60.1 | 98.6 | 116.4 | 25.4 | 1.2 | 35.3 | 34.2 | 45.4 | 190.1 | 109.9 | 25.2 | 1.1 |

The B/C profile difference is consistent with the main path: QK/PV
lose substantial generic runtime validation work, while MLP b1 and
QKV remain dominated by the same pack/guard/download/submit envelope.
The direct C upload implementation is not faster than the runtime's
pack path on every shape; it is reported as measured, not hidden.

## Complete-model transmission probe

The targeted complete model is Tiny ViT b8 with 256 images, batch 8,
three rounds. Only the qk/pv group calls are replaced with the
experimental native-C direct context; every Host action and all other
plans remain the current compiled graph. All 96 paired outputs are
bitwise equal. This is a subgraph integration probe, not a fully native
C model implementation.

| arm | median app µs | p10 app µs |
|---|---:|---:|
| current compiled graph | 4002.6 | 3954.7 |
| qk/pv direct-C substitution | 4415.8 | 4352.5 |

Paired current/variant ratio: median **0.9050×**, p10 **0.8820×**; bitwise equal: **YES**. Adapter setup: 2406.2 µs.

| NPU node | current p10 µs | variant p10 µs | current−variant µs |
|---|---:|---:|---:|
| attn_out | 192.2 | 197.7 | -5.5 |
| classifier | 112.9 | 113.7 | -0.9 |
| down | 252.3 | 252.6 | -0.3 |
| gate_up | 265.4 | 272.7 | -7.3 |
| patch | 195.7 | 194.8 | +0.9 |
| pv | 586.5 | 508.4 | +78.2 |
| qk | 555.9 | 527.9 | +28.0 |
| qkv | 250.0 | 247.9 | +2.0 |
| **sum of measured NPU nodes** | **2410.9** | **2315.8** | **+95.1** |

Measured app p10 difference (current→variant): **+397.8 µs**; the measured NPU-node
sum changes by **-95.1 µs**. The
remaining approximately
**+492.9 µs**
is outside the replaced group nodes (Host actions/graph/output
materialisation or integration overhead). This is a measured
residual, not a sum used as the headline result.

A same-program two-runner control (identical compiled graph,
no adapters) shows no arm-placement bias:
paired median **0.9991×**, p10 **0.9904×**; app p10 3949.7 vs 3948.2 µs, bitwise equal.

The direct context removes qk/pv group-call time but the complete
model is slower. Therefore the isolated B/C gain does **not**
transmit to this model with the current integration shape. A fully
native C Host-action/lifetime implementation is a different
experiment; these data do not justify authorizing it as a redesign.

## Output equality

- comparison rows: 175, differences: 0
- A0/A1/B/C/D final outputs are bitwise equal: YES

D result validation/download is outside its timed interval; its
validation time is recorded separately in `VERIFY_US` logs.
