# Invalidated performance fields (retained historical report)

The accuracy, raw logits, top-1 agreement and weight hashes in this campaign
remain valid.  Its **performance fields are invalid** and must not be cited.

Reason: `evaluate_models.py` in this campaign started `wall_seconds` before
Model execution and then included CPU FP32 and declared-path reference
calculation in the same interval.  `throughput_images_s` divided by that mixed
interval.  It also compared an NPU batch-32/batch-8 execution with a CPU baseline
that processed all 10,000 images in one NumPy matmul, not with the same static
batch schedule.

Known contaminated values in `model-results/core-1/model_eval.json`:

| Model | historical wall | of which CPU reference | accumulated PlanCall |
|---|---:|---:|---:|
| MNIST MLP b32 | 6.223 s | 2.205 s | 0.815 s |
| Tiny Transformer b8 | 65.294 s | 8.287 s | 34.159 s |

Consequently the historical 1,607 images/s and 153 images/s values are not model
throughput, and the old conclusion that both models were slower than CPU is
withdrawn.  The report is kept unchanged for audit history.

Corrected schema-v2 timing, matched-batch CPU baseline and the final verifier
result are in `results/20261024T065201Z-models/` and
`MODEL-MILESTONE-STATUS.md`.
