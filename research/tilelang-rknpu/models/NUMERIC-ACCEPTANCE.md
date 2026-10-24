# Numeric acceptance policy and evidence

## Current policy: train-calibrated, test-accepted

The independent numeric acceptance boundary uses two fixed 32-image subsets:

| Role | Official split | Selection | Labels used |
|---|---|---|---|
| Calibration | MNIST train | seed 20261024, RNG choice without replacement | **no** |
| Acceptance | MNIST test | seed 20261025, RNG choice without replacement | **no** |

The split identity, source image-file SHA-256, selected indices, per-image raw-byte
SHA-256, expected arrays and observation-file SHA-256 are recorded in
`models/numeric_acceptance_train.json`.

Temporal order on RK3588:

1. Run calibration observations on the train subset.
2. Freeze `models/numeric_acceptance_train.json` on the Host using train-only
   observations.
3. Record the frozen spec SHA-256 in the acceptance observation metadata.
4. Run a new acceptance observation on the disjoint test subset.
5. Run `tools/verify_numeric_acceptance.py`.
6. Re-check the existing full 10,000-image campaign raw logits against the same
   frozen profile limits.

The acceptance run is **not blind** in the historical sense: the full test set
and previous task-accuracy results were inspected earlier.  The claim is narrower:
the threshold policy was frozen without using test data, and the later test
observation uses disjoint samples.

## Frozen limit policy

For every device-derived FP16 output:

```text
atol = ceil_to_2^-12(max(2 * train_observed_max_abs_error,
                         2 * f16_spacing(train_expected_absmax)))
rtol = 0
```

Host-only exact layout/normalization outputs use only
`ceil_to_2^-12(2 * train_observed_max_abs_error)`.

- Both the observed error and the expected-magnitude FP16 floor are computed from
  the train calibration set only.
- No test-set maximum, top-1 result or task accuracy enters the formula.
- The floor is a representational-scale bound, not a theoretical end-to-end error
  guarantee.
- If post-freeze test acceptance fails, the required action is to locate the
  numeric-path or implementation defect.  Re-running calibration on test maxima to
  enlarge limits is forbidden.

## Files

Current independent evidence:

- `numeric_acceptance_train.json` — frozen policy, 33 layer gates + 5 profile
  raw-logit gates, calibration/freeze hashes.
- `numeric_calibration_train.npz` — fixed train images, expected MLP hidden/
  Transformer layer values and profile logits.
- `numeric_acceptance_test.npz` — fixed disjoint test images and expected values.
- `models/compiled/mnist_mlp_diag_{l1_bias,l1_relu,l2_bias,l2_relu,l3_bias}.rkplan.json`
  — real-TileLang diagnostic plans.
- `results/20261024T091634Z-numeric/calibration/observation.npz` — post-build
  train hardware observations used to freeze the policy.
- `results/20261024T091634Z-numeric/acceptance/observation.npz` — post-freeze
  test hardware observations.
- `results/20261024T091634Z-numeric/verify-summary.json` — independent verifier
  PASS, including calibration/acceptance disjointness and the existing full-test
  raw-logit recheck.

## Historical test envelope v1

`numeric_acceptance.json`, `numeric_golden.npz` and
`evidence/numeric-calibration/` are retained as a **regression envelope only**.
They were calibrated from test-split observations and therefore do not satisfy
the independent calibration requirement.  They remain useful for reproducing the
first-round model behavior and historical verifier, but cannot be cited as
independent numerical acceptance.

## MLP hidden layers

The production `mnist_mlp_b*` graph is unchanged and still downloads only final
logits.  Numeric diagnostics call separate real-TileLang plans:

- layer 1 `matmul(W1)+b1` -> pre-ReLU export;
- layer 1 bias + exact ReLU -> ReLU export;
- layer 2 consumes the **hardware-produced ReLU1** -> pre-ReLU export;
- layer 2 bias + exact ReLU -> ReLU export;
- layer 3 consumes the **hardware-produced ReLU2** -> final 16-wide
  preactivation+bias export.

Each diagnostic constant is the frozen learned weight or the same static bias
tile/padding used by the production profile.  The diagnostic plans do not alter
the production plan, its single final download, or its timing evidence.

## Non-claims

The numeric limits are an empirical RK3588 envelope for the frozen package,
weights, plans and execution paths.  They are not a proof for other devices,
other compilers, dynamic shapes, quantized variants or unseen input distributions.
