# RK3588 board admission

overall: **PASS**
driver expected: `RKNPU driver: v0.9.8`
profile hash: `954d071351959ddbe2bb7351de196c6b45f6b5bcd6f81f603b649e3e62eb692d`

| core | cases | declared numeric | guards | chain/separate bit-equal |
| --- | --- | --- | --- | --- |
| 1 | 10/10 | 7 | 3 | 3/3 |
| 2 | 10/10 | 7 | 3 | 3/3 |
| 4 | 10/10 | 7 | 3 | 3/3 |

## Numeric detail

Declared acceptance: `abs(got-declared) <= atol + rtol*abs(declared)`.
The FP64 mathematical reference is reported for context and is not the acceptance predicate.

| core | case | declared max abs | FP64 math max abs | mismatch/elements | tasks/submits | bit exact |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | matmul_chain | 0 | 0.0004817205481 | 0/3072 | 1/1 | yes |
| 1 | projection_chain | 0.00048828125 | 0.0003553947812 | 250/1664 | 5/4 | no |
| 1 | projection_separate | 0.00048828125 | 0.0003553947812 | 250/1664 | 5/5 | no |
| 1 | ffn_chain | 0.000244140625 | 0.0002377687311 | 27/1024 | 7/6 | no |
| 1 | ffn_separate | 0.000244140625 | 0.0002377687311 | 27/1024 | 7/7 | no |
| 1 | ragged_chain | 0 | 0.0004865511364 | 0/858 | 2/1 | yes |
| 1 | ragged_separate | 0 | 0.0004865511364 | 0/858 | 2/2 | yes |
| 2 | matmul_chain | 0 | 0.0004817205481 | 0/3072 | 1/1 | yes |
| 2 | projection_chain | 0.00048828125 | 0.0003553947812 | 250/1664 | 5/4 | no |
| 2 | projection_separate | 0.00048828125 | 0.0003553947812 | 250/1664 | 5/5 | no |
| 2 | ffn_chain | 0.000244140625 | 0.0002377687311 | 27/1024 | 7/6 | no |
| 2 | ffn_separate | 0.000244140625 | 0.0002377687311 | 27/1024 | 7/7 | no |
| 2 | ragged_chain | 0 | 0.0004865511364 | 0/858 | 2/1 | yes |
| 2 | ragged_separate | 0 | 0.0004865511364 | 0/858 | 2/2 | yes |
| 4 | matmul_chain | 0 | 0.0004817205481 | 0/3072 | 1/1 | yes |
| 4 | projection_chain | 0.00048828125 | 0.0003553947812 | 250/1664 | 5/4 | no |
| 4 | projection_separate | 0.00048828125 | 0.0003553947812 | 250/1664 | 5/5 | no |
| 4 | ffn_chain | 0.000244140625 | 0.0002377687311 | 27/1024 | 7/6 | no |
| 4 | ffn_separate | 0.000244140625 | 0.0002377687311 | 27/1024 | 7/7 | no |
| 4 | ragged_chain | 0 | 0.0004865511364 | 0/858 | 2/1 | yes |
| 4 | ragged_separate | 0 | 0.0004865511364 | 0/858 | 2/2 | yes |

## Artifact verification

- source bundle SHA-256 recorded: `5e3cc34faf8dd10f5fe4a4dc13d04776fd989a170ac87a2c40cae0b6e80ffad6`
- source bundle SHA-256 recomputed: `5e3cc34faf8dd10f5fe4a4dc13d04776fd989a170ac87a2c40cae0b6e80ffad6`
- source bundle bytes independently verified: `True`
- loaded library SHA-256 (recorded remote): `f3eb16cc70f46b3f77289e023bf02e67dabdb7cedcce09eb2fd93c37dba2c51e`; independently rehashed: `False`

## Chain/separate raw-output comparison

- core 1 `ffn_chain/ffn_separate`: bit_equal=True, max_abs=0
- core 1 `projection_chain/projection_separate`: bit_equal=True, max_abs=0
- core 1 `ragged_chain/ragged_separate`: bit_equal=True, max_abs=0
- core 2 `ffn_chain/ffn_separate`: bit_equal=True, max_abs=0
- core 2 `projection_chain/projection_separate`: bit_equal=True, max_abs=0
- core 2 `ragged_chain/ragged_separate`: bit_equal=True, max_abs=0
- core 4 `ffn_chain/ffn_separate`: bit_equal=True, max_abs=0
- core 4 `projection_chain/projection_separate`: bit_equal=True, max_abs=0
- core 4 `ragged_chain/ragged_separate`: bit_equal=True, max_abs=0

## Guard detail

- core 1 `matmul_guard_nonfinite_output`: code=-4, output_unchanged=True
- core 1 `projection_guard_nan_constant`: code=-4, output_unchanged=False
- core 1 `projection_guard_nan_input`: code=-4, output_unchanged=True
- core 2 `matmul_guard_nonfinite_output`: code=-4, output_unchanged=True
- core 2 `projection_guard_nan_constant`: code=-4, output_unchanged=False
- core 2 `projection_guard_nan_input`: code=-4, output_unchanged=True
- core 4 `matmul_guard_nonfinite_output`: code=-4, output_unchanged=True
- core 4 `projection_guard_nan_constant`: code=-4, output_unchanged=False
- core 4 `projection_guard_nan_input`: code=-4, output_unchanged=True

All requested cases passed on all three single-core masks.
