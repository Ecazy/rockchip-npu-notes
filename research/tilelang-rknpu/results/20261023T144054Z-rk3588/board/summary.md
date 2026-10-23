# RK3588 board admission

overall: **PASS**
driver expected: `RKNPU driver: v0.9.8`
profile hash: `954d071351959ddbe2bb7351de196c6b45f6b5bcd6f81f603b649e3e62eb692d`

| core | cases | declared numeric | guards | chain/separate bit-equal | CLI |
| --- | --- | --- | --- | --- | --- |
| 1 | 27/27 | 24 | 3 | 3/3 | PASS |
| 2 | 27/27 | 24 | 3 | 3/3 | PASS |
| 4 | 27/27 | 24 | 3 | 3/3 | PASS |

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
| 1 | broadcast_scalar_add_chain | 0 | 0.0007135464111 | 0/264 | 2/2 | yes |
| 1 | broadcast_scalar_mul_chain | 0 | 5.744796363e-05 | 0/264 | 2/2 | yes |
| 1 | broadcast_channel_add_separate | 0 | 0.0003492308315 | 0/264 | 2/2 | yes |
| 1 | broadcast_channel_mul_separate | 0 | 0.0001023275327 | 0/264 | 2/2 | yes |
| 1 | independent_silu_chain | 0.00048828125 | 0.0006567516348 | 26/264 | 2/1 | no |
| 1 | large_lut_256x256_chain | 0.00048828125 | 0.0004171909833 | 5082/65536 | 4/2 | no |
| 1 | dual_output_chain | 0 | 0.0001146747381 | 0/640 | 2/2 | yes |
| 1 | mntile_512x384x512_chain | 0.00048828125 | 0.0005963100248 | 94/196608 | 8/4 | no |
| 1 | k_even_512_chain | 0 | 0.0005320787313 | 0/264 | 2/1 | yes |
| 1 | k_odd_257_chain | 0 | 0.000684034414 | 0/264 | 2/1 | yes |
| 1 | projection_multitile_300_chain | 0.0009765625 | 0.0005600726839 | 24938/90000 | 29/20 | no |
| 1 | ffn_multitile_300_chain | 0.00048828125 | 0.0002868636144 | 11118/90000 | 41/24 | no |
| 1 | branch_multitile_300_chain | 0.000244140625 | 0.000478411861 | 54/150000 | 12/6 | no |
| 1 | activation_multitile_300_chain | 0.0009765625 | 0.0006963116353 | 7030/90000 | 9/4 | no |
| 1 | projection_tilelang_staged_chain | 0.0009765625 | 0.0006698468003 | 39758/131072 | 22/16 | no |
| 1 | ffn_tilelang_staged_chain | 0.00048828125 | 0.0002685549643 | 18659/131072 | 30/24 | no |
| 1 | branch_tilelang_staged_chain | 0.000244140625 | 0.000308618628 | 139/393216 | 6/6 | no |
| 2 | matmul_chain | 0 | 0.0004817205481 | 0/3072 | 1/1 | yes |
| 2 | projection_chain | 0.00048828125 | 0.0003553947812 | 250/1664 | 5/4 | no |
| 2 | projection_separate | 0.00048828125 | 0.0003553947812 | 250/1664 | 5/5 | no |
| 2 | ffn_chain | 0.000244140625 | 0.0002377687311 | 27/1024 | 7/6 | no |
| 2 | ffn_separate | 0.000244140625 | 0.0002377687311 | 27/1024 | 7/7 | no |
| 2 | ragged_chain | 0 | 0.0004865511364 | 0/858 | 2/1 | yes |
| 2 | ragged_separate | 0 | 0.0004865511364 | 0/858 | 2/2 | yes |
| 2 | broadcast_scalar_add_chain | 0 | 0.0007135464111 | 0/264 | 2/2 | yes |
| 2 | broadcast_scalar_mul_chain | 0 | 5.744796363e-05 | 0/264 | 2/2 | yes |
| 2 | broadcast_channel_add_separate | 0 | 0.0003492308315 | 0/264 | 2/2 | yes |
| 2 | broadcast_channel_mul_separate | 0 | 0.0001023275327 | 0/264 | 2/2 | yes |
| 2 | independent_silu_chain | 0.00048828125 | 0.0006567516348 | 26/264 | 2/1 | no |
| 2 | large_lut_256x256_chain | 0.00048828125 | 0.0004171909833 | 5082/65536 | 4/2 | no |
| 2 | dual_output_chain | 0 | 0.0001146747381 | 0/640 | 2/2 | yes |
| 2 | mntile_512x384x512_chain | 0.00048828125 | 0.0005963100248 | 94/196608 | 8/4 | no |
| 2 | k_even_512_chain | 0 | 0.0005320787313 | 0/264 | 2/1 | yes |
| 2 | k_odd_257_chain | 0 | 0.000684034414 | 0/264 | 2/1 | yes |
| 2 | projection_multitile_300_chain | 0.0009765625 | 0.0005600726839 | 24938/90000 | 29/20 | no |
| 2 | ffn_multitile_300_chain | 0.00048828125 | 0.0002868636144 | 11118/90000 | 41/24 | no |
| 2 | branch_multitile_300_chain | 0.000244140625 | 0.000478411861 | 54/150000 | 12/6 | no |
| 2 | activation_multitile_300_chain | 0.0009765625 | 0.0006963116353 | 7030/90000 | 9/4 | no |
| 2 | projection_tilelang_staged_chain | 0.0009765625 | 0.0006698468003 | 39758/131072 | 22/16 | no |
| 2 | ffn_tilelang_staged_chain | 0.00048828125 | 0.0002685549643 | 18659/131072 | 30/24 | no |
| 2 | branch_tilelang_staged_chain | 0.000244140625 | 0.000308618628 | 139/393216 | 6/6 | no |
| 4 | matmul_chain | 0 | 0.0004817205481 | 0/3072 | 1/1 | yes |
| 4 | projection_chain | 0.00048828125 | 0.0003553947812 | 250/1664 | 5/4 | no |
| 4 | projection_separate | 0.00048828125 | 0.0003553947812 | 250/1664 | 5/5 | no |
| 4 | ffn_chain | 0.000244140625 | 0.0002377687311 | 27/1024 | 7/6 | no |
| 4 | ffn_separate | 0.000244140625 | 0.0002377687311 | 27/1024 | 7/7 | no |
| 4 | ragged_chain | 0 | 0.0004865511364 | 0/858 | 2/1 | yes |
| 4 | ragged_separate | 0 | 0.0004865511364 | 0/858 | 2/2 | yes |
| 4 | broadcast_scalar_add_chain | 0 | 0.0007135464111 | 0/264 | 2/2 | yes |
| 4 | broadcast_scalar_mul_chain | 0 | 5.744796363e-05 | 0/264 | 2/2 | yes |
| 4 | broadcast_channel_add_separate | 0 | 0.0003492308315 | 0/264 | 2/2 | yes |
| 4 | broadcast_channel_mul_separate | 0 | 0.0001023275327 | 0/264 | 2/2 | yes |
| 4 | independent_silu_chain | 0.00048828125 | 0.0006567516348 | 26/264 | 2/1 | no |
| 4 | large_lut_256x256_chain | 0.00048828125 | 0.0004171909833 | 5082/65536 | 4/2 | no |
| 4 | dual_output_chain | 0 | 0.0001146747381 | 0/640 | 2/2 | yes |
| 4 | mntile_512x384x512_chain | 0.00048828125 | 0.0005963100248 | 94/196608 | 8/4 | no |
| 4 | k_even_512_chain | 0 | 0.0005320787313 | 0/264 | 2/1 | yes |
| 4 | k_odd_257_chain | 0 | 0.000684034414 | 0/264 | 2/1 | yes |
| 4 | projection_multitile_300_chain | 0.0009765625 | 0.0005600726839 | 24938/90000 | 29/20 | no |
| 4 | ffn_multitile_300_chain | 0.00048828125 | 0.0002868636144 | 11118/90000 | 41/24 | no |
| 4 | branch_multitile_300_chain | 0.000244140625 | 0.000478411861 | 54/150000 | 12/6 | no |
| 4 | activation_multitile_300_chain | 0.0009765625 | 0.0006963116353 | 7030/90000 | 9/4 | no |
| 4 | projection_tilelang_staged_chain | 0.0009765625 | 0.0006698468003 | 39758/131072 | 22/16 | no |
| 4 | ffn_tilelang_staged_chain | 0.00048828125 | 0.0002685549643 | 18659/131072 | 30/24 | no |
| 4 | branch_tilelang_staged_chain | 0.000244140625 | 0.000308618628 | 139/393216 | 6/6 | no |

## Artifact verification

- source bundle SHA-256 recorded: `de0e51f886f40791a4f7431c24da467fadbd82104834d8aa6c817702925560e2`
- source bundle SHA-256 recomputed: `de0e51f886f40791a4f7431c24da467fadbd82104834d8aa6c817702925560e2`
- source bundle bytes independently verified: `True`
- loaded library SHA-256 (recorded remote): `019b42d3007eaa3ba2420cae93233b7fc7f609e36e05da0071f59436b354b049`; independently rehashed: `False`

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
