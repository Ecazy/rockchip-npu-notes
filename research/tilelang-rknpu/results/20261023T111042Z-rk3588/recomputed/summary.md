# Offline recomputed metrics

overall: **PASS**

The FP64 math reference is unquantized and retained as float64. Declared-path acceptance uses the recorded allclose policy.

| core | case | declared max abs | FP64 math max abs | declared allclose |
| --- | --- | --- | --- | --- |
| 1 | activation_multitile_300_chain | 0.0009765625 | 0.0006963116353 | True |
| 1 | branch_multitile_300_chain | 0.000244140625 | 0.000478411861 | True |
| 1 | broadcast_channel_add_separate | 0 | 0.0003492308315 | True |
| 1 | broadcast_channel_mul_separate | 0 | 0.0001023275327 | True |
| 1 | broadcast_scalar_add_chain | 0 | 0.0007135464111 | True |
| 1 | broadcast_scalar_mul_chain | 0 | 5.744796363e-05 | True |
| 1 | dual_output_chain | 0 | 0.0001146747381 | True |
| 1 | ffn_chain | 0.000244140625 | 0.0002377687311 | True |
| 1 | ffn_multitile_300_chain | 0.00048828125 | 0.0002868636144 | True |
| 1 | ffn_separate | 0.000244140625 | 0.0002377687311 | True |
| 1 | independent_silu_chain | 0.00048828125 | 0.0006567516348 | True |
| 1 | k_even_512_chain | 0 | 0.0005320787313 | True |
| 1 | k_odd_257_chain | 0 | 0.000684034414 | True |
| 1 | large_lut_256x256_chain | 0.00048828125 | 0.0004171909833 | True |
| 1 | matmul_chain | 0 | 0.0004817205481 | True |
| 1 | mntile_512x384x512_chain | 0.00048828125 | 0.0005963100248 | True |
| 1 | projection_chain | 0.00048828125 | 0.0003553947812 | True |
| 1 | projection_multitile_300_chain | 0.00048828125 | 0.000400118945 | True |
| 1 | projection_separate | 0.00048828125 | 0.0003553947812 | True |
| 1 | ragged_chain | 0 | 0.0004865511364 | True |
| 1 | ragged_separate | 0 | 0.0004865511364 | True |
| 2 | activation_multitile_300_chain | 0.0009765625 | 0.0006963116353 | True |
| 2 | branch_multitile_300_chain | 0.000244140625 | 0.000478411861 | True |
| 2 | broadcast_channel_add_separate | 0 | 0.0003492308315 | True |
| 2 | broadcast_channel_mul_separate | 0 | 0.0001023275327 | True |
| 2 | broadcast_scalar_add_chain | 0 | 0.0007135464111 | True |
| 2 | broadcast_scalar_mul_chain | 0 | 5.744796363e-05 | True |
| 2 | dual_output_chain | 0 | 0.0001146747381 | True |
| 2 | ffn_chain | 0.000244140625 | 0.0002377687311 | True |
| 2 | ffn_multitile_300_chain | 0.00048828125 | 0.0002868636144 | True |
| 2 | ffn_separate | 0.000244140625 | 0.0002377687311 | True |
| 2 | independent_silu_chain | 0.00048828125 | 0.0006567516348 | True |
| 2 | k_even_512_chain | 0 | 0.0005320787313 | True |
| 2 | k_odd_257_chain | 0 | 0.000684034414 | True |
| 2 | large_lut_256x256_chain | 0.00048828125 | 0.0004171909833 | True |
| 2 | matmul_chain | 0 | 0.0004817205481 | True |
| 2 | mntile_512x384x512_chain | 0.00048828125 | 0.0005963100248 | True |
| 2 | projection_chain | 0.00048828125 | 0.0003553947812 | True |
| 2 | projection_multitile_300_chain | 0.00048828125 | 0.000400118945 | True |
| 2 | projection_separate | 0.00048828125 | 0.0003553947812 | True |
| 2 | ragged_chain | 0 | 0.0004865511364 | True |
| 2 | ragged_separate | 0 | 0.0004865511364 | True |
| 4 | activation_multitile_300_chain | 0.0009765625 | 0.0006963116353 | True |
| 4 | branch_multitile_300_chain | 0.000244140625 | 0.000478411861 | True |
| 4 | broadcast_channel_add_separate | 0 | 0.0003492308315 | True |
| 4 | broadcast_channel_mul_separate | 0 | 0.0001023275327 | True |
| 4 | broadcast_scalar_add_chain | 0 | 0.0007135464111 | True |
| 4 | broadcast_scalar_mul_chain | 0 | 5.744796363e-05 | True |
| 4 | dual_output_chain | 0 | 0.0001146747381 | True |
| 4 | ffn_chain | 0.000244140625 | 0.0002377687311 | True |
| 4 | ffn_multitile_300_chain | 0.00048828125 | 0.0002868636144 | True |
| 4 | ffn_separate | 0.000244140625 | 0.0002377687311 | True |
| 4 | independent_silu_chain | 0.00048828125 | 0.0006567516348 | True |
| 4 | k_even_512_chain | 0 | 0.0005320787313 | True |
| 4 | k_odd_257_chain | 0 | 0.000684034414 | True |
| 4 | large_lut_256x256_chain | 0.00048828125 | 0.0004171909833 | True |
| 4 | matmul_chain | 0 | 0.0004817205481 | True |
| 4 | mntile_512x384x512_chain | 0.00048828125 | 0.0005963100248 | True |
| 4 | projection_chain | 0.00048828125 | 0.0003553947812 | True |
| 4 | projection_multitile_300_chain | 0.00048828125 | 0.000400118945 | True |
| 4 | projection_separate | 0.00048828125 | 0.0003553947812 | True |
| 4 | ragged_chain | 0 | 0.0004865511364 | True |
| 4 | ragged_separate | 0 | 0.0004865511364 | True |
