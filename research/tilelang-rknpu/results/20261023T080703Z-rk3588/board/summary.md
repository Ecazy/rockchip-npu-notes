# RK3588 board admission

overall: **PASS**
driver expected: `RKNPU driver: v0.9.8`

| core | cases | numeric | guards |
| --- | --- | --- | --- |
| 1 | 10/10 | 7 | 3 |
| 2 | 10/10 | 7 | 3 |
| 4 | 10/10 | 7 | 3 |

## Numeric detail

| core | case | max abs | mismatched/elements | bit exact |
| --- | --- | --- | --- | --- |
| 1 | matmul_chain | 0 | 0/3072 | yes |
| 1 | projection_chain | 0.00048828125 | 250/1664 | no |
| 1 | projection_separate | 0.00048828125 | 250/1664 | no |
| 1 | ffn_chain | 0.000244140625 | 27/1024 | no |
| 1 | ffn_separate | 0.000244140625 | 27/1024 | no |
| 1 | ragged_chain | 0 | 0/858 | yes |
| 1 | ragged_separate | 0 | 0/858 | yes |
| 2 | matmul_chain | 0 | 0/3072 | yes |
| 2 | projection_chain | 0.00048828125 | 250/1664 | no |
| 2 | projection_separate | 0.00048828125 | 250/1664 | no |
| 2 | ffn_chain | 0.000244140625 | 27/1024 | no |
| 2 | ffn_separate | 0.000244140625 | 27/1024 | no |
| 2 | ragged_chain | 0 | 0/858 | yes |
| 2 | ragged_separate | 0 | 0/858 | yes |
| 4 | matmul_chain | 0 | 0/3072 | yes |
| 4 | projection_chain | 0.00048828125 | 250/1664 | no |
| 4 | projection_separate | 0.00048828125 | 250/1664 | no |
| 4 | ffn_chain | 0.000244140625 | 27/1024 | no |
| 4 | ffn_separate | 0.000244140625 | 27/1024 | no |
| 4 | ragged_chain | 0 | 0/858 | yes |
| 4 | ragged_separate | 0 | 0/858 | yes |

All requested cases passed on all three single-core masks.
