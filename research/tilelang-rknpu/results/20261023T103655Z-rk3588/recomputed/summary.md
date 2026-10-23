# Offline recomputed metrics

overall: **PASS**

The FP64 math reference is unquantized and retained as float64. Declared-path acceptance uses the recorded allclose policy.

| core | case | declared max abs | FP64 math max abs | declared allclose |
| --- | --- | --- | --- | --- |
| 1 | ffn_chain | 0.000244140625 | 0.0002377687311 | True |
| 1 | ffn_separate | 0.000244140625 | 0.0002377687311 | True |
| 1 | matmul_chain | 0 | 0.0004817205481 | True |
| 1 | projection_chain | 0.00048828125 | 0.0003553947812 | True |
| 1 | projection_separate | 0.00048828125 | 0.0003553947812 | True |
| 1 | ragged_chain | 0 | 0.0004865511364 | True |
| 1 | ragged_separate | 0 | 0.0004865511364 | True |
| 2 | ffn_chain | 0.000244140625 | 0.0002377687311 | True |
| 2 | ffn_separate | 0.000244140625 | 0.0002377687311 | True |
| 2 | matmul_chain | 0 | 0.0004817205481 | True |
| 2 | projection_chain | 0.00048828125 | 0.0003553947812 | True |
| 2 | projection_separate | 0.00048828125 | 0.0003553947812 | True |
| 2 | ragged_chain | 0 | 0.0004865511364 | True |
| 2 | ragged_separate | 0 | 0.0004865511364 | True |
| 4 | ffn_chain | 0.000244140625 | 0.0002377687311 | True |
| 4 | ffn_separate | 0.000244140625 | 0.0002377687311 | True |
| 4 | matmul_chain | 0 | 0.0004817205481 | True |
| 4 | projection_chain | 0.00048828125 | 0.0003553947812 | True |
| 4 | projection_separate | 0.00048828125 | 0.0003553947812 | True |
| 4 | ragged_chain | 0 | 0.0004865511364 | True |
| 4 | ragged_separate | 0 | 0.0004865511364 | True |
