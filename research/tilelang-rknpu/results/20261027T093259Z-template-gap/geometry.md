# Representative GEMM geometry

| manifest | plan | sums | tasks/submits | logical MACs | physical MACs | pad ratio | peak B |
|---|---|---|---:|---:|---:|---:|---:|
| mnist_mlp_b1 | mlp | 8 | 11/8 | 296,960 | 3,588,096 | 12.083 | 829,184 |
| mnist_mlp_b8 | mlp | 8 | 11/8 | 2,375,680 | 3,588,096 | 1.510 | 829,184 |
| mnist_mlp_b32 | mlp | 8 | 11/8 | 9,502,720 | 9,568,256 | 1.007 | 937,984 |
| tiny_vit_b1 | qkv | 1 | 1/1 | 196,608 | 196,608 | 1.000 | 32,768 |
| tiny_vit_b8 | qkv | 1 | 1/1 | 1,572,864 | 1,572,864 | 1.000 | 90,112 |
| tiny_vit_b1 | gate_up | 1 | 1/1 | 262,144 | 262,144 | 1.000 | 43,008 |
| tiny_vit_b8 | gate_up | 1 | 1/1 | 2,097,152 | 2,097,152 | 1.000 | 114,688 |
| tiny_vit_b1 | down | 1 | 1/1 | 131,072 | 131,072 | 1.000 | 22,528 |
| tiny_vit_b8 | down | 1 | 1/1 | 1,048,576 | 1,048,576 | 1.000 | 65,536 |
| tiny_vit_b1 | classifier | 1 | 1/1 | 1,024 | 24,576 | 24.000 | 6,400 |
| tiny_vit_b8 | classifier | 1 | 1/1 | 8,192 | 24,576 | 3.000 | 6,400 |
| tiny_vit_b1 | attn_qk_qkv_group | 1 | 4/1 | 16,384 | 65,536 | 4.000 | 16,384 |
| tiny_vit_b8 | attn_qk_qkv_group | 1 | 32/1 | 131,072 | 524,288 | 4.000 | 131,072 |
| tiny_vit_b1 | attn_pv_qkv_group | 1 | 4/1 | 16,384 | 65,536 | 4.000 | 16,384 |
| tiny_vit_b8 | attn_pv_qkv_group | 1 | 32/1 | 131,072 | 524,288 | 4.000 | 131,072 |

## Submit-level detail

| manifest | plan | submit | mode/recipe | tasks | logical K | task geometry (M×K×N × count) | source tiling |
|---|---|---:|---|---:|---:|---|---|
| mnist_mlp_b1 | mlp | 0 | chain/linear_k_acc_v1 | 4 | 1024 | 1×256×256 × 1; 1×256×256 × 3 | {"block_m": 1, "block_n": 64, "coverage": "exact_once", "grid_x": 4, "grid_y": 1} |
| mnist_mlp_b1 | mlp | 3 | chain/linear_k_acc_v1 | 1 | 256 | 1×256×128 × 1 | {"block_m": 1, "block_n": 64, "coverage": "exact_once", "grid_x": 2, "grid_y": 1} |
| mnist_mlp_b1 | mlp | 6 | chain/linear_k_acc_v1 | 1 | 128 | 1×128×16 × 1 | {"block_m": 1, "block_n": 16, "coverage": "exact_once", "grid_x": 1, "grid_y": 1} |
| mnist_mlp_b8 | mlp | 0 | chain/linear_k_acc_v1 | 4 | 1024 | 8×256×256 × 1; 8×256×256 × 3 | {"block_m": 8, "block_n": 64, "coverage": "exact_once", "grid_x": 4, "grid_y": 1} |
| mnist_mlp_b8 | mlp | 3 | chain/linear_k_acc_v1 | 1 | 256 | 8×256×128 × 1 | {"block_m": 8, "block_n": 64, "coverage": "exact_once", "grid_x": 2, "grid_y": 1} |
| mnist_mlp_b8 | mlp | 6 | chain/linear_k_acc_v1 | 1 | 128 | 8×128×16 × 1 | {"block_m": 8, "block_n": 16, "coverage": "exact_once", "grid_x": 1, "grid_y": 1} |
| mnist_mlp_b32 | mlp | 0 | chain/linear_k_acc_v1 | 4 | 1024 | 32×256×256 × 1; 32×256×256 × 3 | {"block_m": 32, "block_n": 64, "coverage": "exact_once", "grid_x": 4, "grid_y": 1} |
| mnist_mlp_b32 | mlp | 3 | chain/linear_k_acc_v1 | 1 | 256 | 32×256×128 × 1 | {"block_m": 32, "block_n": 64, "coverage": "exact_once", "grid_x": 2, "grid_y": 1} |
| mnist_mlp_b32 | mlp | 6 | chain/linear_k_acc_v1 | 1 | 128 | 32×128×16 × 1 | {"block_m": 32, "block_n": 16, "coverage": "exact_once", "grid_x": 1, "grid_y": 1} |
| tiny_vit_b1 | qkv | 0 | chain/linear_k_acc_v1 | 1 | 64 | 16×64×192 × 1 | null |
| tiny_vit_b8 | qkv | 0 | chain/linear_k_acc_v1 | 1 | 64 | 128×64×192 × 1 | null |
| tiny_vit_b1 | gate_up | 0 | chain/linear_k_acc_v1 | 1 | 64 | 16×64×256 × 1 | null |
| tiny_vit_b8 | gate_up | 0 | chain/linear_k_acc_v1 | 1 | 64 | 128×64×256 × 1 | null |
| tiny_vit_b1 | down | 0 | chain/linear_k_acc_v1 | 1 | 128 | 16×128×64 × 1 | null |
| tiny_vit_b8 | down | 0 | chain/linear_k_acc_v1 | 1 | 128 | 128×128×64 × 1 | null |
| tiny_vit_b1 | classifier | 0 | chain/linear_k_acc_v1 | 1 | 64 | 1×64×16 × 1 | null |
| tiny_vit_b8 | classifier | 0 | chain/linear_k_acc_v1 | 1 | 64 | 8×64×16 × 1 | null |
| tiny_vit_b1 | attn_qk_qkv_group | 0 | chain/independent_mm_v1 | 4 | 16 | 16×16×16 × 4 | null |
| tiny_vit_b8 | attn_qk_qkv_group | 0 | chain/independent_mm_v1 | 32 | 16 | 16×16×16 × 32 | null |
| tiny_vit_b1 | attn_pv_qkv_group | 0 | chain/independent_mm_v1 | 4 | 16 | 16×16×16 × 4 | null |
| tiny_vit_b8 | attn_pv_qkv_group | 0 | chain/independent_mm_v1 | 32 | 16 | 16×16×16 × 32 | null |

`logical MACs` and `physical MACs` use the compiled task views; the
physical count includes the CUBE/WEIGHT padding. Register content is
not represented in RKPlan: the template is the fixed `gen_matmul_fp16`
recipe with one task per K/M/N tile. No alternative register budget is
currently selectable from the compiler.
