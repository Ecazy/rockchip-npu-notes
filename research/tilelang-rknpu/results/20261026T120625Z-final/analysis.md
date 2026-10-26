# Final model delivery analysis

- status: **PASS**
- full test images: 10000
- rounds: 3

## Profile comparison

| profile | baseline warm ms | current warm ms | speedup | baseline img/s | current img/s | acc b/c | top1 b/c |
|---|---:|---:|---:|---:|---:|---:|---:|
| mnist_mlp.b32 | 2.881 | 2.246 | 1.283x | 11056.0 | 14181.3 | 0.98310/0.98310 | 1.00000/1.00000 |
| mnist_mlp.b8 | 2.027 | 1.559 | 1.300x | 3946.4 | 5131.6 | 0.98310/0.98310 | 1.00000/1.00000 |
| mnist_mlp.b1 | 2.031 | 1.373 | 1.479x | 492.4 | 728.1 | 0.98310/0.98310 | 1.00000/1.00000 |
| tiny_vit.b8 | 34.631 | 10.603 | 3.266x | 231.0 | 754.5 | 0.98210/0.98210 | 1.00000/1.00000 |
| tiny_vit.b1 | 7.138 | 3.329 | 2.144x | 140.1 | 300.4 | 0.98210/0.98210 | 1.00000/1.00000 |

## Full evaluation wall

- baseline: 224.213606421 s
- current: 148.203594978 s
- ratio: 1.5128756252794222x

## Creation cost (runner setup)

| profile | baseline ms | current ms | ratio |
|---|---:|---:|---:|
| mnist_mlp.b32 | 36.542 | 37.216 | 0.982x |
| mnist_mlp.b8 | 17.492 | 17.991 | 0.972x |
| mnist_mlp.b1 | 18.396 | 18.706 | 0.983x |
| tiny_vit.b8 | 18.003 | 62.206 | 0.289x |
| tiny_vit.b1 | 18.160 | 25.102 | 0.723x |

## Structure

| profile | baseline task/submit | current task/submit | baseline upload/download | current upload/download |
|---|---:|---:|---:|---:|
| mnist_mlp.b32 | 3454/2512 | 3454/2512 | 1256/314 | 1256/314 |
| mnist_mlp.b8 | 13750/10000 | 13750/10000 | 5000/1250 | 5000/1250 |
| mnist_mlp.b1 | 110000/80000 | 110000/80000 | 40000/10000 | 40000/10000 |
| tiny_vit.b8 | 88750/88750 | 87500/10000 | 168750/88750 | 167500/87500 |
| tiny_vit.b1 | 150000/150000 | 140000/80000 | 230000/150000 | 220000/140000 |

Packed task-logit tolerances are listed in the JSON output.
