# SmolLM2 instruction-prompt divergence diagnostic

One same-prefix greedy position was probed per prompt. No V5 gate or observation was modified.

| Prompt | Backend vs archived official | Backend vs corrected official | First divergence | Backend top1 | Declared top1 | Official top1 | Backend margin | Declared margin | Official margin | Classification |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| acc-1 | 32/32 | 32/32 | None | 49 | 49 | 49 | 3.17188 | 3.17188 | 3.19091 | three_way_top1_agreement |
| acc-2 | 3/32 | 32/32 | 0 | 4590 | 4590 | 4590 | 0.03125 | 0.0625 | 0.0790329 | three_way_top1_agreement |
| acc-3 | 1/32 | 32/32 | 0 | 2705 | 2705 | 2705 | 0.46875 | 0.546875 | 0.502281 | three_way_top1_agreement |
| acc-4 | 0/32 | 32/32 | 0 | 2705 | 2705 | 2705 | 0.71875 | 0.734375 | 0.694544 | three_way_top1_agreement |
| acc-5 | 12/32 | 32/32 | 7 | 3384 | 3384 | 3384 | 0.125 | 0.125 | 0.131701 | three_way_top1_agreement |
| acc-6 | 32/32 | 32/32 | None | 504 | 504 | 504 | 1.60938 | 1.60938 | 1.58593 | three_way_top1_agreement |
| acc-7 | 21/32 | 32/32 | 20 | 28 | 28 | 28 | 0.140625 | 0.140625 | 0.13814 | three_way_top1_agreement |
| acc-8 | 1/32 | 32/32 | 0 | 4590 | 4590 | 4590 | 0.375 | 0.390625 | 0.390238 | three_way_top1_agreement |

## Finding

The archived official free-generation path called `model.generate()` without an attention mask.  Because token id 2 is both the chat template's delimiter/EOS and the tokenizer pad id, transformers derived an attention mask with zeros at real prompt positions and then derived position_ids as `cumsum(mask)-1`, resetting RoPE positions after each such token.  Replaying the official model with an explicit all-ones attention mask restores the same-prefix greedy top1 and, in this finite diagnostic, matches the backend for all 256 generated positions across the 8 prompts.  The archived 102/256 match count is therefore not evidence of backend matrix deviation.

This is a reference-tooling defect, not a V5 gate change.  The corrected official generation is kept in a separate delivery directory; V5 inputs, spec, thresholds and sealed observations remain untouched.
