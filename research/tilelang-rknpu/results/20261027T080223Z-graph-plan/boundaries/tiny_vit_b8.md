# Host-boundary ledger

- graph: `tiny_vit_b8`
- profile ledger: `tiny_vit.b8`

| node | action | causes | measured ms | removable now ms | trace-only outputs | trace-write estimate ms | necessary added cost | graph device gap |
|---|---|---|---:|---:|---|---:|---|---|
| patchify | `mnist.patchify_norm.v1` | numeric_semantics, physical_layout, current_implementation | 0.149 | 0.000 | - | - |
| patch_pad | `tensor.pad_2d.v1` | physical_layout, current_implementation | 0.050 | 0.000 | - | - |
| position_add | `tiny_vit.position_add.v1` | numeric_semantics, missing_device_operator | 0.126 | 0.000 | - | exact-shape FP16 FW add is available; broadcast position-add is not |
| rmsnorm1 | `rmsnorm.last_axis.v1` | numeric_semantics, missing_device_operator | 0.161 | 0.000 | - | device row reduction + rsqrt + coefficient materialisation recipe |
| softmax | `attention.scaled_softmax.v1` | numeric_semantics, missing_device_operator | 0.285 | 0.000 | - | device stable max/exp/sum/normalize row reduction and bounded-domain numeric contract |
| merge_heads | `tiny_vit.merge_heads.v1` | physical_layout, current_implementation | 0.056 | 0.000 | - | - |
| residual_attention | `tensor.add.v1` | numeric_semantics | 0.128 | 0.000 | - | - |
| rmsnorm2 | `rmsnorm.last_axis.v1` | numeric_semantics, missing_device_operator | 0.181 | 0.000 | - | device row reduction + rsqrt + coefficient materialisation recipe |
| swiglu | `swiglu.packed_gate_up.v1` | numeric_semantics, missing_device_operator | 0.402 | 0.000 | gate, up | device exact sigmoid/SwiGLU or an independently accepted approximate recipe; current LUT path has a separate numeric policy |
| residual_ffn | `tensor.add.v1` | numeric_semantics | 0.128 | 0.000 | - | - |
| reshape_tokens | `tensor.reshape.v1` | physical_layout | 0.089 | 0.089 | - | - |
| mean_pool | `tensor.mean_axis.v1` | numeric_semantics, missing_device_operator | 0.083 | 0.000 | - | device FP32-preserving token reduction then one FP16 writeback |
| export_logits | `logits.prefix.v1` | numeric_semantics, resource_interface | 0.039 | 0.000 | - | - |

`same_policy_removable_ms` is zero unless the current prototype
provably removes the action (metadata reshape view). Retention of
numeric Host arithmetic is not counted as loss. Trace-only writes are
reported as a counter/bytes estimate but not added to removable time.

## View aliases

- `reshape_tokens`: `h2` -> `h2_tokens` shape `[8, 16, 64]`

## Summary

- production tensors: 25; trace-only tensors: 5
- Host actions: 13; view-elided: 1; measured Host total: 1.876 ms
- explicitly removed by current same-policy prototype: 0.089 ms (not the whole Host total)
