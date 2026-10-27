# Host-boundary ledger

- graph: `tiny_vit_b1`
- profile ledger: `tiny_vit.b1`

| node | action | causes | measured ms | removable now ms | trace-only outputs | trace-write estimate ms | necessary added cost | graph device gap |
|---|---|---|---:|---:|---|---:|---|---|
| patchify | `mnist.patchify_norm.v1` | numeric_semantics, physical_layout, current_implementation | 0.063 | 0.000 | - | - |
| patch_pad | `tensor.pad_2d.v1` | physical_layout, current_implementation | 0.029 | 0.000 | - | - |
| position_add | `tiny_vit.position_add.v1` | numeric_semantics, missing_device_operator | 0.051 | 0.000 | - | exact-shape FP16 FW add is available; broadcast position-add is not |
| rmsnorm1 | `rmsnorm.last_axis.v1` | numeric_semantics, missing_device_operator | 0.071 | 0.000 | - | device row reduction + rsqrt + coefficient materialisation recipe |
| softmax | `attention.scaled_softmax.v1` | numeric_semantics, missing_device_operator | 0.086 | 0.000 | - | device stable max/exp/sum/normalize row reduction and bounded-domain numeric contract |
| merge_heads | `tiny_vit.merge_heads.v1` | physical_layout, current_implementation | 0.030 | 0.000 | - | - |
| residual_attention | `tensor.add.v1` | numeric_semantics | 0.043 | 0.000 | - | - |
| rmsnorm2 | `rmsnorm.last_axis.v1` | numeric_semantics, missing_device_operator | 0.073 | 0.000 | - | device row reduction + rsqrt + coefficient materialisation recipe |
| swiglu | `swiglu.packed_gate_up.v1` | numeric_semantics, missing_device_operator | 0.098 | 0.000 | gate, up | device exact sigmoid/SwiGLU or an independently accepted approximate recipe; current LUT path has a separate numeric policy |
| residual_ffn | `tensor.add.v1` | numeric_semantics | 0.044 | 0.000 | - | - |
| reshape_tokens | `tensor.reshape.v1` | physical_layout | 0.048 | 0.048 | - | - |
| mean_pool | `tensor.mean_axis.v1` | numeric_semantics, missing_device_operator | 0.039 | 0.000 | - | device FP32-preserving token reduction then one FP16 writeback |
| export_logits | `logits.prefix.v1` | numeric_semantics, resource_interface | 0.026 | 0.000 | - | - |

`same_policy_removable_ms` is zero unless the current prototype
provably removes the action (metadata reshape view). Retention of
numeric Host arithmetic is not counted as loss. Trace-only writes are
reported as a counter/bytes estimate but not added to removable time.

## View aliases

- `reshape_tokens`: `h2` -> `h2_tokens` shape `[1, 16, 64]`

## Summary

- production tensors: 25; trace-only tensors: 5
- Host actions: 13; view-elided: 1; measured Host total: 0.701 ms
- explicitly removed by current same-policy prototype: 0.048 ms (not the whole Host total)
