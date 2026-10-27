# Host-boundary ledger

- graph: `mnist_mlp_b1`
- profile ledger: `mnist_mlp.b1`

| node | action | causes | measured ms | removable now ms | trace-only outputs | trace-write estimate ms | necessary added cost | graph device gap |
|---|---|---|---:|---:|---|---:|---|---|
| normalize | `mnist.normalize_flatten.v1` | numeric_semantics, current_implementation | 0.064 | 0.000 | - | - |
| pad_k | `tensor.pad_2d.v1` | physical_layout, current_implementation | 0.030 | 0.000 | - | - |
| export_logits | `logits.prefix.v1` | numeric_semantics, resource_interface | 0.028 | 0.000 | - | - |

`same_policy_removable_ms` is zero unless the current prototype
provably removes the action (metadata reshape view). Retention of
numeric Host arithmetic is not counted as loss. Trace-only writes are
reported as a counter/bytes estimate but not added to removable time.

## View aliases

- none

## Summary

- production tensors: 5; trace-only tensors: 0
- Host actions: 3; view-elided: 0; measured Host total: 0.122 ms
- explicitly removed by current same-policy prototype: 0.000 ms (not the whole Host total)
