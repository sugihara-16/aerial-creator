# P2.5 Inspection Report

- Config path: `configs/training/p2_design_grasp_carry.yaml`
- Generated at: `2026-07-08T10:39:35.833945+00:00`
- Trace JSONL: `../candidate_traces/p2_candidate_trace.jsonl`
- Sample count: `1`
- Candidate records: `5`
- Accepted / rejected / selected: `4` / `1` / `1`
- Valid design rate over exported candidates: `0.800`

P2 は learned design ではなく deterministic scaffold です。
P2.5 は learned training / Isaac / π_H / π_L / QP/PID を実行しない。
P2.5 では actuator command も生成しません。

## Variant Counts

| variant | count |
| --- | ---: |
| `central_base_plus_two_grasp_arms` | 1 |
| `chain_grasp` | 1 |
| `symmetric_two_anchor_grasp` | 1 |
| `tri_anchor_support_grasp` | 1 |
| `tri_anchor_support_grasp_closed_loop_probe` | 1 |

## Selected Variant Distribution

| variant | selected_count |
| --- | ---: |
| `tri_anchor_support_grasp` | 1 |

## Required Slot Coverage Summary

| metric | min | mean | max |
| --- | ---: | ---: | ---: |
| `required_slot_coverage` | 1.000 | 1.000 | 1.000 |

## Feasibility Margin Summary

| metric | min | mean | max |
| --- | ---: | ---: | ---: |
| `anchor_coverage` | 1.000 | 1.000 | 1.000 |
| `capability_coverage` | 1.000 | 1.000 | 1.000 |
| `payload_margin` | 5.483 | 6.780 | 8.725 |
| `reachability_margin` | 1.000 | 1.000 | 1.000 |
| `thrust_margin` | 2.830 | 3.130 | 3.417 |

## Violation Code Histogram

| violation_code | count |
| --- | ---: |
| `F_CLOSED_LOOP_REJECT_V1` | 1 |

## Visualization Files

| variant | graph view | simple 3D layout view |
| --- | --- | --- |
| `central_base_plus_two_grasp_arms` | [central_base_plus_two_grasp_arms_graph.svg](../visualization/central_base_plus_two_grasp_arms_graph.svg) | [central_base_plus_two_grasp_arms_layout.svg](../visualization/central_base_plus_two_grasp_arms_layout.svg) |
| `chain_grasp` | [chain_grasp_graph.svg](../visualization/chain_grasp_graph.svg) | [chain_grasp_layout.svg](../visualization/chain_grasp_layout.svg) |
| `symmetric_two_anchor_grasp` | [symmetric_two_anchor_grasp_graph.svg](../visualization/symmetric_two_anchor_grasp_graph.svg) | [symmetric_two_anchor_grasp_layout.svg](../visualization/symmetric_two_anchor_grasp_layout.svg) |
| `tri_anchor_support_grasp` | [tri_anchor_support_grasp_graph.svg](../visualization/tri_anchor_support_grasp_graph.svg) | [tri_anchor_support_grasp_layout.svg](../visualization/tri_anchor_support_grasp_layout.svg) |

## Representative Selected Designs

| episode | variant | source | selected | accepted | score | violations |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `p2_5_trace_0000` | `tri_anchor_support_grasp` | `policy_variant` | 1 | 1 | 11.381 | none |

## Representative Rejected Designs

| episode | variant | source | selected | accepted | score | violations |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `p2_5_trace_0000` | `tri_anchor_support_grasp_closed_loop_probe` | `closed_loop_invalid_probe` | 0 | 0 | -90.219 | F_CLOSED_LOOP_REJECT_V1 |

## Scope Notes

- P2 completion gate は変更していません。
- P2.5 は P3 に進む前の inspection / debugging phase です。
- P2.5 では Isaac、π_H / π_L、QP/PID、actuator command、learned training は未実行です。
- P3 に進む前に、人間が visualization とこの report を確認してください。
