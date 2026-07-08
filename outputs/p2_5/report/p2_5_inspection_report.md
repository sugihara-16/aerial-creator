# P2.5 Inspection Report

- Config path: `configs/training/p2_design_grasp_carry.yaml`
- Generated at: `2026-07-08T11:09:48.536414+00:00`
- Trace JSONL: `../candidate_traces/p2_candidate_trace.jsonl`
- Sample count: `1`
- Candidate records: `5`
- Accepted / rejected / selected: `4` / `1` / `1`
- Valid design rate over exported candidates: `0.800`

P2 は learned design ではなく deterministic scaffold です。
P2.5 inspection path は Isaac / π_H / π_L / QP/PID / actuator command execution を実行しない。
learned models are NOT used in production path。
deterministic P2DesignPolicy / FeasibilityChecker remain source of truth.

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

## Learning Bootstrap Artifacts

- Dataset path: `../datasets/p2_candidate_dataset.jsonl`
- Dataset summary: `../datasets/p2_candidate_dataset_summary.json`
- Train IDs: `../datasets/train_ids.json`
- Val IDs: `../datasets/val_ids.json`
- Train / val sample count: `255` / `65`
- π_D scorer checkpoint: `../training/pi_d_scorer/checkpoint.pt`
- π_D scorer metrics: `../training/pi_d_scorer/metrics.json`
- Feasibility head checkpoint: `../training/feasibility_head/checkpoint.pt`
- Feasibility head metrics: `../training/feasibility_head/metrics.json`

π_D scorer metrics:

| metric | value |
| --- | ---: |
| `num_train_samples` | 255.000000 |
| `num_val_samples` | 65.000000 |
| `selected_accuracy` | 1.000000 |
| `train_loss` | 0.108427 |
| `val_loss` | 0.108393 |

Feasibility head metrics:

| metric | value |
| --- | ---: |
| `binary_accuracy` | 1.000000 |
| `num_train_samples` | 255.000000 |
| `num_val_samples` | 65.000000 |
| `precision` | 1.000000 |
| `recall` | 1.000000 |
| `train_loss` | 0.000125 |
| `val_loss` | 0.000125 |

learned models are NOT used in production path.
deterministic P2DesignPolicy / FeasibilityChecker remain source of truth.

## Scope Notes

- P2 completion gate は変更していません。
- P2.5 は P3 に進む前の inspection / debugging phase です。
- P2.5 learning bootstrap は supervised training のみであり、full RL ではありません。
- P2.5 では Isaac、π_H / π_L、QP/PID、actuator command execution は未実行です。
- learned models are NOT used in production path.
- deterministic P2DesignPolicy / FeasibilityChecker remain source of truth.
- P3 に進む前に、人間が visualization とこの report を確認してください。
