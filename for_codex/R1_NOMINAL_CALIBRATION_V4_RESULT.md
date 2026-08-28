# R1 nominal calibration v4 result

## Decision

`r1_l1_10mm_5deg` is rejected under the approved v4 contract.  This result is
historical evidence only and does not authorize teacher collection or learning.

## Evidence summary

- Controller-free screening: 682 / 682 accepted (22 train buckets, 27 lattice
  points plus 4 interior points per bucket).
- Isaac execution before the monotonic early stop: 10 episodes for the first
  five lattice cases.
- Successful Isaac episodes: 8 / 10.
- Safety failures: 0.
- Fallbacks: 0.
- The rejected candidate timed out in the lift phase because the original
  surface pair did not hold the object orientation.
- A later diagnostic identified surface ports `[10, 18]` with candidate group
  `slot_0:grasp_pair:3`; it passed two independent Isaac replays.  That
  diagnostic is not reclassified as v4 acceptance evidence.

## Immutable bindings

- Formal result:
  `artifacts/p4_full/order9/r1_teacher/calibration_v6_early_tilt_teacher/calibration_result_v4.json`
  (`881b405c5e739e9939b7885f4f9e22a21c1e747734b65b84cfba77a58aeec0e7`)
- Approved v4 protocol:
  `configs/training/order9_r1_nominal_calibration_protocol_v4.yaml`
  (`59b5f9478b6f559b04eb7f378acb458d918aff16a3d656e851aa3ae6d33184ef`)
- v4 approval:
  `for_codex/R1_NOMINAL_CALIBRATION_PROTOCOL_V4_APPROVAL.json`
  (`ca81c4c79a6821b74d77d72d1255f9c21ee36c5694907a45ff4258b37207eced`)
- Teacher override bytes used by v4:
  `configs/training/order9_r1_teacher_overrides_v1.json`
  (`a76f9e90efe3c9bf36a8e864f5640167a41a9210ee024c63ba3b075a6a32ae07`)
- Protected C3 checkpoint:
  `6ea412ccdfe983cb2b030b3514bc8982424522673b49def188d547120984357b`.

The v4 evidence must not be overwritten.  Any correction or speed admission is
published as a new protocol and a new output root.
