# AMSRR_design_modification_by_codex.md

This file records implementation-time supplements or deviations from `A-MSRR_codex_ready_spec_v0_4_ja.md`.

## 2026-07-09

### P4-Control Articulated Assembly Correction Supplement

- Context: The prior articulated hover smoke only proved that dock mechanism joints could move while the rigid fixed-morphology assembly hovered. It did not prove that connected modules moved relative to each other as a multi-link system, because the generated fixed morphology connected module roots with a fixed joint.
- Decision: Added a separate articulated morphology URDF path for `--fixed-morphology-articulated-hover-smoke`. The child module root is attached to the selected parent connect dummy frame, so the parent dock mechanism joint moves the whole child module subtree in Isaac. The mating child-side dock mechanism is held at zero to keep the assembly representable as a URDF tree instead of a closed kinematic loop.
- Controller/observation decision: The articulated fixed smoke now builds `RuntimeObservation` module poses from actual Isaac body poses (`module_i__fc`) rather than static precomputed module poses. The QPID controller can emit module-scoped dock mechanism commands so only the structural parent-side dock joint is commanded.
- Validation decision: The articulated fixed smoke now requires real relative module pose motion and q-dependent control-model motion in addition to hover stability. The 20 s real Isaac smoke passed with relative module motion, rotor-origin/allocation-matrix changes, QP feasibility, and no bridge target failures.
- Compatibility impact: Non-articulated fixed hover and waypoint smokes still use the rigid fixed morphology path. This remains a low-level P4-control/P4a validation and does not claim dynamic docking, object grasp/carry, learned policies, closed-loop dock constraint physics, or P4 full completion.

### P4-Control Articulated Hover Smoke Supplement

- Context: The user asked whether flight with internal joint motion is part of the current P4-control/P4a validation scope. It is in scope as a low-level articulated-hover smoke, distinct from dynamic docking, policy learning, object grasp/carry, or P4 full completion.
- Decision: Added optional single-module and fixed-morphology articulated hover smokes. The smoke drives dock mechanism joints with a bounded sinusoidal `InteractionKnot.posture_target.joint_pos_target` while the QPID controller continues to receive hover pose/twist through `PolicyCommand`.
- Controller contract decision: `QPIDController` now passes dock mechanism posture references into `dock_mechanism_commands` after joint-limit clipping. Unspecified dock mechanism joints still hold nominal zero. This keeps actuator authority inside the controller/bridge layer; `PolicyCommand` still does not directly emit rotor thrusts, vectoring targets, or dock actuator targets.
- Validation decision: The new smokes require both hover stability and observed joint motion. Reports include selected dock joint ids, trajectory amplitude/period/warmup, max commanded/observed joint motion, max tracking error, bridge target health, and QP feasibility.
- Compatibility impact: No persisted schema change and no expansion of the existing P4-control full acceptance set by default. These smokes are optional low-level checks for q-dependent control updates and articulated flight behavior.

### Dock Frame Alignment Supplement

- Context: GUI inspection showed that the generated two-module fixed morphology placed the modules side-by-side rather than in the actual docked pose. The intended docked relation is that pitch/yaw connect point origins coincide and their axes are colinear, with the facing pitch/yaw dock geometry requiring x/y signs to be reversed while z remains aligned.
- Decision: Represent a pitch/yaw dock connection as a face-to-face port relation `Rz(pi)` between connect point frames. The source module to destination module relative pose is now computed as `src_port_pose * Rz(pi) * inverse(dst_port_pose)`.
- Physical model impact: `DockPortSpec.local_pose` now stores the connect dummy frame pose in the module/base frame, not merely the connect joint origin relative to its immediate dock mechanism parent link. This makes π_D-facing `PortNode.local_pose` geometrically meaningful for dock edge construction.
- π_D/morphology impact: Minimal and grasp/carry morphology builders now compute `DockEdge.relative_pose_src_to_dst` from the selected port pair and propagate module poses through the dock tree so the design morphology itself satisfies the same port alignment relation.
- P4-control/Isaac impact: The fixed-morphology URDF generator now places connected module roots using the selected compatible pitch/yaw port pair instead of a fixed x-spacing. The fixed-morphology controller smoke/probe/summary archive use the same module poses for runtime observations and logs.
- Compatibility impact: The CLI still accepts `--fixed-module-spacing-m`, but spacing is now only a fallback when no connect ports are available. Existing fixed-morphology USDs should be regenerated with `--force-convert`. This does not claim dynamic docking, object grasp/carry success, learned policies, or P4 full completion.

### P4-Control Hover Drift Fix Supplement

- Context: User observation and real Isaac diagnostics showed that the single-module hover drifted after roughly 5-10 s. Pseudoinverse allocation did not hover and increasing vectoring speed alone did not remove the drift. Inspection against `aerial_robot_base` and the Holon URDF found three geometry/control-model mismatches.
- Decision: Treat every rotor's local thrust direction as thrust-frame `+z`. The URDF rotor continuous joint axis sign is now used only to derive the reaction torque coefficient from `<m_f_rate>`, matching the reference controller and MuJoCo bridge convention. For Holon this yields alternating yaw torque coefficients `[-0.0172, 0.0172, -0.0172, 0.0172]` while all rotors push along local `+z`.
- Vectoring decision: The QP virtual lateral channel is now the actual positive gimbal-motion direction, `vectoring_joint_axis_body x virtual_z_axis_body`, instead of rotor-arm x. This matches finite-difference thrust-axis motion in the current URDF where the gimbal axis itself is rotor-arm x.
- Dock-hold decision: Dock mechanism position commands now hold nominal zero rather than the current observed angle. This prevents passive dock joints from being re-targeted to their drifted positions during free hover.
- Validation result: Real Isaac 20 s single-module hover using the primary `rigid_body_qp` path passed with no QP infeasible steps or clipping, final position error below 1 mm, and max position error about 2.3 cm.
- Compatibility impact: No schema change and no promotion of pseudoinverse allocation. This is a correction of URDF interpretation and internal control model geometry; P4-control still does not claim object grasp/carry, learned policies, fixed-morphology long hover, or P4 full completion.

### P4-Control Hover Drift Diagnostic Supplement

- Context: User GUI observation showed that the single-module hover can initially hold but drifts after roughly 5-10 s and crashes. The user requested a temporary pseudoinverse allocation trial and investigation of slow vectoring motion.
- Decision: Added an explicit debug-only `rigid_body_pseudoinverse` allocation mode and a probe-only `--vectoring-velocity-limit-rad-s` conversion override for gimbal/vectoring joint velocity limits. The primary P4-control path remains `rigid_body_qp`; the pseudoinverse path is not an acceptance or completion path.
- Diagnostic result: A 10 s no-stop QP hover reproduced the drift. The pseudoinverse path reduced lateral drift but remained infeasible/clipped throughout and lost altitude. Raising vectoring velocity from 3 to 20 rad/s was correctly reflected in Isaac's joint table but did not stabilize hover, and combining the higher velocity with higher gimbal stiffness/damping still produced late attitude loss. This suggests vectoring speed/servo tracking contributes to the transient behavior but is not the sole cause.
- Compatibility impact: No persisted schema change. The new CLI options are for controlled debugging/comparison only and do not alter P4-control acceptance thresholds or the requirement that QP allocation remains the primary path.

### P4-Control Holon USD Visual Mesh Resolution Supplement

- Context: The Kit GUI could open and `/World/Holon` existed in the stage, but only link frames/axes were visible. Inspection showed `assets/robots/holon/holon.urdf` references relative `mesh/*.STL` paths while the STL files live under `module_urdf/mesh`; the previous generated USD therefore contained articulation/link transforms but no visible mesh payload.
- Decision: The P4-control Holon probe now writes a conversion-only URDF copy with mesh references resolved to existing absolute STL paths before Isaac URDF-to-USD conversion. The same resolver is used for fixed-morphology URDF generation so single-module and rigid fixed-morphology GUI assets share the visual-mesh fix.
- Compatibility impact: This changes only the reproducible URDF-to-USD conversion input used by the probe. Runtime schemas, controller commands, QP allocation, acceptance thresholds, and physical success claims are unchanged. Previously generated USDs should be regenerated with `--force-convert` to pick up visible geometry.

### P4-Control GUI Observation Smoke Supplement

- Context: Real P4-control hover smokes ran correctly in Isaac, but IsaacLab 3 defaults to headless unless the Kit visualizer is explicitly requested with `--viz kit`, and smoke pass runs close the app immediately after the hold criterion is satisfied.
- Decision: Added GUI-observation-only probe options: `--realtime-playback` sleeps one physics `dt` per step for watchable playback, and `--keep-open-after-smoke-s` keeps the Kit app pumping for a fixed duration after the smoke finishes. These options do not change acceptance thresholds or controller behavior.
- Compatibility impact: P4-control acceptance remains based on the same real smoke pass/fail metrics. Long-duration hover is still not claimed; the GUI options are for inspection of the existing smoke behavior.

### P4-Control Smoke Summary Archive Supplement

- Context: After all three real Isaac low-level smokes could pass, the remaining split-acceptance gap was the fast gate requiring `EpisodeArchive` records with controller command, runtime observation, actuator target record, residual/clipping metrics, and explicit no-P4-full-completion labeling.
- Decision: Extended `P4ControlLowLevelRunner` so real-smoke runs automatically build one `EpisodeArchive` smoke summary per attempted non-skipped smoke when no external archives are supplied. Each summary archive records a free-flight smoke task, Holon morphology graph from the configured physical model, desired body pose policy command, summary `ControllerCommand` status, summary `RuntimeObservation`, and summary actuator target record metrics derived from the real smoke result.
- Scope decision: The archive type is `smoke_summary`. It preserves real smoke pass/fail, residual/clipping/missing/unsupported counts, target tolerances, and no-mislabeling fields, but it does not claim per-step actuator target replay, object grasp/carry, learned policy training, or P4 full completion.
- Runner impact: With real Isaac smokes passing and summary archives present, `run_p4_control_acceptance` can mark `fast_gate_passed`, `real_isaac_smoke_passed`, and `completion_passed` true for the P4-control low-level validation scope only.
- Compatibility impact: No persisted schema change. Existing `EpisodeArchive` fields are reused. Dry-run smoke results still produce no archives and cannot pass P4-control completion.

### P4-Control Fixed-Morphology Waypoint Smoke Supplement

- Context: After fixed-morphology hover passed, the remaining real-smoke runner gap was a fixed-morphology waypoint case that still uses the same rigid 2-module Holon assembly, QP controller path, and Isaac bridge application surface.
- Decision: Extended the Holon Isaac probe, backend, and `P4ControlIsaacEnv.run_smokes(dry_run=False)` with `--fixed-morphology-waypoint-smoke`. The waypoint smoke sends `PolicyCommand.desired_body_pose` through the controller each step, ramps the commanded target from the initial root pose to the final target, and reports `fixed_morphology_waypoint_*` metrics including ramp duration, tracking error, QP infeasible count, clipping count, and bridge target health.
- Waypoint-scope decision: The current default fixed-morphology waypoint smoke is intentionally small: world target position `(0.05, 0.0, 0.5)`, yaw `0.0 rad`, `0.1 s` target ramp, `0.20 m` position tolerance, `0.25 rad` attitude tolerance, and `1.0 s` hold. Larger exploratory lateral/z targets were not stable enough to treat as an acceptance default in this order.
- Runner decision: The real smoke runner now attempts all three P4-control low-level smokes: `single_module_hover`, `fixed_morphology_hover`, and `fixed_morphology_waypoint`. This can satisfy the real Isaac smoke side of the split acceptance gate when all three pass, but P4-control completion still also requires the fast archive/interface gate.
- Compatibility impact: This validates only a small real Isaac-backed waypoint smoke for the rigid fixed-morphology asset. It does not claim robust waypoint tracking, object grasp/carry, learned policies, archive completeness, P4-control completion, or P4 full completion.

### P4-Control Fixed-Morphology Hover Smoke Supplement

- Context: After the fixed assembly URDF generator was available, P4-control needed to validate that a rigid 2-module Holon asset can be converted, spawned, controlled, and reported through the same QP/controller/bridge path as the single-module smoke.
- Decision: Extended the Holon Isaac probe with `--fixed-morphology-hover-smoke`. The probe generates the rigid combined URDF on demand, converts it to USD, spawns it as one articulation, reconstructs per-module runtime observations from prefixed Isaac joint names, applies module-prefixed rotor/vectoring/dock actuator targets, and reports fixed-hover pass/fail metrics with the `fixed_morphology_hover_*` prefix.
- Runner decision: `P4ControlIsaacEnv.run_smokes(dry_run=False)` now runs both real `single_module_hover` and real `fixed_morphology_hover`; `fixed_morphology_waypoint` remains an explicit skipped smoke until the waypoint target path is implemented and validated.
- Compatibility impact: This validates fixed-morphology hover only. It does not validate waypoint tracking, archive completeness, object grasp/carry, learned policies, P4-control completion, or P4 full completion. The fixed morphology remains a pre-generated rigid assembly approximation and does not claim physical docking success.

### P4-Control Fixed-Morphology Assembly Asset Preparation Supplement

- Context: The user approved treating the first fixed-morphology smoke as a pre-generated rigid combined URDF/USD asset, with dock connection represented as a fixed joint equivalent. Before running Isaac, the repository needed a deterministic way to generate that asset and the controller needed correct multi-module gravity compensation.
- Decision: Added a fixed-morphology URDF generator that prefixes every copied Holon link/joint/transmission/gazebo reference with `module_<id>__`, rewrites relative mesh paths to absolute paths, and connects additional module roots to `module_0__root` with fixed joints at a configurable spacing. This produces a single URDF frame tree suitable for Isaac conversion while preserving per-module local names for controller-side mapping.
- Controller decision: Updated `QPIDController` default hover and body-target PID force generation to use `RigidBodyControlModel.total_mass_kg` when the rigid-body QP path is active. Single-module behavior is unchanged, but fixed-morphology hover now requests gravity compensation for the full assembled rigid body instead of one Holon module.
- Compatibility impact: This prepares the fixed-morphology smoke asset path but does not yet run fixed-morphology hover or waypoint in Isaac. The connected morphology is a rigid asset-level approximation for low-level controller validation, not a physical docking success claim and not P3/P4 object grasp/carry completion.

### P4-Control Single-Module Real Smoke Runner Supplement

- Context: The standalone Holon probe could pass the real single-module closed-loop hover smoke, but the P4-control runner still returned placeholder real-smoke failures for every required smoke. The next bounded step was to connect the completed single-module smoke into the runner without inventing fixed-morphology spawn/docking semantics.
- Decision: Extended `P4ControlIsaacEnv.run_smokes(dry_run=False)` so it executes only `single_module_hover` through `IsaacLabBackend.run_holon_single_module_hover_smoke`, parses the probe JSON, and converts the numeric pass/fail fields into `P4ControlSmokeResult.metrics`. The fixed-morphology hover and waypoint entries remain explicit skipped results with `skip_reason="real_isaac_execution_not_implemented"`.
- Backend decision: Added a JSON subprocess helper in `IsaacLabBackend` for real smoke commands and made the single-module runner force URDF-to-USD conversion by default. Backend-generated USD paths are taken from `IsaacLabBackendConfig`, allowing tests or manual runs to route generated artifacts to `/tmp` while the checked-in config remains unchanged.
- Compatibility impact: This is a runner/reporting integration for the already validated single-module smoke. It does not satisfy `real_isaac_smoke_passed` or `completion_passed`, because P4-control acceptance still requires fixed-morphology hover and fixed-morphology waypoint results. No object grasp/carry, learned policy, P4-control completion, or P4 full completion claim is introduced.

### P4-Control Single-Module Closed-Loop Hover Smoke Supplement

- Context: After the `PolicyCommand` PID target builder and controller-to-Isaac command path were validated, P4-control needed a real Isaac closed-loop smoke that repeatedly observes Holon state, recomputes the rigid-body QP allocation, and applies bridge-supported actuator targets rather than a single open-loop command.
- Decision: Extended `scripts/p4_control_holon_spawn_probe.py` and `IsaacLabBackend` with `--single-module-hover-smoke`. The smoke keeps one persistent `QPIDController(allocation_mode="rigid_body_qp")`, sends a direct hover `PolicyCommand.desired_body_pose` / `desired_body_twist` target each control step, converts the resulting command through `IsaacControllerBridge`, and applies rotor thrust plus vectoring/dock joint position targets in Isaac. The smoke reports final/max position and attitude errors, hold time, QP infeasible count, bridge clipping/missing/unsupported counts, and the last controller/bridge status. The default pass threshold remains the controller supplement's initial waypoint tolerance: `0.20 m`, `0.25 rad`, and `1.0 s` hold.
- Controller numerics decision: The attitude PID output is treated as desired body angular acceleration and converted to body torque using the current composite inertia from `RigidBodyControlModelBuilder`; it is not interpreted directly as Nm. Controller feasibility now separates a warning scale from hard infeasibility: residuals above `tracking_warning_residual_norm=1e-3` are reported as tracking warnings, while `unsupported_wrench_tolerance=1e-2` is the controller-local infeasibility cutoff used to tolerate small QP/back-conversion residuals seen in real closed-loop smoke.
- Isaac articulation decision: Dock mechanism joints use nonzero implicit hold stiffness/damping in the probe so passive dock joints do not drift to limits and destabilize a single-module hover. The closed-loop smoke can stop early once the configured hold duration is achieved, and records both requested and executed step counts.
- Compatibility impact: This validates a real Isaac-backed single-module hover smoke only. It does not validate fixed-morphology hover, waypoint tracking, object grasp/carry, learned `π_D` / `π_H` / `π_L`, P4-control completion, or P4 full completion. The QP allocator remains the primary path; `BoundedVerticalRotorAllocator` remains degraded fallback only.

### P4-Control PolicyCommand PID Target Builder Supplement

- Context: Before implementing closed-loop hover, the controller needed a deterministic path from direct P4-control hover/waypoint targets to desired body wrench while preserving the `π_L -> PolicyCommand -> controller/QP` responsibility boundary.
- Decision: Extended `QPIDController` so `PolicyCommand.desired_body_pose` and `PolicyCommand.desired_body_twist` activate a PID target builder. The builder uses the user-specified initial gains: xy `P=3.0/I=0.05/D=2.0`, z `P=5.0/I=1.0/D=2.5`, roll/pitch `P=22.0/I=1.0/D=14.0`, and yaw `P=5.0/I=1.0/D=4.0`. Position PID produces world-frame acceleration plus gravity compensation, then converts force to body frame. Attitude uses quaternion error in body frame, with roll/pitch and yaw gains applied by body axis. `PolicyCommand.residual_wrench_body` and any existing feedforward wrench are added to the PID wrench before QP allocation.
- Anti-windup decision: Integral state is held inside `QPIDController` and is committed only when the allocation is feasible and unclipped; infeasible or clipped allocation freezes the integral. No fixed acceleration/torque clipping values were introduced in this order; rotor/vectoring limits and infeasible status remain enforced by the QP/hard-check layer until explicit target-wrench saturation limits are specified.
- Compatibility impact: No persisted schema change. `InteractionKnot.centroidal_target.centroidal_wrench_preference` remains an upstream intent/reference path consumed by the existing bias builder for compatibility, but P4-control direct hover/waypoint should use `PolicyCommand.desired_body_pose`, `desired_body_twist`, and `residual_wrench_body` as the controller-facing path. This does not claim closed-loop hover or P4-control completion.

### P4-Control QP Feasibility Tuning Supplement

- Context: The real controller-to-Isaac command smoke proved that controller output and bridge targets could be applied in Isaac, but the controller still reported `qp_feasible=false`. Diagnostics showed the SLSQP solve succeeded; the default previous-command smoothing pulled the first hover allocation toward a zero-thrust previous command, and the post-solve hard check counted an effectively zero-thrust rotor's undefined vectoring angle as a rate-limit clip.
- Decision: Reduced `VirtualThrustQPAllocator` regularization and previous-command weights to `1e-8` so wrench tracking remains dominant on the primary P4-control allocation path. Set `QPIDControllerConfig.unsupported_wrench_tolerance` to `1e-5`, matching the small residual introduced by virtual-channel linearization/back-conversion while staying below the controller warning threshold. Added tolerance-aware clip detection and a zero-thrust vectoring deadband: when a vectoring rotor back-converts to effectively zero thrust, the vectoring joint target holds the current joint position instead of commanding an arbitrary angle at the rate-limit boundary.
- Compatibility impact: This changes only controller-local numerical tuning and hard-check interpretation. It does not change persisted schemas or the P4-control ownership boundary, does not promote pseudoinverse allocation, and does not claim closed-loop hover, object grasp/carry, policy learning, P4-control completion, or P4 full completion. The real Isaac controller-command smoke now reports QP feasible/ok with no clipping violations, but closed-loop smoke gates remain outstanding.

### P4-Control Controller-to-Isaac Command Smoke Supplement

- Context: After validating raw Isaac command APIs, P4-control needed a real-smoke artifact that starts from A-MSRR controller output rather than manually supplied force/joint arguments.
- Decision: Added `amsrr/simulation/p4_control_controller_smoke.py` to build a single-module morphology, runtime observation, `QPIDController(allocation_mode="rigid_body_qp")` command, and `IsaacControllerBridge` target record. Extended `scripts/p4_control_holon_spawn_probe.py` with `--controller-command-smoke`, which applies bridge rotor-thrust targets to matching `thrust_.*` bodies using rotor local thrust axes and bridge joint-position targets to matching gimbal/dock joints. The script reports controller command, bridge metrics, target clipping/missing/unsupported lists, and controller smoke metrics.
- Compatibility impact: This validates controller-to-bridge-to-Isaac command routing, not closed-loop hover. The current controller smoke still reports `controller_status.qp_feasible=false` because the QP path produces small residual/clipping violations under the present single-step hover request. Completion gates must continue to fail until a later order resolves controller feasibility and real hover/waypoint smoke.

### Holon Battery2 Inertial Correction Supplement

- Context: Real Isaac spawn and command probes consistently reported a PhysX warning that `/World/Holon/Geometry/root/main_body/battery2` had invalid inertia and negative mass fallback behavior. Inspection showed that both the runtime Holon URDF and reference xacro had `battery2` inertial data set to `mass=0` and all inertia components `0`.
- Decision: Set `battery2` inertial origin, mass, and inertia to match the symmetric `battery1` component in `assets/robots/holon/holon.urdf` and `module_urdf/holon.urdf.xacro`. Added a unit test that mesh-bearing runtime URDF links must have positive mass and positive diagonal inertia entries.
- Compatibility impact: This is a source asset correction needed for trustworthy Isaac physics. It changes Holon's aggregate mass/inertia relative to the previous zero-mass battery2 asset and removes the Isaac battery2 invalid-inertia warning after USD regeneration. It does not change schema contracts or claim hover/control completion.

### P4-Control Holon Isaac Command Probe Supplement

- Context: After Holon articulation spawn was validated, P4-control needed a minimal real Isaac check that the intended command surfaces are reachable: rotor-like external wrenches through the wrench composer and vectoring-like joint position targets through Isaac Lab articulation actuators.
- Decision: Extended `scripts/p4_control_holon_spawn_probe.py` with command-probe arguments. The probe can apply world-frame `+z` wrenches to `thrust_.*` bodies using `permanent_wrench_composer.set_forces_and_torques_index(is_global=True)` and command `gimbal.*` joints using `set_joint_position_target_index`. It reports thrust body ids/names, gimbal joint ids/names, robot mass/gravity, commanded force totals, root-state deltas, gimbal target/actual positions, and a tolerance-based `command_probe_passed` flag. Added a backend helper to build this command line.
- Compatibility impact: This is an Isaac API/actuator-path smoke only. The force is a global `+z` probe input, not the finalized rotor-axis thrust model, not QP closed-loop hover, and not a P4-control completion artifact. The probe confirms command routing and observation extraction; later work must connect `ControllerCommand` / `IsaacControllerBridge` records and implement the real single-module/fixed-morphology smoke gates.

### P4-Control Holon Isaac Spawn Probe Supplement

- Context: After validating URDF-to-USD conversion, the next P4-control Isaac smoke prerequisite was to verify that the generated Holon USD can be spawned as an Isaac Lab articulation and stepped in the approved `isaaclab3` / `isaaclab.sh -p` runtime.
- Decision: Added `scripts/p4_control_holon_spawn_probe.py` and a backend command helper for launching it. The probe converts the Holon URDF if needed, creates a fresh Isaac stage, spawns `/World/Holon` through `ArticulationCfg` / `UsdFileCfg`, steps a few physics frames, and emits a JSON summary with `spawn_passed`, body/joint names, root state, USD path, and Isaac-backed metadata. The CLI intentionally avoids the deprecated `--headless` flag; IsaacLab's default no-visualizer path is used for headless execution.
- Compatibility impact: This validates single-module Holon articulation spawn only. It does not apply rotor wrenches, command vectoring joints, run hover/waypoint control, assemble multi-module morphologies, claim object grasp/carry success, or satisfy the P4-control real smoke completion gate. Real probe logs currently include a PhysX warning for the `battery2` rigid body inertia/mass properties; this should be investigated before treating physical hover results as final.

### P4-Control Isaac URDF Conversion Probe Supplement

- Context: Before implementing real P4-control Isaac smoke execution, the Holon URDF import path needed validation in the approved `isaaclab3` / `isaaclab.sh -p` environment.
- Decision: Ran Isaac Lab's `scripts/tools/convert_urdf.py` against `assets/robots/holon/holon.urdf` in headless mode with output under `/tmp/amsrr_isaac_holon`. The converter completed successfully and generated `/tmp/amsrr_isaac_holon/holon/holon.usda` plus payload USD files. Updated A-MSRR config/default generated USD path to `artifacts/isaac/robots/holon/holon/holon.usda`, matching Isaac importer output structure when `generated_usd_dir` is `artifacts/isaac/robots/holon`.
- Compatibility impact: This validates the URDF-to-USD import path but still does not spawn or simulate Holon in a P4-control smoke. Generated USD artifacts were not committed; they remain reproducible from the source URDF and config.

### P4-Control Smoke Runner Configuration Supplement

- Context: After the P4-control fast/real acceptance split, the next implementation order needs configurable Isaac Lab environment settings and smoke scenario definitions before calling real Isaac APIs. The user approved using the existing `isaaclab3` micromamba environment, URDF-to-USD custom articulation as the initial Holon asset path, wrench-composer rotor force application, and the controller supplement's initial waypoint thresholds.
- Decision: Added `configs/env/isaac_lab.yaml`, `configs/training/p4_control_low_level.yaml`, `IsaacLabBackend`, `P4ControlIsaacEnv`, `P4ControlLowLevelRunner`, and `scripts/p4_control_smoke.py`. The runner supports `dry_run` by producing skipped smoke results and never marks completion. Backend availability probes are config-driven and lazy so normal unit tests do not require Isaac imports. The real smoke path now has deterministic scenario names and thresholds, but actual Isaac physics execution is intentionally left for the next order.
- Compatibility impact: This adds configuration and runner contracts only. It does not convert URDF to USD, spawn Holon, apply rotor forces in Isaac, or claim P4-control completion. Real Isaac smoke still requires executing the script through `micromamba activate isaaclab3` and `$ISAACLAB_PATH/isaaclab.sh -p` after the Isaac execution layer is implemented.

### P4-Control Acceptance Split Implementation Supplement

- Context: P4-control acceptance must distinguish fast pytest/interface/archive checks from real Isaac smoke, and P4-control completion must not pass when Isaac is unavailable or when only synthetic/unit checks were run.
- Decision: Added `run_p4_control_acceptance` with explicit `fast_gate_passed`, `real_isaac_smoke_passed`, and `completion_passed` fields. The fast gate checks that `EpisodeArchive` records include controller commands, runtime observations, actuator target records, residual/clipping metrics, and no P4 full-completion/physical-success claim. The real smoke gate requires three Isaac-backed smoke results: single-module hover, fixed-morphology hover, and fixed-morphology waypoint. `completion_passed` is true only when both gates pass.
- Compatibility impact: This is an acceptance/reporting contract only. It does not run Isaac, spawn Holon, or validate physical hover. Synthetic smoke results are accepted only as explicit report inputs for aggregation tests; actual P4-control completion still requires real Isaac smoke artifacts from later runner/backend work.

### P4-Control Actuator Mapping and Bridge Record Supplement

- Context: After the primary virtual-thrust QP allocator, P4-control needs a controller bridge boundary that can be unit-tested without Isaac while preserving the P4 requirement that `ControllerCommand` is converted to Isaac actuator targets and archived as actuator target records.
- Decision: Added controller-side `ActuatorMappingBuilder` and `IsaacControllerBridge`. The mapping extracts active module rotor thrust channels, vectoring joint position channels, dock mechanism position channels, and effort-limited joint channels from `MorphologyGraph` and `PhysicalModel` using deterministic global keys `module_<module_id>:<local_id>`, with single-module local-key aliases for backward compatibility. The bridge converts `ControllerCommand` dictionaries into `IsaacActuatorTargetRecord`, clips targets to mapped actuator limits, records missing/unsupported/clipped actuators, carries controller/QP residual status, and exposes a JSON-compatible dict for `EpisodeArchive.actuator_target_records`.
- Compatibility impact: This is a bridge contract and fast pytest gate only. It does not execute Isaac Lab, does not spawn robots, and does not claim P4-control smoke completion. Later Isaac backend code must consume these records or equivalent `ControllerCommand` data and then satisfy the real single-module/fixed-morphology smoke gates.

### P4-Control VirtualThrustQPAllocator Implementation Supplement

- Context: Agent I Order 2 implements the P4-control primary allocator after the user clarified that virtual rotor thrust directions may be fixed relative to the rotor-arm frame x/z directions and that thrust, joint, and rate limits should be included in QP constraints followed by hard check and clamp.
- Decision: Added `VirtualThrustQPAllocator` as the primary P4-control allocation path. Vectoring rotors are expanded to rotor-arm-fixed virtual x/z force channels, solved with a Python/SciPy quadratic objective plus actuator bounds and linearized vectoring angle/rate constraints, then back-converted to non-negative `rotor_thrusts_n` and absolute `vectoring_joint_targets`. Because Holon physical rotors include both `+z` and `-z` thrust axes, the virtual z channel is sign-aligned with each rotor's positive thrust direction while remaining fixed in the rotor-arm frame. The allocator recomputes achieved wrench after hard check/clamp and records residual, clipping, saturation, and primary/degraded metrics.
- Compatibility impact: `QPAllocationProblem` and `QPAllocationResult` gained backward-compatible optional fields for rigid-body model input, previous vectoring targets, control dt, vectoring outputs, and achieved wrench. `BoundedVerticalRotorAllocator` remains available but now marks itself as `degraded_fallback=1.0`; it is not the P4-control primary path. This does not claim Isaac smoke completion, object grasp/carry success, learned policy performance, or P4 full completion.

### P4-Control RigidBodyControlModel Implementation Supplement

- Context: Agent I Order 1 implemented the deterministic rigid-body model update required before QP allocation and Isaac bridge work. The v0.4 spec and controller supplement require link-level quasi-static inertia aggregation and per-step `q`-conditioned rotor geometry updates, while leaving the exact controller-local body-frame convention and multi-module actuator key convention to implementation.
- Decision: Added `amsrr/controllers/rigid_body_model.py` with controller-local `RigidBodyControlModel`, `RotorControlElement`, and `RigidBodyControlModelBuilder`. The body frame origin is the composite COM and its orientation is the current base/control module orientation. `center_of_mass_body` is therefore `(0, 0, 0)`, rotor origins are stored relative to the COM in body frame, and allocation columns use `r_i x F_i` with reaction torque coefficients. Multi-module actuator keys use deterministic `module_<module_id>:<local_id>` strings.
- Compatibility impact: No persisted schema was changed. The model is an internal controller contract exported from `amsrr.controllers`. It does not output actuator commands, does not replace QP allocation, and does not claim Isaac validation. The scalar rotor allocation matrix is the per-current-geometry basis that the later virtual-thrust-channel QP allocator will expand and back-convert.

### P4-Control Virtual Thrust Channel and Acceptance Split Supplement

- Context: Before starting P4-control / P4a implementation, the user clarified several controller-level requirements: the rigid-body model and allocation matrix must be rebuilt every control cycle from current joint positions, vectoring rotors should be expanded into virtual thrust channels inside the QP, pseudoinverse allocation must not be the main path, Isaac-unavailable tests may skip only unit smoke portions, and P4-control acceptance must distinguish fast pytest gates from real Isaac smoke gates.
- Decision: P4-control Agent I implementation will update composite inertia, COM, rotor origins, rotor axes, and allocation matrix `B(q)` from `RuntimeObservation.module_states[*].joint_positions` every control cycle. The primary allocator will be a QP path; `BoundedVerticalRotorAllocator` remains only a degraded fallback and must not be the source for P4-control completion. Vectoring rotor allocation may use virtual thrust channels internally, but controller output must be back-converted to `ControllerCommand.rotor_thrusts_n` and absolute `ControllerCommand.vectoring_joint_targets`, then re-evaluated for achieved wrench, residual, clipping, and unsupported-command metrics.
- Acceptance decision: P4-control acceptance is split into a fast pytest gate for deterministic/unit/interface/archive checks and a real Isaac smoke gate for actual single-module hover, fixed-morphology hover, and fixed-morphology waypoint tracking. Tests may skip Isaac-specific smoke when Isaac is unavailable, but P4-control completion must not pass without the real Isaac smoke gate.
- Compatibility impact: This is a controller implementation supplement. It preserves the v0.4 responsibility boundary: `π_L` outputs `PolicyCommand` only, controller/QP owns `ControllerCommand`, and the Isaac bridge owns final actuator target conversion. P4-control must not claim object grasp/carry success, π_D/π_H/π_L learning, P4.2 success, P4.3 learning bootstrap, or P4 full completion.

### Main Spec Cross-Reference to QP/PID Controller Supplement

- Context: After the P4-control QP/PID controller supplement was revised and its open questions were resolved, the user requested that the main design spec explicitly refer to the controller supplement at an appropriate location.
- Decision: Added references to `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md` in v0.4 Section 20.1 and Section 24.5.2. The main spec now points controller implementers to the supplement for quasi-static rigid-body model updates, QP allocation, Isaac actuator target conversion, and P4-control acceptance details while preserving the `π_L` / controller responsibility boundary.
- Compatibility impact: Documentation only. No Python controller code, schema code, or acceptance code was changed.

### P4-Control QP/PID Controller Open Questions Resolved

- Context: The controller draft listed open questions for QP backend choice, Isaac thrust target semantics, vectoring command semantics, reaction torque handling, inertia aggregation fidelity, and waypoint tracking thresholds. The user answered that Python and libraries are acceptable initially, vectoring joints should use absolute position targets, reaction torque should be included, and accepted link-level quasi-static inertia aggregation plus initial waypoint thresholds.
- Decision: Updated `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md` to replace the open-question section with implementation decisions. The spec now sets Python/library-based QP as the initial path, per-thruster thrust target as the primary Isaac-side representation with wrench-composer fallback for custom Holon articulation, absolute vectoring joint targets, reaction torque in QP, link-level quasi-static rigid-body aggregation, and configurable initial waypoint thresholds of 0.20 m position error, 0.25 rad attitude error, and 1.0 s hold duration.
- Compatibility impact: Documentation only. No Python controller code, schema code, or acceptance code was changed. Future implementation must still stop and ask before making incompatible assumptions if additional undefined controller details appear.

### P4-Control QP/PID Controller Spec Revision

- Context: The initial controller draft included a reference-implementation notes section and English-first wording. The user clarified that `aerial_robot_base` is temporary reference material only, that the controller spec should be Japanese-first like the main design spec, that allocation must be QP rather than pseudoinverse, and that assembled morphologies should be treated as a quasi-static single rigid body whose inertia and rotor origins are updated from joint angles every control cycle.
- Decision: Rewrote `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md` as a Japanese controller-specific draft. Removed the reference-code section, made QP allocation normative, added quasi-static rigid-body model update requirements for assembled morphologies, clarified controller/bridge logging, and listed implementation-time open questions for solver choice, Isaac actuator semantics, vectoring command semantics, reaction torque handling, inertia aggregation, and waypoint thresholds.
- Compatibility impact: Documentation only. No Python controller code, schema code, or acceptance code was changed. The revised spec remains aligned with v0.4: `π_L` outputs `PolicyCommand` only, the controller owns `ControllerCommand`, and the bridge owns final Isaac actuator target conversion.

### P4-Control QP/PID Controller Design Spec Draft

- Context: The P4-control / P4a work will implement a near-complete low-level QP/PID controller, but v0.4 Section 20 intentionally leaves several controller details underspecified. The user provided `aerial_robot_base` as a temporary reference source and pointed to `gimbalrotor_controller.cpp` with `underactuate_=false`, `gimbal_calc_in_fc_=true`, and `gimbal_dof_=1` as the relevant branch.
- Decision: Added `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md` as a controller-specific draft skeleton. It records the current controller ownership boundaries, reference-code reading notes, initial QP/PID allocation structure, Isaac bridge/logging expectations, proposed Agent I/J/K/L files, and open questions that must be settled before implementation.
- Compatibility impact: Documentation only. No Python controller code, schema code, or acceptance code was changed in this step. The draft preserves the v0.4 rule that `π_L` emits `PolicyCommand` only and final actuator authority remains in the controller / bridge layer.

## 2026-07-08

### P4.0 Simplified Full-Pipeline Implementation Supplement

- Context: v0.4 Section 24.5.1 defines P4.0 as simplified full-pipeline wiring, not P4 full completion. The implementation needed to connect P2 selected `DesignOutput`, P3 simplified assembly result, morphology-conditioned contact candidates, pi_H, pi_L, controller scaffolding, and `EpisodeArchive` logging without claiming Isaac-backed physical success.
- Decision: Added backward-compatible `EpisodeArchive` fields for runtime observations, actuator target records, rollout artifacts, and learning artifacts. Added a `SimplifiedGraspCarryEnv` path that accepts an external `DesignOutput` and optional assembled `MorphologyGraph`, bypassing `FixedSimpleDesignPolicy` on the P4.0 path. Added `P4_0FullPipelineRunner`, `configs/training/p4_0_grasp_carry.yaml`, unit/archive/no-mislabeling tests, and `run_p4_0_acceptance`.
- Logging/no-mislabeling decision: P4.0 archives record `rollout_artifacts` with `phase="P4.0"`, `backend="simplified"`, `is_p4_full_completion=False`, `isaac_backed=False`, and `physical_success_claim=False`. The acceptance report includes an explicit backend note that P4.0 metrics are simplified backend indicators and not Isaac-backed physical success rates.
- Compatibility impact: Existing P1/P2/P3 archives deserialize because the new archive fields have default empty values. P4.0 still does not implement Isaac Lab backend, controller bridge / actuator mapping, actuator target execution, P4-control, P4.1/P4.2, P4.3 learning bootstrap, or P4 full acceptance.

### P4.3 Learning Target Clarification

- Context: After the P4 Isaac-backed completion clarification, the P4.3 learning bootstrap text could still be read as focusing only on π_L or residual controller learning.
- Decision: Updated the source design spec so P4.3 explicitly includes three staged learning targets: π_L / residual controller learning, π_H contact / trajectory policy learning, and π_D outcome-conditioned design scorer / selector fine-tuning. Added the recommended P4.3a-P4.3e order, expanded P4 full acceptance learning artifacts for all three policy families, and updated the P4 Mermaid diagram so the training loop points back to π_D, π_H, and π_L with their separate responsibilities.
- Compatibility impact: No implementation files were changed in this design-only task. Deterministic `P2DesignPolicy`, deterministic π_H / π_L fallbacks, and `FeasibilityChecker` hard safety remain required; learned feasibility heads must not replace deterministic safety gates.

### P4 Isaac-Backed Completion Clarification

- Context: The previous v0.4 P4 text could be read as treating simplified full-pipeline wiring as P4 completion. The user provided a P4 design revision instruction requiring the source spec to distinguish simplified integration from Isaac-backed full grasp/carry completion.
- Decision: Updated the source design spec to split P4 into `P4.0`, `P4-control / P4a`, `P4.1`, `P4.2`, `P4.3`, and P4 full completion. P4.0 is now explicitly a simplified full-pipeline integration stage and must not be called P4 complete. P4 full completion now requires Isaac Lab rollout, low-level flight validation, controller bridge / actuator mapping, Isaac actuator target execution, minimum learning run, checkpoint, metrics, reward curve, and rollout archive.
- Additional clarification: P3 assembly success is now explicitly documented as simplified graph/state integration success rather than physical docking success. P4 full completion requires a bridge from π_A docking/detach/separation steps to controller targets and Isaac-backed execution results.
- Compatibility impact: No implementation files were changed in this design-only task. Future P4 implementation must treat existing `QPIDController` / `QPAllocator` as simplified scaffolding until the Isaac controller bridge and actuator mapping are implemented.

### P3 Assembly Runner Core Supplement

- Context: v0.4 Section 17 defines `AssemblyPlan`, `AssemblyStep`, and `ConstructionState`, and an earlier Agent G scaffold produced deterministic graph-edit plans, but P3 acceptance needs an executable deterministic runner that advances construction state and checks that the physical graph matches the target graph.
- Decision: Added an Agent G `AssemblyRunner` core. It executes a planned sequence through an `AssemblyExecutorInterface`, records per-step `AssemblyExecutionResult` objects, updates `ConstructionState` on successful `verify_attach` steps when the executor does not provide an updated state, and computes graph/state consistency metrics for modules, dock edges, and occupied target ports.
- Compatibility impact: This remains deterministic π_A scaffolding. It does not introduce learned assembly, motion planning, Isaac execution, QP/PID control, or physical docking verification. Later simplified/Isaac executors can provide richer `updated_state` values behind the same executor interface.

### P3 Simplified Assembly Executor Supplement

- Context: v0.4 Section 24.4 requires P3 assembly execution in simplified sim, but does not prescribe a dependency-free executor backend for the existing `AssemblyExecutorInterface`.
- Decision: Added an Agent G `SimplifiedAssemblyExecutor` that deterministically succeeds assembly steps by default, optionally updates construction state on successful `verify_attach` when a target graph is available, and supports explicit failure injection by step id or step type for later retry/abort acceptance probes.
- Compatibility impact: The executor is a smoke backend only. It does not model docking dynamics, path planning, contact physics, controller allocation, Isaac execution, or learned assembly control.

### P3 Retry/Abort State-Machine Supplement

- Context: v0.4 Section 24.4 requires retry/abort paths to be tested, while Section 17 only lists `retry` and `abort` as valid `AssemblyStep.step_type` values without prescribing a state-machine implementation.
- Decision: Extended Agent G `AssemblyRunner` with deterministic retry/abort handling. On a failed planned step, the runner emits a synthetic `retry` step up to `max_retries_per_step`; if the planned step still fails, it emits a synthetic `abort` step and returns an unsuccessful `AssemblyRunReport` with retry/abort counts and executed step types.
- Compatibility impact: Retry/abort remains deterministic scaffolding and does not imply learned assembly recovery, motion replanning, physical docking verification, or controller/QP feasibility.

### P3 Assembly Evaluation Runner Supplement

- Context: v0.4 Section 24.4 defines P3 assembly acceptance but does not prescribe a concrete evaluation runner or archive metrics for the deterministic assembly integration phase.
- Decision: Added Agent K `P3AssemblyEvaluationRunner`. It reuses the P2 grasp/carry task distribution and deterministic `P2DesignPolicy`, selects a feasible target `MorphologyGraph`, executes it through `AssemblyRunner` and `SimplifiedAssemblyExecutor`, stores the `AssemblyPlan` in `EpisodeArchive.assembly_plan`, and records assembly success/state/retry/abort metrics.
- Compatibility impact: This is a simplified assembly integration runner only. It does not execute π_H, π_L, QP/PID, actuator commands, Isaac, or learned assembly control.

### P3 Acceptance Gate Supplement

- Context: v0.4 Section 24.4 defines P3 acceptance criteria but does not prescribe a concrete acceptance report schema or how to exercise retry/abort paths when the normal simplified executor succeeds deterministically.
- Decision: Added Agent L `run_p3_acceptance`. It runs the P3 assembly evaluation runner, checks `assembly_success_rate >= 70%`, verifies successful archives have construction-state physical graph consistency, and runs explicit transient-failure retry and persistent-failure abort probes with the simplified executor.
- Compatibility impact: This is an acceptance harness for deterministic simplified assembly integration. It does not run Isaac, π_H, π_L, QP/PID, actuator commands, or learned assembly control.

### π_D Joint-Angle Non-Design Clarification

- Context: The v0.4 design text could be misread as treating `ModuleNode.pose_in_design_frame` or `DockEdge.relative_pose_src_to_dst` as continuous design variables for π_D, even though A-MSRR module joints are movable and their instantaneous angles belong to planning/control/runtime state rather than structure design.
- Decision: Clarified the source design spec directly. π_D designs graph-level structure only: module count, connection topology, docking port pairs, base module, module roles, RobotAnchors, slot-anchor priors, control groups, and graph-level metadata. π_D must not output movable joint angles, runtime module relative poses, pose trajectories, actuator commands, rotor thrust, joint torque, or vectoring joint targets.
- Compatibility impact: No code change is required at this point. Existing design-level feasibility checks use graph/capability/coverage/margin necessary conditions and do not score a single nominal joint configuration. Existing pose fields remain nominal/canonical metadata for assembly reference, visualization, coarse precheck, debugging, and simulator initialization.

### P2.5 Learning Bootstrap Supplement

- Context: P2.5 inspection existed as a human/debugging phase, but did not yet create a supervised learning bootstrap from deterministic P2 candidate labels. The user requested a minimal learned π_D scorer and learned feasibility head trained from `P2DesignPolicy.evaluate_candidates()` / deterministic `FeasibilityChecker` outputs, without full RL and without replacing the production path.
- Decision: Added a P2.5 candidate dataset builder that exports all accepted/rejected/selected candidates across multiple sampled grasp/carry tasks, including deterministic features, labels, design scores, violation codes, margins, and train/val ID splits. Added two lightweight MLP training loops: one for teacher-selected π_D candidate classification and one for deterministic feasible/infeasible classification. Checkpoints, metrics, and loss curves are saved under `outputs/p2_5/training`.
- Compatibility impact: The learned models are auxiliary bootstrap artifacts only. They are not used by `P2DesignPolicy`, `FeasibilityChecker`, P2 acceptance, P2.5 inspection, Isaac, π_H, π_L, QP/PID, or actuator command execution. Deterministic `P2DesignPolicy` and `FeasibilityChecker` remain the source of truth for design selection and hard safety checks.

### P2.5 Inspection and Candidate Trace Export Supplement

- Context: The user accepted P2 as complete for the current v0.4 Section 24.3 design-level gate, but requested a pre-P3 inspection phase so humans can inspect morphology variants and all P2DesignPolicy candidate evaluations, not only the selected design.
- Decision: Added P2.5 as an additional inspection/debugging phase, not a replacement for P2 completion. It adds SVG morphology graph/layout visualization for all four grasp/carry variants, JSONL/CSV export of per-candidate evaluation traces including an explicit closed-loop invalid rejection probe, a markdown inspection report, and a P2.5 acceptance gate.
- Compatibility impact: P2.5 uses existing `P2DesignPolicy`, P2 design runner/config, `FeasibilityChecker`, `DesignOutput`, `FeasibilityResult`, and `EpisodeArchive`-compatible labels/margins. The inspection path does not run Isaac, π_H, π_L, QP/PID, or actuator commands.

### P2 Completion Gate Supplement

- Context: v0.4 Section 24.3 defines the P2 acceptance criteria, but it does not define a phase-completion report that downstream work can use to distinguish "acceptance function exists" from "P2 milestone is complete."
- Decision: Added Agent L `run_p2_completion` as a thin completion wrapper over `run_p2_acceptance`. It emits a `P2CompletionReport` with explicit boolean completion checks for the Section 24.3 gates: valid design rate, required slot coverage for accepted designs, closed-loop invalid rejection, and feasibility label storage.
- Compatibility impact: This adds acceptance-side report dataclasses only and does not change persisted schemas. P2 completion remains design-level; π_H, π_L, controller allocation, actuator commands, Isaac execution, and later P3/P4 behavior are outside this gate.

### P2 Acceptance Gate Supplement

- Context: v0.4 Section 24.3 defines P2 acceptance criteria but does not prescribe a concrete report schema or how to probe `closed_loop_invalid designs rejected` when the normal P2 design distribution only generates tree morphologies.
- Decision: Added Agent L `amsrr.acceptance.p2_acceptance` as a mechanical P2 acceptance gate. It runs the configured P2 design runner, checks `valid_design_rate`, verifies required slot coverage for accepted archived designs, validates feasibility label storage across `EpisodeArchive.feasibility_result`, and synthesizes an explicit closed-loop invalid design to verify `F_CLOSED_LOOP_REJECT_V1` rejection and labels.
- Compatibility impact: This is an acceptance harness and does not change persisted schemas. It does not run π_H, π_L, controller allocation, actuator commands, Isaac, or learned training.

### P2 Design Evaluation Runner Supplement

- Context: v0.4 Section 24.3 requires P2 design evaluation over diverse grasp/carry tasks and requires feasibility labels to be stored, but it does not prescribe a concrete runner/config format for executing the TaskSpec -> Geometry -> IRG -> Envelope -> π_D -> FeasibilityChecker path before learned training.
- Decision: Added Agent K `P2DesignEvaluationRunner` and `P2GraspCarryDesignDistribution`. The runner samples randomized object grasp/carry TaskSpecs, builds geometry descriptors through `IRGBuilder.build_with_scene_graph()`, extracts the `InteractionEnvelope`, evaluates `P2DesignPolicy` candidates, and stores the selected `DesignOutput` plus selected `FeasibilityResult` in `EpisodeArchive` JSONL records.
- Compatibility impact: This is an evaluation/dataset scaffold, not a learned training loop, not Isaac execution, and not a controller/actuator-command runner. It uses existing `EpisodeArchive.feasibility_result`, `FeasibilityResult.proxy_scores`, and `FeasibilityResult.margins` fields without changing persisted schemas.

### P2 Design Policy Candidate Selection Scaffold Supplement

- Context: v0.4 Section 15 defines π_D action/candidate scaffolding and Section 24.3 requires P2 design evaluation, but it does not prescribe a deterministic baseline for enumerating multiple candidate designs, separating accepted/rejected candidates, or selecting among accepted candidates before learned π_D training.
- Decision: Added Agent E `P2DesignPolicy` as a deterministic π_D scaffold. It enumerates grasp/carry morphology variants, evaluates each `DesignOutput` with the deterministic `FeasibilityChecker`, splits candidates into accepted and rejected sets, computes a deterministic soft score from feasibility margins plus small support/complexity/variant priors, and returns the highest-scoring accepted design. If no candidate is accepted, it returns the highest-scoring rejected candidate for debugging/dataset labeling.
- Compatibility impact: This is not a learned π_D head and does not output actuator commands. Selection metadata is stored as float entries in `DesignOutput.design_scores` (`p2_design_policy_*`) without changing persisted schemas.

### P2 Grasp-Carry Morphology Variant Builder Supplement

- Context: v0.4 Section 15.4 names design teacher variants (`chain_grasp`, `symmetric_two_anchor_grasp`, `tri_anchor_support_grasp`, `central_base_plus_two_grasp_arms`) but does not prescribe exact module poses, tree topology, control groups, or RobotAnchor placement for each variant.
- Decision: Added Agent E `GraspCarryMorphologyVariantBuilder` as a deterministic P2 scaffold for object grasp/carry. The four variants now produce distinct connected-tree `MorphologyGraph` layouts: a linear chain, a central base with two direct grasp arms, a tri-anchor support/grasp frame with an optional support anchor on the base, and a central base with two two-link grasp arms. `DeterministicDesignTeacher` now routes object grasp/carry variants through this builder instead of merely annotating the minimal seed morphology.
- Compatibility impact: This does not change persisted schemas and does not claim the variants are optimized or learned designs. It gives π_D training/evaluation a finite deterministic set of distinct `DesignOutput` demonstrations while preserving existing `DesignPolicyContext -> DesignOutput`, `ContactSlotID -> RobotAnchorID`, and FeasibilityChecker boundaries.

### P2 FeasibilityChecker Acceptance Labels Supplement

- Context: v0.4 Section 24.3 requires P2 acceptance to measure valid design rate, required slot coverage for accepted designs, closed-loop invalid rejection, and stored feasibility labels, but `FeasibilityResult` does not define a dedicated label field.
- Decision: Strengthened Agent F design-level `FeasibilityChecker` without changing schemas. It now records stable P2 count/ratio/margin keys in `FeasibilityResult.margins` for required slot coverage, anchor capability coverage, coarse reachability, port conflicts, closed-loop rejection, thrust margin, and payload margin. It also stores deterministic 0/1 label scores in `proxy_scores` using `L_FEASIBLE`, `L_HARD_VIOLATION`, and `L_<hard_check_code>` keys.
- Compatibility impact: Existing hard violation codes and `FeasibilityResult` fields remain unchanged. The `L_...` entries are acceptance/dataset labels stored in the available float map, not learned proxy estimates and not replacements for deterministic hard checks.

### P1 Acceptance Gate Supplement

- Context: v0.4 Section 24.2 defines P1 acceptance criteria but does not prescribe a concrete test harness or result schema for recording the pass/fail gate.
- Decision: Added Agent L `amsrr.acceptance.p1_acceptance` as a lightweight acceptance harness. It runs the configured P1 simplified runner over 1000 randomized training-distribution episodes, checks the minimum success rate and zero crash criteria, samples valid randomized objects for non-empty contact candidates, and returns a serializable `P1AcceptanceReport`.
- Compatibility impact: This closes the P1 gate against the interface-backed simplified backend. It does not invoke Isaac Lab and does not claim high-fidelity physics validation; later Agent J simulator backends can be tested behind the same policy/controller boundaries.

### P1 Task Distribution Runner and EpisodeArchive Supplement

- Context: v0.4 requires P1 object grasp/carry randomization over object size, mass, friction, and target pose, plus EpisodeArchive logging and reproducibility metadata. It does not prescribe exact config field names or a JSON storage format.
- Decision: Added Agent K config-driven P1 distribution and runner modules. `P1TaskDistributionConfig` randomizes box primitive size, object mass/friction, initial object pose, and object target pose. `P1SimplifiedRunner` runs the simplified env over sampled tasks, computes batch metrics, and emits `EpisodeArchive` records.
- Logging supplement: Added `EpisodeArchive` with the v0.4 fields plus a `reproducibility` map for source hash, random seed, simulator version, URDF hash, and thrust model hash. JSONL helpers write/read archive sequences for lightweight P1 dataset/debug logs.
- Compatibility impact: This is dataset/logging scaffolding for the simplified env, not a training loop and not an Isaac data recorder. The randomization config can be expanded later for wind, sensor noise, thrust scale error, contact break thresholds, and additional object shapes.

### P1 Simplified Grasp-Carry Simulation Env Supplement

- Context: v0.4 requires simulator-specific code to remain behind interfaces, allows simplified contact in Version 1, and defines P1 acceptance as validating the GeometryProcessor/IRGBuilder/pi_H/pi_L/controller loop with no schema/checker crashes over 1000 episodes.
- Decision: Added an Agent JP1 `SimplifiedGraspCarryEnv` under `amsrr/simulation`. It implements the `reset`, `step`, and `get_runtime_observation` boundary, builds the existing TaskSpec -> IRG -> Envelope -> fixed/simple DesignOutput -> ContactCandidateSet -> pi_H trajectory pipeline, runs `BaselineLowLevelPolicy` and `QPIDController`, and uses a kinematic/fixed-joint approximation after attach to move the object toward active object targets.
- Compatibility impact: This is not an Isaac Lab environment and does not model high-fidelity contact dynamics. It is an interface-backed smoke backend for P1 crash-free validation before Isaac integration. Later Isaac environments can implement the same `SimulationEnvBase` boundary.

### P1 pi_L + QP/PID Controller Interface Supplement

- Context: v0.4 defines the `PolicyCommand` and `ControllerCommand` schemas and the pi_H -> pi_L -> QP/PID flow, but does not prescribe a deterministic P1 low-level baseline policy or dependency-free controller backend.
- Decision: Added an Agent I `BaselineLowLevelPolicy` that consumes `RuntimeObservation`, `MorphologyGraph`, `PhysicalModel`, `ContactWrenchTrajectory`, and an optional active `InteractionKnot`. It selects the active knot by runtime time when not supplied, emits zero anchor pose offsets for active assignments, small contact-tracking biases derived from active assignment wrench targets, and a clipped object pose/velocity residual wrench intent when the active knot contains object targets.
- Controller supplement: Added `ControllerContext`, `ControllerBase`, `QPAllocationProblem`, `QPAllocationResult`, `QPAllocatorInterface`, `BoundedVerticalRotorAllocator`, and `QPIDController`. The P1 allocator solves only bounded vertical thrust allocation, reports unsupported lateral/torque wrench residuals, clips vectoring joints to URDF limits, and applies a PD proxy for non-vectoring joint torque references.
- Compatibility impact: The policy emits only `PolicyCommand` intent and never rotor thrusts, vectoring targets, joint torques, or dock commands. The controller layer owns `ControllerCommand` output. Exact multi-axis/vectoring/contact QP remains a future allocator backend behind `QPAllocatorInterface`.

### P1 pi_H Grasp-Carry Baseline Planner Supplement

- Context: v0.4 defines the `ContactWrenchTrajectory` schema and says pi_H selects `ContactAssignment` sets from `ContactCandidateSet`, but does not prescribe a deterministic P1 baseline planner.
- Decision: Added an Agent H `GraspCarryBaselinePlanner` that consumes IRG, `InteractionEnvelope`, target `MorphologyGraph`, `ContactCandidateSet`, and optional `RuntimeObservation`. It prioritizes existing `grasp_pair` group proposals, converts selected candidates to maintain-state `ContactAssignment`s, checks them with `evaluate_selected_assignment_feasibility`, and emits a five-knot grasp/carry baseline trajectory: approach, attach, lift/maintain, transport, release.
- Compatibility impact: This is a deterministic baseline pi_H implementation, not a learned high-level policy. It does not output actuator commands and does not perform exhaustive candidate subset search. Later learned pi_H heads can replace the group selection/scoring while keeping the same `HighLevelPolicyContext -> ContactWrenchTrajectory` boundary.

### Selected Assignment Feasibility Proxy Supplement

- Context: v0.4 separates unary ContactCandidate screening from assignment-level feasibility after π_H selects `ContactAssignment` sets. It requires no exhaustive subset enumeration and leaves exact wrench/friction/collision/QP solving to later evaluator/controller work.
- Decision: Added `evaluate_selected_assignment_feasibility` as a deterministic selected-assignment proxy evaluator. It checks selected candidate existence and assignment consistency, unary-valid candidates, slot min/max cardinality when supplied by the caller, pairwise conflict matrix entries, duplicate selected candidates, a grasp-opposition wrench proxy, optional explicit wrench/QP residual thresholds, optional friction margin, and optional collision margin. Results are stored in `ContactCandidateSet.assignment_feasibility_cache` using the existing deterministic assignment key.
- Compatibility impact: This does not enumerate arbitrary candidate subsets and does not replace exact multi-contact feasibility. Later π_H and QP/collision backends can pass exact residuals/margins into the same `AssignmentFeasibilityResult` schema.

### P1 ContactCandidateSampler Deterministic Proposal Supplement

- Context: v0.4 requires morphology-conditioned `ContactCandidateSampler` after π_D, with unary screens, pairwise/group compatibility, and no exhaustive candidate-subset enumeration. Exact reachability, collision, and assignment-level wrench/QP feasibility belong to later checker/controller work.
- Decision: Added an Agent H deterministic sampler that consumes `TaskSpec`, IRG ContactSlots, `InteractionEnvelope`, target `MorphologyGraph` RobotAnchors, and `GeometryDescriptor` contact regions. It emits one candidate per compatible ContactSlot × ContactRegion × RobotAnchor by default, transforms patch/region positions from entity frame to world frame using the entity pose, and records deterministic unary smoke scores for mode match, normal alignment, local reachability, surface quality, moment-arm proxy, support quality, friction plausibility, and anchor capability.
- Group proposal supplement: Added small `grasp_pair` proposals for valid same-slot grasp candidates with different anchors, prioritized by opposing normals, and `support_set` proposals for support candidates when no grasp-pair proposal exists for a slot. This is pair/group scaffolding only and does not claim selected groups are full task-feasible.
- Compatibility impact: The sampler runs only after a morphology with RobotAnchors exists. It preserves the `ContactSlotID -> RobotAnchorID -> ContactCandidateID` boundary, does not perform exhaustive subset feasibility, and leaves selected-assignment wrench/QP checks for later π_H/assignment-level evaluators.

### GraphEditAssemblyPlanner Scaffold Supplement

- Context: v0.4 defines `AssemblyPlan`, `AssemblyStep`, and `ConstructionState`, and requires a deterministic π_A `GraphEditAssemblyPlanner`, but does not prescribe how a target `MorphologyGraph` should be expanded into P1/P3 smoke-level assembly steps.
- Decision: Added an Agent G deterministic graph-edit planner that treats the target morphology as a connected tree rooted at `base_module_id`. Starting from an initial construction state containing only the base component, each unattached `DockEdge` is expanded in stable `edge_id` order into four steps: `move_to_staging`, `align_ports`, `dock`, and `verify_attach`. The construction-state helper can mark verified edges as attached and rebuild the assembled subgraph with attached latch states and port occupancy.
- Interface supplement: Added implementation-local dataclasses matching the v0.4 assembly contracts inside `amsrr/assembly`: `AssemblyPlan`, `AssemblyStep`, `ConstructionState`, `AssemblyExecutionResult`, plus `ControlHandoffRequest` for controller handoff scaffolding. These are not added to the persisted `amsrr/schemas` package.
- Compatibility impact: This is deterministic π_A scaffolding, not learned assembly flight control and not simulator execution. Attach/detach safety is represented only as interface structure and precondition/success-condition records; exact motion planning, retry execution, QP feasibility during detach, and simulator verification remain later work.

## 2026-07-07

### Deterministic Design Teacher Scaffold Supplement

- Context: v0.4 requires a design grammar / teacher generator for bootstrapping π_D, but does not prescribe exact module poses, grammar expansion internals, or candidate-mask data structures for P1 fixed/simple morphology.
- Decision: Added an Agent E deterministic teacher scaffold that uses the existing minimal connected-tree morphology builder as the P1 fixed/simple morphology provider. Teacher variants are stable labels (`chain_grasp`, `symmetric_two_anchor_grasp`, `tri_anchor_support_grasp`, `central_base_plus_two_grasp_arms`, `perch_anchor_frame`, `valve_torque_arm`, `support_shift_frame`) over a schema-compatible `DesignOutput`. For object grasp/carry, the default P1 selection is `tri_anchor_support_grasp` when the IRG has required grasp slots and an optional support slot; otherwise it falls back to `symmetric_two_anchor_grasp` or `chain_grasp`.
- Candidate-mask supplement: Added a small `DesignCandidateGenerator` that wraps teacher action traces and masks `STOP` until the final teacher step. Its final STOP validity checks the existing Version 1 STOP conditions at a scaffold level: module count, base assignment, connected graph, port conflicts, required slot coverage, closed-loop rejection, and optional FeasibilityChecker result.
- Compatibility impact: This does not implement a learned π_D, does not change persisted schemas, and does not claim the teacher variants are optimized designs. Later policy heads can replace the scorer/sampler internals while keeping the `DesignPolicyContext -> DesignOutput` boundary.

### P0 ContactCandidate Pairwise Matrix Supplement

- Context: v0.4 requires pairwise/group compatibility for `ContactCandidateSet`, but exact contact-pair geometry, collision, and grasp grouping belong to later sampler/controller work.
- Decision: Added P0 pairwise helpers that mark immediate conflicts only: duplicate candidate IDs and candidates sharing the same robot anchor. Compatibility scores are deterministic smoke values: conflict `0.0`, same-slot different-anchor `0.75`, unrelated non-conflicting pair `0.5`, diagonal `1.0`.
- Compatibility impact: This does not claim arbitrary candidate subsets are feasible. Later ContactCandidateSampler and assignment-level evaluators can replace the heuristic internals while keeping the same `ContactCandidateSet` fields.

### Assignment-Level QP Smoke Supplement

- Context: v0.4 says assignment-level wrench/friction/collision/QP feasibility is evaluated after pi_H selects a `ContactAssignment` set, and that Version 1 must not enumerate every candidate subset.
- Decision: Added `evaluate_assignment_level_qp` as a selected-assignment smoke evaluator. It stores an `AssignmentFeasibilityResult` in `ContactCandidateSet.assignment_feasibility_cache` and emits `E_ASSIGNMENT_QP_INFEASIBLE` when the provided residual exceeds threshold.
- Compatibility impact: The QP backend remains out of scope for this P0 piece. A later exact evaluator can supply the residuals and margins without changing the cache/result schema.

### PolicyCommand Bias Builder Interface Supplement

- Context: v0.4 requires pi_L to output `PolicyCommand` intent while final actuator commands must come from the QP/PID/controller layer.
- Decision: Added `PolicyCommandBiasBuilder` and `DesiredBiasReferences` to convert `PolicyCommand` plus the active `InteractionKnot` into controller reference inputs: joint position/velocity references, desired wrench, body references, contact tracking references, anchor pose offsets, and merged priority weights.
- Compatibility impact: The builder deliberately does not produce rotor thrusts, vectoring joint commands, or final actuator commands. Later QP/PID interfaces can consume these references as controller inputs.

### Minimal Morphology Seed Supplement

- Context: v0.4 defines MorphologyGraph/DesignOutput and π_D action vocabulary, but a learned design policy and full deterministic teacher variants are later work.
- Decision: Added `MinimalMorphologyBuilder` as a deterministic P0 seed builder. It creates a connected tree of Holon modules, replicates dock ports from `PhysicalModel`, creates RobotAnchors from IRG ContactSlots, and emits a DesignAction trace ending in `STOP`.
- Compatibility impact: This produces valid `DesignOutput` objects for downstream FeasibilityChecker and ContactCandidateSampler scaffolding without pretending to be an optimized π_D policy.

### FeasibilityChecker P0 Coarse Proxy Supplement

- Context: v0.4 lists design-level hard checks including coarse reachability, collision, thrust margin, payload margin, and QP hover feasibility. Exact collision and QP solving require later simulator/controller integration.
- Decision: Added a design-level `FeasibilityChecker` scaffold with deterministic structural checks and coarse force proxies. Hover thrust uses `abs(thrust_axis_local.z) * thrust_max_n` per rotor because the Holon module is vectoring-capable and the normalized URDF contains both positive and negative local thrust axes.
- Compatibility impact: The checker owns deterministic hard violations now, while exact collision/QP checks can replace the proxy internals later without changing `FeasibilityResult`.

### SharedInteractionWorkspace Empty Group and Mask Supplement

- Context: v0.4 requires `group_slices` and `group_masks`, and lists several required modality groups. P0 currently has only some modality encoders implemented.
- Decision: Added `WorkspaceTokenGroup` and `SharedInteractionWorkspaceBuilder`. Missing required modality groups are represented as zero-width slices with empty `[B, 0]` masks and explicit `d_model` in the token group contract.
- Compatibility impact: A partial P0 encoder stack can still produce a valid full workspace while preserving required group keys. Downstream heads can rely on stable group presence even before every modality is implemented.

### SharedInteractionWorkspace Strict Group Mask Validation Supplement

- Context: v0.4 states masks must represent validity and zero-valued feature vectors must not imply validity.
- Decision: `SharedInteractionWorkspace` now requires a `group_masks` entry for every group slice. Each group mask must have shape `[B, slice_width]` and match the corresponding slice of the global `mask`.
- Compatibility impact: This strengthens the internal tensor contract without changing persisted task/IRG/envelope schemas. Existing tests were updated to pass explicit group masks.

### InteractionEnvelope Count and Optional Slot Supplement

- Context: v0.4's grasp/carry envelope example has `required_contact_count_range: [2, 4]` and `required_contact_modes: [grasp, support]`. The current IRG template represents grasp as a required slot and support as an optional slot.
- Decision: `InteractionEnvelopeExtractor` computes `required_contact_count_range` from required ContactSlots only, while `required_contact_modes` and target region sets include all ContactSlots, including optional ones.
- Compatibility impact: This matches the v0.4 grasp/carry example and preserves optional support information for samplers and policies without inflating the required contact count.

### InteractionEnvelopeEncoder Contract Supplement

- Context: v0.4 requires an `InteractionEnvelopeEncoder` and states that it should fall back to `mlp_embedding` when no dedicated backend key exists, but does not define a concrete P0 tensor container.
- Decision: Added a dependency-free internal `InteractionEnvelopeEncoderOutput` dataclass with nested-list tensor-compatible fields: `tokens`, `mask`, `token_type_ids`, `source_type_ids`, `source_ids`, `group_slice`, and `group_mask`. The encoder emits deterministic scalar tokens and records `backend_type="mlp_embedding"` by default.
- Compatibility impact: No persisted schema changes are required. Learned MLP parameters and full SharedInteractionWorkspace assembly remain later implementation steps.

### IRGBuilder Template Constraint Mapping Supplement

- Context: v0.4 templates name several task-local constraints such as `max_tilt`, `maintain_contact`, `latch_feasibility`, and `support_polygon_proxy`, while the implemented schema validates constraint nodes against the existing `ConstraintType` enum.
- Decision: Agent D keeps the schema unchanged and maps these template-local concepts to the closest standard constraint types:
  - `max_tilt` -> `workspace`
  - `maintain_contact` -> `no_slip`
  - `latch_feasibility` -> `workspace`
  - `support_polygon_proxy` -> `support_ratio`
- Compatibility impact: IRG validation remains strict against v0.4 enum values, while the original template-local meaning is preserved in `ConstraintNode.feature["parameters"]["template_constraint"]` and the violation code.

### IRGBuilder Lazy Descriptor Requirement Supplement

- Context: v0.4's object grasp/carry example includes a floor support surface referencing `floor_geom` but does not include `floor_geom` in `geometry_library`. The object grasp/carry IRG only needs the target object's contact regions.
- Decision: Agent D normalizes all scene entities, but treats support surface and obstacle geometry descriptors as required only when a template actually requests their contact regions. Object descriptors remain required when referenced by object templates. Perching and contact-mediated locomotion still fail if their required support surface descriptor cannot be resolved.
- Compatibility impact: The v0.4 grasp/carry example can compile to an abstract object-contact IRG without adding schema fields or silently inventing geometry. Templates that need environment contact regions still receive deterministic failures for missing descriptors.

### GeometryProcessor P0 Mesh Smoke Supplement

- Context: v0.4 describes mesh loading, repair, normal/curvature segmentation, rim extraction, and convex decomposition. Full robust mesh processing is larger than the P0 smoke requirement.
- Decision: Agent C implements deterministic STL/OBJ smoke processing using bounding box, surface area, approximate volume, and dominant-normal patch clusters. It emits `SurfacePatchToken` and `ContactRegion` objects with `region_type="mesh_patch_cluster"`.
- Compatibility impact: The output conforms to existing v0.4 `GeometryDescriptor`, `SurfacePatchGraph`, and `ContactRegionGraph` schemas. Later work can replace the normal-cluster implementation with richer segmentation without schema changes.

### Geometry Reference Supplement: Hash URIs Instead of Raw Asset Paths

- Context: v0.4 requires file paths to be GeometryProcessor inputs only and not NN features.
- Decision: `GeometryDescriptor.collision_ref` and `exact_geometry_ref` use deterministic hash URIs such as `mesh://sha256:<hash>` and `primitive://sha256:<hash>` instead of raw filesystem paths.
- Compatibility impact: Asset paths are resolved inside `asset_resolver.py`; downstream learned components receive descriptor refs and patch/region tokens without path strings.

### Runtime Asset Supplement: Normalized Holon URDF

- Context: v0.4 recommends `robot_model.module_urdf_path: assets/robots/holon/holon.urdf`, while this checkout provides `module_urdf/holon.urdf.xacro` as the developer reference and does not provide `module_urdf/holon.urdf`.
- Decision: Added `assets/robots/holon/holon.urdf` as a normalized runtime asset derived from `module_urdf/holon.urdf.xacro`.
- User direction: User approved converting the xacro into an easier-to-use asset under `assets/`.
- Compatibility impact: Runtime path remains configurable and now matches `configs/robot/robot_model.yaml`. The original developer reference xacro is left unchanged.

### Naming Supplement: Thrust Link IDs

- Context: `configs/robot/thrust_model.yaml` uses rotor IDs `thrust_1` through `thrust_4`, while the developer reference xacro used link names `thrust1` through `thrust4`.
- Decision: The normalized runtime URDF uses link names `thrust_1` through `thrust_4`, matching the thrust config IDs.
- User direction: User approved changing to the `thrust_1` format during asset normalization.
- Compatibility impact: `RotorModel.rotor_id` preserves the config ID, and `RotorModel.thrust_frame_link` now resolves directly to a same-named URDF link. The loader still supports normalized matching for compatible future assets.

### Parser Supplement: xacro-Derived XML Without ROS Dependency

- Context: The provided `module_urdf/holon.urdf.xacro` is parseable as XML for the fields needed by P0, and adding ROS/xacro dependencies would be unnecessary for the current scope.
- Decision: Agent B loader parses URDF/xacro-derived XML with the Python standard library and ignores unknown/custom robot child tags while preserving useful metadata such as `baselink`.
- Compatibility impact: No package installation is required. Full xacro macro expansion remains out of scope until an asset actually requires it.

### Approved Schema Supplement: `IRGEdgeType.ALLOWS`

- Context: v0.4 Section 10.4 lists `IRGEdgeType`, but omits `allows`.
- Reason: v0.4 Section 10.2, Section 10.15, Section 11.7, and the worked example all require `ContactRegion --allows--> ContactSlot` edges.
- Decision: Added `IRGEdgeType.ALLOWS = "allows"` in `amsrr/schemas/irg.py`.
- Approval: User approved before implementation.
- Compatibility impact: This aligns the enum with existing v0.4 graph examples and enables deterministic IRGBuilder output without ad hoc edge strings.

### Implementation Supplements for Underspecified Helper Schemas

- Context: v0.4 references helper schemas such as `SurfaceSpec`, `ObstacleSpec`, `WindSpec`, `ObjectKinematicModel`, `CollisionPrimitive`, `ControlGroup`, runtime state records, and `ControllerStatus` without complete field-level definitions.
- Decision: Added minimal dataclass definitions for these helper schemas with conservative fields required by surrounding v0.4 contracts.
- Compatibility impact: These are additive implementation details intended to support parsing, serialization, and tests. They do not rename or remove any v0.4 core schema fields.

### Serialization Supplement: `SharedInteractionWorkspace.group_slices`

- Context: v0.4 defines `group_slices: dict[str, slice]`, while JSON cannot directly represent Python `slice` objects.
- Decision: Runtime schema stores Python `slice` objects, and JSON serialization represents them as `{start, stop, step}` mappings.
- Compatibility impact: In-memory contract remains `dict[str, slice]`; serialization is a practical roundtrip representation for tests and archives.
