# WORKLOG.md

## Global Worklog

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control controller supplement and user-provided initial PID gains
- Work package / Agent label: Agent I boundary: P4-control Order 12 PolicyCommand PID target builder
- Summary: Added the deterministic controller-side PID target builder needed before closed-loop hover. `PolicyCommand.desired_body_pose` / `desired_body_twist` now generate a desired body wrench with gravity compensation, user-specified xy/z/roll-pitch/yaw PID gains, body-frame quaternion attitude error, feedforward/residual wrench addition, target tracking metrics, and integral anti-windup that commits only after feasible unclipped allocation.
- Files changed:
  - `amsrr/controllers/qpid_controller.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None.
- Upstream dependencies used: P4-control controller supplement Section 6, user-provided PID gains, existing `PolicyCommand` body target/residual wrench fields, `RuntimeObservation` pose/twist, and Agent I QP allocator contract.
- Downstream impact: The upcoming single-module closed-loop Isaac smoke can reuse one persistent `QPIDController` instance and feed direct hover/waypoint targets through `PolicyCommand` instead of bypassing π_L/controller ownership boundaries.
- Tests added or run:
  - Added `test_qpid_controller_builds_pid_wrench_from_policy_body_target_and_feedforward`.
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_qpid_controller.py -q`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr scripts -q`
  - `git diff --check`
- Commands run:
  - `sed -n ... amsrr/controllers/qpid_controller.py`
  - `sed -n ... tests/unit/controllers/test_qpid_controller.py`
  - `rg -n "root_pose_w|root_quat_w|xyzw" /home/leus/IsaacLab/source/isaaclab -S` to confirm IsaacLab 3 quaternion ordering is XYZW
- Tests run: Controller unit tests passed: 9 passed. Full unit suite passed: 122 passed, 1 skipped. Compileall and diff check passed.
- Assumptions: No fixed acceleration or torque saturation values were introduced because the user specified gains but not wrench saturation limits; QP actuator bounds and infeasible/clipped metrics remain the safety enforcement layer for this order.
- Blockers: None for PID target builder. Closed-loop Isaac hover is still not implemented and no P4-control completion is claimed.
- Next steps: Commit Order 12, then implement the real single-module closed-loop smoke using a persistent controller and direct `PolicyCommand` hover target.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control Isaac environment recommendation
- Work package / Agent label: Agent I/J boundary: P4-control Order 11 QP feasibility tuning for controller-command smoke
- Summary: Resolved the controller smoke's false QP infeasible status. The QP solver was succeeding, but default smoothing weights pulled the first hover allocation toward the previous zero-thrust command and the post-solve hard check counted a zero-thrust vectoring angle singularity as clipping. Reduced the primary allocator regularization/previous-command weights, raised controller-level unsupported-wrench tolerance only to the small back-conversion residual scale, and held vectoring joints at current position when the back-converted rotor thrust is effectively zero.
- Files changed:
  - `amsrr/controllers/qp_allocator_interface.py`
  - `amsrr/controllers/qpid_controller.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `tests/unit/simulation/test_p4_control_controller_smoke.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None.
- Upstream dependencies used: Agent I `VirtualThrustQPAllocator`, controller hard check/clamp rules, Agent J controller-command smoke helper, corrected Holon URDF/USD, and real Isaac command probe path.
- Downstream impact: Controller-to-Isaac command smoke now reports QP feasible/ok with no residual or clipping violations under the single-module initial hover command. This remains a command-routing/QP-feasibility smoke, not closed-loop hover or P4-control completion.
- Tests added or run:
  - Added `test_qpid_controller_rigid_body_qp_hover_is_feasible_with_default_tolerance`.
  - Strengthened `test_single_module_controller_command_smoke_builds_bridge_record` to assert QP feasible/status ok, residual `< 1e-5`, and no clipping.
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_qpid_controller.py tests/unit/simulation/test_p4_control_controller_smoke.py -q`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr scripts -q`
  - `git diff --check`
- Commands run:
  - `sed -n ... amsrr/controllers/qp_allocator_interface.py`
  - `sed -n ... amsrr/controllers/qpid_controller.py`
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/leus/amsrr python3 - <<'PY' ...` for QP weight, residual, and clipping diagnostics
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/leus/amsrr:$PYTHONPATH /home/leus/IsaacLab/isaaclab.sh -p /home/leus/amsrr/scripts/p4_control_holon_spawn_probe.py --config /home/leus/amsrr/configs/env/isaac_lab.yaml --force-convert --generated-usd-dir /tmp/amsrr_isaac_holon_controller_feasible --generated-usd-path /tmp/amsrr_isaac_holon_controller_feasible/holon/holon.usda --steps 80 --controller-command-smoke`
- Tests run: Focused controller/smoke tests passed: 9 passed. Full unit suite passed: 121 passed, 1 skipped. Compileall and diff check passed. The first sandboxed Isaac run failed because no CUDA GPU was visible; the approved external Isaac run passed with `command_probe_passed=true`, `controller_status.status="ok"`, `controller_status.qp_feasible=true`, `controller_qp_feasible=1.0`, `allocation_residual_norm ~= 3.83e-6`, `clipped_target_count=0.0`, `violation_count=0.0`, no missing/unsupported bridge targets, and no battery2 invalid-inertia warning.
- Assumptions: A `1e-5` controller unsupported-wrench tolerance is a numerical back-conversion tolerance, not a physical tracking-success threshold. Near-zero-thrust vectoring angles are actuator-neutral, so holding current joint position is preferable to commanding an arbitrary rate-limit boundary.
- Blockers: None for controller-command QP feasibility. This does not validate closed-loop hover, fixed-morphology hover, waypoint tracking, object carry, learned policies, or P4 full completion.
- Next steps: Commit Order 11, then proceed toward the real closed-loop single-module hover smoke path unless a method-level undefined item appears.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control Isaac environment recommendation
- Work package / Agent label: Agent I/J/K boundary: P4-control Order 10 controller-to-Isaac command smoke
- Summary: Added a unit-testable controller command smoke builder and connected it to the real Isaac Holon probe. The new path builds a single-module morphology/runtime observation, computes a `QPIDController` command with `allocation_mode="rigid_body_qp"`, converts it through `IsaacControllerBridge`, and applies the bridge record to Isaac thrust bodies and gimbal/dock joints via the existing probe script.
- Files changed:
  - `amsrr/simulation/p4_control_controller_smoke.py`
  - `amsrr/simulation/isaac_lab_backend.py`
  - `scripts/p4_control_holon_spawn_probe.py`
  - `tests/unit/simulation/test_p4_control_controller_smoke.py`
  - `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added controller smoke helper dataclasses/functions and backend command helper `holon_controller_command_probe_command`.
- Upstream dependencies used: `QPIDController`, `VirtualThrustQPAllocator`, `ActuatorMapping`, `IsaacControllerBridge`, corrected Holon URDF/USD, Isaac Lab wrench composer and joint target APIs.
- Downstream impact: Later real-smoke runner work can consume the controller smoke bundle and bridge record instead of hand-authored force/joint commands. The same application path can be reused for single-module closed-loop hover once QP feasibility and control loop updates are fixed.
- Tests added or run:
  - Added `tests/unit/simulation/test_p4_control_controller_smoke.py`.
  - Updated `tests/unit/simulation/test_p4_control_isaac_env.py` for the controller command probe command contract.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_control_controller_smoke.py tests/unit/simulation/test_p4_control_isaac_env.py tests/unit/training/test_p4_control_runner.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr scripts -q`
  - `git diff --check`
- Commands run:
  - `sed -n ... amsrr/controllers/qpid_controller.py`
  - `sed -n ... amsrr/controllers/isaac_controller_bridge.py`
  - `sed -n ... amsrr/simulation/p4_control_isaac_env.py`
  - `PYTHONPATH=/home/leus/amsrr python3 - <<'PY' ...` to inspect command/bridge record contents
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONPATH=/home/leus/amsrr:$PYTHONPATH /home/leus/IsaacLab/isaaclab.sh -p /home/leus/amsrr/scripts/p4_control_holon_spawn_probe.py --config /home/leus/amsrr/configs/env/isaac_lab.yaml --force-convert --generated-usd-dir /tmp/amsrr_isaac_holon_controller --generated-usd-path /tmp/amsrr_isaac_holon_controller/holon/holon.usda --steps 80 --controller-command-smoke`
  - `find amsrr scripts tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: New/related controller smoke tests passed: 9 passed. Full unit suite passed: 120 passed, 1 skipped. Compileall and diff check passed. Real Isaac controller-command smoke passed with `command_probe_passed=true`, `controller_command_smoke=true`, `controller_bridge_missing_actuators=[]`, `controller_bridge_unsupported_actuators=[]`, `controller_bridge_clipped_targets=[]`, bridge target count `12`, and no battery2 invalid-inertia warning.
- Assumptions: This order intentionally filters raw controller joint-torque commands out of the bridge record because this smoke validates P4-control rotor thrust, vectoring joint target, and dock joint position surfaces. The raw zero/rotor/fixed joint torque command behavior remains visible in `raw_joint_torque_command_count`.
- Blockers: None for controller-to-Isaac command routing. The controller QP still reports `qp_feasible=false` with small residual/clipping violations (`allocation_residual_norm ~= 0.00692`), so this is not a hover pass or completion artifact.
- Next steps: Commit Order 10, then investigate/resolve the single-module hover QP feasibility and closed-loop control update path before attempting real P4-control smoke acceptance.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control Isaac environment recommendation
- Work package / Agent label: Agent B/J boundary: P4-control Order 9 Holon battery2 inertial correction
- Summary: Investigated the recurring real Isaac PhysX warning for `battery2`. The runtime Holon URDF and reference xacro had `battery2` inertial data set to zero mass and zero inertia. Copied the symmetric `battery1` inertial origin/mass/inertia to `battery2`, added a unit guard for mesh-bearing runtime URDF links, regenerated USD under `/tmp`, and verified the battery2 warning is gone in a real Isaac command probe.
- Files changed:
  - `assets/robots/holon/holon.urdf`
  - `module_urdf/holon.urdf.xacro`
  - `tests/unit/robot_model/test_urdf_loader.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None.
- Upstream dependencies used: Real Isaac probe logs from Orders 7/8; Holon URDF/xacro inertial data; `battery1` symmetric inertial values.
- Downstream impact: Real Isaac smoke should regenerate USD from the corrected URDF before physical tests. Holon's aggregate mass/inertia changes from the previous zero-mass battery2 asset, so controller mass/gravity compensation and QP allocation should use the corrected physical model.
- Tests added or run:
  - Added `test_asset_urdf_inertials_are_positive_for_physics_import`.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/robot_model -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr scripts -q`
  - `git diff --check`
- Commands run:
  - `rg -n "battery2|battery1|inertial|mass|ixx|iyy|izz" assets/robots/holon module_urdf configs/robot tests/unit/robot_model`
  - `sed -n ... assets/robots/holon/holon.urdf`
  - `sed -n ... module_urdf/holon.urdf.xacro`
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONPATH=/home/leus/amsrr:$PYTHONPATH /home/leus/IsaacLab/isaaclab.sh -p /home/leus/amsrr/scripts/p4_control_holon_spawn_probe.py --config /home/leus/amsrr/configs/env/isaac_lab.yaml --force-convert --generated-usd-dir /tmp/amsrr_isaac_holon_inertial --generated-usd-path /tmp/amsrr_isaac_holon_inertial/holon/holon.usda --steps 20 --hover-force-scale 0.5 --gimbal-target-rad 0.1`
  - `find amsrr scripts tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Robot model unit tests passed: 8 passed, 1 skipped. Full unit suite passed: 119 passed, 1 skipped. Compileall and diff check passed. Real Isaac command probe with regenerated USD passed with `command_probe_passed=true`, `robot_mass_kg=1.8667402267456055`, `num_bodies=25`, `num_joints=12`, max gimbal target error `0.014317 rad`, and no battery2 invalid inertia/negative mass warning in the log.
- Assumptions: `battery2` is the symmetric counterpart of `battery1`; until a CAD-exported inertial override is provided, mirroring `battery1` inertial properties is the most conservative correction.
- Blockers: None for the inertial correction. Remaining Isaac warnings are general extension/USD/TGS notices, not the prior battery2 invalid-mass warning.
- Next steps: Commit Order 9, then wire corrected physical model/controller output into real Isaac single-module closed-loop smoke.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control Isaac environment recommendation
- Work package / Agent label: Agent J/K boundary: P4-control Order 8 Holon Isaac wrench and joint command probe
- Summary: Extended the real Isaac Holon probe from spawn-only to command-path validation. The probe now applies world-frame `+z` wrenches to the four `thrust_.*` bodies and position targets to `gimbal.*` joints, then reports command ids, force totals, root-state deltas, gimbal actual/target positions, and a tolerance-based command pass flag.
- Files changed:
  - `amsrr/simulation/isaac_lab_backend.py`
  - `scripts/p4_control_holon_spawn_probe.py`
  - `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added `IsaacLabBackend.holon_command_probe_command` and command-probe JSON fields in the Isaac-only script.
- Upstream dependencies used: IsaacLab `WrenchComposer.set_forces_and_torques_index`, `Articulation.set_joint_position_target_index`, Holon real body names `thrust_1..4`, Holon real gimbal joints `gimbal1..4`.
- Downstream impact: The next real-smoke order can connect `ControllerCommand` / `IsaacControllerBridge` outputs to the same Isaac body/joint command surfaces. The command probe also gives a concrete observation extraction payload for root state and joint state, but still does not evaluate closed-loop hover.
- Tests added or run:
  - Updated `tests/unit/simulation/test_p4_control_isaac_env.py` to assert the command-probe command line contract.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_control_isaac_env.py tests/unit/training/test_p4_control_runner.py -q`
  - `python3 -m py_compile scripts/p4_control_holon_spawn_probe.py amsrr/simulation/isaac_lab_backend.py tests/unit/simulation/test_p4_control_isaac_env.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr scripts -q`
  - `git diff --check`
- Commands run:
  - `sed -n ... /home/leus/IsaacLab/source/isaaclab/isaaclab/utils/wrench_composer.py`
  - `sed -n ... /home/leus/IsaacLab/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py`
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONPATH=/home/leus/amsrr:$PYTHONPATH /home/leus/IsaacLab/isaaclab.sh -p /home/leus/amsrr/scripts/p4_control_holon_spawn_probe.py --config /home/leus/amsrr/configs/env/isaac_lab.yaml --convert-if-missing --generated-usd-dir /tmp/amsrr_isaac_holon_spawn --generated-usd-path /tmp/amsrr_isaac_holon_spawn/holon/holon.usda --steps 80 --hover-force-scale 0.5 --gimbal-target-rad 0.1`
  - `find amsrr scripts tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Related P4-control env/runner tests passed: 8 passed. Full unit suite passed: 118 passed, 1 skipped. Compileall, py_compile, and diff check passed. Real Isaac command probe passed with `command_probe_passed=true`, `thrust_body_names=["thrust_1","thrust_2","thrust_3","thrust_4"]`, `gimbal_joint_names=["gimbal1","gimbal2","gimbal3","gimbal4"]`, total commanded force `13.0386 N`, and max gimbal target error `0.001052 rad` against a `0.02 rad` tolerance.
- Assumptions: The command probe's `hover-force-scale` applies a global `+z` wrench for Isaac API validation only. It is not the final rotor-axis force semantics, not QP allocation, and not a hover success criterion.
- Blockers: None for command API routing. The real logs still show the `battery2` PhysX inertia/mass warning, and the half-hover open-loop probe falls as expected; both should be handled before claiming physical hover performance.
- Next steps: Commit Order 8, then wire `ControllerCommand` / `IsaacControllerBridge` outputs into the real Isaac probe or runner and begin single-module closed-loop smoke implementation.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control Isaac environment recommendation
- Work package / Agent label: Agent J boundary: P4-control Order 7 Holon Isaac articulation spawn probe
- Summary: Added a real Isaac Lab spawn probe for the generated Holon USD. The probe runs under `isaaclab3` / `isaaclab.sh -p`, optionally converts the URDF, creates a fresh stage, spawns `/World/Holon` as an `Articulation`, steps a few frames, and emits JSON metadata. Real execution passed with Holon reported as 25 bodies and 12 joints.
- Files changed:
  - `amsrr/simulation/isaac_lab_backend.py`
  - `scripts/p4_control_holon_spawn_probe.py`
  - `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added an Isaac backend helper method for constructing the Holon spawn probe command.
- Upstream dependencies used: AGENTS.md micromamba/IsaacLab instructions; IsaacLab `AppLauncher`, `SimulationContext`, `ArticulationCfg`, `UsdFileCfg`, `UrdfConverter`; Holon URDF from Agent B.
- Downstream impact: Later real-smoke code can rely on the generated Holon USD being spawnable as an Isaac articulation and can consume the body/joint names from the probe. The next real-smoke order still needs wrench/force application, joint target application, and controller observation extraction.
- Tests added or run:
  - Updated `tests/unit/simulation/test_p4_control_isaac_env.py` to assert the spawn probe command contract and avoid deprecated `--headless`.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_control_isaac_env.py tests/unit/training/test_p4_control_runner.py -q`
  - `python3 -m py_compile scripts/p4_control_holon_spawn_probe.py amsrr/simulation/isaac_lab_backend.py tests/unit/simulation/test_p4_control_isaac_env.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr scripts -q`
  - `git diff --check`
- Commands run:
  - `sed -n ... /home/leus/IsaacLab/scripts/tutorials/01_assets/add_new_robot.py`
  - `sed -n ... /home/leus/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/quadcopter.py`
  - `sed -n ... /home/leus/IsaacLab/scripts/tools/convert_urdf.py`
  - `sed -n ... /home/leus/IsaacLab/source/isaaclab/isaaclab/assets/asset_base.py`
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONPATH=/home/leus/amsrr:$PYTHONPATH /home/leus/IsaacLab/isaaclab.sh -p /home/leus/amsrr/scripts/p4_control_holon_spawn_probe.py --config /home/leus/amsrr/configs/env/isaac_lab.yaml --force-convert --generated-usd-dir /tmp/amsrr_isaac_holon_spawn --generated-usd-path /tmp/amsrr_isaac_holon_spawn/holon/holon.usda --steps 3`
  - `find amsrr scripts tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Related P4-control env/runner tests passed: 8 passed. Full unit suite passed: 118 passed, 1 skipped. Compileall, py_compile, and diff check passed. Real Isaac spawn probe passed with `spawn_passed=true`, `isaac_backed=true`, `converted=true`, `num_bodies=25`, `num_joints=12`, and root pose near `[0, 0, 0.496]` after 3 steps.
- Assumptions: Generated USD artifacts are reproducible outputs and were written under `/tmp/amsrr_isaac_holon_spawn` for the real probe, not committed. The spawn probe is a prerequisite smoke and not a hover/control result.
- Blockers: None for single-module articulation spawn. Real Isaac logs still warn about `battery2` invalid inertia/negative mass fallback; investigate or correct URDF inertial data before trusting physical hover performance.
- Next steps: Commit Order 7, then implement a minimal Isaac wrench/joint-command probe using the spawned Holon articulation and the existing controller bridge records.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control Isaac environment recommendation
- Work package / Agent label: Agent J boundary: P4-control Order 6 Isaac URDF conversion probe
- Summary: Probed the real IsaacLab URDF converter path for Holon. The `isaaclab3` environment exposes `isaaclab` and `torch`; `isaaclab.sh -p scripts/p4_control_smoke.py --probe` reports the backend available in Isaac Python; `convert_urdf.py` successfully converted `assets/robots/holon/holon.urdf` to `/tmp/amsrr_isaac_holon/holon/holon.usda`. Updated generated USD config/default path to match Isaac importer's output layout.
- Files changed:
  - `configs/env/isaac_lab.yaml`
  - `amsrr/simulation/isaac_lab_backend.py`
  - `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None.
- Upstream dependencies used: AGENTS.md micromamba/IsaacLab instructions; IsaacLab `scripts/tools/convert_urdf.py`; Holon URDF from Agent B.
- Downstream impact: Real smoke implementation can rely on URDF conversion succeeding and should expect generated USD under `<generated_usd_dir>/holon/holon.usda`.
- Tests added or run:
  - Updated generated USD path assertion in `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_control_isaac_env.py tests/unit/training/test_p4_control_runner.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
- Commands run:
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && python -c ...`
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONPATH=/home/leus/amsrr:$PYTHONPATH /home/leus/IsaacLab/isaaclab.sh -p /home/leus/amsrr/scripts/p4_control_smoke.py --probe`
  - `sed -n ... /home/leus/IsaacLab/scripts/tools/convert_urdf.py`
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONPATH=/home/leus/amsrr:$PYTHONPATH /home/leus/IsaacLab/isaaclab.sh -p /home/leus/IsaacLab/scripts/tools/convert_urdf.py /home/leus/amsrr/assets/robots/holon/holon.urdf /tmp/amsrr_isaac_holon --headless`
  - `find /tmp/amsrr_isaac_holon -maxdepth 4 -type f -printf ...`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Related P4-control env/runner tests passed: 8 passed. Full unit suite passed: 118 passed, 1 skipped. Compileall and diff check passed. Isaac URDF conversion probe passed and produced `/tmp/amsrr_isaac_holon/holon/holon.usda`.
- Assumptions: Generated USD artifacts are reproducible outputs and are not committed. The repo config records where generated USD should live when produced under the workspace.
- Blockers: None for URDF conversion. Next real-smoke order still needs Isaac articulation spawn, force/wrench application, joint command application, and observation extraction implementation.
- Next steps: Commit Order 6, then implement or probe a minimal Isaac articulation spawn script for the generated Holon USD.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control Isaac environment recommendation
- Work package / Agent label: Agent J/K boundary: P4-control Order 5 smoke runner configuration and dry-run harness
- Summary: Added configurable P4-control Isaac Lab backend settings, low-level smoke scenario config, smoke environment boundary, dry-run runner, and CLI probe/dry-run script. The runner defines the three required smoke names and thresholds while keeping real Isaac execution unimplemented and completion false unless later real smoke artifacts are supplied.
- Files changed:
  - `configs/env/isaac_lab.yaml`
  - `configs/training/p4_control_low_level.yaml`
  - `amsrr/simulation/isaac_lab_backend.py`
  - `amsrr/simulation/p4_control_smoke.py`
  - `amsrr/simulation/p4_control_isaac_env.py`
  - `amsrr/simulation/__init__.py`
  - `amsrr/training/p4_control_runner.py`
  - `amsrr/training/__init__.py`
  - `scripts/p4_control_smoke.py`
  - `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `tests/unit/training/test_p4_control_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added simulation/training config/result dataclasses and shared P4-control smoke-result contract.
- Upstream dependencies used: AGENTS.md Isaac Lab micromamba instructions; v0.4 Sections 23.5 and 24.5.2; controller supplement Sections 9, 11, and 13; user approval for URDF-to-USD custom articulation and wrench-composer initial path.
- Downstream impact: Later real Isaac execution code can use the config and scenario contracts. Current dry-run/probe path is safe for fast pytest and cannot claim P4-control completion.
- Tests added or run:
  - Added `tests/unit/simulation/test_p4_control_isaac_env.py`
  - Added `tests/unit/training/test_p4_control_runner.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_control_isaac_env.py tests/unit/training/test_p4_control_runner.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p4_control_acceptance.py -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `PYTHONPATH=/home/leus/amsrr python3 scripts/p4_control_smoke.py --probe`
  - `PYTHONPATH=/home/leus/amsrr python3 scripts/p4_control_smoke.py`
- Commands run:
  - `sed -n ... amsrr/utils/config.py amsrr/simulation/base.py amsrr/training/p4_0_full_pipeline_runner.py`
  - `find configs ...`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_control_isaac_env.py tests/unit/training/test_p4_control_runner.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p4_control_acceptance.py -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `PYTHONPATH=/home/leus/amsrr python3 scripts/p4_control_smoke.py --probe`
  - `PYTHONPATH=/home/leus/amsrr python3 scripts/p4_control_smoke.py`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: New P4-control env/runner tests passed: 8 passed. Full unit suite passed: 118 passed, 1 skipped. Targeted P4-control acceptance tests passed: 2 passed. Compileall, diff check, CLI probe, and CLI dry-run passed.
- Assumptions: Normal repo Python is not the Isaac runtime; `scripts/p4_control_smoke.py --probe` is expected to report Isaac Python modules unavailable unless run through the `isaaclab3` environment / `isaaclab.sh -p` path.
- Blockers: Real Isaac physics execution, URDF conversion, Holon spawn, wrench application, and observation extraction are still unimplemented.
- Next steps: Commit Order 5, then probe the actual `isaaclab3` / `isaaclab.sh -p` environment and proceed to URDF conversion or stop if Isaac API details are still undefined.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control split acceptance requirement
- Work package / Agent label: Agent L boundary: P4-control Order 4 fast/real acceptance split
- Summary: Implemented P4-control acceptance reporting that separates fast archive/interface checks from real Isaac smoke completion. The new report exposes `fast_gate_passed`, `real_isaac_smoke_passed`, and `completion_passed`; completion cannot pass unless single-module hover, fixed-morphology hover, and fixed-morphology waypoint smoke results are all Isaac-backed and passed.
- Files changed:
  - `amsrr/acceptance/p4_control_acceptance.py`
  - `amsrr/acceptance/__init__.py`
  - `tests/acceptance/test_p4_control_acceptance.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added acceptance/report dataclasses only.
- Upstream dependencies used: v0.4 Sections 24.5.2 and 24.5.6; user clarification that fast pytest and real Isaac smoke gates must be separate and completion must not pass without real smoke; existing `EpisodeArchive` P4 fields.
- Downstream impact: Later Agent J/K runners can feed real smoke results into this acceptance gate. Until then, P4-control fast gate may pass but P4-control completion remains false.
- Tests added or run:
  - Added `tests/acceptance/test_p4_control_acceptance.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p4_control_acceptance.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
- Commands run:
  - `git status --short`
  - `sed -n ... amsrr/acceptance/p4_0_acceptance.py tests/acceptance/test_p4_0_acceptance.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p4_control_acceptance.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: P4-control acceptance tests passed: 2 passed. Full acceptance suite passed: 9 passed. Full unit suite passed: 110 passed, 1 skipped. Compileall and diff check passed.
- Assumptions: This order does not execute Isaac; it only evaluates smoke result records supplied by a later real Isaac runner. Synthetic smoke results in tests exercise aggregation only and are not a completion artifact.
- Blockers: Real Isaac single-module/fixed-morphology hover and waypoint smoke remain unimplemented/unrun.
- Next steps: Commit Order 4. The next implementation order should create the P4-control runner/config and/or real Isaac smoke harness; method-level Isaac execution details may require user confirmation before implementation.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control controller bridge requirements
- Work package / Agent label: Agent I/J boundary: P4-control Order 3 actuator mapping and bridge target records
- Summary: Implemented simulator-independent actuator mapping and Isaac target record conversion for P4-control. Added deterministic active actuator channels for rotor thrusts, vectoring joints, dock mechanism joints, and effort-limited joints; added a bridge that converts `ControllerCommand` into clipped actuator target records with missing/unsupported/clipped metrics and controller residual status.
- Files changed:
  - `amsrr/controllers/actuator_mapping.py`
  - `amsrr/controllers/isaac_controller_bridge.py`
  - `amsrr/controllers/__init__.py`
  - `tests/unit/controllers/test_actuator_mapping.py`
  - `tests/unit/controllers/test_isaac_controller_bridge.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added controller-local bridge dataclasses and JSON-compatible target record conversion for use in existing `EpisodeArchive.actuator_target_records`.
- Upstream dependencies used: v0.4 Sections 20.8, 23.5, 24.5.2, 25.1; Agent B `PhysicalModel`; Agent I `ControllerCommand`; existing `EpisodeArchive.actuator_target_records` dict field.
- Downstream impact: Agent J/K/L can consume `ActuatorMapping` and `IsaacActuatorTargetRecord` as the fast-testable bridge contract before real Isaac execution. This does not satisfy the real Isaac smoke gate by itself.
- Tests added or run:
  - Added `tests/unit/controllers/test_actuator_mapping.py`
  - Added `tests/unit/controllers/test_isaac_controller_bridge.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_actuator_mapping.py tests/unit/controllers/test_isaac_controller_bridge.py tests/unit/controllers -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
- Commands run:
  - `git status --short`
  - `rg -n ... actuator_target ...`
  - `sed -n ... for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `sed -n ... amsrr/logging/episode_archive.py amsrr/schemas/runtime.py`
  - `python3 - <<'PY' ... physical model actuator summary ...`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_actuator_mapping.py tests/unit/controllers/test_isaac_controller_bridge.py tests/unit/controllers -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Controller/mapping/bridge tests passed: 16 passed. Full unit suite passed: 110 passed, 1 skipped. Compileall and diff check passed.
- Assumptions: Single-module bridge accepts local command keys for backward compatibility; multi-module bridge requires deterministic global `module_<module_id>:<local_id>` keys. The bridge records target conversion only and does not call Isaac APIs.
- Blockers: None for the fast pytest bridge contract. Real Isaac smoke remains unrun and P4-control completion is not claimed.
- Next steps: Commit Order 3, then proceed to the next implementation order for P4-control runner/config or Isaac smoke harness, stopping if real Isaac environment details are undefined.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control controller supplement and user virtual rotor clarification
- Work package / Agent label: Agent I: P4-control Order 2 primary virtual-thrust QP allocator
- Summary: Implemented the primary P4-control allocator path. Added rotor-arm-fixed virtual x/z thrust channels for vectoring rotors, SciPy-backed quadratic allocation with actuator bounds and linearized vectoring joint/rate constraints, back-conversion to non-negative rotor thrusts and absolute vectoring joint targets, achieved-wrench recomputation after hard check/clamp, and controller integration behind `QPIDControllerConfig(allocation_mode="rigid_body_qp")`.
- Files changed:
  - `amsrr/controllers/rigid_body_model.py`
  - `amsrr/controllers/qp_allocator_interface.py`
  - `amsrr/controllers/qpid_controller.py`
  - `amsrr/controllers/__init__.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `tests/unit/controllers/test_rigid_body_model.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Backward-compatible optional controller-local fields were added to `QPAllocationProblem`, `QPAllocationResult`, and `RigidBodyControlModel`.
- Upstream dependencies used: Agent I Order 1 `RigidBodyControlModel`, v0.4 Sections 20 and 24.5.2, controller supplement Section 7, and user clarification that virtual directions are rotor-arm x/z fixed and limits should be in QP constraints plus hard check/clamp.
- Downstream impact: P4-control now has a QP-primary allocator interface for controller-side unit tests. `BoundedVerticalRotorAllocator` remains available but is explicitly tagged as degraded fallback and must not be used to claim P4-control completion.
- Tests added or run:
  - Added virtual-thrust allocator back-conversion and limit/clamp tests.
  - Added controller integration test for `allocation_mode="rigid_body_qp"`.
  - Updated rigid-body model test for current q and virtual axes.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_qpid_controller.py tests/unit/controllers/test_rigid_body_model.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
- Commands run:
  - `sed -n ... amsrr/controllers/*.py`
  - `rg -n ... for_codex/*.md amsrr tests`
  - `python3 - <<'PY' ... import scipy ...`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_qpid_controller.py tests/unit/controllers/test_rigid_body_model.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Targeted QP/controller/rigid-body tests passed: 10 passed. Controller unit tests passed: 11 passed. Full unit suite passed: 105 passed, 1 skipped. Compileall and diff check passed.
- Assumptions: The virtual z channel is rotor-arm fixed but sign-aligned to each rotor's positive thrust direction, because Holon includes both `+z` and `-z` rotor thrust axes. True rotor thrust magnitude is hard checked after solving because exact `sqrt(Fx^2+Fz^2)` thrust bounds are not linear QP constraints in the virtual channel parameterization.
- Blockers: None for Agent I Order 2 fast pytest gate. Real Isaac smoke is still not run and P4-control completion is not claimed.
- Next steps: Commit Agent I Order 2, then proceed to the next implementation order, likely controller bridge / actuator mapping, unless method-level details are undefined.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control controller supplement and virtual thrust channel supplement
- Work package / Agent label: Agent I: P4-control Order 1 RigidBodyControlModel
- Summary: Implemented deterministic per-control-step rigid-body model update for P4-control. Added controller-local `RigidBodyControlModel`, `RotorControlElement`, and `RigidBodyControlModelBuilder` that compute link-level composite mass/COM/inertia, current rotor origins and axes, scalar rotor allocation columns, vectoring joint axes, dock actuator ids, and actuator limits from `PhysicalModel`, `MorphologyGraph`, and `RuntimeObservation.module_states[*].joint_positions`.
- Files changed:
  - `amsrr/controllers/rigid_body_model.py`
  - `amsrr/controllers/__init__.py`
  - `tests/unit/controllers/test_rigid_body_model.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None to persisted schemas. Added controller-local dataclasses and exports.
- Upstream dependencies used: v0.4 Sections 20, 24.5.2, 26.9, 27.1; `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`; user P4-control clarification; Agent B `PhysicalModel`; Agent I controller scaffolds.
- Downstream impact: The next Agent I order can build the primary QP allocator on top of `RigidBodyControlModel.allocation_matrix_body` and rotor/vectoring metadata, while keeping `BoundedVerticalRotorAllocator` as degraded fallback only.
- Tests added or run:
  - Added `tests/unit/controllers/test_rigid_body_model.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_rigid_body_model.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
- Commands run:
  - `git status --short`
  - `sed -n ... amsrr/controllers/*.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_rigid_body_model.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers -q`
  - `python3 -m compileall amsrr -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Targeted rigid-body model tests passed: 3 passed. Controller unit tests passed: 8 passed. Full unit suite passed: 102 passed, 1 skipped in 4.83s. Compileall passed.
- Assumptions: Controller-local body frame uses composite COM as origin and base/control module orientation as attitude. Multi-module actuator ids use `module_<module_id>:<local_id>` strings.
- Blockers: None.
- Next steps: Commit Agent I Order 1, then proceed to Agent I Order 2 primary QP allocator unless method-level undefined issues appear.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control virtual thrust channel supplement
- Work package / Agent label: P4-Control / P4a planning: virtual thrust channel and acceptance split
- Summary: Recorded the user's P4-control implementation clarifications before code changes. The supplement fixes per-control-step `q`-conditioned rigid-body model and allocation matrix updates, QP-primary allocation, vectoring rotor virtual thrust channel expansion with back-conversion to `ControllerCommand`, degraded-only status for `BoundedVerticalRotorAllocator`, and separate fast pytest versus real Isaac smoke gates.
- Files changed:
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Documentation/planning only.
- Upstream dependencies used: User clarification, v0.4 Sections 20 and 24.5.2, and `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`.
- Downstream impact: Agent I implementation must rebuild composite inertia, COM, rotor origins, rotor axes, and allocation matrix `B(q)` every control cycle from `RuntimeObservation` joint positions. Agent L must keep fast pytest checks separate from real Isaac smoke completion.
- Tests added or run: No tests added; documentation-only supplement. `git diff --check` passed.
- Commands run:
  - `git diff --check`
  - `git diff --stat`
  - `git diff -- for_codex/AMSRR_design_modification_by_codex.md for_codex/WORKLOG.md`
- Assumptions: Virtual thrust channels are internal QP variables only; public output remains `ControllerCommand`.
- Blockers: None.
- Next steps: Commit this planning supplement, then start Agent I Order 1 `RigidBodyControlModel`.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus controller supplement cross-reference
- Work package / Agent label: Documentation integration: main spec reference to P4-control QP/PID supplement
- Summary: Inserted references to the controller supplement into the main v0.4 design spec. Section 20.1 now points controller implementers to `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md` for QP/PID, quasi-static rigid-body model update, QP allocation, Isaac bridge, and P4-control acceptance details. Section 24.5.2 also references the supplement and summarizes the resolved P4-control implementation decisions.
- Files changed:
  - `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Documentation only.
- Upstream dependencies used: User request, revised controller supplement, v0.4 Sections 20 and 24.5.2.
- Downstream impact: Future P4-control work should consult the controller supplement before changing Agent I/J/K/L implementation files. The main spec still preserves the original `π_L` intent-only rule.
- Tests added or run: No tests added or run; documentation-only change. `git diff --check` was run.
- Commands run:
  - `sed -n ... for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `git diff --check`
- Assumptions: The supplement is a controller-specific supplement, not a replacement for the full v0.4 source of truth.
- Blockers: None.
- Next steps: Proceed to detailed implementation planning for Agent I only after any remaining controller supplement details are accepted.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus controller design draft v0.1 resolved decisions
- Work package / Agent label: Agent I/J/K/L planning: P4-control QP/PID controller decision recording
- Summary: Recorded the user's answers to the controller-spec open questions. The controller spec now fixes Python/library-based QP as the initial backend, per-thruster thrust target as the primary Isaac representation with wrench-composer fallback for custom Holon articulation, absolute vectoring joint targets, reaction torque inclusion, link-level quasi-static inertia aggregation, and configurable initial waypoint thresholds.
- Files changed:
  - `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Documentation only.
- Upstream dependencies used: User answers for controller open questions; local Isaac Lab examples/docs showing both per-thruster multirotor support and wrench-composer force application paths.
- Downstream impact: Agent I can implement link-level rigid-body model update and QP allocation without asking these questions again. Agent J bridge should preserve per-rotor thrust target records even if the custom Holon Isaac backend applies them through wrench composer.
- Tests added or run: No tests added or run; documentation-only decision recording. `git diff --check` was run after edits.
- Commands run:
  - `sed -n ... /home/leus/IsaacLab/scripts/demos/quadcopter.py`
  - `sed -n ... /home/leus/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/quadcopter_env.py`
  - `sed -n ... /home/leus/IsaacLab/source/isaaclab_contrib/isaaclab_contrib/assets/multirotor/multirotor.py`
  - `sed -n ... for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`
  - `git diff --check`
- Assumptions: For Isaac, the A-MSRR archive contract uses per-thruster thrust targets as the stable representation even if a backend implementation applies equivalent forces through wrench composer.
- Blockers: None for this documentation update.
- Next steps: Begin Agent I implementation planning for `rigid_body_model.py`, QP allocation, and bridge-facing actuator target records.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus revised controller design draft v0.1
- Work package / Agent label: Agent I/J/K/L planning: P4-control QP/PID controller specification revision
- Summary: Revised the controller-specific spec per user guidance. Removed the reference-implementation notes section, rewrote the draft in Japanese, made QP allocation normative instead of pseudoinverse allocation, and added the quasi-static assembled-morphology rule: every control step updates inertia, CoM, rotor origins, and rotor axes from joint angles, then controls the whole morphology as a single rigid body.
- Files changed:
  - `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Documentation only.
- Upstream dependencies used: User clarification, v0.4 Sections 20, 23.5, 24.5.2, 25, 26.9-26.10, 27.1, and the existing controller draft.
- Downstream impact: Future Agent I/J/K/L implementation should treat the revised spec as the active controller supplement and should not encode `aerial_robot_base` as a source dependency or spec reference. Allocation is QP-owned; any fallback must be explicitly labeled degraded/non-QP.
- Tests added or run: No tests added or run; documentation-only revision. `git diff --check` was run after edits.
- Commands run:
  - `sed -n ... for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`
  - `sed -n ... for_codex/AMSRR_design_modification_by_codex.md`
  - `sed -n ... for_codex/WORKLOG.md`
  - `git diff --check`
- Assumptions: The unresolved controller details are now explicitly listed as implementation-before-coding questions in the spec.
- Blockers: None for this documentation revision.
- Next steps: Resolve the listed open questions, then begin Agent I implementation of rigid-body model update, QP allocation, and Isaac bridge contracts.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus controller design draft v0.1
- Work package / Agent label: Agent I/J/K/L planning: P4-control QP/PID controller specification draft
- Summary: Read the temporary `aerial_robot_base` gimbal rotor controller reference at a high level under the requested branch assumptions (`underactuate_=false`, `gimbal_calc_in_fc_=true`, `gimbal_dof_=1`) and added a controller-specific design spec skeleton for the upcoming near-complete QP/PID controller work.
- Files changed:
  - `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Documentation only.
- Upstream dependencies used: v0.4 Sections 20, 23.5, 24.5.2, 25, 26.9-26.10, 27.1; existing `amsrr/controllers` scaffolds; reference file `/home/leus/ros2/aerial_robot_base_ws/src/aerial_robot_base/robots/gimbalrotor/src/control/gimbalrotor_controller.cpp`.
- Downstream impact: Future P4-control implementation can refine the controller draft before editing Agent I/J/K/L source files. The draft highlights required decisions around frames, allocation matrix equations, actuator target records, Isaac bridge semantics, and QP fallback behavior.
- Tests added or run: No tests added or run; this step created documentation only.
- Commands run:
  - `wc -l /home/leus/ros2/aerial_robot_base_ws/src/aerial_robot_base/robots/gimbalrotor/src/control/gimbalrotor_controller.cpp`
  - `rg -n ... /home/leus/ros2/aerial_robot_base_ws/src/aerial_robot_base/robots/gimbalrotor/src/control/gimbalrotor_controller.cpp`
  - `sed -n ... /home/leus/ros2/aerial_robot_base_ws/src/aerial_robot_base/robots/gimbalrotor/src/control/gimbalrotor_controller.cpp`
  - `sed -n ... /home/leus/ros2/aerial_robot_base_ws/src/aerial_robot_base/robots/gimbalrotor/include/gimbalrotor/control/gimbalrotor_controller.h`
- Assumptions: The reference reading was intentionally shallow and branch-specific; exact frame conventions, Isaac actuator semantics, and QP solver choice remain open in the draft.
- Blockers: None for the documentation skeleton.
- Next steps: Refine the controller spec into concrete equations/config/tests, then start Agent I controller allocation and Isaac bridge implementation.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.0 implementation order
- Work package / Agent label: Agent K/L: P4.0 final docs and verification
- Summary: Completed P4.0 documentation handoff after Orders 1-5. Added the P4.0 implementation supplement to the design modification log, confirmed full unit and acceptance suites pass, and kept P4.0 explicitly scoped to simplified full-pipeline wiring rather than P4 full completion.
- Files changed:
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: Documentation only in this order. Earlier Order 1 added backward-compatible `EpisodeArchive` fields.
- Upstream dependencies used: Completed P4.0 Orders 1-5, v0.4 Sections 24.5.1, 25.1, 26.10, and 27.3.
- Downstream impact: Future work can start P4-control / controller bridge / actuator mapping with P4.0 simplified acceptance as a prerequisite, without confusing P4.0 with Isaac-backed P4 full completion.
- Tests added or run: No new tests added in this order.
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance -q`
- Tests run: Full unit suite passed: 99 passed, 1 skipped in 4.67s. Full acceptance suite passed: 7 passed in 118.22s.
- Assumptions: P4.0 acceptance remains a simplified backend wiring gate only; Isaac-backed physical success metrics belong to later P4-control / P4.1 / P4.2 / P4 full acceptance.
- Blockers: None.
- Next steps: Proceed to controller bridge / actuator mapping and P4-control Isaac low-level flight validation before any P4 full completion claim.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.0 implementation order
- Work package / Agent label: Agent L: P4.0 simplified acceptance gate
- Summary: Added a P4.0 acceptance gate that runs the simplified full-pipeline runner, verifies P2 selected design and P3 assembly result usage, checks contact candidate / pi_H / pi_L / controller / archive completeness, records simplified success/drop/collision/QP metrics, and enforces no-mislabeling as non-Isaac and non-P4-full.
- Files changed:
  - `amsrr/acceptance/p4_0_acceptance.py`
  - `amsrr/acceptance/__init__.py`
  - `amsrr/training/p4_0_full_pipeline_runner.py`
  - `tests/acceptance/test_p4_0_acceptance.py`
- Schema/interface changes: None to persisted schemas. Added acceptance-side report/criteria dataclasses and metric aliases `collision_rate` / `qp_infeasible_rate`.
- Upstream dependencies used: Order 3 runner, Order 4 runner tests, v0.4 Section 24.5.1 P4.0 simplified acceptance and no-mislabeling text.
- Downstream impact: P4.0 can now be mechanically accepted without claiming P4 full completion. Later P4-control / P4.1 work must remain separate.
- Tests added or run: Added P4.0 acceptance test.
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p4_0_acceptance.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p4_0_full_pipeline_runner.py tests/acceptance/test_p4_0_acceptance.py -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
- Tests run: P4.0 acceptance targeted test passed: 1 passed. P4.0 unit + acceptance passed: 3 passed. Compileall passed. `git diff --check` passed.
- Assumptions: Because v0.4 does not set a P4.0 success-rate threshold, this gate checks wiring/archive completeness and metric recording rather than adding a new success threshold.
- Blockers: None.
- Next steps: Order 6, update docs / WORKLOG / design modification log and run final verification.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.0 implementation order
- Work package / Agent label: Agent K/L: P4.0 unit, archive completeness, and no-mislabeling tests
- Summary: Added P4.0 runner unit tests that execute the simplified full pipeline, validate archive completeness for design/feasibility/assembly/trajectory/policy/controller/runtime/reward records, and verify explicit no-mislabeling metadata stating that P4.0 is simplified, not Isaac-backed, and not P4 full completion.
- Files changed:
  - `tests/unit/training/test_p4_0_full_pipeline_runner.py`
- Schema/interface changes: None.
- Upstream dependencies used: Order 3 P4.0 runner, Order 1 archive compatibility, Order 2 simplified env injection, and v0.4 P4.0 no-mislabeling requirement.
- Downstream impact: Order 5 acceptance can build on these tested runner/archive invariants rather than duplicating every low-level field assertion.
- Tests added or run: Added P4.0 runner archive/no-mislabeling unit tests.
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p4_0_full_pipeline_runner.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_simplified_grasp_carry_env.py tests/unit/training/test_p1_runner.py tests/unit/training/test_p4_0_full_pipeline_runner.py -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
- Tests run: P4.0 runner tests passed: 2 passed. Related env/P1/P4.0 tests passed: 9 passed. Compileall passed. `git diff --check` passed.
- Assumptions: P4.0 archives intentionally leave `actuator_target_records` empty because no Isaac actuator target conversion is performed in the simplified backend.
- Blockers: None.
- Next steps: Order 5, implement the P4.0 simplified acceptance gate over the tested runner.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.0 implementation order
- Work package / Agent label: Agent K: P4.0 full-pipeline runner
- Summary: Added a P4.0 simplified full-pipeline runner that wires P2 selected `DesignOutput`, P3 simplified assembly result, morphology-conditioned contact candidates, baseline pi_H trajectory, baseline pi_L policy commands, controller commands, rewards, metrics, and `EpisodeArchive` logging. The runner records explicit simplified-backend / not-Isaac / not-P4-full metadata.
- Files changed:
  - `amsrr/training/p4_0_full_pipeline_runner.py`
  - `amsrr/training/__init__.py`
  - `configs/training/p4_0_grasp_carry.yaml`
- Schema/interface changes: None to persisted schemas. Uses the additive `EpisodeArchive` fields from Order 1.
- Upstream dependencies used: P2 design distribution/policy, P3 assembly runner/executor semantics, Order 2 simplified env external design injection, Agent H pi_H baseline, Agent I pi_L/controller scaffolds, v0.4 Section 24.5.1 P4.0 requirements.
- Downstream impact: Order 4 can add archive completeness and no-mislabeling tests against the new runner. Order 5 can implement the P4.0 acceptance gate over this runner.
- Tests added or run: No unit test files added in this order; import/config smoke and compile checks passed.
- Commands run:
  - `python3 -m compileall amsrr -q`
  - `python3 -c "from amsrr.training import load_p4_0_full_pipeline_runner_config, P4_0FullPipelineRunner; ..."`
  - `git diff --check`
- Tests run: Compileall passed. P4.0 runner config/import smoke passed. `git diff --check` passed.
- Assumptions: P3 `AssemblyRunReport.final_state.physical_graph` is the simplified assembled morphology for P4.0 wiring only and does not imply physical docking success.
- Blockers: None.
- Next steps: Order 4, add unit/archive/no-mislabeling tests for the P4.0 runner.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.0 implementation order
- Work package / Agent label: Agent JP1/K: simplified env external DesignOutput / assembled morphology injection
- Summary: Added a P4.0-compatible injection path to `SimplifiedGraspCarryEnv` so callers can provide a selected external `DesignOutput` and optional assembled morphology. The existing P1 fixed/simple default path remains unchanged, while external design paths bypass `FixedSimpleDesignPolicy`.
- Files changed:
  - `amsrr/simulation/simplified_grasp_carry_env.py`
  - `tests/unit/simulation/test_simplified_grasp_carry_env.py`
- Schema/interface changes: None to persisted schemas. Added optional concrete-env arguments and an internal build-artifact `design_source` label.
- Upstream dependencies used: P4.0 requirement to use P2 selected morphology / P3 assembled morphology downstream and avoid `FixedSimpleDesignPolicy` fixed path.
- Downstream impact: The P4.0 runner can instantiate or reset the simplified env with the P2 selected design and P3 assembled morphology before contact candidate generation, π_H, π_L, and controller execution.
- Tests added or run: Added external design/assembled morphology injection test.
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_simplified_grasp_carry_env.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p1_runner.py -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
- Tests run: Simplified env tests passed: 4 passed. P1 runner tests passed: 3 passed. Compileall passed. `git diff --check` passed.
- Assumptions: In P4.0, "assembled morphology" is represented by the successful P3 construction state's physical graph or an equivalent `MorphologyGraph`; this does not claim physical docking success.
- Blockers: None.
- Next steps: Order 3, implement the P4.0 full-pipeline runner over P2 selected design, P3 assembly result, contact candidates, π_H, π_L, controller, rewards, metrics, and archive logging.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.0 implementation order
- Work package / Agent label: Agent A/K: P4.0 archive compatibility
- Summary: Added backward-compatible P4 archive fields to `EpisodeArchive` for runtime observations, actuator target records, rollout artifacts, and learning artifacts. Existing P1/P2/P3 archive construction remains valid because the new fields use defaults.
- Files changed:
  - `amsrr/logging/episode_archive.py`
  - `tests/unit/training/test_p1_runner.py`
- Schema/interface changes: Additive `EpisodeArchive` interface fields with default empty list/dict values.
- Upstream dependencies used: v0.4 Section 25.1 EpisodeArchive contract and P4.0/P4 logging requirements.
- Downstream impact: P4.0 can archive simplified rollout records now, while later P4-control / Isaac-backed runs can fill runtime observations and actuator target records without changing the archive type again.
- Tests added or run: Added legacy archive default-field restoration assertions in the P1 runner archive roundtrip test.
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p1_runner.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p2_design_runner.py tests/unit/training/test_p3_assembly_runner.py -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
- Tests run: P1 runner tests passed: 3 passed. P2/P3 runner tests passed: 5 passed. Compileall passed. `git diff --check` passed.
- Assumptions: P1-P4.0 simplified archives may leave `runtime_observations` and `actuator_target_records` empty unless a runner explicitly records them; Isaac-backed P4 must populate them per the source spec.
- Blockers: None.
- Next steps: Order 2, add external `DesignOutput` / assembled morphology injection to the simplified env without using `FixedSimpleDesignPolicy` on the P4.0 path.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.3 learning target clarification
- Work package / Agent label: P4.3 learning design revision / source-spec update
- Summary: Clarified that P4.3 learning bootstrap targets π_L/residual controller learning, π_H contact/trajectory policy learning, and π_D outcome-conditioned design scorer/selector fine-tuning, not π_L alone. Added P4.3a-P4.3e recommended order, expanded P4 full acceptance learning artifacts for all three policy families, and updated the P4 Mermaid diagram so the training loop points back to π_D, π_H, and π_L with their separate output responsibilities.
- Files changed:
  - `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: Source spec only. No Python implementation files were changed.
- Upstream dependencies used: User-provided追加修正 request, current P4.3 design text, v0.4 Sections 15, 19, 20, 24.5, and P2.5 learning bootstrap status.
- Downstream impact: Future P4.3 implementation must collect deterministic Isaac rollout datasets, then stage learning through π_L/residual control, π_H trajectory/contact policy, and π_D scorer fine-tuning before any optional joint fine-tuning. Deterministic fallbacks and `FeasibilityChecker` hard safety remain required.
- Tests added or run: No tests added; this is a design-spec revision only.
- Commands run:
  - `rg -n "P4.3|learning bootstrap|Training loop|π_D|π_H|π_L|P4 full acceptance|minimum learning" for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `sed -n ... for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `sed -n ... for_codex/AMSRR_design_modification_by_codex.md`
  - `sed -n ... for_codex/WORKLOG.md`
  - `git status --short`
  - `rg -n "P4\\.3a|P4\\.3b|P4\\.3c|P4\\.3d|P4\\.3e|π_L / residual controller|π_H contact / trajectory|π_D outcome-conditioned|updates π_D|updates π_H|updates π_L|deterministic safety gate|FeasibilityChecker" for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `git diff --check`
  - `git diff --stat`
- Tests run: Documentation verification only. `rg` verification found the new P4.3 learning-target terms and training-loop update arrows in the source spec. `git diff --check` passed.
- Assumptions: P2.5 π_D scorer can be used as an initializer or auxiliary model, but deterministic `P2DesignPolicy` and `FeasibilityChecker` remain the production fallback and hard-safety source of truth.
- Blockers: None.
- Next steps: When P4.3 implementation starts, collect deterministic Isaac rollout datasets before staging π_L/residual, π_H, and π_D scorer learning.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4 Isaac-backed completion clarification
- Work package / Agent label: P4 design revision / source-spec update
- Summary: Updated the source design spec per the user-provided P4 design revision instruction. P4 is now split into P4.0 simplified full-pipeline integration, P4-control/P4a low-level Isaac flight validation, P4.1 Isaac backend smoke, P4.2 Isaac deterministic full grasp/carry rollout, P4.3 Isaac learning bootstrap, and P4 full completion. The spec now states that P4.0 is necessary but not P4 complete, and that P4 full completion requires Isaac-backed rollout plus minimum learning run artifacts.
- Files changed:
  - `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: Source spec only. Future implementation will need EpisodeArchive additions for `runtime_observations`, `actuator_target_records`, rollout artifacts, and learning artifacts; no Python implementation files were changed in this task.
- Upstream dependencies used: User-provided `/home/leus/Downloads/p4_design_revision_instruction.md`, v0.4 Sections 17, 20, 23, 24, 25, 26, and 27.
- Downstream impact: P4 implementation must not mark simplified backend acceptance as P4 complete. Future P4 work must implement controller bridge / actuator mapping, π_A docking/detach/separation handoff to controller targets, P4-control Isaac low-level flight validation, Isaac-backed rollout, and a minimum learning run before P4 full completion.
- Tests added or run: No tests added; this is a design-spec revision only.
- Commands run:
  - `wc -l /home/leus/Downloads/p4_design_revision_instruction.md`
  - `sed -n ... /home/leus/Downloads/p4_design_revision_instruction.md`
  - `rg -n "P4|full grasp|SimplifiedGraspCarryEnv|π_H|π_L|QP|Controller|Simulation|Training Curriculum|Acceptance|Agent J|Agent K|Agent L|Implementation order|EpisodeArchive" for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `sed -n ... for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `rg -n "P4.0|P4-control|low-level flight|Isaac|Controller bridge|actuator mapping|P4 full completion" for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `git diff --check`
  - `find amsrr -type d -name __pycache__ -prune -exec rm -rf {} +`
  - `git status --short`
  - `git diff --stat`
- Tests run: Documentation verification only. `rg` verification found the new P4 terms in the source spec. `git diff --check` passed.
- Assumptions: The P4 revision changes the source design contract but intentionally does not implement any P4 code yet.
- Blockers: None.
- Next steps: When implementation resumes, begin with P4.0 simplified full-pipeline integration, then implement controller bridge / actuator mapping and P4-control Isaac low-level flight validation before claiming P4 full completion.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P3 assembly integration supplements
- Work package / Agent label: P3 final verification and handoff
- Summary: Completed final verification after the P3 assembly runner/executor/retry/acceptance sequence. Full unit and acceptance suites passed, compile checks passed, and diff whitespace checks passed.
- Files changed:
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None.
- Upstream dependencies used: Completed P3 order 1-5 commits, full unit suite, full acceptance suite, and existing AGENTS.md handoff rules.
- Downstream impact: P3 deterministic assembly integration is now mechanically checked. Future P4 work can start from the P3 acceptance gate, while remembering that this remains simplified assembly integration and does not run Isaac, π_H, π_L, QP/PID, or actuator commands.
- Tests added or run: No new tests added in this final handoff step.
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Full unit suite passed: 96 passed, 1 skipped. Full acceptance suite passed: 6 passed in 115.39s. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Assumptions: P3 acceptance is satisfied by deterministic simplified assembly integration per v0.4 Section 24.4. Physical docking, Isaac, π_H, π_L, QP/PID, actuator commands, and full grasp/carry task execution remain P4/later work.
- Blockers: None.
- Next steps: Proceed to P4 full grasp/carry integration after reviewing P3 acceptance outputs and deciding whether Isaac-backed assembly validation is needed before P4.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P3 Acceptance Gate Supplement
- Work package / Agent label: Agent L: P3 acceptance gate
- Summary: Added a P3 acceptance gate for Section 24.4. It runs the P3 assembly evaluation runner, checks assembly success rate, verifies construction-state/physical-graph consistency for successful assemblies, and exercises explicit retry and abort probes through the simplified executor.
- Files changed:
  - `amsrr/acceptance/p3_acceptance.py`
  - `amsrr/acceptance/__init__.py`
  - `tests/acceptance/test_p3_acceptance.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None to persisted schemas. Added acceptance-side `P3AcceptanceCriteria`, `P3AcceptanceReport`, and `run_p3_acceptance`.
- Upstream dependencies used: v0.4 Section 24.4; Agent K P3 runner; Agent G assembly runner/retry/abort and simplified executor; P2 design distribution/policy for probe target graphs.
- Downstream impact: P3 can now be mechanically checked before moving to P4 full grasp/carry integration.
- Tests added or run:
  - Added `test_p3_acceptance_section_24_4`
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p3_assembly_runner.py tests/acceptance/test_p3_acceptance.py -q`
  - `python3 -m compileall amsrr -q`
- Tests run: P3 runner and P3 acceptance targeted tests passed: 3 passed. `python3 -m compileall amsrr -q` passed.
- Assumptions: Retry/abort path testing uses explicit deterministic failure probes because the normal simplified executor succeeds deterministically.
- Blockers: None.
- Next steps: Run P3 acceptance and related targeted tests, commit order 5, then perform final docs/worklog verification.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P3 Assembly Evaluation Runner Supplement
- Work package / Agent label: Agent K: P3 assembly evaluation runner
- Summary: Added a P3 assembly evaluation runner/config that samples grasp/carry tasks, reuses deterministic P2 design selection, executes the selected target morphology through `AssemblyRunner` and `SimplifiedAssemblyExecutor`, stores `AssemblyPlan` in `EpisodeArchive.assembly_plan`, and records assembly success/state/retry/abort metrics.
- Files changed:
  - `amsrr/training/p3_assembly_runner.py`
  - `amsrr/training/__init__.py`
  - `configs/training/p3_assembly_grasp_carry.yaml`
  - `tests/unit/training/test_p3_assembly_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None to persisted schemas. Added training-side `P3AssemblyRunnerConfig`, `P3AssemblyRunnerResult`, and `P3AssemblyEvaluationRunner`.
- Upstream dependencies used: P2 task distribution/config, `P2DesignPolicy`, `FeasibilityChecker` labels through selected candidate results, Agent G assembly runner/executor, and `EpisodeArchive`.
- Downstream impact: Agent L P3 acceptance can aggregate assembly success rate, retry/abort coverage, and construction-state consistency from runner archives/reports.
- Tests added or run:
  - Added `test_p3_assembly_runner_collects_successful_assembly_archives`
  - Added `test_p3_assembly_runner_config_loader`
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p2_design_runner.py tests/unit/training/test_p3_assembly_runner.py tests/unit/assembly/test_graph_edit_planner.py tests/unit/assembly/test_assembly_runner.py tests/unit/assembly/test_simplified_executor.py -q`
  - `python3 -m compileall amsrr -q`
- Tests run: P3 runner plus related P2/assembly targeted tests passed: 17 passed. `python3 -m compileall amsrr -q` passed.
- Assumptions: P3 assembly runner remains simplified and intentionally does not run contact candidates, π_H, π_L, QP/PID, actuator commands, or Isaac.
- Blockers: None.
- Next steps: Run targeted Agent K tests, commit order 4, then implement P3 acceptance gate.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P3 Retry/Abort State-Machine Supplement
- Work package / Agent label: Agent G: P3 retry/abort behavior
- Summary: Extended `AssemblyRunner` with deterministic retry/abort behavior. Failed planned steps now emit synthetic `retry` steps up to a configurable retry limit, then emit a synthetic `abort` step if the planned step still fails. `AssemblyRunReport` now records retry/abort counts, aborted status, and executed step types. The simplified executor can now fail matching steps once for transient failure tests.
- Files changed:
  - `amsrr/assembly/assembly_runner.py`
  - `amsrr/assembly/simplified_executor.py`
  - `tests/unit/assembly/test_assembly_runner.py`
  - `tests/unit/assembly/test_simplified_executor.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None to persisted schemas. Extended assembly-local runner/executor dataclasses only.
- Upstream dependencies used: Existing Agent G runner/executor scaffolding and v0.4 valid AssemblyStep types `retry` and `abort`.
- Downstream impact: P3 runner/acceptance can now measure retry and abort path coverage directly from `AssemblyRunReport`.
- Tests added or run:
  - Added `test_assembly_runner_can_disable_retry_for_single_failure_stop`
  - Added `test_simplified_executor_fail_once_allows_runner_retry_success`
  - Updated failure-path tests to assert synthetic retry/abort records.
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/assembly/test_graph_edit_planner.py tests/unit/assembly/test_assembly_runner.py tests/unit/assembly/test_simplified_executor.py -q`
  - `python3 -m compileall amsrr -q`
- Tests run: Agent G targeted assembly tests passed: 12 passed. `python3 -m compileall amsrr -q` passed.
- Assumptions: Retry/abort steps are synthetic runtime steps and are not inserted into the source `AssemblyPlan.steps`, preserving the original deterministic graph-edit plan.
- Blockers: None.
- Next steps: Run targeted Agent G tests, commit order 3, then implement P3 assembly evaluation runner.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P3 Simplified Assembly Executor Supplement
- Work package / Agent label: Agent G: P3 simplified assembly executor
- Summary: Added a deterministic `SimplifiedAssemblyExecutor` backend for the assembly executor interface. It succeeds assembly steps by default, can return updated construction state on `verify_attach`, records per-step smoke metrics, and supports explicit failure injection for later retry/abort probes.
- Files changed:
  - `amsrr/assembly/simplified_executor.py`
  - `amsrr/assembly/__init__.py`
  - `tests/unit/assembly/test_simplified_executor.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None to persisted schemas. Added assembly-local `SimplifiedAssemblyExecutorConfig` and `SimplifiedAssemblyExecutor`.
- Upstream dependencies used: Existing Agent G `AssemblyRunner`, `AssemblyExecutorInterface`, `mark_edge_attached`, and v0.4 P3 simplified sim acceptance guidance.
- Downstream impact: Order 3 retry/abort behavior and Order 4 P3 runner can use the simplified executor to exercise success and failure paths without Isaac or controller dependencies.
- Tests added or run:
  - Added `test_simplified_executor_runs_full_assembly_and_returns_updated_state`
  - Added `test_simplified_executor_can_inject_step_type_failure`
  - Added `test_simplified_executor_success_without_target_graph_uses_runner_state_transition`
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/assembly/test_graph_edit_planner.py tests/unit/assembly/test_assembly_runner.py tests/unit/assembly/test_simplified_executor.py -q`
  - `python3 -m compileall amsrr -q`
- Tests run: Agent G targeted assembly tests passed: 10 passed. `python3 -m compileall amsrr -q` passed.
- Assumptions: Simplified executor metrics are smoke values only and do not imply physical docking feasibility.
- Blockers: None.
- Next steps: Run targeted Agent G tests, commit order 2, then implement retry/abort behavior.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P3 Assembly Runner Core Supplement
- Work package / Agent label: Agent G: P3 assembly state execution core
- Summary: Added a deterministic `AssemblyRunner` core that runs `AssemblyPlan` steps through an `AssemblyExecutorInterface`, updates `ConstructionState` after successful `verify_attach` steps, records per-step results, and reports final physical-graph consistency metrics against the target `MorphologyGraph`.
- Files changed:
  - `amsrr/assembly/assembly_runner.py`
  - `amsrr/assembly/__init__.py`
  - `tests/unit/assembly/test_assembly_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None to persisted schemas. Added assembly-local `AssemblyRunnerConfig` and `AssemblyRunReport` dataclasses.
- Upstream dependencies used: v0.4 Sections 17 and 24.4; existing Agent G `GraphEditAssemblyPlanner`, `ConstructionState`, `AssemblyExecutorInterface`, and P2 grasp/carry morphology variants.
- Downstream impact: P3 simplified executor and acceptance work can now execute deterministic assembly plans and evaluate whether construction-state physical graph changes match the target graph.
- Tests added or run:
  - Added `test_assembly_runner_completes_plan_and_updates_construction_state`
  - Added `test_assembly_runner_stops_on_failed_step_without_completing_graph`
  - Added `test_assembly_runner_resumes_from_partial_construction_state`
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/assembly/test_graph_edit_planner.py tests/unit/assembly/test_assembly_runner.py -q`
  - `python3 -m compileall amsrr -q`
- Tests run: Agent G targeted assembly tests passed: 7 passed. `python3 -m compileall amsrr -q` passed.
- Assumptions: Successful `verify_attach` is the deterministic point at which the core can mark a target dock edge attached if the executor does not provide a richer updated state.
- Blockers: None.
- Next steps: Run targeted Agent G tests, commit order 1, then implement the simplified assembly executor.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus user-requested P2.5 learning bootstrap
- Work package / Agent label: P2.5: Supervised learning bootstrap for π_D scorer and feasibility head
- Summary: Added a P2.5 learning bootstrap that turns deterministic `P2DesignPolicy` candidate evaluations and `FeasibilityChecker` labels into a supervised dataset, trains a minimal learned π_D candidate scorer, trains a minimal learned feasibility head, saves checkpoints/metrics/loss curves, and updates the P2.5 report/acceptance gate. This is not full RL and does not replace deterministic design selection or hard safety checks.
- Files changed:
  - `amsrr/training/p2_candidate_trace_export.py`
  - `amsrr/training/p2_learning_dataset.py`
  - `amsrr/training/p2_learned_scorer.py`
  - `amsrr/training/p2_feasibility_head_training.py`
  - `amsrr/reporting/p2_5_inspection_report.py`
  - `amsrr/acceptance/p2_5_inspection.py`
  - `amsrr/acceptance/p2_5_learning_bootstrap.py`
  - `amsrr/acceptance/__init__.py`
  - `tests/unit/training/test_p2_learning_dataset.py`
  - `tests/unit/training/test_p2_learned_scorer.py`
  - `tests/unit/training/test_p2_feasibility_head_training.py`
  - `tests/unit/reporting/test_p2_5_inspection_report.py`
  - `tests/acceptance/test_p2_5_inspection.py`
  - `tests/acceptance/test_p2_5_learning_bootstrap.py`
  - `outputs/p2_5/datasets/p2_candidate_dataset.jsonl`
  - `outputs/p2_5/datasets/p2_candidate_dataset_summary.json`
  - `outputs/p2_5/datasets/train_ids.json`
  - `outputs/p2_5/datasets/val_ids.json`
  - `outputs/p2_5/training/pi_d_scorer/checkpoint.pt`
  - `outputs/p2_5/training/pi_d_scorer/metrics.json`
  - `outputs/p2_5/training/pi_d_scorer/loss_curve.csv`
  - `outputs/p2_5/training/feasibility_head/checkpoint.pt`
  - `outputs/p2_5/training/feasibility_head/metrics.json`
  - `outputs/p2_5/training/feasibility_head/loss_curve.csv`
  - `outputs/p2_5/report/p2_5_inspection_report.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None to persisted schemas. Added training/acceptance helper dataclasses only.
- Upstream dependencies used: Existing P2 task distribution/config, `P2DesignPolicy.evaluate_candidates()`/`evaluate_design_outputs()`, deterministic `FeasibilityChecker`, P2.5 candidate trace export, and P2.5 report/acceptance scaffolding.
- Dataset output: `outputs/p2_5/datasets/p2_candidate_dataset.jsonl`
- Dataset counts: 320 candidate records from 64 task samples; train=255, val=65; accepted=256, rejected=64, selected=64.
- Dataset labels/features: all normal P2 candidates plus closed-loop invalid probes are stored with selected/accepted/feasible labels, teacher scores, design scores, violation labels/codes, feasibility margins, slot/capability coverage, thrust/payload/reachability margins, module count, and dock edge count.
- Training commands:
  - `python3 -m amsrr.training.p2_learning_dataset --config configs/training/p2_design_grasp_carry.yaml --output-dir outputs/p2_5/datasets --sample-count 64 --seed 0`
  - `python3 -m amsrr.training.p2_learned_scorer --dataset outputs/p2_5/datasets/p2_candidate_dataset.jsonl --train-ids outputs/p2_5/datasets/train_ids.json --val-ids outputs/p2_5/datasets/val_ids.json --output-dir outputs/p2_5/training/pi_d_scorer --epochs 40 --seed 0`
  - `python3 -m amsrr.training.p2_feasibility_head_training --dataset outputs/p2_5/datasets/p2_candidate_dataset.jsonl --train-ids outputs/p2_5/datasets/train_ids.json --val-ids outputs/p2_5/datasets/val_ids.json --output-dir outputs/p2_5/training/feasibility_head --epochs 40 --seed 1`
- π_D scorer checkpoint: `outputs/p2_5/training/pi_d_scorer/checkpoint.pt`
- π_D scorer metrics: train_loss=0.10842715948820114, val_loss=0.10839308053255081, selected_accuracy=1.0, num_train_samples=255, num_val_samples=65.
- Feasibility head checkpoint: `outputs/p2_5/training/feasibility_head/checkpoint.pt`
- Feasibility head metrics: train_loss=0.00012452361988835037, val_loss=0.00012500998855102807, binary_accuracy=1.0, precision=1.0, recall=1.0, num_train_samples=255, num_val_samples=65.
- Report update: `outputs/p2_5/report/p2_5_inspection_report.md` now records dataset paths/counts, scorer/head checkpoint paths, metrics, and explicitly states that learned models are NOT used in production path and deterministic `P2DesignPolicy` / `FeasibilityChecker` remain source of truth.
- Tests added or run:
  - Added `test_p2_learning_dataset_builds_records_and_split`
  - Added `test_p2_learned_scorer_training_writes_checkpoint_and_metrics`
  - Added `test_p2_feasibility_head_training_writes_checkpoint_and_metrics`
  - Added `test_p2_5_learning_bootstrap_acceptance_gate`
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p2_learning_dataset.py tests/unit/training/test_p2_learned_scorer.py tests/unit/training/test_p2_feasibility_head_training.py tests/unit/reporting/test_p2_5_inspection_report.py tests/acceptance/test_p2_5_learning_bootstrap.py -q`
  - `python3 -m amsrr.training.p2_learning_dataset --config configs/training/p2_design_grasp_carry.yaml --output-dir outputs/p2_5/datasets --sample-count 64 --seed 0`
  - `python3 -m amsrr.training.p2_learned_scorer --dataset outputs/p2_5/datasets/p2_candidate_dataset.jsonl --train-ids outputs/p2_5/datasets/train_ids.json --val-ids outputs/p2_5/datasets/val_ids.json --output-dir outputs/p2_5/training/pi_d_scorer --epochs 40 --seed 0`
  - `python3 -m amsrr.training.p2_feasibility_head_training --dataset outputs/p2_5/datasets/p2_candidate_dataset.jsonl --train-ids outputs/p2_5/datasets/train_ids.json --val-ids outputs/p2_5/datasets/val_ids.json --output-dir outputs/p2_5/training/feasibility_head --epochs 40 --seed 1`
  - `python3 -m amsrr.reporting.p2_5_inspection_report --trace-dir outputs/p2_5/candidate_traces --visualization-dir outputs/p2_5/visualization --output-dir outputs/p2_5/report --config configs/training/p2_design_grasp_carry.yaml --dataset-dir outputs/p2_5/datasets --training-dir outputs/p2_5/training`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
- Tests run: Targeted learning/report/acceptance tests passed: 5 passed. Full unit suite passed: 86 passed, 1 skipped. Full acceptance suite passed: 5 passed in 89.94s. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Production-path status: The learned π_D scorer and learned feasibility head are not used in production path. Deterministic `P2DesignPolicy` remains the design-selection source of truth, and deterministic `FeasibilityChecker` remains the hard safety/source-of-truth checker.
- Explicitly not executed: full RL, Isaac, π_H, π_L, QP/PID, actuator command execution.
- Assumptions: Minimal MLPs are sufficient for bootstrap acceptance because this task proves dataset -> training -> checkpoint -> metrics wiring rather than model quality. The high accuracy reflects deterministic teacher labels and simple diagnostic features, not a claim of general learned policy performance.
- Blockers: None.
- Next steps: Commit P2.5 learning bootstrap changes if accepted, then proceed toward P3 only after human review of P2.5 report/visualizations and confirmation that learned artifacts remain auxiliary.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 clarification
- Work package / Agent label: Design clarification: π_D structural scope and joint-angle non-design rule
- Summary: Clarified the source design spec to state that π_D designs graph-level A-MSRR structure only and must not treat movable joint angles or runtime module relative poses as design freedoms. Clarified `ModuleNode.pose_in_design_frame` and `DockEdge.relative_pose_src_to_dst` as nominal/canonical metadata, not optimized joint configurations, and clarified that design-level feasibility must not score a single nominal joint configuration.
- Files changed:
  - `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None.
- Upstream dependencies used: User clarification, current P2/P2.5 implementation, Section 14 MorphologyGraph, Section 15 π_D, and Section 16 FeasibilityChecker.
- Downstream impact: Future π_D, feasibility, visualization, and report work must treat pose fields as canonical/nominal metadata only. Joint-angle optimization and runtime relative-pose trajectories belong to π_H, π_L, QP/PID, controller/runtime state, or simulator logic.
- Tests added or run: No code tests added; this is a source-spec/documentation clarification.
- Commands run:
  - `git status --short`
  - `rg -n "π_D|MorphologyGraph|pose_in_design_frame|DockEdge|Design-level|Feasibility|feasibility|relative_pose" for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `rg -n "pose_in_design_frame|relative_pose_src_to_dst|joint angle|joint_angle|joint target|vectoring|rotor thrust|torque" amsrr tests for_codex -g '*.py' -g '*.md'`
  - `sed -n ...` inspections for Sections 14, 15, 16, and worked example text
- Tests run: Not run because no source code changed. Existing code inspection indicates current P2 design-level checker does not score single joint angles or a single nominal module-relative pose.
- Assumptions: Current P2 scaffold `pose_in_design_frame` values are nominal layout/reference values for visualization/debugging and not learned or optimized continuous joint outputs.
- Blockers: None.
- Next steps: If desired, regenerate P2.5 report text to include this clarification, but functional code changes are not required for correctness.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus user-requested P2.5 inspection phase
- Work package / Agent label: P2.5: Post-P2 inspection, visualization, and candidate trace export
- Summary: Added P2.5 as an additional pre-P3 inspection/debugging phase that visualizes all P2 grasp/carry morphology variants, exports every evaluated candidate including accepted/rejected/selected labels, generates a human-readable inspection report, and provides a P2.5 acceptance gate.
- Files changed:
  - `amsrr/training/p2_inspection_context.py`
  - `amsrr/training/p2_candidate_trace_export.py`
  - `amsrr/visualization/__init__.py`
  - `amsrr/visualization/p2_morphology.py`
  - `amsrr/reporting/__init__.py`
  - `amsrr/reporting/p2_5_inspection_report.py`
  - `amsrr/acceptance/__init__.py`
  - `amsrr/acceptance/p2_5_inspection.py`
  - `tests/unit/visualization/test_p2_morphology_visualization.py`
  - `tests/unit/training/test_p2_candidate_trace_export.py`
  - `tests/unit/reporting/test_p2_5_inspection_report.py`
  - `tests/acceptance/test_p2_5_inspection.py`
  - `outputs/p2_5/visualization/*.svg`
  - `outputs/p2_5/candidate_traces/p2_candidate_trace.jsonl`
  - `outputs/p2_5/candidate_traces/p2_candidate_summary.csv`
  - `outputs/p2_5/report/p2_5_inspection_report.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None to persisted schemas. Added inspection/report/acceptance-side helper dataclasses only.
- Upstream dependencies used: Existing P2 completion, Agent E grasp/carry variants and `P2DesignPolicy`, Agent F feasibility labels/margins, Agent K P2 design config/distribution, `DesignOutput`, `FeasibilityResult`, and current P2 runner context.
- Downstream impact: P3 should not start until a human has inspected `outputs/p2_5/report/p2_5_inspection_report.md` and the SVG visualizations. P2 completion semantics remain unchanged; P2.5 is an additional inspection gate.
- Generated visualization files:
  - `outputs/p2_5/visualization/chain_grasp_graph.svg`
  - `outputs/p2_5/visualization/chain_grasp_layout.svg`
  - `outputs/p2_5/visualization/symmetric_two_anchor_grasp_graph.svg`
  - `outputs/p2_5/visualization/symmetric_two_anchor_grasp_layout.svg`
  - `outputs/p2_5/visualization/tri_anchor_support_grasp_graph.svg`
  - `outputs/p2_5/visualization/tri_anchor_support_grasp_layout.svg`
  - `outputs/p2_5/visualization/central_base_plus_two_grasp_arms_graph.svg`
  - `outputs/p2_5/visualization/central_base_plus_two_grasp_arms_layout.svg`
- Candidate trace outputs:
  - `outputs/p2_5/candidate_traces/p2_candidate_trace.jsonl`
  - `outputs/p2_5/candidate_traces/p2_candidate_summary.csv`
- Inspection report: `outputs/p2_5/report/p2_5_inspection_report.md`
- Candidate counts in generated trace: 5 records total; 4 accepted; 1 rejected; 1 selected.
- Representative violation code: `F_CLOSED_LOOP_REJECT_V1` from the explicit `tri_anchor_support_grasp_closed_loop_probe` rejected candidate.
- Tests added or run:
  - Added `test_p2_morphology_visualization_outputs_graph_and_layout_svgs`
  - Added `test_p2_candidate_trace_export_writes_all_candidates_and_probe`
  - Added `test_p2_5_inspection_report_contains_summary_and_scope_notes`
  - Added `test_p2_5_inspection_acceptance_gate`
- Commands run:
  - Read attached request text from `/home/leus/.codex/attachments/.../pasted-text.txt`
  - `git status --short`, `git diff --stat`, `find ...`, `sed -n ...`, `rg ...`, and `git log ...` inspections
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/visualization/test_p2_morphology_visualization.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p2_candidate_trace_export.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/reporting/test_p2_5_inspection_report.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p2_5_inspection.py -q`
  - `python3 -m amsrr.visualization.p2_morphology --config configs/training/p2_design_grasp_carry.yaml --output-dir outputs/p2_5/visualization`
  - `python3 -m amsrr.training.p2_candidate_trace_export --config configs/training/p2_design_grasp_carry.yaml --output-dir outputs/p2_5/candidate_traces`
  - `python3 -m amsrr.reporting.p2_5_inspection_report --trace-dir outputs/p2_5/candidate_traces --visualization-dir outputs/p2_5/visualization --output-dir outputs/p2_5/report --config configs/training/p2_design_grasp_carry.yaml`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: P2.5 targeted tests passed individually: visualization 1 passed, trace export 1 passed, report 1 passed, P2.5 acceptance 1 passed. Full unit suite passed: 83 passed, 1 skipped. Full acceptance suite passed: 4 passed in 88.72s. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- P2.5 explicitly not executed: Isaac, π_H, π_L, QP/PID, actuator commands, learned training.
- Assumptions: The normal P2 variant set currently yields accepted candidates for the default sample, so P2.5 appends an explicit closed-loop invalid probe through `P2DesignPolicy.evaluate_design_outputs()` to externalize a rejected candidate and its labels without changing P2 completion.
- Blockers: None.
- Next steps: Commit final P2.5 report/acceptance changes. Human review of the report and SVGs is recommended before P3.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent L: P2 completion gate
- Summary: Added a P2 milestone completion wrapper that runs the Section 24.3 P2 acceptance gate and emits explicit boolean completion checks for valid design rate, required slot coverage, closed-loop invalid rejection, and feasibility label storage.
- Files changed:
  - `amsrr/acceptance/__init__.py`
  - `amsrr/acceptance/p2_completion.py`
  - `tests/acceptance/test_p2_completion.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None to persisted schemas. Added acceptance-side `P2CompletionCriteria`, `P2CompletionReport`, and `run_p2_completion`.
- Upstream dependencies used: v0.4 Section 24.3; existing `run_p2_acceptance`, `P2AcceptanceReport`, P2 design runner archives, and Agent F feasibility labels/margins.
- Downstream impact: Downstream P3/P4 work can treat `run_p2_completion(...).passed` as the local P2 milestone gate before assembly/end-to-end integration. This remains design-level and does not run π_H, π_L, QP/PID, actuator commands, Isaac, or learned training.
- Tests added or run:
  - Added `test_p2_completion_milestone_section_24_3`
- Commands run:
  - `git status --short`, `sed -n ...`, `rg -n ...`, `ls -la ...`, and `git log -5 --oneline` inspections for spec/worklog/acceptance state
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p2_completion.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Targeted P2 completion test passed: 1 passed in 23.62s. Full unit suite passed: 80 passed, 1 skipped. Full acceptance suite passed: 3 passed in 88.20s. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Assumptions: P2 completion is defined as successful Section 24.3 design-level acceptance. It intentionally does not imply assembly execution, π_H/π_L/controller execution, Isaac Sim execution, or full grasp/carry success; those begin in P3/P4.
- Blockers: None.
- Next steps: Commit P2 completion changes if accepted, then advance to P3 assembly integration.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent L: P2 acceptance gate
- Summary: Added a mechanical P2 acceptance gate for Section 24.3 that runs the P2 design evaluation runner, checks `valid_design_rate >= 70%`, verifies accepted-design required slot coverage, probes closed-loop invalid rejection, and validates feasibility label storage in `EpisodeArchive.feasibility_result`.
- Files changed:
  - `amsrr/acceptance/__init__.py`
  - `amsrr/acceptance/p2_acceptance.py`
  - `tests/acceptance/test_p2_acceptance.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Added acceptance-side `P2AcceptanceCriteria`, `P2AcceptanceReport`, and `run_p2_acceptance`.
- Upstream dependencies used: v0.4 Section 24.3; Agent K P2 design runner/archive output; Agent E P2 `P2DesignPolicy` and grasp/carry variants; Agent F `FeasibilityChecker` labels/margins.
- Downstream impact: P2 now has a reproducible pass/fail gate before moving to later assembly/end-to-end phases. The gate remains design-level only and does not run π_H, π_L, QP/PID/controller commands, Isaac, or learned training.
- Tests added or run:
  - Added `test_p2_acceptance_section_24_3`
- Commands run:
  - `sed -n ...`, `rg -n ...`, `git status --short`, `git diff --stat`, and `git log -3 --oneline` inspections for acceptance, feasibility labels, Section 24.3, and current commit format
  - `git add ...`
  - `git commit -m "[P2][Agent K] Add design evaluation runner"`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p2_acceptance.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Targeted P2 acceptance test passed: 1 passed in 23.45s. Full unit suite passed: 80 passed, 1 skipped. Full acceptance suite passed: 2 passed in 64.82s. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Assumptions: `required_slot_coverage >= 90% for accepted designs` is enforced as a minimum over accepted archived designs, which is stricter than an average-only interpretation. The normal P2 distribution produces tree morphologies, so closed-loop invalid rejection is tested through an explicit synthetic closed-loop design probe.
- Blockers: None.
- Next steps: Commit Agent L P2 acceptance changes if accepted, then continue to the next P2/P3 work package.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent K: P2 design evaluation runner
- Summary: Added a P2 design-evaluation runner that samples diverse grasp/carry TaskSpecs, runs TaskSpec -> Geometry/IRG -> InteractionEnvelope -> P2 π_D candidate evaluation -> FeasibilityChecker, and stores selected `DesignOutput` plus selected `FeasibilityResult` labels/margins in `EpisodeArchive` JSONL records.
- Files changed:
  - `amsrr/training/__init__.py`
  - `amsrr/training/p2_design_distribution.py`
  - `amsrr/training/p2_design_runner.py`
  - `configs/training/p2_design_grasp_carry.yaml`
  - `tests/unit/training/test_p2_design_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Existing `EpisodeArchive.feasibility_result`, `FeasibilityResult.proxy_scores`, `FeasibilityResult.margins`, and `DesignOutput.design_scores` are used unchanged.
- Upstream dependencies used: v0.4 Sections 23.4, 24.3, 25.1, 26.10; Agent E P2 variant builder and `P2DesignPolicy`; Agent F P2 feasibility labels/margins; existing `IRGBuilder`, `InteractionEnvelopeExtractor`, `PhysicalModel`, and `EpisodeArchive` logging.
- Downstream impact: P2 acceptance and dataset generation can now read archived design-level labels directly from `EpisodeArchive.feasibility_result`. The runner remains design-level only and does not run π_H, π_L, controller allocation, actuator commands, Isaac, or learned training.
- Tests added or run:
  - Added `test_p2_design_distribution_randomizes_and_marks_metadata`
  - Added `test_p2_design_runner_collects_feasibility_archives`
  - Added `test_p2_design_runner_config_loader`
- Commands run:
  - `sed -n ...`, `rg --files ...`, `rg -n ...`, `git status --short`, and `git diff --stat` inspections for training runners, policy/checker interfaces, schema/logging utilities, config files, and worklogs
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p2_design_runner.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Targeted P2 design runner tests passed: 3 passed. Full unit suite passed: 80 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Assumptions: P2 object diversity reuses and slightly widens the P1 box grasp/carry randomization fields for this slice. The runner archives the selected candidate's feasibility labels; full per-candidate dataset rows can be added later if P2 training needs rejected-candidate supervision beyond the current selected-design archive.
- Blockers: None.
- Next steps: Add an Agent L P2 acceptance gate over this runner, or extend archive output to store per-candidate feasibility traces if required by the training dataset format.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent E: P2 π_D candidate selection scaffold
- Summary: Added deterministic P2 design-policy scaffold that enumerates multiple grasp/carry candidate morphologies, evaluates each with `FeasibilityChecker`, separates accepted/rejected candidates, computes deterministic soft scores, and returns the best accepted design.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/design_policy_p2.py`
  - `tests/unit/policies/test_p2_design_policy.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Existing `DesignOutput.design_scores` stores P2 selection metadata as float keys with `p2_design_policy_*` prefixes.
- Upstream dependencies used: v0.4 Sections 15, 16, 24.3, 26.5, 27.1; Agent E grasp/carry variant builder; Agent F P2 FeasibilityChecker labels/margins; existing `DesignPolicyContext` and `DesignOutput` schemas.
- Downstream impact: P2 runner/acceptance can now call `P2DesignPolicy.evaluate_candidates()` to obtain all candidates plus accepted/rejected splits, or `design()` to get the deterministic selected design. Later learned π_D heads can replace scoring while preserving the candidate/evaluation boundary.
- Tests added or run:
  - Added `test_p2_design_policy_enumerates_variants_and_selects_best_accepted`
  - Added `test_p2_design_policy_splits_rejected_candidates_with_feasibility_checker`
  - Added `test_p2_design_policy_falls_back_to_best_rejected_when_none_accepted`
- Commands run:
  - `sed -n ...` inspections for design policy, teacher, candidate generator, package exports, and existing tests
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_p2_design_policy.py tests/unit/policies/test_design_teacher.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Targeted P2 design policy/design teacher tests passed: 6 passed. Full unit suite passed: 77 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Assumptions: This is deterministic π_D scaffolding, not a learned policy head. The soft score is a hand-coded P2 baseline combining feasibility margins with small support/complexity/variant priors; it is documented as replaceable by learned scoring later.
- Blockers: None.
- Next steps: Continue with Agent K/L P2 design runner and acceptance gate, or add dataset logging around the new candidate selection results.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent F: P2 design-level FeasibilityChecker strengthening
- Summary: Strengthened design-level `FeasibilityChecker` for P2 acceptance by stabilizing hard-check labels and numeric margins for slot coverage, anchor capability, closed-loop rejection, port conflicts, thrust/payload margins, and coarse reachability.
- Files changed:
  - `amsrr/feasibility/checker.py`
  - `tests/unit/feasibility/test_feasibility_checker.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Existing `FeasibilityResult` schema is unchanged; P2 labels are stored as float entries in `proxy_scores` with `L_...` keys, and acceptance margins are stored in the existing `margins` map.
- Upstream dependencies used: v0.4 Sections 16.2-16.8, 24.3, 26.6, 27.1; existing Agent E P2 grasp/carry morphology variants, IRG ContactSlot and CapabilityRequirement edges, PhysicalModel thrust data, and MorphologyGraph/DesignOutput schemas.
- Downstream impact: P2 runners/acceptance can aggregate `L_FEASIBLE`, `L_<hard_check_code>`, required-slot coverage ratios, closed-loop rejection, port conflict counts, thrust margin, and payload margin directly from archived `FeasibilityResult` records.
- Tests added or run:
  - Added `test_p2_feasibility_checker_records_acceptance_margins_for_variant`
  - Added `test_p2_feasibility_checker_uses_capability_requirement_force_label`
  - Added `test_p2_feasibility_checker_records_port_conflict_margins`
  - Added `test_p2_feasibility_checker_records_reachability_margins`
  - Updated existing missing slot coverage and closed-loop tests to assert labels/margins
- Commands run:
  - `sed -n ...` and `rg -n ...` inspections for spec Sections 16/24/26, checker, IRG templates, and tests
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/feasibility/test_feasibility_checker.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/feasibility/test_feasibility_checker.py -q` passed: 7 passed. Full unit suite passed: 74 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Assumptions: `FeasibilityResult` has no dedicated label field in v0.4, so deterministic P2 labels are represented as `proxy_scores["L_..."]` floats. These labels do not replace hard violations and are intended for acceptance/dataset aggregation.
- Blockers: None.
- Next steps: Continue with Agent E P2 candidate/evaluation policy scaffolding or Agent K/L P2 design runner and acceptance gate.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent E: P2 grasp/carry morphology variant builder
- Summary: Implemented real deterministic P2 grasp/carry morphology variants for `chain_grasp`, `symmetric_two_anchor_grasp`, `tri_anchor_support_grasp`, and `central_base_plus_two_grasp_arms`, and routed object grasp/carry `DeterministicDesignTeacher` output through the new variant builder.
- Files changed:
  - `amsrr/morphology/__init__.py`
  - `amsrr/morphology/grasp_carry_designs.py`
  - `amsrr/policies/design_teacher.py`
  - `tests/unit/morphology/test_grasp_carry_variants.py`
  - `tests/unit/policies/test_design_teacher.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Existing `MorphologyGraph`, `DesignOutput`, `DesignAction`, `RobotAnchor`, and `ControlGroup` schemas were used unchanged.
- Upstream dependencies used: v0.4 Sections 14, 15.3, 15.4, 16, 24.3, 26.5, 27.1; existing IRG ContactSlot semantics, PhysicalModel dock ports/capability token, FeasibilityChecker, and design teacher/candidate trace boundaries.
- Downstream impact: P2 design evaluation can now sample/evaluate distinct teacher morphology demonstrations instead of four labels over one minimal seed graph. ContactCandidateSampler and FeasibilityChecker continue to consume the same schema objects.
- Tests added or run:
  - Added `test_grasp_carry_variants_build_distinct_feasible_morphologies`
  - Added `test_grasp_carry_variant_topology_shapes`
  - Added `test_grasp_carry_variants_cover_required_slot_min_count`
  - Updated `test_design_teacher_selects_p1_grasp_support_variant`
- Commands run:
  - `sed -n ...` inspections for morphology builder, design teacher, and existing tests
  - `python3 -c "from amsrr.robot_model.physical_model_builder import build_physical_model_from_config; ..."` to inspect Holon dock ports
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/morphology/test_grasp_carry_variants.py tests/unit/policies/test_design_teacher.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Targeted morphology/design teacher tests passed: 6 passed. Full unit suite passed: 70 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Assumptions: Exact variant poses/topologies are not specified by v0.4, so this implementation defines deterministic scaffold layouts for P2 teacher/evaluation use. These variants are not optimized morphology search results and are not learned π_D outputs yet.
- Blockers: None.
- Next steps: Continue with Agent F P2 FeasibilityChecker strengthening or Agent E P2 candidate/evaluation policy scaffolding, then add the P2 acceptance runner/gate.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent L: P1 tests and acceptance
- Summary: Added an explicit P1 acceptance gate for v0.4 Section 24.2 using the configured simplified grasp/carry runner, EpisodeArchive JSONL output, and randomized contact-candidate smoke checks.
- Files changed:
  - `amsrr/acceptance/__init__.py`
  - `amsrr/acceptance/p1_acceptance.py`
  - `tests/acceptance/test_p1_acceptance.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added acceptance-side `P1AcceptanceCriteria`, `P1AcceptanceReport`, and `run_p1_acceptance`.
- Upstream dependencies used: v0.4 Sections 24.2, 25.1, 26.12, 27.3; existing P1 task distribution config, `P1SimplifiedRunner`, `EpisodeArchive`, fixed/simple design policy, ContactCandidateSampler, pi_H baseline, pi_L baseline, and QPID controller.
- Downstream impact: P1 has a reproducible pass/fail acceptance harness before Isaac Lab integration. Later simulator backends can add equivalent acceptance coverage without changing the Section 24.2 criteria.
- Tests added or run:
  - Added `test_p1_acceptance_section_24_2`
- Commands run:
  - `sed -n ...` and `rg -n ...` inspections for spec Sections 24/26/27, acceptance ownership, runner/env/logging modules, and existing tests
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p1_acceptance.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p1_acceptance.py -q` passed: 1 passed in 41.26s. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 67 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Assumptions: Section 24.2 can be evaluated on the interface-backed simplified env for P1. Isaac Lab remains a later simulator-backend validation step, not a prerequisite for this acceptance gate.
- Blockers: None.
- Next steps: Commit Agent L acceptance changes if accepted, then move to the next post-P1 work package.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent K: P1 task distribution, runner, metrics, and EpisodeArchive logging
- Summary: Implemented P1 order 8 task randomization config, grasp/carry task distribution, simplified env runner, EpisodeArchive schema/logging, batch metrics, and archive JSONL roundtrip tests.
- Files changed:
  - `amsrr/logging/__init__.py`
  - `amsrr/logging/episode_archive.py`
  - `amsrr/training/__init__.py`
  - `amsrr/training/p1_task_distribution.py`
  - `amsrr/training/p1_runner.py`
  - `configs/training/p1_grasp_carry_distribution.yaml`
  - `tests/unit/training/test_p1_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added logging/training-side `EpisodeArchive`, `P1TaskDistributionConfig`, `P1TaskSample`, `P1GraspCarryTaskDistribution`, `P1RunnerConfig`, `P1RunnerResult`, and `P1SimplifiedRunner`.
- Upstream dependencies used: v0.4 Sections 23.4, 24.2, 25.1, 25.3, 26.10; existing TaskSpec, IRG, InteractionEnvelope, DesignOutput, PolicyCommand, ControllerCommand, simplified env, and config/hash utilities.
- Downstream impact: P1 simplified runs can now be sampled from a configured object distribution, summarized by metrics, and serialized as EpisodeArchive JSONL records for later dataset/training work.
- Tests added or run:
  - Added `test_p1_distribution_randomizes_configured_fields`
  - Added `test_p1_runner_collects_metrics_and_archives`
  - Added `test_p1_runner_config_loader`
- Commands run:
  - `sed -n ...` and `rg -n ...` inspections for spec Sections 23/24/25/26, TaskSpec schemas, simplified env, config/hash utilities, and existing tests
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p1_runner.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p1_runner.py -q` passed: 3 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 67 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: P1 randomization currently covers box primitive size, object mass, object friction, initial object pose, and target pose. Object shape stays box for this slice; wind, sensor noise, thrust scale error, and contact break threshold randomization are deferred.
- Blockers: None.
- Next steps: Continue with broader dataset/logging integration or Isaac Lab backend binding after this P1 runner is accepted.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent JP1: simplified grasp-carry simulation env
- Summary: Implemented P1 order 7 interface-backed simplified grasp/carry environment that runs the existing TaskSpec -> IRG -> Envelope -> fixed/simple morphology -> ContactCandidateSampler -> pi_H -> pi_L -> QPID controller loop without Isaac dependencies, plus 1000-episode crash-free unit coverage.
- Files changed:
  - `amsrr/simulation/__init__.py`
  - `amsrr/simulation/base.py`
  - `amsrr/simulation/simplified_grasp_carry_env.py`
  - `tests/unit/simulation/test_simplified_grasp_carry_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added simulation-side `SimulationEnvBase`, `SimplifiedGraspCarryEnvConfig`, `SimplifiedGraspCarryBuildArtifacts`, `SimplifiedEpisodeResult`, `SimplifiedBatchRunResult`, `SimplifiedGraspCarryEnv`, and `run_crash_free_episodes`.
- Upstream dependencies used: v0.4 Sections 23, 24.2, 25.1, 26.10; Agent D IRGBuilder/EnvelopeExtractor; Agent E fixed/simple design policy; Agent H ContactCandidateSampler and pi_H baseline; Agent I pi_L and QPID controller interfaces.
- Downstream impact: P1 can validate the schema/runtime/controller loop before Isaac Lab integration. Later Agent J Isaac environments can implement the same `SimulationEnvBase` boundary while reusing policy/controller interfaces.
- Tests added or run:
  - Added `test_simplified_grasp_carry_env_matches_base_protocol`
  - Added `test_simplified_grasp_carry_env_runs_policy_controller_episode`
  - Added `test_simplified_grasp_carry_1000_episodes_crash_free`
- Commands run:
  - `sed -n ...` and `rg -n ...` inspections for spec Sections 23/24/25/26, existing pi_H/pi_L/controller tests, and worklog/design notes
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_simplified_grasp_carry_env.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_simplified_grasp_carry_env.py -q` passed: 3 passed, including 1000 simplified episodes with 0 crashes. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 64 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: The P1 simplified env uses kinematic/fixed-joint contact after attach, high-level object target tracking, deterministic small initial XY jitter, and controller status checks. It is not an Isaac Lab environment and does not model high-fidelity contact, friction, aerodynamic, or collision dynamics.
- Blockers: None.
- Next steps: Continue with Isaac Lab environment integration or dataset/logging once P1 simplified env behavior is accepted.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent I: pi_L + QP/PID interfaces
- Summary: Implemented P1 order 6 Agent I interfaces: deterministic pi_L baseline, controller context/base protocol, QP allocator interface, dependency-free bounded vertical rotor allocator, QPID controller scaffold, package exports, and policy/controller unit tests.
- Files changed:
  - `amsrr/controllers/__init__.py`
  - `amsrr/controllers/controller_base.py`
  - `amsrr/controllers/qp_allocator_interface.py`
  - `amsrr/controllers/qpid_controller.py`
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/low_level_policy_base.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `tests/unit/policies/test_low_level_baseline.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added policy-side `LowLevelPolicyContext`, `LowLevelPolicyBase`, `BaselineLowLevelPolicyConfig`, `BaselineLowLevelPolicy`, and `select_active_knot`; added controller-side `ControllerContext`, `ControllerBase`, `QPAllocationProblem`, `QPAllocationResult`, `QPAllocatorInterface`, `BoundedVerticalRotorAllocator`, `RotorAllocationSpec`, `QPIDControllerConfig`, and `QPIDController`.
- Upstream dependencies used: v0.4 Sections 20, 26.9, 27.1, 28.11; existing `PolicyCommand`, `RuntimeObservation`, `PhysicalModel`, `ContactWrenchTrajectory`, and `PolicyCommandBiasBuilder`.
- Downstream impact: P1 simplified grasp-carry simulation can consume deterministic `PolicyCommand` and `ControllerCommand` outputs through stable interfaces. Later learned pi_L heads and exact QP backends can replace the baseline/allocator without changing the context boundaries.
- Tests added or run:
  - Added `test_baseline_low_level_policy_outputs_policy_command`
  - Added `test_baseline_low_level_policy_selects_knot_from_runtime_time`
  - Added `test_baseline_low_level_policy_suppresses_residual_when_controller_infeasible`
  - Added `test_select_active_knot_rejects_empty_trajectory`
  - Added `test_bounded_vertical_rotor_allocator_feasible_and_unsupported_residual`
  - Added `test_bounded_vertical_rotor_allocator_reports_infeasible_clip`
  - Added `test_qpid_controller_outputs_controller_command`
  - Added `test_qpid_controller_reports_infeasible_vertical_wrench`
- Commands run:
  - `sed -n ...` and `rg -n ...` inspections for spec Section 20/26/27, policy schemas, pi_H planner, controller bias builder, physical-model builder, and existing tests
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_low_level_baseline.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_qpid_controller.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_low_level_baseline.py -q` passed: 4 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_qpid_controller.py -q` passed: 4 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 61 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Assumptions: P1 pi_L baseline is a deterministic tracking-intent scaffold. Object pose error is converted to a clipped residual wrench proxy; contact tracking bias is a small scaled copy of active assignment wrench targets. The P1 controller allocator supports bounded vertical thrust allocation only and reports unsupported lateral/torque wrench residuals as metrics/violations.
- Blockers: None.
- Next steps: Continue with P1 order 7, simplified grasp-carry simulation environment.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent H: pi_H baseline planner
- Summary: Implemented a deterministic P1 grasp/carry high-level planner that selects feasible contact assignments from `ContactCandidateSet` group proposals, caches assignment feasibility labels, and emits a schema-valid `ContactWrenchTrajectory`.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/high_level_policy_base.py`
  - `amsrr/policies/contact_wrench_trajectory.py`
  - `tests/unit/policies/test_high_level_baseline.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added policy-side `HighLevelPolicyContext`, `HighLevelPolicyBase`, `BaselineTrajectoryPlannerConfig`, and `GraspCarryBaselinePlanner`.
- Upstream dependencies used: v0.4 Sections 19, 26.8, 27.1, 28.10; Agent H ContactCandidateSampler; selected assignment feasibility evaluator; existing policy schemas.
- Downstream impact: Agent I pi_L baseline can now consume a deterministic `ContactWrenchTrajectory` with approach/attach/maintain/release assignments, posture anchor targets, object goal targets, and priority weights.
- Tests added or run:
  - Added `test_grasp_carry_baseline_planner_outputs_contact_wrench_trajectory`
  - Added `test_select_feasible_assignments_uses_grasp_pair_group`
- Commands run:
  - `sed -n ...` inspections for spec Section 19, Agent H deliverables, policy schemas, and existing tests
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_high_level_baseline.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_high_level_baseline.py -q` passed: 2 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 53 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: P1 pi_H baseline prefers `grasp_pair` proposals and emits a fixed five-knot grasp/carry schedule. It is deterministic scaffold logic, not a learned high-level policy or exhaustive assignment search.
- Blockers: None.
- Next steps: Continue with implementation order item 15 / P1 order 6, Agent I pi_L baseline policy and controller interface work.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent H/F: Selected assignment feasibility proxy
- Summary: Implemented selected-assignment feasibility evaluation for π_H-selected `ContactAssignment` sets, including candidate consistency, slot cardinality, pairwise conflict, grasp-opposition wrench proxy, friction/collision/QP residual hooks, cache updates, exports, and unit tests.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/assignment_feasibility.py`
  - `tests/unit/policies/test_contact_candidate_interfaces.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added a policy-side evaluator function that returns the existing `AssignmentFeasibilityResult` schema.
- Upstream dependencies used: v0.4 Sections 18.6, 18.7, 19.3, Appendix B.4, Appendix C; existing `ContactCandidateSet`, `ContactAssignment`, pairwise conflict matrices, and assignment-feasibility cache.
- Downstream impact: Agent H π_H baseline can evaluate only its selected assignments and cache infeasible selections without enumerating arbitrary candidate subsets. Later exact QP/collision/wrench evaluators can pass residuals/margins through the same result schema.
- Tests added or run:
  - Added `test_selected_assignment_feasibility_accepts_opposing_grasp_pair`
  - Added `test_selected_assignment_feasibility_rejects_cardinality_and_pair_conflict`
  - Added `test_selected_assignment_feasibility_rejects_non_opposing_grasp_normals`
- Commands run:
  - `sed -n ...` inspections for spec Sections 18.6, 18.7, 19.3, Appendix B/C, and existing assignment feasibility code
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_contact_candidate_interfaces.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_contact_candidate_interfaces.py -q` passed: 5 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 51 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: Assignment-level hard checks here are deterministic proxies: selected cardinality, pair conflicts, friction margin, and opposing-normal grasp proxy. They are not exact wrench closure, exact collision, or exact QP solving.
- Blockers: None.
- Next steps: Continue with implementation order item 14, Agent H π_H trajectory schema/baseline planner.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent H: ContactCandidateSampler
- Summary: Implemented a deterministic morphology-conditioned `ContactCandidateSampler` for P1 grasp/carry, optional group proposal support in `build_contact_candidate_set`, package exports, and unit tests covering non-empty candidate generation, grasp-pair proposals, anchor association preservation, and serialization.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/contact_candidate_set.py`
  - `amsrr/policies/contact_candidate_sampler.py`
  - `tests/unit/policies/test_contact_candidate_sampler.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Extended the existing contact-candidate helper function with optional `group_proposals` and `sampler_version` arguments while preserving prior defaults.
- Upstream dependencies used: v0.4 Sections 18, 24.2, 26.8, 27.1, 28.9; Agent D IRGBuilder and EnvelopeExtractor; Agent E fixed/simple `DesignOutput`; GeometryProcessor descriptors and ContactRegionGraph; existing ContactCandidate schemas.
- Downstream impact: Agent H π_H baseline planner can now consume finite morphology-conditioned `ContactCandidateSet` objects with slot coverage, pairwise matrices, and small grasp/support group proposals.
- Tests added or run:
  - Added `test_contact_candidate_sampler_returns_non_empty_grasp_carry_candidates`
  - Added `test_contact_candidate_sampler_builds_grasp_pair_group_proposals`
  - Added `test_contact_candidate_sampler_uses_robot_anchor_associations`
- Commands run:
  - `sed -n ...` inspections for spec Section 18, Agent H deliverables, geometry, IRG, morphology, and existing candidate helpers
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_contact_candidate_sampler.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_contact_candidate_sampler.py -q` passed: 3 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 48 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: P1 sampler emits deterministic smoke candidates and unary scores, not exact reachability/collision/QP feasibility. Default quota is one candidate per ContactSlot × ContactRegion × RobotAnchor. Grasp-pair proposals are small pairwise/group hints and are not task-feasibility proofs.
- Blockers: None.
- Next steps: Continue with implementation order item 14, Agent H π_H trajectory schema/baseline planner and selected-assignment feasibility interface.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent G: π_A GraphEditAssemblyPlanner
- Summary: Implemented deterministic graph-edit assembly planning over target `MorphologyGraph` dock edges, construction-state helpers, control handoff request scaffolding, executor interface records, package exports, and Agent G unit tests.
- Files changed:
  - `amsrr/assembly/__init__.py`
  - `amsrr/assembly/construction_state.py`
  - `amsrr/assembly/graph_edit_planner.py`
  - `amsrr/assembly/control_handoff.py`
  - `amsrr/assembly/executor_interface.py`
  - `tests/unit/assembly/test_graph_edit_planner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No changes to existing persisted schema modules. Added implementation-local assembly dataclasses/interfaces that match v0.4 Section 17 contracts inside `amsrr/assembly`.
- Upstream dependencies used: v0.4 Sections 17, 26.7, 27.1; existing `MorphologyGraph`, `DockEdge`, `Violation`, `MinimalMorphologyBuilder`, IRGBuilder, and PhysicalModel builder.
- Downstream impact: P1/P3 can now derive deterministic assembly step sequences from fixed/simple target morphologies. Agent H can proceed to ContactCandidateSampler using assembled/target graph contracts without needing learned assembly.
- Tests added or run:
  - Added `test_initial_construction_state_contains_base_only`
  - Added `test_graph_edit_planner_builds_deterministic_attach_sequence`
  - Added `test_graph_edit_planner_resumes_from_construction_state`
  - Added `test_control_handoff_request_for_docking_step`
- Commands run:
  - `sed -n ...` and `rg -n ...` inspections for spec Section 17, 26.7, 27.1, existing schemas, and tests
  - `mkdir -p amsrr/assembly tests/unit/assembly`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/assembly/test_graph_edit_planner.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/assembly/test_graph_edit_planner.py -q` passed: 4 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 45 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: Target morphology is treated as a connected tree rooted at `base_module_id` for v1/P1 scaffold planning. Each new dock edge expands to `move_to_staging -> align_ports -> dock -> verify_attach`. Exact assembly motion planning, retry execution, learned assembly, and simulator verification are out of scope for this slice.
- Blockers: None.
- Next steps: Continue with implementation order item 13, Agent H ContactCandidateSampler and ContactCandidateSet group proposal generation.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent E: Deterministic design teacher and π_D scaffolding
- Summary: Implemented the P1 fixed/simple morphology provider surface for π_D by adding a `DesignPolicyContext`, fixed-simple baseline design policy, deterministic design teacher variants, and a small action-candidate/STOP-mask generator over teacher traces.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/design_policy_base.py`
  - `amsrr/policies/design_candidate_generator.py`
  - `amsrr/policies/design_teacher.py`
  - `tests/unit/policies/test_design_teacher.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added policy-side interface/scaffold modules that consume existing `TaskSpec`, `InteractionRequirementGraph`, `InteractionEnvelope`, `PhysicalModel`, and `DesignOutput` schemas.
- Upstream dependencies used: v0.4 Sections 14, 15, 24.2, 26.5, 27.1; existing Agent E/F minimal morphology builder; Agent F FeasibilityChecker; Agent D IRGBuilder and EnvelopeExtractor; Agent B PhysicalModel.
- Downstream impact: Agent G can consume a deterministic target `DesignOutput` for assembly planning; Agent H can consume fixed/simple morphology and RobotAnchors for ContactCandidateSampler implementation; later learned π_D heads can replace the teacher scorer while keeping the same `DesignPolicyContext -> DesignOutput` boundary.
- Tests added or run:
  - Added `test_design_teacher_selects_p1_grasp_support_variant`
  - Added `test_design_candidate_trace_masks_stop_until_final_step`
  - Added `test_fixed_simple_design_policy_outputs_feasible_stop`
- Commands run:
  - `sed -n ...` inspections for spec Sections 14, 15, 26.5, 27.1, schema, morphology builder, and existing tests
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_design_teacher.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_design_teacher.py -q` passed: 3 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 41 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: Teacher variants are deterministic labels over the existing minimal connected-tree morphology scaffold. P1 object grasp/carry defaults to `tri_anchor_support_grasp` when the IRG contains required grasp slots plus an optional support slot. The candidate generator is an action-mask scaffold, not a learned scorer.
- Blockers: None.
- Next steps: Continue with implementation order item 12, Agent G π_A GraphEditAssemblyPlanner, then Agent H ContactCandidateSampler.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent H/I: P0 interface-only smoke pieces
- Summary: Implemented P0 interface-only helpers for ContactCandidateSet pairwise compatibility, assignment-level QP infeasibility reporting, and PolicyCommand-to-QP/PID reference bias building.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/contact_candidate_set.py`
  - `amsrr/policies/assignment_feasibility.py`
  - `amsrr/controllers/__init__.py`
  - `amsrr/controllers/policy_command_builder.py`
  - `tests/unit/policies/test_contact_candidate_interfaces.py`
  - `tests/unit/controllers/test_policy_command_builder.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added interface helper modules that consume existing `ContactCandidateSet`, `AssignmentFeasibilityResult`, `ContactAssignment`, `InteractionKnot`, and `PolicyCommand` schemas.
- Upstream dependencies used: v0.4 Sections 18, 19, 20, 26.8, 26.9, 27.2, 28.9, 28.10, 28.11, Appendix B.4; existing policy/contact candidate schemas.
- Downstream impact: ContactCandidateSampler, π_H trajectory planners, π_L policies, and controller backends have smoke-tested interface contracts for candidate pairwise matrices, selected-assignment feasibility cache entries, and desired bias references.
- Tests added or run:
  - Added `test_contact_candidate_pairwise_conflict_matrix`
  - Added `test_assignment_level_qp_infeasible_case`
  - Added `test_policy_command_bias_builder`
- Commands run:
  - `rg -n ...` and `sed -n ...` inspections for spec Sections 18-20, 26.8, 26.9, and 27.2
  - `mkdir -p amsrr/policies amsrr/controllers tests/unit/policies tests/unit/controllers`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 38 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: Pairwise conflict is limited to immediate candidate conflicts such as shared robot anchor; no exhaustive subset feasibility is performed. Assignment-level QP infeasibility only evaluates a selected assignment set. PolicyCommandBiasBuilder emits references for QP/PID and never final actuator commands.
- Blockers: None.
- Next steps: P0 Section 27.2 unit-test smoke coverage is now complete; later phases can implement full ContactCandidateSampler, π_H baseline trajectory planner, π_L baseline policy, and controller interfaces.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent E/F: Minimal MorphologyGraph + Feasibility hard-check scaffolding
- Summary: Implemented a deterministic minimal MorphologyGraph/DesignOutput builder and a design-level FeasibilityChecker scaffold for schema, connected graph, module count, port compatibility, closed-loop rejection, required slot coverage, coarse reachability, thrust margin, payload margin, and hover proxy checks.
- Files changed:
  - `amsrr/morphology/__init__.py`
  - `amsrr/morphology/graph.py`
  - `amsrr/feasibility/__init__.py`
  - `amsrr/feasibility/checker.py`
  - `amsrr/feasibility/violation_codes.py`
  - `tests/unit/morphology/test_minimal_morphology_builder.py`
  - `tests/unit/feasibility/test_feasibility_checker.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added implementation modules that consume existing `MorphologyGraph`, `DesignOutput`, and `FeasibilityResult` schemas.
- Upstream dependencies used: v0.4 Sections 14, 15.2, 15.3, 16, 26.5, 26.6, 27.1, 27.2, 28.6, 28.7; Agent B PhysicalModel; Agent D IRGBuilder.
- Downstream impact: Later design-policy scaffolding, assembly planning, contact candidate sampling, and assignment-level feasibility can consume a deterministic seed morphology and checker result.
- Tests added or run:
  - Added `test_minimal_morphology_builder_grasp_carry_design_output`
  - Added `test_minimal_morphology_design_output_roundtrip`
  - Added `test_feasibility_checker_accepts_minimal_design`
  - Added `test_feasibility_checker_rejects_missing_required_slot_coverage`
  - Added `test_feasibility_checker_rejects_closed_loop_v1`
- Commands run:
  - `rg -n ...` and `sed -n ...` inspections for spec Sections 14/16/26/28, schemas, and robot model utilities
  - `mkdir -p amsrr/morphology amsrr/feasibility/checks tests/unit/morphology tests/unit/feasibility`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 35 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: Minimal morphology is a deterministic seed/teacher scaffold, not an optimized policy output. Coarse thrust margin uses `abs(thrust_axis_local.z) * thrust_max_n` for the vectoring-capable Holon proxy. Coarse collision and QP hover are represented by necessary-condition scaffold checks, not exact simulation/QP.
- Blockers: None.
- Next steps: Continue with design policy scaffolding / deterministic teacher generator, then assembly planning and contact candidate sampling.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent A/L: SharedInteractionWorkspace tensor/mask contract
- Summary: Implemented workspace token group schema, strict group mask/slice validation, recommended learned-query specs, and a SharedInteractionWorkspaceBuilder that assembles modality token groups into a padded shared workspace with required empty groups.
- Files changed:
  - `amsrr/schemas/workspace.py`
  - `amsrr/encoders/__init__.py`
  - `amsrr/encoders/workspace_builder.py`
  - `tests/unit/schemas/test_workspace.py`
  - `tests/unit/encoders/test_workspace_builder.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: Strengthened internal workspace validation by adding `WorkspaceTokenGroup`, `OPTIONAL_WORKSPACE_GROUPS`, `WORKSPACE_GROUPS`, required `group_masks` shape checks, optional `contact_candidates`, and `recommended_learned_query_specs`.
- Upstream dependencies used: v0.4 Sections 21.6, 21.7, 26.1, 27.1, 27.2; prior InteractionEnvelopeEncoder token-group output.
- Downstream impact: Future modality encoders can produce `WorkspaceTokenGroup` objects and use `SharedInteractionWorkspaceBuilder` to assemble a single tensor/mask/source-id contract for π_D/π_H/π_L/critic/feasibility heads.
- Tests added or run:
  - Added `test_workspace_rejects_group_mask_mismatch`
  - Added `test_workspace_token_group_shapes`
  - Added `test_learned_query_spec_contract`
  - Added `test_workspace_builder_assembles_required_group_slices`
  - Added `test_workspace_builder_supports_optional_contact_candidate_group`
  - Added `test_workspace_builder_rejects_mismatched_d_model`
  - Added `test_empty_workspace_token_group_contract`
- Commands run:
  - `rg -n ...` and `sed -n ...` inspections for spec Section 21, workspace schema, and encoder outputs
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 30 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: Empty modality groups are represented as `[B, 0]` nested-list rows plus explicit `d_model`, then become zero-width slices in the assembled workspace. Query specs are contracts only; learned query parameters are not implemented here.
- Blockers: None.
- Next steps: Implementation order item 9 can build MorphologyGraph and DesignOutput; later modality encoders can feed additional non-empty groups into the workspace builder.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent D/A: InteractionEnvelopeExtractor + InteractionEnvelopeEncoder
- Summary: Implemented deterministic InteractionEnvelope extraction from IRG and a dependency-free InteractionEnvelopeEncoder contract that emits padded token tensors, masks, token type ids, source type ids, and source ids for the `interaction_envelope` workspace group.
- Files changed:
  - `amsrr/irg/__init__.py`
  - `amsrr/irg/envelope_extractor.py`
  - `amsrr/encoders/__init__.py`
  - `amsrr/encoders/interaction_envelope_encoder.py`
  - `tests/unit/irg/test_envelope_extractor.py`
  - `tests/unit/encoders/test_interaction_envelope_encoder.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added internal encoder output contract `InteractionEnvelopeEncoderOutput` for P0 token/mask/source-id handoff.
- Upstream dependencies used: v0.4 Sections 13, 21, 26.4, 27.1, 27.2, 28.5; Agent A schemas and workspace tensor shape helpers; Agent D IRGBuilder output.
- Downstream impact: π_D/π_H scaffolding and future SharedInteractionWorkspace assembly can consume deterministic envelope tokens. ContactCandidateSampler can use envelope target region sets, contact count ranges, and modes without reinterpreting TaskSpec directly.
- Tests added or run:
  - Added `test_interaction_envelope_extract`
  - Added `test_interaction_envelope_extracts_all_task_families`
  - Added `test_interaction_envelope_encoder_contract`
  - Added `test_interaction_envelope_encoder_batch_padding`
- Commands run:
  - `rg -n ...` and `sed -n ...` inspections for spec Sections 13/21, schemas, and IRG templates
  - `mkdir -p amsrr/encoders tests/unit/encoders`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 23 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: Required contact count range aggregates required ContactSlots only; optional slots still contribute contact mode and target-region tokens. The encoder implements the deterministic contract and `mlp_embedding` fallback metadata, not learned parameters.
- Blockers: None.
- Next steps: Implementation order item 8 can assemble modality token groups into full SharedInteractionWorkspace and learned query pooling contracts.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent D: IRGBuilder + InteractionTemplates
- Summary: Implemented deterministic IRGBuilder, SceneGraph normalization, IRG structural validator, and all five P0 task-family templates: free-flight navigation, object grasp/carry, valve operation, perching manipulation, and contact-mediated locomotion.
- Files changed:
  - `amsrr/irg/__init__.py`
  - `amsrr/irg/irg_builder.py`
  - `amsrr/irg/validator.py`
  - `amsrr/irg/templates/__init__.py`
  - `amsrr/irg/templates/base.py`
  - `amsrr/irg/templates/free_flight.py`
  - `amsrr/irg/templates/object_grasp_carry.py`
  - `amsrr/irg/templates/valve_operation.py`
  - `amsrr/irg/templates/perching_manipulation.py`
  - `amsrr/irg/templates/contact_mediated_locomotion.py`
  - `tests/unit/irg/test_irg_builder.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Existing TaskSpec, GeometryDescriptor, InteractionRequirementGraph, IRGNode, IRGEdge, PhaseType, ConstraintType, CapabilityType, and ContactMode schemas were used unchanged.
- Upstream dependencies used: v0.4 Sections 10, 11, 12, 26.4, 27.1, 27.2, 28.3, 28.4; Agent A schemas; Agent C GeometryProcessor outputs.
- Downstream impact: Agent E EnvelopeExtractor and downstream policy/feasibility work can now consume valid IRGs for every P0 task family. IRGs remain abstract and do not include final contact poses, robot anchors, morphology, trajectories, or actuator commands.
- Tests added or run:
  - Added `test_phase_label_to_phase_type_mapping`
  - Added `test_irg_builder_grasp_carry_valid`
  - Added `test_irg_builder_all_task_families_smoke`
- Commands run:
  - `find amsrr/irg tests/unit/irg -type f | sort`
  - `sed -n ...` inspections for IRGBuilder, templates, validator, schemas, and spec sections
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `python3 - <<'PY' ...` smoke inspection of object grasp/carry IRG node and edge counts
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 19 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: Template-local phase labels are preserved in `phase_label` and mapped into existing `PhaseType` values. Template-local constraint concepts that are not v0.4 `ConstraintType` enum values are represented by the closest standard enum and preserved in `parameters["template_constraint"]`.
- Blockers: None.
- Next steps: Agent E EnvelopeExtractor should compute compact summaries from these IRGs without treating the envelope as source of truth. The Section 26.4 `envelope_extractor.py` item remains for the next work package because this task explicitly targeted item 6, IRGBuilder and all task templates.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent C: GeometryProcessor
- Summary: Implemented deterministic GeometryProcessor for primitives and mesh smoke, including asset resolution, primitive analytic surface decomposition, STL/OBJ mesh summary loading, surface patch graph construction, and contact region graph construction.
- Files changed:
  - `amsrr/geometry/__init__.py`
  - `amsrr/geometry/asset_resolver.py`
  - `amsrr/geometry/surface_patch_graph.py`
  - `amsrr/geometry/contact_region_extractor.py`
  - `amsrr/geometry/geometry_processor.py`
  - `tests/unit/geometry/test_geometry_processor.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Existing `GeometryDescriptor`, `GlobalShapeFeatures`, `SurfacePatchGraph`, `ContactRegionGraph`, `SurfacePatchToken`, and `ContactRegion` schemas were used unchanged.
- Upstream dependencies used: v0.4 Sections 8.1-8.10, 26.3, 27.1, 27.2; Agent A schemas; existing mesh assets under `module_urdf/mesh/`.
- Downstream impact: Agent D IRGBuilder can consume primitive and mesh `GeometryDescriptor` outputs. P0 now has box primitive regions and mesh smoke coverage.
- Tests added or run:
  - Added `test_geometry_processor_box_regions`
  - Added `test_geometry_processor_mesh_smoke`
- Commands run:
  - `mkdir -p amsrr/geometry tests/unit/geometry assets/objects/primitives assets/objects/meshes`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 16 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: P0 mesh support is a deterministic smoke implementation, not full mesh repair/segmentation. Mesh descriptors expose hashed refs such as `mesh://sha256:<hash>` instead of raw asset paths.
- Blockers: None.
- Next steps: Agent D IRGBuilder and templates can use these descriptors; later mesh work can replace smoke normal-cluster aggregation with richer segmentation without schema changes.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent B: URDF / PhysicalModel
- Summary: Implemented URDF/xacro XML loader, thrust model YAML loader, PhysicalModel builder, ModuleCapabilityToken builder, normalized runtime Holon URDF asset, and Agent B unit tests.
- Files changed:
  - `assets/robots/holon/holon.urdf`
  - `amsrr/robot_model/__init__.py`
  - `amsrr/robot_model/urdf_loader.py`
  - `amsrr/robot_model/thrust_model.py`
  - `amsrr/robot_model/physical_model_builder.py`
  - `tests/unit/robot_model/test_urdf_loader.py`
  - `tests/unit/robot_model/test_thrust_model.py`
  - `tests/unit/robot_model/test_physical_model_builder.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Existing `PhysicalModel`, `LinkModel`, `JointModel`, `RotorModel`, `DockPortSpec`, and `ModuleCapabilityToken` schemas were used unchanged.
- Upstream dependencies used: v0.4 Sections 3.1, 3.2, 9.1-9.8, 26.2, 27.1, 27.2; `module_urdf/README_for_codex.md`; existing Agent A schemas.
- Downstream impact: Agent C/D/F and later controller work can now load a configurable runtime URDF path and receive structured `PhysicalModel` plus compact module capability features.
- Tests added or run:
  - Added `test_urdf_parse_holon_if_present`
  - Added `test_urdf_parse_holon_xacro_reference`
  - Added `test_asset_urdf_uses_config_thrust_link_names`
  - Added `test_thrust_model_loads_config`
  - Added `test_thrust_model_rejects_duplicate_rotor_ids`
  - Added `test_physical_model_total_mass_positive`
  - Added `test_physical_model_rotors_and_dock_ports`
  - Added `test_module_capability_token_from_physical_model`
- Commands run:
  - `mkdir -p assets/robots/holon`
  - `cp module_urdf/holon.urdf.xacro assets/robots/holon/holon.urdf`
  - `perl -0pi -e 's/thrust([1-4])\b/thrust_$1/g' assets/robots/holon/holon.urdf`
  - `mkdir -p amsrr/robot_model tests/unit/robot_model`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 14 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: `module_urdf/holon.urdf.xacro` can be parsed as XML without ROS/xacro macro expansion. Runtime asset path remains configurable and uses `assets/robots/holon/holon.urdf`. `thrust_1` config IDs are preserved as schema rotor IDs.
- Blockers: None. `module_urdf/holon.urdf` is absent, so its explicit test is skipped by design.
- Next steps: Agent C GeometryProcessor for primitives and mesh smoke, or Agent D IRGBuilder once geometry descriptors exist.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Repository organization / handoff documentation
- Summary: Moved Codex-facing project documents into `for_codex/` and prepared current implementation files for git tracking.
- Files changed:
  - `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `for_codex/AGENTS.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None.
- Upstream dependencies used: User request to move Codex-facing documents under `for_codex/` and commit current implementation.
- Downstream impact: Future coding assistants should read `for_codex/AGENTS.md`, `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`, and `for_codex/WORKLOG.md` from the new directory.
- Tests added or run: Reran current unit tests before commit.
- Commands run:
  - `git status --short`
  - `git status --short --untracked-files=all`
  - `git log --oneline --max-count=5`
  - `git ls-files`
  - `mkdir -p for_codex`
  - `git mv A-MSRR_codex_ready_spec_v0_4_ja.md AGENTS.md AMSRR_design_modification_by_codex.md WORKLOG.md for_codex/`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 7 passed.
- Assumptions: Documentation relocation does not change runtime import paths or test behavior.
- Blockers: None.
- Next steps: Stage and commit moved documentation plus current schema/config/test implementation.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent A: Schemas and validation; Agent A/L support: config loading, hashing, initial test harness
- Summary: Implemented schema-first dataclass models, strict JSON/YAML coercion and validation helpers, config loading, stable hashing, minimum robot/training config files, and unit tests for schema roundtrip, TaskSpec parsing, workspace masks, and config hashing.
- Files changed:
  - `amsrr/__init__.py`
  - `amsrr/schemas/__init__.py`
  - `amsrr/schemas/common.py`
  - `amsrr/schemas/task_spec.py`
  - `amsrr/schemas/geometry.py`
  - `amsrr/schemas/irg.py`
  - `amsrr/schemas/interaction_envelope.py`
  - `amsrr/schemas/morphology.py`
  - `amsrr/schemas/physical_model.py`
  - `amsrr/schemas/runtime.py`
  - `amsrr/schemas/policies.py`
  - `amsrr/schemas/feasibility.py`
  - `amsrr/schemas/workspace.py`
  - `amsrr/schemas/contact_candidates.py`
  - `amsrr/utils/__init__.py`
  - `amsrr/utils/config.py`
  - `amsrr/utils/hashing.py`
  - `configs/robot/robot_model.yaml`
  - `configs/robot/thrust_model.yaml`
  - `configs/training/p0_schema_tests.yaml`
  - `tests/conftest.py`
  - `tests/unit/schemas/test_task_spec.py`
  - `tests/unit/schemas/test_schema_roundtrip.py`
  - `tests/unit/schemas/test_workspace.py`
  - `tests/unit/utils/test_config_hashing.py`
- Schema/interface changes: Initial schema/interface implementation. Added approved supplement `IRGEdgeType.ALLOWS = "allows"` because v0.4 uses `allows` edges in diagrams and examples but omits it from the enum listing.
- Upstream dependencies used: v0.4 Sections 7, 8, 9, 10, 13, 14, 16, 18, 19, 20, 21, 23, 25, 26.1, 27.1, 27.2; AGENTS.md implementation rules.
- Downstream impact: Agent B/C/D can now consume stable dataclass schemas and config/hash utilities. IRGBuilder can emit `contact_region --allows--> contact_slot` edges without inventing a local edge string.
- Tests added or run:
  - Added `test_task_spec_parse_grasp_carry_yaml`
  - Added `test_task_spec_rejects_missing_grasp_carry_mass`
  - Added `test_schema_roundtrip_json`
  - Added `test_irg_edge_type_includes_allows`
  - Added `test_shared_interaction_workspace_tensor_shapes`
  - Added `test_padded_tensor_masks`
  - Added `test_config_loading_and_hashing`
- Commands run:
  - `python3 --version`
  - `python3 -m pytest --version`
  - `python3 -c "import yaml; print(yaml.__version__)"`
  - `python3 -m pytest tests/unit -q` failed before collection due external pytest plugin `launch_testing` hook incompatibility.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 7 passed.
  - `python3 -m compileall amsrr -q` passed.
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +` removed generated Python cache directories after tests/compile checks.
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed; `python3 -m compileall amsrr -q` passed.
- Assumptions: No new package installation; use standard-library dataclasses plus installed PyYAML. Spec examples may omit optional fields, so optional schema fields default to `None` where needed for the provided YAML.
- Blockers: None for Agent A items 1-2. Full P0 tests for URDF, GeometryProcessor, IRGBuilder, EnvelopeExtractor, and downstream policy/controller behavior remain unimplemented.
- Next steps: Agent B URDF/PhysicalModel loader and/or Agent C primitive GeometryProcessor, using the schemas added here.

---

## Work Package Logs

### P4-Control / P4a: QP/PID Controller Specification

#### 2026-07-09
- Scope: Implement Agent I Order 12 `PolicyCommand` PID target builder.
- Files changed:
  - `amsrr/controllers/qpid_controller.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Controller supplement PID target-builder section, user-provided PID gains, `PolicyCommand` direct body target fields, runtime pose/twist, and QP allocation result status.
- Implemented: PID gains in `QPIDControllerConfig`, body-pose/twist target wrench generation, world-frame gravity/position PID force converted to body frame, quaternion body-frame attitude PID torque, feedforward residual wrench addition, target tracking metrics, and allocation-gated integral anti-windup.
- Not implemented: Closed-loop Isaac execution, fixed-morphology hover, waypoint smoke, explicit desired-wrench saturation limits, or P4-control completion.
- Schema/interface changes: None.
- Downstream impact: Agent J/K closed-loop smoke can now compute `ControllerCommand` from observed pose/twist and `PolicyCommand.desired_body_pose` without bypassing the controller boundary.
- Tests added: `test_qpid_controller_builds_pid_wrench_from_policy_body_target_and_feedforward`.
- Tests passed: Controller unit tests passed: 9 passed. Full unit suite passed: 122 passed, 1 skipped. Compileall and `git diff --check` passed.
- Handoff notes: IsaacLab 3 root poses use XYZW quaternions, matching A-MSRR `Pose7D`. Integral state is held by the controller instance, so closed-loop smoke must reuse the controller across steps.
- Open questions: None for this order.

#### 2026-07-09
- Scope: Tune Agent I/J controller-command QP feasibility after the first real controller-to-Isaac smoke.
- Files changed:
  - `amsrr/controllers/qp_allocator_interface.py`
  - `amsrr/controllers/qpid_controller.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `tests/unit/simulation/test_p4_control_controller_smoke.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Agent I virtual-thrust QP allocator and RigidBodyControlModel, Agent J controller-command smoke, Holon actuator limits and real Isaac command surface.
- Implemented: Reduced QP smoothing/regularization weights so the primary objective tracks hover wrench before smoothing, set controller residual tolerance to `1e-5`, added tolerance-aware hard-check clipping detection, and held vectoring targets at current position for effectively zero-thrust rotors.
- Not implemented: Closed-loop hover controller, fixed-morphology smoke, waypoint smoke, object grasp/carry, or P4 full completion.
- Schema/interface changes: None.
- Downstream impact: The controller-command smoke can now be used as a clean precondition for closed-loop work because QP infeasibility is no longer caused by numerical tuning or zero-thrust vectoring angle singularity.
- Tests added: Rigid-body QP hover feasibility unit assertion and strengthened controller-command smoke assertions.
- Tests passed: Focused tests passed: 9 passed. Full unit suite passed: 121 passed, 1 skipped. Compileall and `git diff --check` passed. Real Isaac controller-command smoke passed with QP feasible/ok and no bridge target violations after an approved external run.
- Handoff notes: `BoundedVerticalRotorAllocator` remains only degraded fallback. The `1e-5` tolerance is still below the controller warning threshold and should not be reused as a success metric for real hover.
- Open questions: None for this tuning order.

#### 2026-07-09
- Scope: Probe real IsaacLab URDF conversion for Holon and correct generated USD path.
- Files changed:
  - `configs/env/isaac_lab.yaml`
  - `amsrr/simulation/isaac_lab_backend.py`
  - `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: AGENTS.md `isaaclab3` environment, IsaacLab `convert_urdf.py`, Holon URDF, and Order 5 backend config.
- Implemented: Verified `isaaclab` Python module visibility under micromamba, verified `isaaclab.sh -p` can run the P4-control probe script, ran IsaacLab URDF conversion to `/tmp`, and corrected generated USD expected path to `<generated_usd_dir>/holon/holon.usda`.
- Not implemented: Workspace USD generation, committed USD assets, articulation spawn, force application, joint target execution, or real smoke simulation.
- Schema/interface changes: None.
- Downstream impact: Later spawn code should consume `artifacts/isaac/robots/holon/holon/holon.usda` after conversion rather than expecting `holon.usd` directly in the generated root.
- Tests added: Updated generated USD path assertion.
- Tests passed: Related env/runner tests passed: 8 passed. Full unit suite passed: 118 passed, 1 skipped. Compileall and `git diff --check` passed. Real Isaac URDF conversion probe passed.
- Handoff notes: The converter warned that `--headless` is deprecated and that omitting `--viz` is now default headless. Future command builders should avoid depending on `--headless`.
- Open questions: None for conversion. Real spawn/control APIs are next.

#### 2026-07-09
- Scope: Implement Agent J/K boundary Order 5 smoke runner configuration and dry-run harness.
- Files changed:
  - `configs/env/isaac_lab.yaml`
  - `configs/training/p4_control_low_level.yaml`
  - `amsrr/simulation/isaac_lab_backend.py`
  - `amsrr/simulation/p4_control_smoke.py`
  - `amsrr/simulation/p4_control_isaac_env.py`
  - `amsrr/simulation/__init__.py`
  - `amsrr/training/p4_control_runner.py`
  - `amsrr/training/__init__.py`
  - `scripts/p4_control_smoke.py`
  - `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `tests/unit/training/test_p4_control_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: AGENTS.md Isaac environment notes, P4-control controller supplement, prior acceptance split, and approved recommendations for URDF-to-USD custom articulation plus wrench-composer thrust application.
- Implemented: Config-driven IsaacLab backend probe, URDF conversion command construction, low-level smoke thresholds/scenarios, shared smoke result dataclass, dry-run smoke runner, CLI probe/dry-run entry point, and tests proving dry-run cannot pass completion.
- Not implemented: Real Isaac API execution, URDF-to-USD conversion run, Holon spawn, rotor force application, joint command application, runtime observation extraction, or real smoke pass/fail measurement.
- Schema/interface changes: None to persisted schemas. Added simulation/training-side contracts.
- Downstream impact: The next order can run the CLI under `isaaclab3` / `isaaclab.sh -p` to validate environment imports and then implement real execution behind the existing scenario/result contract.
- Tests added: P4-control backend/env config tests and P4-control runner dry-run tests.
- Tests passed: New P4-control env/runner tests passed: 8 passed. Full unit suite passed: 118 passed, 1 skipped. Targeted P4-control acceptance tests passed: 2 passed. Compileall, `git diff --check`, CLI probe, and CLI dry-run passed.
- Handoff notes: Running the CLI with normal repo Python reports `isaac_python_modules_unavailable_in_current_interpreter`, which is expected. Real probes should use the AGENTS.md micromamba/IsaacLab launch path.
- Open questions: None for config/dry-run. Real Isaac execution details remain to be probed in the next order.

#### 2026-07-09
- Scope: Implement Agent L boundary Order 4 P4-control fast/real acceptance split.
- Files changed:
  - `amsrr/acceptance/p4_control_acceptance.py`
  - `amsrr/acceptance/__init__.py`
  - `tests/acceptance/test_p4_control_acceptance.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: P4-control acceptance requirements, user clarification on real Isaac smoke gating, existing P4.0 acceptance/report style, and `EpisodeArchive` P4 logging fields.
- Implemented: `P4ControlSmokeResult`, `P4ControlAcceptanceReport`, `run_p4_control_acceptance`, fast archive checks for controller/runtime/actuator/residual metrics, real smoke aggregation for the three required Isaac smoke cases, and tests proving fast gate does not imply completion.
- Not implemented: Real Isaac smoke runner, Isaac environment spawn/step, hover stabilization, fixed-morphology waypoint tracking, or artifact collection from actual Isaac.
- Schema/interface changes: None to persisted schemas. Added acceptance-only dataclasses and exports.
- Downstream impact: Later P4-control smoke runners can call this acceptance function with real Isaac-backed smoke results. The gate will keep completion false if smoke is missing, skipped, synthetic-only, or failed.
- Tests added: `test_p4_control_fast_gate_does_not_complete_without_real_isaac_smoke` and `test_p4_control_completion_requires_all_real_isaac_smokes`.
- Tests passed: P4-control acceptance tests passed: 2 passed. Full acceptance suite passed: 9 passed. Full unit suite passed: 110 passed, 1 skipped. Compileall and `git diff --check` passed.
- Handoff notes: This order intentionally does not claim P4-control completion. It codifies the split gate so later real smoke artifacts cannot be accidentally replaced by unit tests.
- Open questions: Real Isaac execution details remain for the next runner/backend order.

#### 2026-07-09
- Scope: Implement Agent I/J boundary Order 3 actuator mapping and bridge target records.
- Files changed:
  - `amsrr/controllers/actuator_mapping.py`
  - `amsrr/controllers/isaac_controller_bridge.py`
  - `amsrr/controllers/__init__.py`
  - `tests/unit/controllers/test_actuator_mapping.py`
  - `tests/unit/controllers/test_isaac_controller_bridge.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: v0.4 controller bridge requirements, controller supplement bridge/logging section, `PhysicalModel`, `MorphologyGraph`, `ControllerCommand`, and `EpisodeArchive.actuator_target_records`.
- Implemented: Active actuator channel extraction for rotors/vectoring joints/dock joints/joint efforts, deterministic global actuator ids, single-module local aliases, actuator limit clipping, `IsaacControllerBridge` conversion to target records, missing/unsupported/clipped metrics, controller residual/QP status propagation, and JSON-compatible record roundtrip tests.
- Not implemented: Isaac Lab API calls, robot spawning, force application, P4-control runner, real single-module hover, fixed-morphology hover, or waypoint tracking smoke.
- Schema/interface changes: None to persisted schemas. Added controller-local bridge dataclasses and exports.
- Downstream impact: Later Isaac backend and runner code can use these records as the stable bridge contract and store their dict form in `EpisodeArchive.actuator_target_records`.
- Tests added: `test_actuator_mapping_builds_single_module_aliases_and_limits`, `test_actuator_mapping_uses_global_keys_for_multiple_modules`, `test_clip_to_channel_reports_clipped_value`, `test_isaac_controller_bridge_converts_and_clips_targets`, and target-record roundtrip test.
- Tests passed: Controller/mapping/bridge tests passed: 16 passed. Full unit suite passed: 110 passed, 1 skipped. Compileall and `git diff --check` passed.
- Handoff notes: This is a fast pytest bridge contract only. P4-control completion still requires real Isaac smoke with these records or equivalent actuator targets actually executed.
- Open questions: None for this bridge-contract order.

#### 2026-07-09
- Scope: Implement Agent I Order 2 primary virtual-thrust QP allocator and controller integration.
- Files changed:
  - `amsrr/controllers/rigid_body_model.py`
  - `amsrr/controllers/qp_allocator_interface.py`
  - `amsrr/controllers/qpid_controller.py`
  - `amsrr/controllers/__init__.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `tests/unit/controllers/test_rigid_body_model.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Agent I Order 1 rigid-body model, P4-control controller supplement, user clarification on rotor-arm-fixed x/z virtual channels and QP constraints plus hard check/clamp.
- Implemented: `VirtualThrustQPAllocator`, optional rigid-body-model allocation inputs, optional vectoring outputs and achieved wrench in allocation results, virtual x/z channel construction, vectoring joint limit/rate linear constraints, thrust/joint hard check and clamp after back-conversion, QP/degraded metrics, controller `allocation_mode="rigid_body_qp"`, and focused unit tests.
- Not implemented: Isaac controller bridge, actuator target record conversion, P4-control runner, real Isaac single-module/fixed-morphology smoke, or waypoint tracking smoke.
- Schema/interface changes: None to persisted schemas. Controller-local dataclasses gained backward-compatible optional fields.
- Downstream impact: Agent J/K/L can now consume a primary QP controller path for fast tests and bridge development, while treating `BoundedVerticalRotorAllocator` as degraded fallback only.
- Tests added: `test_virtual_thrust_qp_allocator_back_converts_vectoring_channel`, `test_virtual_thrust_qp_allocator_applies_limits_and_hard_clamp`, `test_qpid_controller_can_select_rigid_body_qp_primary_path`, plus rigid-body virtual axis/current-q assertions.
- Tests passed: Targeted QP/controller/rigid-body tests passed: 10 passed. Controller unit tests passed: 11 passed. Full unit suite passed: 105 passed, 1 skipped. Compileall and `git diff --check` passed.
- Handoff notes: Virtual z is rotor-arm fixed but sign-aligned with the rotor's positive thrust axis. Exact thrust magnitude bounds are rechecked after solving because the virtual-channel magnitude constraint is nonlinear; QP still includes actuator box bounds and vectoring angle/rate linear constraints.
- Open questions: None for the fast pytest gate. P4-control completion remains blocked on real Isaac smoke in later orders.

#### 2026-07-09
- Scope: Implement Agent I Order 1 deterministic `RigidBodyControlModel` update.
- Files changed:
  - `amsrr/controllers/rigid_body_model.py`
  - `amsrr/controllers/__init__.py`
  - `tests/unit/controllers/test_rigid_body_model.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: `PhysicalModel`, `MorphologyGraph`, `RuntimeObservation`, Holon URDF-derived joints/links/rotors, P4-control controller supplement, and user virtual-thrust-channel clarification.
- Implemented: Link-tree forward kinematics from current joint positions, `fc`/baselink-relative module transforms, composite COM and link-level inertia aggregation, body frame at COM with base module orientation, current rotor origin/axis extraction, scalar rotor allocation matrix columns, vectoring joint axes, dock actuator ids, active actuator limits, controller package exports, and unit tests.
- Not implemented: Virtual thrust channel QP expansion, QP solve, back-conversion to `ControllerCommand`, Isaac controller bridge, P4-control runner, or real Isaac smoke.
- Schema/interface changes: None to persisted schemas. Added controller-local internal dataclasses only.
- Downstream impact: The QP allocator can now consume current `B(q)`-style rotor geometry and per-module actuator keys without re-parsing URDF or hard-coding robot paths.
- Tests added: `test_rigid_body_model_builds_single_module_allocation_matrix`, `test_rigid_body_model_updates_rotor_axis_from_joint_position`, `test_rigid_body_model_handles_multiple_modules_with_unique_actuator_ids`.
- Tests passed: Targeted rigid-body tests passed: 3 passed. Controller unit tests passed: 8 passed. Full unit suite passed: 102 passed, 1 skipped. Compileall passed.
- Handoff notes: Zero-axis fixed joints are intentionally skipped in the joint-axis map while still participating in kinematic transforms. Rotor allocation columns are scalar-thrust columns; virtual channel expansion belongs to the next QP allocator order.
- Open questions: None currently known.

#### 2026-07-09
- Scope: Record P4-control virtual thrust channel, QP-primary allocation, and split acceptance-gate clarifications before implementation.
- Files changed:
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: User clarification, v0.4 controller/P4-control sections, and the controller supplement.
- Implemented: Documentation supplement requiring per-step `q`-conditioned rigid-body model and `B(q)` updates, virtual thrust channels inside QP with back-conversion to rotor thrust/vectoring targets, degraded-only bounded vertical fallback, and separate fast pytest / real Isaac smoke acceptance gates.
- Not implemented: Controller code, rigid-body model, QP allocator, Isaac bridge, P4-control runner, or acceptance gate.
- Schema/interface changes: None.
- Downstream impact: Agent I/J/K/L implementation should treat these clarified requirements as active P4-control constraints.
- Tests added: None.
- Tests passed: `git diff --check` passed.
- Handoff notes: P4-control completion requires real Isaac smoke; Isaac-unavailable skips are acceptable only for non-completion unit smoke paths.
- Open questions: None currently known.

#### 2026-07-09
- Scope: Add main-spec cross-references to the P4-control QP/PID controller supplement.
- Files changed:
  - `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: User request, revised controller supplement, v0.4 Section 20 and Section 24.5.2.
- Implemented: Main spec references to `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md` from low-level control and P4-control sections.
- Not implemented: Controller code, schema changes, Isaac bridge, P4-control runner, or acceptance gate.
- Schema/interface changes: None.
- Downstream impact: Controller implementers now have an explicit pointer from the source design spec to the controller supplement.
- Tests added: None.
- Tests passed: Not run; documentation-only change. `git diff --check` passed.
- Handoff notes: The cross-reference does not weaken the main spec rule that `π_L` outputs intent only.
- Open questions: None currently known at this documentation level.

#### 2026-07-09
- Scope: Record resolved controller implementation decisions before coding.
- Files changed:
  - `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: User answers to open questions, revised P4-control controller spec, local Isaac Lab multirotor/quadcopter examples.
- Implemented: Initial Python/library QP backend decision, per-thruster thrust target primary representation with wrench-composer fallback, absolute vectoring joint targets, reaction torque inclusion, link-level quasi-static inertia aggregation, and initial waypoint thresholds.
- Not implemented: Controller code, QP backend, Isaac bridge, P4-control runner, or acceptance gate.
- Schema/interface changes: None.
- Downstream impact: Agent I/J can now implement against resolved controller assumptions; additional undefined details still require user confirmation before incompatible assumptions are encoded.
- Tests added: None.
- Tests passed: Not run; documentation-only decision recording. `git diff --check` passed after edits.
- Handoff notes: Archive per-rotor thrust targets even if Isaac execution uses wrench composer internally.
- Open questions: None currently known at this planning level.

#### 2026-07-09
- Scope: Revise the P4-control controller spec after user clarification.
- Files changed:
  - `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: User clarification, v0.4 controller/P4-control sections, existing controller draft.
- Implemented: Japanese-first controller spec, no reference-implementation notes section, QP allocation as the required allocator path, quasi-static rigid-body model update for assembled morphologies, ControllerCommand / bridge / archive logging requirements, and implementation-before-coding questions.
- Not implemented: Controller code, QP solver backend, rigid-body model code, actuator mapping, Isaac bridge, P4-control runner, or acceptance gate.
- Schema/interface changes: None.
- Downstream impact: Agent I implementation should build deterministic model-update and QP allocation code from the revised spec and keep any non-QP fallback explicitly marked as degraded.
- Tests added: None.
- Tests passed: Not run; documentation-only revision. `git diff --check` passed after edits.
- Handoff notes: Combined morphology control assumes joint motion is quasi-static. Each control cycle updates inertia/CoM/rotor geometry from current joint angles, then controls the current shape as one rigid body.
- Open questions: QP solver/backend choice, Isaac actuator semantics, vectoring target semantics, reaction torque treatment, inertia aggregation fidelity, waypoint error thresholds.

#### 2026-07-09
- Scope: Create a controller-specific draft spec and read the gimbal rotor reference controller at a high level before implementation.
- Files changed:
  - `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: v0.4 controller/P4-control sections, current Agent I controller scaffolds, temporary `aerial_robot_base` reference source.
- Implemented: Draft controller spec skeleton with reference branch notes, QP/PID layer placeholders, allocation model placeholders, Isaac bridge/logging expectations, proposed Agent I/J/K/L files, and open questions.
- Not implemented: Controller code, actuator mapping, Isaac bridge, P4-control runner, acceptance gate, QP solver backend, or Isaac execution.
- Schema/interface changes: None.
- Downstream impact: Agent I implementation should begin only after this draft is refined into concrete equations and config/test requirements.
- Tests added: None.
- Tests passed: Not run; documentation-only task.
- Handoff notes: Keep `π_L` authority limited to `PolicyCommand`; controller/bridge remains final actuator authority. The reference branch sends base thrust plus torque allocation matrix to FC rather than directly publishing gimbal commands.
- Open questions: Frame conventions, actuator target semantics in Isaac, reaction torque handling, solver/fallback choice, multi-module inertia aggregation, and waypoint error thresholds.

### P4.0 Implementation: Simplified Full-Pipeline Integration

#### 2026-07-08
- Scope: Order 6 final docs, design modification log, and full verification.
- Files changed:
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: P4.0 Orders 1-5 and full suite verification.
- Implemented: P4.0 design modification supplement and final worklog handoff entry with full unit/acceptance results.
- Not implemented: Isaac Lab backend, controller bridge / actuator mapping, P4-control, P4.1/P4.2 deterministic Isaac rollout, P4.3 learning bootstrap, or P4 full acceptance.
- Schema/interface changes: None in this order.
- Downstream impact: Next work package should start with controller bridge / actuator mapping and P4-control, using P4.0 acceptance as the simplified wiring prerequisite.
- Tests added: None.
- Tests passed: Full unit suite passed: 99 passed, 1 skipped in 4.67s. Full acceptance suite passed: 7 passed in 118.22s.
- Handoff notes: P4.0 is accepted as simplified full-pipeline wiring only; it must not be reported as Isaac-backed full grasp/carry completion.
- Open questions: None currently.

#### 2026-07-08
- Scope: Order 5 P4.0 simplified acceptance gate.
- Files changed:
  - `amsrr/acceptance/p4_0_acceptance.py`
  - `amsrr/acceptance/__init__.py`
  - `amsrr/training/p4_0_full_pipeline_runner.py`
  - `tests/acceptance/test_p4_0_acceptance.py`
- Upstream dependencies: P4.0 runner/archive tests, P4.0 no-mislabeling metadata, v0.4 Section 24.5.1 and Section 27.3 acceptance items.
- Implemented: `P4_0AcceptanceCriteria`, `P4_0AcceptanceReport`, `run_p4_0_acceptance`, P2/P3 usage checks, FixedSimple absence check, candidate/trajectory/policy/controller/archive completeness checks, simplified metric recording checks, no-mislabeling checks, backend-note report field, acceptance export, and acceptance test.
- Not implemented: Isaac controller bridge, actuator mapping, P4-control, P4.1/P4.2/P4.3, or P4 full acceptance.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P4.0 simplified full-pipeline wiring can be accepted independently, and later P4 stages can require this gate as a prerequisite.
- Tests added: `test_p4_0_acceptance_simplified_full_pipeline`.
- Tests passed: P4.0 acceptance targeted test passed: 1 passed. P4.0 unit + acceptance passed: 3 passed. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: The report explicitly says the metrics are not Isaac-backed physical success rates; this is not P4 full completion.
- Open questions: None currently.

#### 2026-07-08
- Scope: Order 4 P4.0 unit, archive completeness, and no-mislabeling tests.
- Files changed:
  - `tests/unit/training/test_p4_0_full_pipeline_runner.py`
- Upstream dependencies: Order 3 runner and config, `EpisodeArchive` defaults, simplified env external design injection, P2/P3 scaffolds.
- Implemented: Unit test coverage for P4.0 runner config loading, full simplified pipeline archive contents, P2 selected design usage, P3 assembly result usage, contact candidates, trajectory records, policy/controller commands, runtime observations, rewards, and explicit no-Isaac/no-P4-full rollout metadata.
- Not implemented: P4.0 acceptance gate, report schema, full acceptance test, Isaac backend, or learning bootstrap.
- Schema/interface changes: None.
- Downstream impact: Agent L acceptance can aggregate runner metrics and check phase-level pass/fail criteria with confidence that per-archive fields are complete.
- Tests added: `test_p4_0_full_pipeline_runner_archives_full_simplified_pipeline`, `test_p4_0_full_pipeline_runner_config_loader`.
- Tests passed: P4.0 runner tests passed: 2 passed. Related env/P1/P4.0 tests passed: 9 passed. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: The test asserts `rollout_artifacts["is_p4_full_completion"] is False`, `isaac_backed is False`, and `physical_success_claim is False`.
- Open questions: None currently.

#### 2026-07-08
- Scope: Order 3 P4.0 full-pipeline runner implementation.
- Files changed:
  - `amsrr/training/p4_0_full_pipeline_runner.py`
  - `amsrr/training/__init__.py`
  - `configs/training/p4_0_grasp_carry.yaml`
- Upstream dependencies: P2 selected design path, P3 simplified assembly result, Order 2 env injection, ContactCandidateSampler, `GraspCarryBaselinePlanner`, `BaselineLowLevelPolicy`, `QPIDController`, and `EpisodeArchive`.
- Implemented: `P4_0FullPipelineRunnerConfig`, `P4_0FullPipelineRunnerResult`, config loader, `P4_0FullPipelineRunner`, deterministic episode sampling, P2 selection, P3 assembly execution, simplified rollout execution, reward/metric aggregation, archive writing, and explicit simplified-backend no-P4-full metadata.
- Not implemented: Unit/archive/no-mislabeling tests, P4.0 acceptance gate, Isaac backend, controller bridge, actuator mapping, or learning bootstrap.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Order 4 can assert the runner uses P2/P3 outputs, generates candidates/trajectory/policy/controller records, and does not label P4.0 as Isaac-backed or full P4 completion.
- Tests added: None in this order.
- Tests passed: `python3 -m compileall amsrr -q` passed. P4.0 config/import smoke passed. `git diff --check` passed.
- Handoff notes: Archive `rollout_artifacts["note"]` states that P4.0 metrics are simplified backend indicators, not Isaac-backed physical success rates.
- Open questions: None currently.

#### 2026-07-08
- Scope: Order 2 simplified env external `DesignOutput` / assembled morphology injection.
- Files changed:
  - `amsrr/simulation/simplified_grasp_carry_env.py`
  - `tests/unit/simulation/test_simplified_grasp_carry_env.py`
- Upstream dependencies: P4.0 selected design / assembled morphology handoff requirement, existing `SimplifiedGraspCarryEnv`, P2 deterministic `P2DesignPolicy`, and P3 simplified assembly boundary.
- Implemented: Optional `design_output` and `assembled_morphology` injection on env construction/reset, internal `design_source` labeling, external-design build path that bypasses `FixedSimpleDesignPolicy`, and assembled morphology replacement while preserving design metadata.
- Not implemented: P4.0 runner, archive completeness checks, P4.0 acceptance, Isaac backend, controller bridge, or actuator mapping.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Agent K can build P4.0 episodes using P2 selected `DesignOutput` and P3 assembled `MorphologyGraph` before sampling contacts and planning trajectories.
- Tests added: `test_simplified_grasp_carry_env_accepts_external_design_output`.
- Tests passed: Simplified env tests passed: 4 passed. P1 runner tests passed: 3 passed. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: Existing P1 `FixedSimpleDesignPolicy` path remains the default for callers that do not provide an external design.
- Open questions: None currently.

#### 2026-07-08
- Scope: Order 1 archive compatibility for P4.0/P4 logging fields.
- Files changed:
  - `amsrr/logging/episode_archive.py`
  - `tests/unit/training/test_p1_runner.py`
- Upstream dependencies: v0.4 Section 25.1, P4.0 simplified archive requirements, existing P1/P2/P3 runner archive behavior.
- Implemented: Defaulted `runtime_observations`, `actuator_target_records`, `rollout_artifacts`, and `learning_artifacts` on `EpisodeArchive`; added a legacy dict restoration check for archives missing those fields.
- Not implemented: P4.0 runner, simplified env injection, P4.0 acceptance, Isaac actuator target conversion, or learned training artifacts.
- Schema/interface changes: Additive archive fields only; existing archives deserialize with defaults.
- Downstream impact: Later P4.0 runner can store trajectory/policy/controller/reward metrics immediately and can optionally include simplified runtime observations, while P4-control/Isaac work can fill actuator records.
- Tests added: Legacy `EpisodeArchive.from_dict` default restoration assertions in `test_p1_runner_collects_metrics_and_archives`.
- Tests passed: P1 runner tests passed: 3 passed. P2/P3 runner tests passed: 5 passed. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: Keep P4.0 no-mislabeling checks separate: these fields enable P4 logging but do not imply Isaac-backed rollout or P4 full completion.
- Open questions: None currently.

### P4.3 Design Revision: Learning Target Clarification

#### 2026-07-08
- Scope: Clarify P4.3 learning bootstrap targets in the source design spec only.
- Files changed:
  - `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: User追加修正 request, existing P4 Isaac-backed completion clarification, P2.5 learned π_D scorer / feasibility head notes, and v0.4 π_D / π_H / π_L ownership boundaries.
- Implemented: Three staged P4.3 learning target families, P4.3a-P4.3e recommended sequence, expanded P4 full acceptance learning artifacts for π_L/residual control, π_H, and π_D scorer fine-tuning, and updated Mermaid training-loop arrows to π_D / π_H / π_L.
- Not implemented: Any training code, checkpoints, policy heads, dataset builders, acceptance code, or Isaac rollout code.
- Schema/interface changes: Source spec only.
- Downstream impact: Future P4.3 work must not interpret learning bootstrap as π_L-only. Learned π_D scorer usage remains outcome-conditioned scoring/ranking only, π_H learning owns contact assignment / trajectory timing, and π_L learning owns PolicyCommand / residual intent.
- Tests added: None.
- Tests passed: Documentation verification only: required P4.3 terms were found in the revised source spec; `git diff --check` passed.
- Handoff notes: Learned models may enter production only through deterministic safety gates; hard feasibility remains owned by `FeasibilityChecker`.
- Open questions: None currently.

### P4 Design Revision: Isaac-Backed Full Completion Clarification

#### 2026-07-08
- Scope: Revise the source design spec only, clarifying P4 staging and preventing simplified full-pipeline acceptance from being treated as P4 completion.
- Files changed:
  - `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: User-provided P4 design revision instruction, v0.4 Sections 17, 20, 23, 24, 25, 26, 27, and current P1/P2/P3 handoff state.
- Implemented: P4 phase split, P4.0 simplified integration scope, P4-control low-level Isaac flight validation prerequisites, controller bridge / actuator mapping requirements, π_A docking/detach/separation bridge requirement, Isaac backend requirements, split P4 acceptance, P4 learning bootstrap requirements, P4 mermaid flow, EpisodeArchive P4 logging fields, Agent I/J/K/L P4 ownership notes, and revised implementation order.
- Not implemented: Any P4 code, Isaac Lab backend, controller bridge, actuator mapping, P4 runner, P4 acceptance gate, or learning run.
- Schema/interface changes: Source spec only. Future schema/code changes are implied for P4 archive logging, but no implementation module was changed.
- Downstream impact: Future P4 implementation must proceed through P4.0, P4-control/P4a, P4.1, P4.2, P4.3, and P4 full acceptance rather than claiming completion after simplified backend wiring.
- Tests added: None.
- Tests passed: Documentation checks only: required P4 terms were found in the revised source spec; `git diff --check` passed.
- Handoff notes: P2.5 learned models remain auxiliary and deterministic `P2DesignPolicy` / `FeasibilityChecker` fallback remains required at P4 start.
- Open questions: None currently.

### P2.5: Post-P2 Inspection, Visualization, and Candidate Trace Export

#### 2026-07-08
- Scope: Add a pre-P3 inspection/debugging phase without replacing the existing P2 completion gate.
- Files changed:
  - `amsrr/training/p2_inspection_context.py`
  - `amsrr/training/p2_candidate_trace_export.py`
  - `amsrr/visualization/__init__.py`
  - `amsrr/visualization/p2_morphology.py`
  - `amsrr/reporting/__init__.py`
  - `amsrr/reporting/p2_5_inspection_report.py`
  - `amsrr/acceptance/__init__.py`
  - `amsrr/acceptance/p2_5_inspection.py`
  - `tests/unit/visualization/test_p2_morphology_visualization.py`
  - `tests/unit/training/test_p2_candidate_trace_export.py`
  - `tests/unit/reporting/test_p2_5_inspection_report.py`
  - `tests/acceptance/test_p2_5_inspection.py`
  - `outputs/p2_5/`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: User-requested P2.5 phase, existing P2 design policy/variants, FeasibilityChecker labels/margins, P2 design config/distribution, and Section 24.3 completion.
- Implemented: SVG morphology graph/layout visualization for all four P2 variants, JSONL/CSV per-candidate trace export, explicit closed-loop rejected probe, markdown inspection report, and P2.5 acceptance gate.
- Not implemented: Isaac, π_H, π_L, QP/PID, actuator commands, learned training, P3 assembly integration.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Human reviewers can inspect `outputs/p2_5/report/p2_5_inspection_report.md`, SVG layouts, and candidate traces before P3. P2 completion remains unchanged.
- Tests added: `test_p2_morphology_visualization_outputs_graph_and_layout_svgs`, `test_p2_candidate_trace_export_writes_all_candidates_and_probe`, `test_p2_5_inspection_report_contains_summary_and_scope_notes`, `test_p2_5_inspection_acceptance_gate`.
- Tests passed: Targeted P2.5 tests passed individually. Full unit suite passed: 83 passed, 1 skipped. Full acceptance suite passed: 4 passed in 88.72s. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: The generated trace contains 5 records: four normal P2 policy variants plus one closed-loop invalid probe, with counts accepted=4, rejected=1, selected=1.
- Open questions: Human review of P2.5 visualization/report is still recommended before P3 starts.

### Agent E: P2 π_D Candidate Selection Scaffold

#### 2026-07-08
- Scope: Add deterministic P2 π_D scaffold that enumerates candidate morphology designs, labels them with FeasibilityChecker results, and deterministically selects a design by soft score.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/design_policy_p2.py`
  - `tests/unit/policies/test_p2_design_policy.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Agent E grasp/carry morphology variants, Agent F P2 FeasibilityChecker labels/margins, existing `DesignPolicyContext`, `DesignOutput`, and v0.4 π_D / P2 acceptance guidance.
- Implemented: `P2DesignPolicyConfig`, `P2DesignCandidateEvaluation`, `P2DesignSelection`, `P2DesignPolicy`, variant enumeration, candidate feasibility evaluation, accepted/rejected split, deterministic soft scoring, selected-design annotation, and package exports.
- Not implemented: Learned π_D neural scoring, policy-gradient training, replay/dataset generation, P2 runner/acceptance gate, or simulator execution.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P2 runners can inspect `selection.candidates`, `accepted_candidates`, `rejected_candidates`, and `selected_candidate`, while callers that only need a design can use `P2DesignPolicy.design(context)`.
- Tests added: `test_p2_design_policy_enumerates_variants_and_selects_best_accepted`, `test_p2_design_policy_splits_rejected_candidates_with_feasibility_checker`, `test_p2_design_policy_falls_back_to_best_rejected_when_none_accepted`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_p2_design_policy.py tests/unit/policies/test_design_teacher.py -q` passed: 6 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 77 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: The current hand-coded soft score intentionally prefers accepted candidates first, then balances slot/capability coverage, reachability, thrust/payload margins, optional support, variant prior, and complexity. It is a deterministic baseline for P2 before learned scoring.
- Open questions: None currently.

### Agent F: P2 Design-Level FeasibilityChecker

#### 2026-07-08
- Scope: Strengthen design-level feasibility outputs for P2 grasp/carry design evaluation and acceptance aggregation.
- Files changed:
  - `amsrr/feasibility/checker.py`
  - `tests/unit/feasibility/test_feasibility_checker.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: v0.4 hard-check list and P2 acceptance criteria, existing `FeasibilityResult`, IRG `CapabilityRequirement --applies_to--> ContactSlot` edges, Agent E P2 grasp/carry morphology variants, and PhysicalModel thrust data.
- Implemented: Checker version `p2_agent_f_design_v1`, stable `L_FEASIBLE` / `L_HARD_VIOLATION` / `L_<hard_check_code>` labels, coverage and capability ratios, CapabilityRequirement min-force checks, reachability ratios, port conflict counts, closed-loop rejection margins, detailed thrust/payload force margins, and metadata violation counts.
- Not implemented: Exact collision checking, exact QP hover solve, learned feasibility head, P2 runner/acceptance harness, or simulator validation.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Agent K/L P2 runners can store and aggregate feasibility labels directly from `EpisodeArchive.feasibility_result`. Agent E design policy work can use deterministic rejection labels for candidate evaluation.
- Tests added: `test_p2_feasibility_checker_records_acceptance_margins_for_variant`, `test_p2_feasibility_checker_uses_capability_requirement_force_label`, `test_p2_feasibility_checker_records_port_conflict_margins`, `test_p2_feasibility_checker_records_reachability_margins`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/feasibility/test_feasibility_checker.py -q` passed: 7 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 74 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: `proxy_scores["L_..."]` entries are deterministic labels encoded in the available float map, not learned proxy values. Hard safety remains owned by `hard_violations` and `feasible`.
- Open questions: None currently.

### Agent E: P2 Grasp-Carry Morphology Variant Builder

#### 2026-07-08
- Scope: Implement P2 order 1 real object grasp/carry morphology variants as distinct `MorphologyGraph` outputs, without changing schemas or downstream policy/controller interfaces.
- Files changed:
  - `amsrr/morphology/__init__.py`
  - `amsrr/morphology/grasp_carry_designs.py`
  - `amsrr/policies/design_teacher.py`
  - `tests/unit/morphology/test_grasp_carry_variants.py`
  - `tests/unit/policies/test_design_teacher.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: v0.4 MorphologyGraph/DesignOutput schemas, π_D teacher variant names, STOP validity constraints, IRG ContactSlots, Holon PhysicalModel dock ports, and existing FeasibilityChecker hard-check scaffold.
- Implemented: `GraspCarryMorphologyVariant`, `GraspCarryMorphologyVariantBuilder`, `build_grasp_carry_variant_design_output`, four deterministic connected-tree layouts, variant-specific module roles/poses/edges/control groups, required/optional RobotAnchor placement, design action traces, and design teacher routing for object grasp/carry variants.
- Not implemented: Learned π_D scorer/sampler, optimized morphology search, P2 design runner, P2 acceptance gate, exact collision/QP feasibility, or Isaac execution.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Future P2 design datasets and feasibility labeling can distinguish topology variants while preserving existing `DesignOutput` and `RobotAnchor` contracts. Existing P1 simplified flow continues to use `FixedSimpleDesignPolicy` through the same interface.
- Tests added: `test_grasp_carry_variants_build_distinct_feasible_morphologies`, `test_grasp_carry_variant_topology_shapes`, `test_grasp_carry_variants_cover_required_slot_min_count`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/morphology/test_grasp_carry_variants.py tests/unit/policies/test_design_teacher.py -q` passed: 6 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 70 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: `central_base_plus_two_grasp_arms` requires enough module budget for a five-module two-link-arm layout. The default grasp/carry teacher selection still chooses `tri_anchor_support_grasp` when an optional support slot exists and `max_modules >= 3`.
- Open questions: None currently.

### Agent L: Tests and Acceptance

#### 2026-07-08
- Scope: Final P3 verification and handoff after order 1-5 implementation commits.
- Files changed:
  - `for_codex/WORKLOG.md`
- Upstream dependencies: P3 Agent G runner/executor/retry work, Agent K P3 runner, Agent L P3 acceptance, and full repo tests.
- Implemented: Final worklog handoff entry with full verification commands and results.
- Not implemented: No new functionality in this handoff step.
- Schema/interface changes: None.
- Downstream impact: Future P4 work can treat P3 acceptance as passing in this checkout.
- Tests added: None.
- Tests passed: Full unit suite passed: 96 passed, 1 skipped. Full acceptance suite passed: 6 passed in 115.39s. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: P3 remains simplified deterministic assembly integration; P4 must still integrate contact candidates, π_H, π_L, QP/PID/controller execution, and full grasp/carry success criteria.
- Open questions: Whether to add Isaac-backed assembly validation before or during P4 remains a planning decision, not a blocker for the current simplified P3 gate.

#### 2026-07-08
- Scope: Add P3 order 5 acceptance gate for v0.4 Section 24.4.
- Files changed:
  - `amsrr/acceptance/p3_acceptance.py`
  - `amsrr/acceptance/__init__.py`
  - `tests/acceptance/test_p3_acceptance.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Agent K P3 runner, Agent G assembly runner/executor, P2 design distribution/policy, and `EpisodeArchive` JSONL roundtrip helpers.
- Implemented: `P3AcceptanceCriteria`, `P3AcceptanceReport`, `run_p3_acceptance`, assembly success-rate gate, construction-state consistency gate, explicit retry probe, explicit abort probe, archive roundtrip validation, and acceptance test.
- Not implemented: P4 full grasp/carry, Isaac execution, learned assembly, π_H/π_L/QP/PID execution, or actuator commands.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P3 deterministic assembly integration now has a reproducible pass/fail milestone gate.
- Tests added: `test_p3_acceptance_section_24_4`.
- Tests passed: P3 runner and P3 acceptance targeted tests passed: 3 passed. `python3 -m compileall amsrr -q` passed.
- Handoff notes: The acceptance gate intentionally treats retry/abort probes separately from normal success-rate episodes so deterministic success runs do not need random failures.
- Open questions: None currently.

#### 2026-07-08
- Scope: Mark P2 complete by wrapping the Section 24.3 design-level acceptance gate in an explicit milestone completion report.
- Files changed:
  - `amsrr/acceptance/__init__.py`
  - `amsrr/acceptance/p2_completion.py`
  - `tests/acceptance/test_p2_completion.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: `run_p2_acceptance`, Agent K P2 design runner, Agent E P2 design policy/variants, Agent F feasibility labels, and v0.4 Section 24.3.
- Implemented: `P2CompletionCriteria`, `P2CompletionReport`, `run_p2_completion`, explicit completion checks, and a 1000-episode P2 completion acceptance test.
- Not implemented: P3 assembly execution, P4 end-to-end grasp/carry success, Isaac Sim execution, learned π_D training, π_H/π_L/controller execution inside the P2 gate.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Future work can use `run_p2_completion(...).passed` as the local signal that the P2 design-level milestone is complete before advancing to P3/P4.
- Tests added: `test_p2_completion_milestone_section_24_3`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p2_completion.py -q` passed: 1 passed in 23.62s. Full unit suite passed: 80 passed, 1 skipped. Full acceptance suite passed: 3 passed in 88.20s. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: The completion report mirrors Section 24.3 exactly and deliberately does not claim actuator-command or simulator-task success.
- Open questions: None currently.

#### 2026-07-08
- Scope: Implement P2 Section 24.3 acceptance reporting and tests.
- Files changed:
  - `amsrr/acceptance/__init__.py`
  - `amsrr/acceptance/p2_acceptance.py`
  - `tests/acceptance/test_p2_acceptance.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: v0.4 P2 acceptance criteria, Agent K P2 design runner, Agent E P2 design policy and variants, Agent F FeasibilityChecker labels/margins, and `EpisodeArchive`.
- Implemented: `P2AcceptanceCriteria`, `P2AcceptanceReport`, `run_p2_acceptance`, Section 24.3 metric checks, synthetic closed-loop invalid probe, archive label validation, and a 1000-episode acceptance test.
- Not implemented: P2 completion wrapper in this entry, learned π_D training, Isaac validation, π_H/π_L/controller execution, or assembly integration.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P2 design-level validity and feasibility-label persistence can be checked mechanically before later phases.
- Tests added: `test_p2_acceptance_section_24_3`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p2_acceptance.py -q` passed: 1 passed in 23.45s. Full unit suite passed: 80 passed, 1 skipped. Full acceptance suite passed: 2 passed in 64.82s. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: Closed-loop rejection is tested with an explicit synthetic invalid design because the normal P2 candidate builders intentionally emit connected trees.
- Open questions: None currently.

#### 2026-07-08
- Scope: Implement P1 Section 24.2 acceptance reporting and tests.
- Files changed:
  - `amsrr/acceptance/__init__.py`
  - `amsrr/acceptance/p1_acceptance.py`
  - `tests/acceptance/test_p1_acceptance.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: v0.4 P1 acceptance criteria, `P1SimplifiedRunner`, P1 task distribution config, `EpisodeArchive`, `SimplifiedGraspCarryEnv`, ContactCandidateSampler, pi_H/pi_L/controller baselines.
- Implemented: `P1AcceptanceCriteria`, `P1AcceptanceReport`, `run_p1_acceptance`, and a 1000-episode acceptance test that checks success rate, zero crashes, non-empty contact candidates on randomized valid objects, and archive roundtrip counts.
- Not implemented: Isaac Lab backend validation, learned pi_L training, held-out object evaluation, and high-fidelity contact physics checks.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P1 completion can now be verified by running `tests/acceptance/test_p1_acceptance.py`; future simulator backends can reuse the same criteria.
- Tests added: `test_p1_acceptance_section_24_2`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p1_acceptance.py -q` passed: 1 passed in 41.26s. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 67 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: Acceptance currently targets the simplified backend explicitly. Keep Isaac Lab checks separate under Agent J / simulator integration.
- Open questions: None currently.

### Agent K: P1 Task Distribution, Runner, Metrics, and Logging

#### 2026-07-08
- Scope: Add P3 order 4 assembly evaluation runner, config, archive metrics, and tests.
- Files changed:
  - `amsrr/training/p3_assembly_runner.py`
  - `amsrr/training/__init__.py`
  - `configs/training/p3_assembly_grasp_carry.yaml`
  - `tests/unit/training/test_p3_assembly_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: P2 design distribution/policy, Agent G assembly runner/executor, PhysicalModel builder, IRGBuilder, InteractionEnvelopeExtractor, and `EpisodeArchive` JSONL helpers.
- Implemented: `P3_ASSEMBLY_RUNNER_VERSION`, `P3AssemblyRunnerConfig`, `P3AssemblyRunnerResult`, `load_p3_assembly_runner_config`, `P3AssemblyEvaluationRunner`, P3 config file, assembly archive metrics, and unit tests.
- Not implemented: P3 acceptance gate, retry/abort acceptance probes, Isaac execution, π_H/π_L/QP/PID execution, or actuator commands.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Agent L can implement P3 Section 24.4 acceptance over this runner.
- Tests added: `test_p3_assembly_runner_collects_successful_assembly_archives`, `test_p3_assembly_runner_config_loader`.
- Tests passed: P3 runner plus related P2/assembly targeted tests passed: 17 passed. `python3 -m compileall amsrr -q` passed.
- Handoff notes: The runner stores the source `AssemblyPlan` as JSON-compatible data in `EpisodeArchive.assembly_plan`; full `AssemblyRunReport` records remain runtime evaluation objects.
- Open questions: None currently.

#### 2026-07-08
- Scope: Implement P2 design evaluation distribution, runner, metrics, and EpisodeArchive feasibility-label logging.
- Files changed:
  - `amsrr/training/__init__.py`
  - `amsrr/training/p2_design_distribution.py`
  - `amsrr/training/p2_design_runner.py`
  - `configs/training/p2_design_grasp_carry.yaml`
  - `tests/unit/training/test_p2_design_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: v0.4 P2 acceptance criteria and logging guidance, Agent E P2 design policy/variants, Agent F FeasibilityChecker labels/margins, IRGBuilder, InteractionEnvelopeExtractor, PhysicalModel, and EpisodeArchive.
- Implemented: P2 grasp/carry design distribution, config loader, design evaluation runner, selected-design archive writing, feasibility label/margin metric extraction, P2 config file, package exports, and unit tests.
- Not implemented: P2 completion wrapper in this entry, per-candidate archive rows, learned training loops, Isaac recorder, π_H/π_L/controller execution, or actuator-command logging.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Agent L P2 acceptance/completion can aggregate design-level validity and labels directly from archived `EpisodeArchive.feasibility_result` values.
- Tests added: `test_p2_design_distribution_randomizes_and_marks_metadata`, `test_p2_design_runner_collects_feasibility_archives`, `test_p2_design_runner_config_loader`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p2_design_runner.py -q` passed: 3 passed. Full unit suite passed: 80 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: The runner archives the selected candidate's feasibility result; rejected-candidate supervision is available through `P2DesignSelection` but not yet emitted as separate dataset rows.
- Open questions: None currently.

#### 2026-07-08
- Scope: Implement P1 order 8 task distribution, runner, metrics, and EpisodeArchive logging for the simplified grasp/carry backend.
- Files changed:
  - `amsrr/logging/__init__.py`
  - `amsrr/logging/episode_archive.py`
  - `amsrr/training/__init__.py`
  - `amsrr/training/p1_task_distribution.py`
  - `amsrr/training/p1_runner.py`
  - `configs/training/p1_grasp_carry_distribution.yaml`
  - `tests/unit/training/test_p1_runner.py`
- Upstream dependencies: v0.4 domain randomization and EpisodeArchive guidance, existing TaskSpec and policy/controller schemas, `SimplifiedGraspCarryEnv`, and config/hash utilities.
- Implemented: Config-loaded P1 grasp/carry distribution, object size/mass/friction/initial-pose/target-pose sampling, per-episode runner over the simplified env, batch success/crash/failure metrics, EpisodeArchive dataclass, reproducibility metadata, JSONL write/read helpers, and unit tests.
- Not implemented: Learned training loop, replay buffer, dataset sharding, Isaac recorder, wind/sensor/thrust-scale randomization, non-box object shape sampling, or large-scale filesystem dataset management.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Simplified P1 runs now produce archives and metrics suitable for debugging and later dataset/training integration.
- Tests added: `test_p1_distribution_randomizes_configured_fields`, `test_p1_runner_collects_metrics_and_archives`, `test_p1_runner_config_loader`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p1_runner.py -q` passed: 3 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 67 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: `EpisodeArchive` includes a `reproducibility` map as an implementation supplement for Section 25.3. Config currently lives at `configs/training/p1_grasp_carry_distribution.yaml`.
- Open questions: None currently.

### Agent JP1: Simplified Grasp-Carry Simulation Env

#### 2026-07-08
- Scope: Implement P1 order 7 simplified grasp/carry simulation environment for interface-backed crash-free validation before Isaac Lab binding.
- Files changed:
  - `amsrr/simulation/__init__.py`
  - `amsrr/simulation/base.py`
  - `amsrr/simulation/simplified_grasp_carry_env.py`
  - `tests/unit/simulation/test_simplified_grasp_carry_env.py`
- Upstream dependencies: v0.4 simplified contact and P1 acceptance requirements, existing TaskSpec/RuntimeObservation schemas, IRGBuilder, InteractionEnvelopeExtractor, fixed/simple design policy, ContactCandidateSampler, GraspCarryBaselinePlanner, BaselineLowLevelPolicy, and QPIDController.
- Implemented: `SimulationEnvBase`, simplified reset/step/get-runtime-observation boundary, deterministic pipeline build artifacts, kinematic/fixed-joint grasp attach approximation, active object-target tracking, contact-state emission, task-progress metrics, per-episode result summaries, batch 1000-episode runner, and unit tests.
- Not implemented: Isaac Lab/Isaac Sim integration, physics contact solver, friction/slip dynamics, collision geometry stepping, stochastic actuator faults, dataset archive writer, or training loop integration.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P1 can validate no schema/checker/controller crashes over 1000 simplified episodes. Later simulator backends can implement `SimulationEnvBase` while preserving existing policy/controller contracts.
- Tests added: `test_simplified_grasp_carry_env_matches_base_protocol`, `test_simplified_grasp_carry_env_runs_policy_controller_episode`, `test_simplified_grasp_carry_1000_episodes_crash_free`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_simplified_grasp_carry_env.py -q` passed: 3 passed, including 1000 simplified episodes with 0 crashes. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 64 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: The env deliberately keeps simulator-specific dependencies out of `amsrr/simulation`; the current backend is deterministic and suitable for interface smoke and acceptance checks, not physics validation.
- Open questions: None currently.

### Agent I: pi_L + QP/PID Interfaces

#### 2026-07-08
- Scope: Implement P1 order 6 Agent I interfaces that map active pi_H knots/runtime observations to `PolicyCommand`, then to controller-owned `ControllerCommand` outputs.
- Files changed:
  - `amsrr/controllers/__init__.py`
  - `amsrr/controllers/controller_base.py`
  - `amsrr/controllers/qp_allocator_interface.py`
  - `amsrr/controllers/qpid_controller.py`
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/low_level_policy_base.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `tests/unit/policies/test_low_level_baseline.py`
- Upstream dependencies: v0.4 Section 20 pi_L/controller split, Agent H `ContactWrenchTrajectory`, existing runtime/physical-model/policy schemas, and `PolicyCommandBiasBuilder`.
- Implemented: `LowLevelPolicyContext`, `LowLevelPolicyBase`, `BaselineLowLevelPolicyConfig`, `BaselineLowLevelPolicy`, runtime-time active knot selection, object target residual wrench proxy, active contact tracking bias, controller-status residual suppression, `ControllerContext`, `ControllerBase`, QP allocator problem/result/backend protocol, bounded vertical rotor allocator, QPID controller scaffold, vectoring joint clipping, PD joint torque proxy, dock-mechanism hold commands, and focused tests.
- Not implemented: Learned pi_L head, exact multi-axis/vectoring/contact QP, OSQP/C++ backend, high-fidelity object/contact dynamics, simulator execution, or training integration.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P1 simplified grasp-carry simulation can run through pi_H trajectory, pi_L intent, desired-reference builder, and controller command scaffolding without introducing a simulator dependency yet.
- Tests added: `test_baseline_low_level_policy_outputs_policy_command`, `test_baseline_low_level_policy_selects_knot_from_runtime_time`, `test_baseline_low_level_policy_suppresses_residual_when_controller_infeasible`, `test_select_active_knot_rejects_empty_trajectory`, `test_bounded_vertical_rotor_allocator_feasible_and_unsupported_residual`, `test_bounded_vertical_rotor_allocator_reports_infeasible_clip`, `test_qpid_controller_outputs_controller_command`, `test_qpid_controller_reports_infeasible_vertical_wrench`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_low_level_baseline.py -q` passed: 4 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_qpid_controller.py -q` passed: 4 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 61 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: The pi_L baseline intentionally emits residual intent only. `ControllerCommand` fields are produced only in the controller layer. The current allocator is a simplified bounded vertical allocator and reports unsupported wrench residuals for future exact QP replacement.
- Open questions: None currently.

### Agent H: pi_H Baseline Planner

#### 2026-07-08
- Scope: Implement a deterministic baseline pi_H planner for P1 grasp/carry after ContactCandidateSampler and selected-assignment feasibility.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/high_level_policy_base.py`
  - `amsrr/policies/contact_wrench_trajectory.py`
  - `tests/unit/policies/test_high_level_baseline.py`
- Upstream dependencies: `ContactCandidateSet` group proposals, selected-assignment feasibility, IRG state targets/contact slots, InteractionEnvelope, MorphologyGraph, and existing policy schemas.
- Implemented: `HighLevelPolicyContext`, `HighLevelPolicyBase`, `BaselineTrajectoryPlannerConfig`, `GraspCarryBaselinePlanner`, `select_feasible_assignments`, five-knot deterministic grasp/carry trajectory generation, object goal extraction, free-anchor pose targets, wrench target scaffolding, and feasibility cache integration.
- Not implemented: Learned pi_H heads, trajectory optimization, multi-knot re-planning from live observations, exact wrench/QP optimization, contact schedule search beyond group-proposal attempts, or simulator execution.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: pi_L and controller interfaces can now consume active `InteractionKnot`s and `ContactAssignment`s from a full `ContactWrenchTrajectory`.
- Tests added: `test_grasp_carry_baseline_planner_outputs_contact_wrench_trajectory`, `test_select_feasible_assignments_uses_grasp_pair_group`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 53 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: Planner selection uses `evaluate_selected_assignment_feasibility`, so infeasible attempted groups are cached on the candidate set. The returned trajectory never contains final actuator commands.
- Open questions: None currently.

### Agent H/F: Selected Assignment Feasibility Proxy

#### 2026-07-08
- Scope: Add assignment-level feasibility checks for selected `ContactAssignment` sets only, without subset enumeration.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/assignment_feasibility.py`
  - `tests/unit/policies/test_contact_candidate_interfaces.py`
- Upstream dependencies: Existing `ContactCandidateSet`, selected `ContactAssignment`, pairwise conflict matrix, candidate unary validity, and v0.4 assignment-level feasibility guidance.
- Implemented: `evaluate_selected_assignment_feasibility`, violation code constants, slot min/max cardinality checks, assignment/candidate consistency checks, selected pairwise conflict checks, duplicate selected-candidate checks, grasp-opposition residual proxy, friction margin proxy, optional explicit wrench/QP/collision residual hooks, and deterministic cache update.
- Not implemented: Exact force closure, exact support polygon/contact support ratio, full multi-contact collision, exact QP allocation, or π_H trajectory generation.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: π_H can now select candidate assignments and receive deterministic feasibility/cache labels before later exact solver integration.
- Tests added: `test_selected_assignment_feasibility_accepts_opposing_grasp_pair`, `test_selected_assignment_feasibility_rejects_cardinality_and_pair_conflict`, `test_selected_assignment_feasibility_rejects_non_opposing_grasp_normals`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 51 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: The existing `evaluate_assignment_level_qp` smoke helper remains backward compatible. The new evaluator should be used when π_H has a selected assignment set and wants cardinality/pairwise/wrench-proxy labels in addition to optional QP residual labels.
- Open questions: None currently.

### Agent H: ContactCandidateSampler

#### 2026-07-08
- Scope: Implement P1 morphology-conditioned contact candidate sampling and group-proposal scaffolding only.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/contact_candidate_set.py`
  - `amsrr/policies/contact_candidate_sampler.py`
  - `tests/unit/policies/test_contact_candidate_sampler.py`
- Upstream dependencies: `TaskSpec`, IRG ContactSlots, `InteractionEnvelope`, `MorphologyGraph` RobotAnchors, `GeometryDescriptor` / ContactRegionGraph, and existing `ContactCandidateSet` helper functions.
- Implemented: `ContactCandidateSamplerConfig`, `ContactCandidateSampler`, deterministic candidate IDs, entity-pose world transform, unary smoke scores, compatible-anchor filtering, `build_group_proposals`, `grasp_pair` group proposals, `support_set` fallback proposals, optional group-proposal support in `build_contact_candidate_set`, and package exports.
- Not implemented: Learned candidate encoder/scorer, task-specific advanced sampling quotas, exact reachability, exact local collision/clearance, assignment-level wrench/friction/QP feasibility, π_H selection, or simulator/runtime contact verification.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: π_H baseline can now select over finite candidate pools with slot coverage and group hints. Assignment-level evaluators can later populate `assignment_feasibility_cache`.
- Tests added: `test_contact_candidate_sampler_returns_non_empty_grasp_carry_candidates`, `test_contact_candidate_sampler_builds_grasp_pair_group_proposals`, `test_contact_candidate_sampler_uses_robot_anchor_associations`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 48 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: Candidate generation preserves `ContactSlotID -> RobotAnchorID -> ContactCandidateID`; candidates are generated only for anchors already associated with the slot by π_D. Group proposals deliberately do not imply full task feasibility.
- Open questions: None currently.

### Agent G: π_A GraphEditAssemblyPlanner

#### 2026-07-08
- Scope: Add P3 order 3 retry/abort behavior to the deterministic assembly runner.
- Files changed:
  - `amsrr/assembly/assembly_runner.py`
  - `amsrr/assembly/simplified_executor.py`
  - `tests/unit/assembly/test_assembly_runner.py`
  - `tests/unit/assembly/test_simplified_executor.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Existing Agent G assembly runner/executor and v0.4 `AssemblyStep.step_type` values.
- Implemented: Configurable retry limit, synthetic retry steps, synthetic abort steps, retry/abort counts, aborted status, executed step-type tracing, and fail-once support in the simplified executor.
- Not implemented: Motion replanning, learned recovery policy, detach release gates, physical docking verification, controller/QP integration, or Isaac execution.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P3 acceptance can test successful transient retry and persistent-failure abort paths without changing source assembly plans.
- Tests added: `test_assembly_runner_can_disable_retry_for_single_failure_stop`, `test_simplified_executor_fail_once_allows_runner_retry_success`.
- Tests passed: Agent G targeted assembly tests passed: 12 passed. `python3 -m compileall amsrr -q` passed.
- Handoff notes: Runtime retry/abort steps are represented in `AssemblyRunReport.executed_step_types`; `AssemblyPlan.steps` remains the source graph-edit plan.
- Open questions: None currently.

#### 2026-07-08
- Scope: Add P3 order 2 simplified assembly executor backend.
- Files changed:
  - `amsrr/assembly/simplified_executor.py`
  - `amsrr/assembly/__init__.py`
  - `tests/unit/assembly/test_simplified_executor.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Agent G runner/core state transitions, `AssemblyExecutorInterface`, and existing construction-state helpers.
- Implemented: `SimplifiedAssemblyExecutorConfig`, `SimplifiedAssemblyExecutor`, default successful step execution, `verify_attach` updated-state return when target graph is provided, per-step smoke metrics, and deterministic failure injection by step id/type.
- Not implemented: Retry/abort state-machine execution, P3 runner/acceptance, Isaac execution, physical docking dynamics, or controller/QP integration.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P3 runner and acceptance can use this executor for deterministic success probes and controlled failure probes.
- Tests added: `test_simplified_executor_runs_full_assembly_and_returns_updated_state`, `test_simplified_executor_can_inject_step_type_failure`, `test_simplified_executor_success_without_target_graph_uses_runner_state_transition`.
- Tests passed: Agent G targeted assembly tests passed: 10 passed. `python3 -m compileall amsrr -q` passed.
- Handoff notes: Failure injection is executor-local; policy-level retry/abort handling remains the next order.
- Open questions: None currently.

#### 2026-07-08
- Scope: Add P3 order 1 assembly execution core on top of the existing graph-edit planner.
- Files changed:
  - `amsrr/assembly/assembly_runner.py`
  - `amsrr/assembly/__init__.py`
  - `tests/unit/assembly/test_assembly_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Existing Agent G planner/state dataclasses, `AssemblyExecutorInterface`, v0.4 Section 17 contracts, and P2 morphology variants.
- Implemented: `AssemblyRunnerConfig`, `AssemblyRunReport`, `AssemblyRunner`, automatic successful `verify_attach` state transition, final target graph consistency metrics, success/failure report serialization, and focused unit tests.
- Not implemented: Simplified executor, retry/abort policy execution, P3 runner/acceptance, Isaac execution, or physical docking verification.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Later P3 work can plug in deterministic simplified executors and aggregate assembly success/state-consistency metrics.
- Tests added: `test_assembly_runner_completes_plan_and_updates_construction_state`, `test_assembly_runner_stops_on_failed_step_without_completing_graph`, `test_assembly_runner_resumes_from_partial_construction_state`.
- Tests passed: Agent G targeted assembly tests passed: 7 passed. `python3 -m compileall amsrr -q` passed.
- Handoff notes: `state_matches_target` checks assembled module IDs, dock edge endpoint/port keys, and occupied target ports; target edge latch states remain target metadata rather than the equality criterion.
- Open questions: None currently.

#### 2026-07-08
- Scope: Implement implementation-order item 12: deterministic π_A assembly planner and construction/execution interface scaffolding.
- Files changed:
  - `amsrr/assembly/__init__.py`
  - `amsrr/assembly/construction_state.py`
  - `amsrr/assembly/graph_edit_planner.py`
  - `amsrr/assembly/control_handoff.py`
  - `amsrr/assembly/executor_interface.py`
  - `tests/unit/assembly/test_graph_edit_planner.py`
- Upstream dependencies: Agent E target `MorphologyGraph` / `DesignOutput`, v0.4 assembly contracts, `Violation`, and existing schema serialization helpers.
- Implemented: `AssemblyStep`, `AssemblyPlan`, `ConstructionState`, `initial_construction_state`, `construction_state_from_current_graph`, `mark_edge_attached`, `GraphEditAssemblyPlanner`, `AssemblyPlannerConfig`, `ControlHandoffManager`, `ControlHandoffRequest`, `AssemblyExecutionResult`, and `AssemblyExecutorInterface`.
- Not implemented: Learned assembly policy, simulator executor, path/motion planner, retry/abort state-machine execution, detach execution gates, QP/PID controller integration, or physical docking verification.
- Schema/interface changes: None to existing persisted schemas. Added assembly-local dataclasses/interfaces matching v0.4.
- Downstream impact: Later P1/P3 code can request deterministic assembly plans for target morphologies and can hand assembly steps to simulator/controller interfaces once those exist.
- Tests added: `test_initial_construction_state_contains_base_only`, `test_graph_edit_planner_builds_deterministic_attach_sequence`, `test_graph_edit_planner_resumes_from_construction_state`, `test_control_handoff_request_for_docking_step`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 45 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: The planner expands each target dock edge into four deterministic steps and returns no `next_step` when the target graph already has no remaining dock edges. Construction subgraphs keep only assembled modules/edges, while unattached modules remain in `ConstructionState.unattached_modules` and singleton components.
- Open questions: None currently.

### Agent E: Deterministic Design Teacher + π_D Scaffolding

#### 2026-07-07
- Scope: Implement implementation-order item 11 for P1 fixed/simple morphology: deterministic design teacher and π_D scaffolding only.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/design_policy_base.py`
  - `amsrr/policies/design_candidate_generator.py`
  - `amsrr/policies/design_teacher.py`
  - `tests/unit/policies/test_design_teacher.py`
- Upstream dependencies: Existing `MinimalMorphologyBuilder`, `DesignOutput`, IRG ContactSlot semantics, InteractionEnvelopeExtractor, PhysicalModel builder, FeasibilityChecker, and v0.4 π_D action vocabulary.
- Implemented: `DesignPolicyContext`, `DesignPolicyBase` protocol, `FixedSimpleDesignPolicy`, `DesignTeacherVariant`, `DeterministicDesignTeacher`, `DesignTeacherExample`, `DesignCandidateGenerator`, `DesignActionCandidate`, `DesignCandidateStep`, P1 grasp/support teacher variant selection, and STOP-mask smoke checks.
- Not implemented: Learned π_D scoring/sampling, optimized teacher geometry variants, policy training, assembly planning, contact candidate sampling, π_H, π_L, QP/PID controller behavior, simulator integration.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P1 and Agent H can now request a stable fixed/simple `DesignOutput` with RobotAnchors before generating contact candidates. Agent G can plan assembly against the same target graph.
- Tests added: `test_design_teacher_selects_p1_grasp_support_variant`, `test_design_candidate_trace_masks_stop_until_final_step`, `test_fixed_simple_design_policy_outputs_feasible_stop`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 41 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: Teacher trace STOP is masked until the final teacher step. Final STOP validity performs scaffold checks and optionally respects a FeasibilityChecker result. π_D still emits `DesignOutput` only and never controller or actuator commands.
- Open questions: None currently.

### Agent H/I: P0 Interface-Only Smoke Pieces

#### 2026-07-07
- Scope: Add the remaining P0 Section 27.2 smoke pieces for contact candidates, selected assignment feasibility, and policy command bias references.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/contact_candidate_set.py`
  - `amsrr/policies/assignment_feasibility.py`
  - `amsrr/controllers/__init__.py`
  - `amsrr/controllers/policy_command_builder.py`
  - `tests/unit/policies/test_contact_candidate_interfaces.py`
  - `tests/unit/controllers/test_policy_command_builder.py`
- Upstream dependencies: Existing contact candidate and policy schemas, v0.4 candidate/π_H/π_L/controller interface contracts.
- Implemented: `build_pairwise_conflict_matrix`, `build_pairwise_compatibility_score`, `build_contact_candidate_set`, deterministic `assignment_key_from_assignments`, `evaluate_assignment_level_qp`, and `PolicyCommandBiasBuilder`.
- Not implemented: Full morphology-conditioned candidate sampling, learned candidate encoder/scorer, π_H baseline planner, π_L baseline policy, actual QP allocation, PID/controller actuator outputs.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Future Agent H/I implementations can replace helpers with richer implementations while keeping tested schema boundaries and no direct actuator output from π_L.
- Tests added: `test_contact_candidate_pairwise_conflict_matrix`, `test_assignment_level_qp_infeasible_case`, `test_policy_command_bias_builder`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 38 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: `ASSIGNMENT_QP_INFEASIBLE_CODE` is `E_ASSIGNMENT_QP_INFEASIBLE`, matching v0.4 Appendix C. `PolicyCommandBiasBuilder` merges π_H priority weights with π_L command weights, with PolicyCommand taking precedence.
- Open questions: None currently.

### Agent E/F: Minimal MorphologyGraph + Feasibility Hard-Check Scaffolding

#### 2026-07-07
- Scope: Build a connected minimal MorphologyGraph/DesignOutput from TaskSpec + IRG + PhysicalModel, and evaluate design-level hard feasibility checks.
- Files changed:
  - `amsrr/morphology/__init__.py`
  - `amsrr/morphology/graph.py`
  - `amsrr/feasibility/__init__.py`
  - `amsrr/feasibility/checker.py`
  - `amsrr/feasibility/violation_codes.py`
  - `tests/unit/morphology/test_minimal_morphology_builder.py`
  - `tests/unit/feasibility/test_feasibility_checker.py`
- Upstream dependencies: Agent B PhysicalModel and ModuleCapabilityToken, Agent D IRG ContactSlots, v0.4 MorphologyGraph/DesignOutput/FeasibilityResult schemas.
- Implemented: Minimal module chain generation, dock port replication/compatibility masking, structural dock edges, robot anchor creation from ContactSlots, slot-anchor binding priors, design action trace, violation code constants, and design-level hard checks for required P0 validity conditions.
- Not implemented: Learned π_D, candidate enumeration policy head, deterministic design teacher variants beyond the minimal seed, exact collision checking, exact QP hover feasibility, assignment-level feasibility, simulator integration.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: ContactCandidateSampler can start from known RobotAnchors and slot-anchor priors; later FeasibilityChecker work can refine coarse checks without changing result schema.
- Tests added: Morphology builder and feasibility checker tests listed in the global entry.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 35 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: The checker version is `p0_agent_ef_v1`. The minimal builder creates optional slot anchors too when a ContactSlot allows them, but required coverage is checked against required slots and their `min_count_group`.
- Open questions: None currently.

### Agent A/L: SharedInteractionWorkspace Tensor/Mask Contract

#### 2026-07-07
- Scope: Define and validate the internal NN tensor contract that fuses per-modality token groups with masks, source ids, group slices, and learned-query specs.
- Files changed:
  - `amsrr/schemas/workspace.py`
  - `amsrr/encoders/__init__.py`
  - `amsrr/encoders/workspace_builder.py`
  - `tests/unit/schemas/test_workspace.py`
  - `tests/unit/encoders/test_workspace_builder.py`
- Upstream dependencies: Agent A workspace schema foundation, Agent D/A InteractionEnvelopeEncoder output, v0.4 SharedInteractionWorkspace and LearnedQuerySpec contract.
- Implemented: `WorkspaceTokenGroup`, stricter `SharedInteractionWorkspace` group mask validation, optional contact candidate group support, recommended query specs, empty group factory, encoder-output-to-group adapter, and shared workspace assembly.
- Not implemented: Learned query tensors/parameters, attention pooling, fusion encoder, policy heads, modality-specific encoders beyond the existing InteractionEnvelopeEncoder.
- Schema/interface changes: Internal workspace schema validation was strengthened. `group_masks` are now required for every group slice and must match the corresponding global mask slice.
- Downstream impact: Heads can rely on `source_ids` and `group_slices` to map outputs back to source schema ids. π_H contexts can opt into the optional `contact_candidates` group.
- Tests added: Workspace group/mask/query tests and workspace builder tests listed in the global entry.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 30 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: `SharedInteractionWorkspaceBuilder` fills missing required groups with zero-width empty groups, so partial modality implementations can still produce a valid full workspace.
- Open questions: None currently.

### Agent D/A: InteractionEnvelopeExtractor + InteractionEnvelopeEncoder

#### 2026-07-07
- Scope: Aggregate compact interaction requirements from IRG and expose deterministic encoder tokens for the envelope modality.
- Files changed:
  - `amsrr/irg/__init__.py`
  - `amsrr/irg/envelope_extractor.py`
  - `amsrr/encoders/__init__.py`
  - `amsrr/encoders/interaction_envelope_encoder.py`
  - `tests/unit/irg/test_envelope_extractor.py`
  - `tests/unit/encoders/test_interaction_envelope_encoder.py`
- Upstream dependencies: Agent A `InteractionEnvelope` and `SharedInteractionWorkspace` shape helper schemas; Agent D IRG node/edge conventions; v0.4 envelope and encoder contracts.
- Implemented: Contact count range aggregation, contact mode aggregation, target region set extraction, wrench summary extraction, support/vertical thrust ratio summary hooks, precision/duration/capability extraction, branch option extraction for future fallback/mutually-exclusive IRGs, padded envelope token contract with masks and source ids.
- Not implemented: Full multimodal SharedInteractionWorkspace assembly, learned MLP/Transformer modules, query pooling parameters, generic constraint-threshold schema beyond fields currently available in `InteractionEnvelope`.
- Schema/interface changes: No persisted schema changes. Added internal encoder output dataclass for the interaction-envelope modality.
- Downstream impact: Future policy scaffolding can use envelope token groups without raw dict reinterpretation. Full workspace assembly remains the next implementation-order step.
- Tests added: `test_interaction_envelope_extract`, `test_interaction_envelope_extracts_all_task_families`, `test_interaction_envelope_encoder_contract`, `test_interaction_envelope_encoder_batch_padding`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 23 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: `InteractionEnvelopeEncoder` defaults to `backend_type="mlp_embedding"` when no dedicated backend key is provided. It emits deterministic scalar features; learned weights belong to later model code.
- Open questions: None currently.

### Agent D: IRGBuilder + InteractionTemplates

#### 2026-07-07
- Scope: Compile TaskSpec plus GeometryDescriptor-derived contact regions into a single typed InteractionRequirementGraph for all P0 task families.
- Files changed:
  - `amsrr/irg/__init__.py`
  - `amsrr/irg/irg_builder.py`
  - `amsrr/irg/validator.py`
  - `amsrr/irg/templates/__init__.py`
  - `amsrr/irg/templates/base.py`
  - `amsrr/irg/templates/free_flight.py`
  - `amsrr/irg/templates/object_grasp_carry.py`
  - `amsrr/irg/templates/valve_operation.py`
  - `amsrr/irg/templates/perching_manipulation.py`
  - `amsrr/irg/templates/contact_mediated_locomotion.py`
  - `tests/unit/irg/test_irg_builder.py`
- Upstream dependencies: Agent A schema dataclasses and enum validation, Agent C GeometryProcessor contact regions, v0.4 IRG and template contracts.
- Implemented: Deterministic node IDs and edge ordering, task/phase/contact-region/contact-slot/wrench/state/constraint/capability node generation, typed cross edges, structural validation, phase-label mapping, and smoke-valid IRGs for all five P0 task families.
- Not implemented: InteractionEnvelope extraction, task-aware geometry re-extraction beyond current GeometryProcessor descriptors, exact valve rim/handle segmentation, final contact/candidate selection, robot anchor assignment, morphology generation, trajectory generation, actuator commands.
- Schema/interface changes: None.
- Downstream impact: Envelope extraction can derive contact count ranges, modes, region sets, wrench requirements, state targets, constraints, and capability requirements directly from the IRG.
- Tests added: `test_phase_label_to_phase_type_mapping`, `test_irg_builder_grasp_carry_valid`, `test_irg_builder_all_task_families_smoke`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 19 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: `IRGBuilder.build_with_scene_graph()` returns both IRG and normalized SceneGraph for debugging. Non-template-required environment/obstacle descriptors are resolved lazily, so the v0.4 grasp/carry example can build even though it references `floor_geom` without declaring that geometry.
- Open questions: None currently.

### Agent C: GeometryProcessor

#### 2026-07-07
- Scope: Convert `GeometrySpec` references into `GeometryDescriptor`, learning-side patch/region tokens, and hashed exact/collision geometry refs for P0 primitives and mesh smoke.
- Files changed:
  - `amsrr/geometry/__init__.py`
  - `amsrr/geometry/asset_resolver.py`
  - `amsrr/geometry/surface_patch_graph.py`
  - `amsrr/geometry/contact_region_extractor.py`
  - `amsrr/geometry/geometry_processor.py`
  - `tests/unit/geometry/test_geometry_processor.py`
- Upstream dependencies: Agent A schema dataclasses, v0.4 GeometryProcessor contract, existing `module_urdf/mesh/battery_1.STL` smoke asset.
- Implemented: Primitive analytic decomposition for box, sphere, cylinder, and capsule; box face region coverage; STL binary/ascii and OBJ smoke mesh summary; normal-cluster mesh patch aggregation; path-free descriptor refs; deterministic surface/contact graph edge construction.
- Not implemented: Full mesh repair, curvature estimation, rim extraction, convex decomposition, SDF surface sampling, point cloud reconstruction, task-template-specific rim/edge extraction.
- Schema/interface changes: None.
- Downstream impact: IRGBuilder can request object surface contact regions for primitives and receive non-empty mesh patch clusters for mesh objects.
- Tests added: `test_geometry_processor_box_regions`, `test_geometry_processor_mesh_smoke`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 16 passed, 1 skipped.
- Handoff notes: `GeometryDescriptor.collision_ref` and `exact_geometry_ref` use hash URIs rather than raw filesystem paths. Asset paths remain resolver inputs only.
- Open questions: None currently.

### Agent B: URDF / PhysicalModel

#### 2026-07-07
- Scope: Parse Holon URDF/xacro XML, load thrust limits, build `PhysicalModel`, derive dock ports and rotor models, and report module capability features.
- Files changed:
  - `assets/robots/holon/holon.urdf`
  - `amsrr/robot_model/__init__.py`
  - `amsrr/robot_model/urdf_loader.py`
  - `amsrr/robot_model/thrust_model.py`
  - `amsrr/robot_model/physical_model_builder.py`
  - `tests/unit/robot_model/test_urdf_loader.py`
  - `tests/unit/robot_model/test_thrust_model.py`
  - `tests/unit/robot_model/test_physical_model_builder.py`
- Upstream dependencies: Agent A schemas, `configs/robot/robot_model.yaml`, `configs/robot/thrust_model.yaml`, `module_urdf/holon.urdf.xacro`, `module_urdf/README_for_codex.md`.
- Implemented: XML loader for URDF/xacro-derived files, link/joint/inertial/mesh extraction, frame-tree validation, rotor and dock candidate reporting, thrust model validation, runtime `PhysicalModel` builder, dock port derivation from connect point joints, rotor vectoring joint association, capability token derivation.
- Not implemented: Full xacro macro expansion, transform-accurate aggregate inertia, non-mesh collision primitive reconstruction, external metadata config for dock ports beyond name-pattern derivation.
- Schema/interface changes: None.
- Downstream impact: Feasibility and controller work can consume exact link/joint/rotor/dock schema objects. Design/policy work can consume `ModuleCapabilityToken`.
- Tests added: Agent B tests listed above.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 14 passed, 1 skipped.
- Handoff notes: The normalized runtime URDF is at `assets/robots/holon/holon.urdf`, matching `configs/robot/robot_model.yaml`. The original developer reference xacro remains under `module_urdf/`.
- Open questions: None currently.

### Repository Organization: Codex Handoff Docs

#### 2026-07-07
- Scope: Move Codex-facing specification, instructions, design modification log, and worklog under `for_codex/`.
- Files changed:
  - `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `for_codex/AGENTS.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: User request.
- Implemented: Documentation relocation and git staging/commit preparation.
- Not implemented: No source code changes in this worklog entry.
- Schema/interface changes: None.
- Downstream impact: Future handoff readers should look under `for_codex/`.
- Tests added: None.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 7 passed.
- Handoff notes: Keep design-spec deviations in `for_codex/AMSRR_design_modification_by_codex.md`, separate from chronological worklog entries.
- Open questions: None.

### Agent A: Schemas and Validation

#### 2026-07-07
- Scope: Implement schema dataclasses, enums, serialization/deserialization, and validation helpers for P0 foundation.
- Files changed:
  - `amsrr/schemas/common.py`
  - `amsrr/schemas/task_spec.py`
  - `amsrr/schemas/geometry.py`
  - `amsrr/schemas/irg.py`
  - `amsrr/schemas/interaction_envelope.py`
  - `amsrr/schemas/morphology.py`
  - `amsrr/schemas/physical_model.py`
  - `amsrr/schemas/runtime.py`
  - `amsrr/schemas/policies.py`
  - `amsrr/schemas/feasibility.py`
  - `amsrr/schemas/workspace.py`
  - `amsrr/schemas/contact_candidates.py`
- Upstream dependencies: v0.4 schema sections and P0 acceptance requirements.
- Implemented: Strict dataclass `from_dict` / `to_dict` / JSON roundtrip, enum coercion, nested schema coercion, TaskSpec validation, abstract ContactSlot guard, phase_type validation, workspace tensor shape checks.
- Not implemented: URDF parsing, geometry processing, IRG building, envelope extraction logic, controller/QP logic, simulator integration.
- Schema/interface changes: Initial implementation plus approved `IRGEdgeType.ALLOWS` supplement.
- Downstream impact: Downstream work packages should import schema objects from `amsrr.schemas.*` rather than redefining local dataclasses.
- Tests added: Schema roundtrip, TaskSpec parsing/validation, workspace shape/mask, `allows` edge presence.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 7 passed.
- Handoff notes: `ContactSlotNode` validation rejects final contact/candidate fields to preserve the IRG abstraction boundary. `SharedInteractionWorkspace` stores `group_slices` as Python `slice` objects and serializes them as JSON mappings.
- Open questions: None currently.

### Agent A/L: Config Loading, Hashing, and Test Harness

#### 2026-07-07
- Scope: Add minimum config and hash utilities needed before robot/geometry/IRG work.
- Files changed:
  - `amsrr/utils/config.py`
  - `amsrr/utils/hashing.py`
  - `configs/robot/robot_model.yaml`
  - `configs/robot/thrust_model.yaml`
  - `configs/training/p0_schema_tests.yaml`
  - `tests/conftest.py`
  - `tests/unit/utils/test_config_hashing.py`
- Upstream dependencies: v0.4 Appendix E minimum example files and Section 25 reproducibility metadata.
- Implemented: YAML/JSON config loading, deterministic canonical JSON hashing, SHA-256 file hashing, minimum robot/thrust/P0 config files.
- Not implemented: Full config schema classes, config merge/override system, command-line runners.
- Schema/interface changes: None beyond initial utility contracts.
- Downstream impact: Robot loader and future dataset/cache keys can use `stable_hash` / `hash_file`.
- Tests added: Config loading and stable hash ordering test.
- Tests passed: Included in 7 passing unit tests.
- Handoff notes: PyYAML is already available in the environment; no dependency install was performed.
- Open questions: None currently.
