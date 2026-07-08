# AMSRR_design_modification_by_codex.md

This file records implementation-time supplements or deviations from `A-MSRR_codex_ready_spec_v0_4_ja.md`.

## 2026-07-09

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
