# WORKLOG.md

## Global Worklog

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
