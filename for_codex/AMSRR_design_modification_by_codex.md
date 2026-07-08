# AMSRR_design_modification_by_codex.md

This file records implementation-time supplements or deviations from `A-MSRR_codex_ready_spec_v0_4_ja.md`.

## 2026-07-08

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
