# WORKLOG.md

## Global Worklog

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
