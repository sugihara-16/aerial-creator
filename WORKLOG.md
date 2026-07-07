# WORKLOG.md

## Global Worklog

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
