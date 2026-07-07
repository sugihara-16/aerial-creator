# AMSRR_design_modification_by_codex.md

This file records implementation-time supplements or deviations from `A-MSRR_codex_ready_spec_v0_4_ja.md`.

## 2026-07-07

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
