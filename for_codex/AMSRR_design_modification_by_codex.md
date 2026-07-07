# AMSRR_design_modification_by_codex.md

This file records implementation-time supplements or deviations from `A-MSRR_codex_ready_spec_v0_4_ja.md`.

## 2026-07-07

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
