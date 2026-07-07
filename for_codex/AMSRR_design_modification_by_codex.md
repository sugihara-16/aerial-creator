# AMSRR_design_modification_by_codex.md

This file records implementation-time supplements or deviations from `A-MSRR_codex_ready_spec_v0_4_ja.md`.

## 2026-07-07

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
