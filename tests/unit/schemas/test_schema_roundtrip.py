from __future__ import annotations


from amsrr.schemas.common import ContactMode
from amsrr.schemas.contact_candidates import (
    AssignmentFeasibilityResult,
    ContactCandidate,
    ContactCandidateSet,
)
from amsrr.schemas.geometry import (
    ContactRegion,
    ContactRegionGraph,
    GeometryDescriptor,
    GlobalShapeFeatures,
    SurfacePatchGraph,
    SurfacePatchToken,
)
from amsrr.schemas.interaction_envelope import (
    CapabilityRequirement,
    InteractionEnvelope,
    TargetRegionSet,
    WrenchSpaceRequirement,
)
from amsrr.schemas.irg import IRGEdge, IRGEdgeType, IRGNode, IRGNodeType, InteractionRequirementGraph
from amsrr.schemas.physical_model import JointModel, LinkModel, PhysicalModel
from amsrr.schemas.policies import ControllerStatus, PolicyCommand
from amsrr.schemas.task_spec import TaskSpec


def test_plain_data_scalar_fast_path_keeps_enum_and_nested_conversion():
    from dataclasses import dataclass
    from enum import IntEnum, Enum
    from pathlib import Path
    from amsrr.schemas.common import canonical_json, to_plain_data
    class Number(IntEnum):
        ONE = 1
    class Text(str, Enum):
        A = 'a'
    @dataclass
    class Nested:
        values: tuple
    value = Nested((None, True, 3, 1.25, 'x', Number.ONE, Text.A, Path('local'),
                    {'slice': slice(1, 4, 2)}))
    expected = {'values': [None, True, 3, 1.25, 'x', 1, 'a', 'local',
                           {'slice': {'start': 1, 'stop': 4, 'step': 2}}]}
    actual = to_plain_data(value)
    assert actual == expected
    assert type(actual['values'][5]) is int and type(actual['values'][6]) is str
    assert canonical_json(value) == canonical_json(expected)


def test_schema_roundtrip_json(grasp_carry_dict: dict) -> None:
    task = TaskSpec.from_dict(grasp_carry_dict)
    assert TaskSpec.from_json(task.to_json()).to_dict() == task.to_dict()

    patch = SurfacePatchToken(
        patch_id=0,
        entity_id="box_01",
        position_object=(0.0, 0.0, 0.075),
        normal_object=(0.0, 0.0, 1.0),
        tangent_u_object=(1.0, 0.0, 0.0),
        tangent_v_object=(0.0, 1.0, 0.0),
        patch_area_m2=0.06,
        mean_curvature=0.0,
        gaussian_curvature=0.0,
        local_thickness_m=None,
        friction=0.6,
        contact_allowed=True,
        allowed_contact_modes=[ContactMode.GRASP],
    )
    region = ContactRegion(
        region_id="box_top",
        entity_id="box_01",
        region_type="face",
        patch_ids=[0],
        pose_object=None,
        normal_summary_object=(0.0, 0.0, 1.0),
        area_m2=0.06,
        curvature_summary=[0.0, 0.0],
        friction=0.6,
        allowed_contact_modes=[ContactMode.GRASP],
    )
    geom = GeometryDescriptor(
        geometry_id="box_geom",
        global_shape_features=GlobalShapeFeatures(
            bbox_m=(0.3, 0.2, 0.15),
            volume_m3=0.009,
            surface_area_m2=0.27,
            approximate_com_object=(0.0, 0.0, 0.0),
            approximate_inertia_diag=(1.0, 1.0, 1.0),
            principal_axes_flat=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            compactness=1.0,
            symmetry_features=[1.0],
        ),
        surface_patch_graph=SurfacePatchGraph(nodes=[patch]),
        contact_region_graph=ContactRegionGraph(nodes=[region]),
        collision_ref="collision://box_geom",
        exact_geometry_ref="exact://box_geom",
    )
    assert GeometryDescriptor.from_json(geom.to_json()).to_dict() == geom.to_dict()

    irg = InteractionRequirementGraph(
        irg_id="irg_001",
        task_id=task.task_id,
        nodes=[
            IRGNode(0, IRGNodeType.TASK, task.task_id, 1.0, True, None, {"task_id": task.task_id}),
            IRGNode(1, IRGNodeType.PHASE, "approach_object", 1.0, True, None, {"phase_type": "approach"}),
            IRGNode(2, IRGNodeType.CONTACT_REGION, "box_top", 1.0, True, None, {}),
            IRGNode(3, IRGNodeType.CONTACT_SLOT, "slot_0", 1.0, True, 1, {"slot_id": 0}),
        ],
        edges=[
            IRGEdge(0, 1, IRGEdgeType.CONTAINS),
            IRGEdge(2, 3, IRGEdgeType.ALLOWS),
        ],
    )
    assert InteractionRequirementGraph.from_json(irg.to_json()).to_dict() == irg.to_dict()

    envelope = InteractionEnvelope(
        envelope_id="env_001",
        task_id=task.task_id,
        required_contact_count_range=(2, 4),
        required_contact_modes=[ContactMode.GRASP, ContactMode.SUPPORT],
        target_region_sets=[TargetRegionSet(entity_id="box_01", region_ids=["box_top"])],
        wrench_space_requirements=[
            WrenchSpaceRequirement(applies_to="object_contact_slots", effect="inward_grasp_force")
        ],
        capability_requirements=[CapabilityRequirement(capability_type="grasp", min_force_n=5.0)],
    )
    assert InteractionEnvelope.from_json(envelope.to_json()).to_dict() == envelope.to_dict()

    physical = PhysicalModel(
        model_id="holon_minimal",
        urdf_path="module_urdf/holon.urdf.xacro",
        links=[
            LinkModel("root", None, 0.1, [1, 0, 0, 1, 0, 1], (0, 0, 0), None, None),
            LinkModel("main_body", "root_joint", 0.2, [1, 0, 0, 1, 0, 1], (0, 0, 0), None, None),
        ],
        joints=[
            JointModel("root_joint", "fixed", "root", "main_body", (0, 0, 0), (0, 0, 0), (0, 0, 1), None, None, None, None)
        ],
        rotors=[],
        dock_ports=[],
        collision_primitives=[],
        aggregate_mass_kg=0.3,
        aggregate_inertia_body=[1, 0, 0, 1, 0, 1],
    )
    assert PhysicalModel.from_json(physical.to_json()).to_dict() == physical.to_dict()

    command = PolicyCommand(
        desired_body_twist=[0, 0, 0, 0, 0, 0],
        desired_anchor_pose_offsets={1: (0, 0, 0, 0, 0, 0, 1)},
        contact_tracking_bias={1: [0.1, 0.0]},
    )
    assert PolicyCommand.from_json(command.to_json()).to_dict() == command.to_dict()

    candidate = ContactCandidate(
        candidate_id=0,
        slot_id=0,
        anchor_id=0,
        target_entity_id="box_01",
        region_id="box_top",
        contact_pose_world=(0, 0, 0, 0, 0, 0, 1),
        contact_frame_world=(0, 0, 0, 0, 0, 0, 1),
        normal_world=(0, 0, 1),
        tangent_basis_world=[1, 0, 0, 0, 1, 0],
        contact_mode=ContactMode.GRASP,
        friction=0.6,
        patch_area_m2=0.01,
        candidate_scores={"normal_alignment": 1.0},
        unary_valid=True,
    )
    candidate_set = ContactCandidateSet(
        set_id="set_001",
        task_id=task.task_id,
        morphology_graph_id="morph_001",
        candidates=[candidate],
        candidate_mask=[True],
        slot_coverage={0: [0]},
        pairwise_conflict_matrix=[[False]],
        pairwise_compatibility_score=[[1.0]],
        group_proposals=[],
        assignment_feasibility_cache={
            "0": AssignmentFeasibilityResult("0", [0], False, ["E_ASSIGNMENT_QP_INFEASIBLE"])
        },
        sampler_version="test",
    )
    assert ContactCandidateSet.from_json(candidate_set.to_json()).to_dict() == candidate_set.to_dict()

    status = ControllerStatus(status="ok", qp_feasible=True)
    assert ControllerStatus.from_json(status.to_json()).to_dict() == status.to_dict()

def test_irg_edge_type_includes_allows() -> None:
    assert IRGEdgeType.ALLOWS.value == "allows"


def test_repeated_schema_loading_retains_strict_validation_and_owned_values():
    from dataclasses import dataclass
    import pytest
    from amsrr.schemas.common import SchemaBase, SchemaValidationError

    @dataclass
    class RepeatedSchema(SchemaBase):
        count: int
        weight: float
        active: bool
        label: str
        values: list[float]

    data = dict(count='3', weight=2, active=True, label='sample', values=[1, 2])
    first = RepeatedSchema.from_dict(data)
    second = RepeatedSchema.from_dict(data)
    assert first.count == 3 and first.weight == 2.0
    first.values[0] = 99.0
    assert second.values == [1.0, 2.0] and data['values'] == [1, 2]
    for field, value in [('count', True), ('weight', True), ('active', 1),
                         ('label', 3), ('values', [False])]:
        with pytest.raises(SchemaValidationError):
            RepeatedSchema.from_dict({**data, field: value})
    with pytest.raises(SchemaValidationError, match='required'):
        RepeatedSchema.from_dict({k: v for k, v in data.items() if k != 'count'})
    RepeatedSchema._allow_extra_fields = True
    RepeatedSchema.from_dict({**data, 'extra': 1})
    RepeatedSchema._allow_extra_fields = False
    with pytest.raises(SchemaValidationError, match='unknown fields'):
        RepeatedSchema.from_dict({**data, 'extra': 1})

