from __future__ import annotations

from dataclasses import replace

import pytest

from amsrr.controllers.qp_allocator_interface import VirtualThrustQPAllocator
from amsrr.feasibility import violation_codes as codes
from amsrr.feasibility.morphology_flight import (
    MorphologyFlightFeasibilityChecker,
    MorphologyFlightFeasibilityConfig,
    collision_geometry_content_hash,
    morphology_collision_aabbs,
    morphology_min_collision_z,
)
from amsrr.morphology.dock_geometry import modules_with_dock_aligned_poses
from amsrr.morphology.graph import MinimalMorphologyBuilder
from amsrr.robot_model.physical_model_builder import (
    build_module_capability_token,
    build_physical_model_from_config,
)
from amsrr.schemas.morphology import ControlGroup, MorphologyGraph

MESH_SEARCH_DIRS = ("module_urdf", "module_urdf/mesh")


class _RecordingAllocator:
    def __init__(self) -> None:
        self.problems = []
        self.delegate = VirtualThrustQPAllocator()

    def allocate(self, problem):
        self.problems.append(problem)
        return self.delegate.allocate(problem)


def _physical_model():
    return build_physical_model_from_config("configs/robot/robot_model.yaml")


def _morphology(
    module_count: int = 2,
    *,
    physical_model=None,
) -> MorphologyGraph:
    physical_model = physical_model or _physical_model()
    capability = build_module_capability_token(physical_model)
    modules = MinimalMorphologyBuilder._build_modules(module_count, capability)
    ports, edges = MinimalMorphologyBuilder._build_ports_and_edges(
        module_count, physical_model.dock_ports
    )
    modules = modules_with_dock_aligned_poses(modules, edges, base_module_id=0)
    return MorphologyGraph(
        graph_id=f"morphology-flight-test-{module_count}",
        modules=modules,
        ports=ports,
        dock_edges=edges,
        robot_anchors=[],
        control_groups=[
            ControlGroup(
                group_id="all_modules",
                module_ids=list(range(module_count)),
                role="whole_body",
            )
        ],
        base_module_id=0,
        is_closed_loop=False,
    )


def _checker(**overrides) -> MorphologyFlightFeasibilityChecker:
    return MorphologyFlightFeasibilityChecker(
        MorphologyFlightFeasibilityConfig(
            mesh_search_dirs=MESH_SEARCH_DIRS,
            **overrides,
        )
    )


def _codes(result) -> set[str]:
    return {item.code for item in result.hard_violations}


def test_real_holon_tree_passes_structure_collision_and_both_hover_qps() -> None:
    result = _checker().check(_morphology(), _physical_model())

    assert result.feasible is True
    assert result.hard_violations == []
    assert result.metadata["collision_geometry_status"] == "available"
    assert result.metadata["collision_geometry_content_hash"] == (
        collision_geometry_content_hash(
            _physical_model(), mesh_search_dirs=MESH_SEARCH_DIRS
        )
    )
    assert (
        result.metadata["collision_method"]
        == "nominal_q_urdf_collision_mesh_module_aabb"
    )
    assert (
        result.metadata["hover_reference_state"] == "design_pose_zero_twist_zero_q_qdot"
    )
    assert result.metadata["hover_velocity_limits_applied"] is False
    assert result.margins["qp_hover_feasible"] == 1.0
    assert result.margins["qp_margin_feasible"] == 1.0
    assert result.margins["qp_hover_qp_primary_path"] == 1.0
    assert result.margins["qp_margin_qp_primary_path"] == 1.0
    assert result.margins["thrust_margin_ratio"] == pytest.approx(0.15)


def test_margin_qp_uses_configured_scaled_gravity_wrench() -> None:
    physical_model = _physical_model()
    morphology = _morphology()
    allocator = _RecordingAllocator()
    checker = MorphologyFlightFeasibilityChecker(
        MorphologyFlightFeasibilityConfig(
            mesh_search_dirs=MESH_SEARCH_DIRS,
            min_thrust_margin_ratio=0.25,
        ),
        allocator=allocator,
    )

    result = checker.check(morphology, physical_model)

    assert result.feasible is True
    assert len(allocator.problems) == 2
    hover_target = allocator.problems[0].desired_wrench_body
    margin_target = allocator.problems[1].desired_wrench_body
    assert hover_target is not None
    assert margin_target == pytest.approx([value * 1.25 for value in hover_target])
    assert result.margins["thrust_margin_threshold"] == pytest.approx(0.25)
    assert result.margins["thrust_margin_ratio"] == pytest.approx(0.25)


def test_nonadjacent_collision_gate_uses_configured_clearance_margin() -> None:
    result = _checker(collision_margin_m=2.0).check(
        _morphology(module_count=3), _physical_model()
    )

    assert result.feasible is False
    assert codes.F_COARSE_COLLISION in _codes(result)
    assert result.margins["coarse_collision_checked_pair_count"] == 1.0
    assert result.margins["coarse_collision_pair_count"] == 1.0
    assert result.metadata["collision_adjacent_pair_exclusion_count"] == 2


def test_collision_geometry_is_a_hard_gate_when_meshes_cannot_be_resolved() -> None:
    checker = MorphologyFlightFeasibilityChecker(MorphologyFlightFeasibilityConfig())

    result = checker.check(_morphology(), _physical_model())

    assert result.feasible is False
    assert codes.F_COARSE_COLLISION in _codes(result)
    assert result.metadata["collision_geometry_status"] == "unavailable"
    assert "was not found" in str(result.metadata["collision_geometry_error"])


def test_port_occupancy_mismatch_stops_downstream_geometry_and_qp_gates() -> None:
    morphology = _morphology()
    occupied_port = next(port for port in morphology.ports if port.occupied)
    morphology.ports = [
        (
            replace(port, occupied=False)
            if port.port_global_id == occupied_port.port_global_id
            else port
        )
        for port in morphology.ports
    ]

    result = _checker().check(morphology, _physical_model())

    assert result.feasible is False
    assert codes.F_PORT_OCCUPANCY in _codes(result)
    assert result.metadata["structural_gate_passed"] is False
    assert result.metadata["collision_geometry_status"] == "skipped_structural_failure"
    assert result.metadata["hover_gate_status"] == "skipped_structural_failure"


@pytest.mark.parametrize("mutation", ["module_type", "capability_token"])
def test_module_identity_and_capability_are_bound_to_physical_model(mutation: str) -> None:
    morphology = _morphology()
    module = morphology.modules[1]
    if mutation == "module_type":
        replacement = replace(module, module_type="not_holon")
    else:
        replacement = replace(
            module,
            capability_token=replace(
                module.capability_token,
                rotor_count=module.capability_token.rotor_count + 1,
            ),
        )
    morphology.modules = [morphology.modules[0], replacement]

    result = _checker().check(morphology, _physical_model())

    assert result.feasible is False
    assert codes.F_SCHEMA_VALID in _codes(result)
    assert result.margins["physical_model_module_binding_valid"] == 0.0


@pytest.mark.parametrize("mutation", ["local_pose", "compatible_mask"])
def test_port_geometry_and_compatibility_are_derived_from_physical_model(
    mutation: str,
) -> None:
    morphology = _morphology()
    target = morphology.ports[0]
    if mutation == "local_pose":
        replacement = replace(
            target,
            local_pose=(target.local_pose[0] + 0.01, *target.local_pose[1:]),
        )
        expected_code = codes.F_PORT_OCCUPANCY
    else:
        replacement = replace(
            target,
            compatible_port_type_mask=[0] * len(target.compatible_port_type_mask),
        )
        expected_code = codes.F_COMPATIBLE_PORT_TYPES
    morphology.ports = [replacement, *morphology.ports[1:]]

    result = _checker().check(morphology, _physical_model())

    assert result.feasible is False
    assert expected_code in _codes(result)
    assert result.metadata["structural_gate_passed"] is False


def test_qp_rejects_low_thrust_model_for_hover_and_margin() -> None:
    physical_model = _physical_model()
    low_thrust_model = replace(
        physical_model,
        rotors=[replace(rotor, thrust_max_n=1.0) for rotor in physical_model.rotors],
    )

    result = _checker().check(
        _morphology(physical_model=low_thrust_model), low_thrust_model
    )

    assert result.feasible is False
    assert codes.F_QP_HOVER_FEASIBILITY in _codes(result)
    assert codes.F_THRUST_MARGIN in _codes(result)
    assert result.margins["qp_hover_feasible"] == 0.0
    assert result.margins["qp_margin_feasible"] == 0.0
    assert result.margins["thrust_margin_ratio"] == -1.0


def test_public_collision_bounds_support_floor_height_initialization() -> None:
    morphology = _morphology()
    physical_model = _physical_model()

    bounds = morphology_collision_aabbs(
        morphology,
        physical_model,
        mesh_search_dirs=MESH_SEARCH_DIRS,
    )
    minimum_z = morphology_min_collision_z(
        morphology,
        physical_model,
        mesh_search_dirs=MESH_SEARCH_DIRS,
    )

    assert set(bounds) == {0, 1}
    assert minimum_z == pytest.approx(min(lower[2] for lower, _ in bounds.values()))
    assert minimum_z < 0.0
