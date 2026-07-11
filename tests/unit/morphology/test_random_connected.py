from __future__ import annotations

from dataclasses import replace

import pytest

from amsrr.geometry.pose_math import FACE_TO_FACE_DOCK_RELATION, compose_pose, inverse_pose
from amsrr.morphology.random_connected import (
    RandomConnectedMorphologyConfig,
    RandomConnectedMorphologyDistribution,
    morphology_structural_hash,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError


def _distribution(
    config: RandomConnectedMorphologyConfig | None = None,
) -> RandomConnectedMorphologyDistribution:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    return RandomConnectedMorphologyDistribution(physical_model, config)


def test_random_connected_sample_is_seed_deterministic_and_serializable() -> None:
    distribution = _distribution()

    first = distribution.sample(seed=1729)
    second = distribution.sample(seed=1729)

    assert first.to_dict() == second.to_dict()
    assert first.graph_id.endswith(morphology_structural_hash(first))
    assert type(first).from_json(first.to_json()).to_dict() == first.to_dict()


def test_random_connected_default_distribution_covers_two_through_eight_modules() -> None:
    distribution = _distribution()
    samples = distribution.samples(seeds=range(100))

    assert {len(sample.modules) for sample in samples} == {2, 3, 4, 5, 6, 7, 8}


@pytest.mark.parametrize("module_count", range(2, 9))
def test_random_connected_candidate_is_port_aligned_tree(module_count: int) -> None:
    distribution = _distribution()
    morphology = distribution.sample(seed=100 + module_count, module_count=module_count)

    assert len(morphology.modules) == module_count
    assert len(morphology.dock_edges) == module_count - 1
    assert morphology.base_module_id == 0
    assert morphology.is_closed_loop is False
    assert [(module.module_id, module.role_id, module.is_base) for module in morphology.modules] == [
        (module_id, "base" if module_id == 0 else "member", module_id == 0)
        for module_id in range(module_count)
    ]
    assert len(morphology.control_groups) == 1
    assert morphology.control_groups[0].group_id == "all_modules"
    assert morphology.control_groups[0].module_ids == list(range(module_count))
    assert morphology.robot_anchors == []

    ports_by_id = {port.port_global_id: port for port in morphology.ports}
    modules_by_id = {module.module_id: module for module in morphology.modules}
    referenced_port_ids: list[int] = []
    neighbors: dict[int, set[int]] = {module_id: set() for module_id in modules_by_id}
    for edge in morphology.dock_edges:
        src_port = ports_by_id[edge.src_port_id]
        dst_port = ports_by_id[edge.dst_port_id]
        referenced_port_ids.extend((src_port.port_global_id, dst_port.port_global_id))
        neighbors[edge.src_module_id].add(edge.dst_module_id)
        neighbors[edge.dst_module_id].add(edge.src_module_id)
        assert {src_port.port_type, dst_port.port_type} == {"pitch_dock", "yaw_dock"}
        src_port_world = compose_pose(
            modules_by_id[edge.src_module_id].pose_in_design_frame,
            compose_pose(src_port.local_pose, FACE_TO_FACE_DOCK_RELATION),
        )
        dst_port_world = compose_pose(
            modules_by_id[edge.dst_module_id].pose_in_design_frame,
            dst_port.local_pose,
        )
        assert dst_port_world == pytest.approx(src_port_world)

    assert len(referenced_port_ids) == len(set(referenced_port_ids))
    assert {port.port_global_id for port in morphology.ports if port.occupied} == set(referenced_port_ids)
    assert all(
        len([port for port in morphology.ports if port.module_id == module_id]) == 4
        for module_id in modules_by_id
    )
    assert _reachable_from_base(neighbors) == set(modules_by_id)


def test_random_connected_fixed_size_produces_diverse_candidates() -> None:
    distribution = _distribution()
    hashes = {
        morphology_structural_hash(distribution.sample(seed=seed, module_count=6))
        for seed in range(20)
    }

    assert len(hashes) >= 8


def test_structural_hash_is_invariant_to_nonbase_ids_edge_direction_and_list_order() -> None:
    morphology = _distribution().sample(seed=37, module_count=6)
    remapped = _relabel_and_reorder(morphology)

    assert morphology_structural_hash(remapped) == morphology_structural_hash(morphology)


def test_random_connected_config_can_narrow_uniform_module_range() -> None:
    distribution = _distribution(RandomConnectedMorphologyConfig(min_modules=4, max_modules=4))

    assert {len(distribution.sample(seed=seed).modules) for seed in range(10)} == {4}
    with pytest.raises(SchemaValidationError, match="configured inclusive range"):
        distribution.sample(seed=0, module_count=3)


@pytest.mark.parametrize(
    "config",
    [
        RandomConnectedMorphologyConfig(min_modules=2, max_modules=2),
        RandomConnectedMorphologyConfig(min_modules=8, max_modules=8),
    ],
)
def test_random_connected_config_accepts_version_one_boundaries(
    config: RandomConnectedMorphologyConfig,
) -> None:
    assert len(_distribution(config).sample(seed=0).modules) == config.min_modules


def test_random_connected_rejects_invalid_config_seed_and_ports() -> None:
    with pytest.raises(SchemaValidationError, match="2 <= min_modules"):
        RandomConnectedMorphologyConfig(min_modules=1)
    with pytest.raises(SchemaValidationError, match="max_modules <= 8"):
        RandomConnectedMorphologyConfig(max_modules=9)
    with pytest.raises(SchemaValidationError, match="non-negative integer"):
        _distribution().sample(seed=-1)

    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    pitch_only_model = replace(
        physical_model,
        dock_ports=[port for port in physical_model.dock_ports if port.port_type == "pitch_dock"],
    )
    with pytest.raises(SchemaValidationError, match="compatible pitch-yaw"):
        RandomConnectedMorphologyDistribution(pitch_only_model)


def _reachable_from_base(neighbors: dict[int, set[int]]) -> set[int]:
    reached: set[int] = set()
    pending = [0]
    while pending:
        module_id = pending.pop()
        if module_id in reached:
            continue
        reached.add(module_id)
        pending.extend(neighbors[module_id] - reached)
    return reached


def _relabel_and_reorder(morphology):
    nonbase_ids = sorted(
        module.module_id for module in morphology.modules if module.module_id != morphology.base_module_id
    )
    module_id_map = {morphology.base_module_id: 0}
    module_id_map.update(zip(nonbase_ids, reversed(nonbase_ids)))

    old_ports = list(reversed(morphology.ports))
    port_id_map = {
        port.port_global_id: 10_000 + index
        for index, port in enumerate(old_ports)
    }
    remapped_ports = [
        replace(
            port,
            port_global_id=port_id_map[port.port_global_id],
            module_id=module_id_map[port.module_id],
        )
        for port in old_ports
    ]
    remapped_modules = [
        replace(module, module_id=module_id_map[module.module_id])
        for module in reversed(morphology.modules)
    ]
    remapped_edges = []
    for edge_id, edge in enumerate(reversed(morphology.dock_edges)):
        remapped_edges.append(
            replace(
                edge,
                edge_id=edge_id,
                src_module_id=module_id_map[edge.dst_module_id],
                src_port_id=port_id_map[edge.dst_port_id],
                dst_module_id=module_id_map[edge.src_module_id],
                dst_port_id=port_id_map[edge.src_port_id],
                relative_pose_src_to_dst=inverse_pose(edge.relative_pose_src_to_dst),
            )
        )
    return replace(
        morphology,
        graph_id="arbitrary-id-not-in-structural-hash",
        modules=remapped_modules,
        ports=remapped_ports,
        dock_edges=remapped_edges,
        control_groups=[],
        base_module_id=0,
    )
