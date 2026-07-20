from __future__ import annotations

from dataclasses import dataclass

import pytest

from amsrr.schemas.morphology import MorphologyGraph
from amsrr.simulation.order9_production_hard_checker_runtime import (
    bind_order9_production_hard_checker,
    build_order9_shadow_worker_command,
    order9_shadow_bucket_from_sample,
    write_order9_shadow_bucket_morphology,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_randomization import Order9ConservativeRandomizer
from amsrr.training.order9_teacher import build_order8_grasp_carry_task_spec
from tests.unit.simulation.test_order9_shadow_executor import _valid_context


@dataclass
class _UnusedExporter:
    def export_shadow_state(self, _context):  # pragma: no cover - handshake test only.
        raise AssertionError("identity handshake must not export live state")


class _DescriptorTransport:
    transport_version = "unit-descriptor-transport-v1"

    def __init__(self, response):
        self.response = response
        self.closed = False

    def request(self, operation, payload):
        assert operation == "describe"
        assert payload == {}
        return self.response

    def close(self):
        self.closed = True


def test_bucket_identity_and_command_bind_every_randomized_physical_value(
    tmp_path,
) -> None:
    graph = _valid_context().morphology_graph
    sample = Order9ConservativeRandomizer().sample(_task(), seed=19, sample_index=2)
    bucket = order9_shadow_bucket_from_sample(
        sample,
        graph,
        support_top_z_m=0.15,
    )
    graph_path = write_order9_shadow_bucket_morphology(tmp_path, graph)
    for name in ("config.yaml", "worker.py", "checkpoint.pt", "robot.usda", "micromamba"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    command = build_order9_shadow_worker_command(
        repository_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        worker_script=tmp_path / "worker.py",
        micromamba_environment="isaaclab3",
        device="cuda:0",
        pi_l_checkpoint_path=tmp_path / "checkpoint.pt",
        pi_l_checkpoint_sha256="a" * 64,
        robot_usd_path=tmp_path / "robot.usda",
        morphology_graph_path=graph_path,
        bucket=bucket,
        control_dt_s=0.02,
        micromamba_executable=tmp_path / "micromamba",
    )

    joined = " ".join(command)
    assert f"--morphology-graph-json {graph_path}" in joined
    assert f"--object-id {bucket.object_id}" in joined
    assert f"--object-geometry {bucket.object_geometry_type}" in joined
    assert "--selected-gripper-friction" in command
    assert "--contact-stiffness" in command
    assert "--contact-damping" in command
    assert bucket.topology_structural_hash in graph_path.name


def test_production_binding_authenticates_descriptor_before_building_checker() -> None:
    config = load_order9_learning_config(
        "configs/training/order9_learning_curriculum.yaml"
    )
    graph = _valid_context().morphology_graph
    sample = Order9ConservativeRandomizer().sample(_task(), seed=5)
    bucket = order9_shadow_bucket_from_sample(
        sample,
        graph,
        support_top_z_m=0.15,
    )
    checkpoint = "b" * 64
    scene = {
        "object_id": bucket.object_id,
        "object_geometry_type": bucket.object_geometry_type,
        "object_size_m": list(bucket.object_size_m),
        "object_mass_kg": bucket.object_mass_kg,
        "object_inertia_body": list(bucket.object_inertia_body),
        "object_friction": bucket.object_friction,
        "selected_gripper_friction": bucket.selected_gripper_friction,
        "contact_stiffness_n_per_m": bucket.contact_stiffness_n_per_m,
        "contact_damping_n_s_per_m": bucket.contact_damping_n_s_per_m,
        "support_center_world_m": [1.0, 0.0, 0.075],
        "support_half_extents_m": [0.275, 0.18, 0.075],
    }
    response = {
        "operation": "describe",
        "accepted": True,
        "worker_version": "unit-worker-v1",
        "pi_l_checkpoint_sha256": checkpoint,
        "descriptor": {
            "topology_structural_hash": bucket.topology_structural_hash,
            "pi_l_checkpoint_sha256": checkpoint,
            "control_dt_s": config.hard_checker.shadow_control_dt_s,
            "maximum_horizon_s": config.hard_checker.shadow_rollout_horizon_s,
            "scene": scene,
        },
    }
    transport = _DescriptorTransport(response)

    runtime = bind_order9_production_hard_checker(
        config=config,
        state_exporter=_UnusedExporter(),
        transport=transport,
        pi_l_checkpoint_sha256=checkpoint,
        bucket=bucket,
    )

    assert runtime.checker.config.evaluation_mode == "production"
    assert runtime.bucket.bucket_hash
    runtime.close()
    assert transport.closed is True


def test_production_binding_rejects_mismatched_object_bucket() -> None:
    config = load_order9_learning_config(
        "configs/training/order9_learning_curriculum.yaml"
    )
    graph = _valid_context().morphology_graph
    bucket = order9_shadow_bucket_from_sample(
        Order9ConservativeRandomizer().sample(_task(), seed=7),
        graph,
        support_top_z_m=0.15,
    )
    checkpoint = "c" * 64
    transport = _DescriptorTransport(
        {
            "operation": "describe",
            "accepted": True,
            "worker_version": "unit-worker-v1",
            "pi_l_checkpoint_sha256": checkpoint,
            "descriptor": {
                "topology_structural_hash": bucket.topology_structural_hash,
                "pi_l_checkpoint_sha256": checkpoint,
                "control_dt_s": config.hard_checker.shadow_control_dt_s,
                "maximum_horizon_s": config.hard_checker.shadow_rollout_horizon_s,
                "scene": {
                    "object_id": bucket.object_id,
                    "object_geometry_type": "sphere",
                },
            },
        }
    )

    with pytest.raises(RuntimeError, match="object_geometry_type"):
        bind_order9_production_hard_checker(
            config=config,
            state_exporter=_UnusedExporter(),
            transport=transport,
            pi_l_checkpoint_sha256=checkpoint,
            bucket=bucket,
        )


def test_morphology_bucket_file_refuses_same_structure_different_graph_bytes(
    tmp_path,
) -> None:
    graph = _valid_context().morphology_graph
    write_order9_shadow_bucket_morphology(tmp_path, graph)
    changed = MorphologyGraph.from_dict(graph.to_dict())
    changed.graph_id += "-different-identity"

    with pytest.raises(FileExistsError, match="different graph"):
        write_order9_shadow_bucket_morphology(tmp_path, changed)


def _task():
    return build_order8_grasp_carry_task_spec(
        object_pose_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
        object_size_m=(0.30, 0.40, 0.15),
        object_mass_kg=1.0,
        object_friction=0.6,
        required_transport_distance_m=0.20,
        support_height_m=0.15,
        max_contact_force_n=30.0,
        max_contact_torque_nm=5.0,
    )
