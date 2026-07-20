from __future__ import annotations

from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import ControllerStatus
from amsrr.schemas.runtime import ObjectRuntimeState, RuntimeObservation, TaskProgressState
from amsrr.training.order9_reward import Order9RewardEngine
from amsrr.training.order9_teacher import build_order8_grasp_carry_task_spec


def test_phase_switches_task_terms_but_never_masks_common_safety() -> None:
    engine = Order9RewardEngine()
    task = _task()
    approach = engine.step(task_spec=task, observation=_observation("approach", x=0.50))
    transport = engine.step(task_spec=task, observation=_observation("transport", x=0.60))

    assert "grasp_maintenance" not in approach.active_task_terms
    assert approach.terms["weighted_grasp_maintenance"] == 0.0
    assert approach.terms["weighted_collision"] < 0.0
    assert "grasp_maintenance" in transport.active_task_terms
    assert transport.terms["weighted_grasp_maintenance"] > 0.0
    assert transport.terms["weighted_collision"] < 0.0


def test_actor_phase_context_is_explicit_and_contains_no_raw_contact() -> None:
    output = Order9RewardEngine().step(
        task_spec=_task(),
        observation=_observation("contact_acquisition", x=0.50),
    )
    features = output.phase_context.actor_features()

    assert output.phase_context.phase_label == "establish_contact"
    assert len(features) == output.phase_context.phase_count + 1
    assert sum(features[:-1]) == 1.0
    assert features[-1] == 0.4
    assert output.raw_contact_used_as_actor_input is False
    assert output.terms["raw_contact_actor_input"] == 0.0
    assert output.terms["raw_contact_reward_or_safety_only"] == 1.0


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


def _observation(phase: str, *, x: float) -> RuntimeObservation:
    morphology = MorphologyGraph(
        graph_id="reward-morphology",
        modules=[],
        ports=[],
        dock_edges=[],
        robot_anchors=[],
        control_groups=[],
        base_module_id=0,
        is_closed_loop=False,
    )
    return RuntimeObservation(
        time_s=1.0,
        morphology_graph=morphology,
        module_states=[],
        object_states=[
            ObjectRuntimeState(
                object_id="order8_object",
                pose_world=(x, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
                twist_world=[0.0] * 6,
            )
        ],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(
            phase_label=phase,
            progress_ratio=0.4,
            metrics={
                "grasp_data_available": 1.0,
                "grasp_maintenance": 1.0,
                "collision_data_available": 1.0,
                "hard_collision": 1.0,
                "slip_data_available": 1.0,
                "slip": 0.0,
            },
        ),
    )
