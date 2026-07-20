from __future__ import annotations

from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.simulation.order9_runtime_state import (
    Order9LiveMainStateExporter,
    Order9SimulationSnapshot,
    restore_order9_controller_and_policy_state,
)
from tests.unit.simulation.test_order9_shadow_executor import _valid_context


_CHECKPOINT = "c" * 64


class _SimulationSource:
    snapshot_source_version = "unit-snapshot-source-v1"

    def capture_order9_simulation_snapshot(self, context):
        return Order9SimulationSnapshot(
            state_id="episode-1:step-42",
            topology_structural_hash=morphology_structural_hash(
                context.morphology_graph
            ),
            simulation_time_s=0.84,
            simulation_state={
                "articulation_root_state": [[0.0] * 13],
                "object_root_state": [[0.0] * 13],
            },
            execution_state={"previous_command": None, "plan_sequence": 3},
            metadata={"environment_index": 7},
        )


class _StateOwner:
    def __init__(self, state, *, checkpoint_sha256=None):
        self.state = state
        self.restored = None
        if checkpoint_sha256 is not None:
            self.checkpoint_sha256 = checkpoint_sha256

    def export_runtime_state(self):
        return dict(self.state)

    def restore_runtime_state(self, payload):
        self.restored = dict(payload)


def test_live_main_state_export_binds_all_future_affecting_state() -> None:
    context = _valid_context()
    controller = _StateOwner({"runtime_state_version": "qpid-unit"})
    policy = _StateOwner(
        {"runtime_state_version": "pi-l-unit", "hidden": [[0.25]]},
        checkpoint_sha256=_CHECKPOINT,
    )
    exporter = Order9LiveMainStateExporter(
        simulation_source=_SimulationSource(),
        controller=controller,
        pi_l_policy=policy,
        checkpoint_sha256=_CHECKPOINT,
    )

    exported = exporter.export_shadow_state(context)

    assert exported.state_id == "episode-1:step-42"
    assert exported.controller_state["qpid"]["runtime_state_version"] == "qpid-unit"
    assert exported.controller_state["trajectory_execution"]["plan_sequence"] == 3
    assert exported.pi_l_state["hidden"] == [[0.25]]
    assert exported.metadata["atomic_state_lock_held"] is True
    assert exported.pi_l_checkpoint_sha256 == _CHECKPOINT


def test_restore_controller_policy_returns_execution_state() -> None:
    context = _valid_context()
    controller = _StateOwner({"runtime_state_version": "qpid-unit"})
    policy = _StateOwner(
        {"runtime_state_version": "pi-l-unit", "hidden": [[0.25]]},
        checkpoint_sha256=_CHECKPOINT,
    )
    state = Order9LiveMainStateExporter(
        simulation_source=_SimulationSource(),
        controller=controller,
        pi_l_policy=policy,
        checkpoint_sha256=_CHECKPOINT,
    ).export_shadow_state(context)
    copied_controller = _StateOwner({})
    copied_policy = _StateOwner({}, checkpoint_sha256=_CHECKPOINT)

    execution = restore_order9_controller_and_policy_state(
        state,
        controller=copied_controller,
        pi_l_policy=copied_policy,
    )

    assert copied_controller.restored == state.controller_state["qpid"]
    assert copied_policy.restored == state.pi_l_state
    assert execution == {"previous_command": None, "plan_sequence": 3}
