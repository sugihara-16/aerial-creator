from __future__ import annotations

import pytest

from scripts.order8_proxy_pad_gui import (
    build_proxy_pad_spawn_command,
    source_command_at_qclose,
    stable_grasp_step_count,
    upgrade_legacy_order8_config,
)
from amsrr.schemas.order8 import (
    ORDER8_NATURAL_CONTACT_CONFIG_VERSION,
    Order8NaturalContactConfig,
)


def _source_command() -> list[str]:
    return [
        "/home/leus/IsaacLab/isaaclab.sh",
        "-p",
        "scripts/p4_control_holon_spawn_probe.py",
        "--steps",
        "1500",
        "--order8-natural-contact",
        "--order8-diagnostic-only",
        "--order8-diagnostic-proxy-pad",
        "--order8-diagnostic-near-contact-base-pose",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "1",
        "--order8-diagnostic-near-contact-joint-positions-json",
        '{"module_0:yaw_dock_mech_joint1": 0.0}',
        "--order8-diagnostic-near-contact-object-pose",
        "1",
        "0",
        "0.2",
        "0",
        "0",
        "0",
        "1",
        "--headless",
        "--realtime-playback",
        "--keep-open-after-smoke-s",
        "5",
    ]


def _runtime_report() -> dict[str, object]:
    return {
        "order8_natural_contact_qclose_checkpoint_base_pose": [
            0.5,
            0.0,
            0.3,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "order8_natural_contact_qclose_checkpoint_joint_positions_rad": {
            "module_0:yaw_dock_mech_joint1": 0.25,
        },
        "order8_natural_contact_qclose_checkpoint_state": {
            "schema_version": "order8_qclose_checkpoint_state_v1",
            "module_root_poses": {"0": [0.0] * 7},
        },
        "order8_natural_contact_simulation_dt_s": 0.02,
        "order8_natural_contact_step_evidence": [
            {
                "time_s": 0.0,
                "grasp_acquired": False,
                "selected_contact_link_ids": [],
            },
            {
                "time_s": 1.2,
                "grasp_acquired": True,
                "selected_contact_link_ids": ["module_1:pad", "module_2:pad"],
            },
        ],
    }


def test_qclose_source_replaces_open_fixture_with_exact_checkpoint() -> None:
    command = source_command_at_qclose(_source_command(), _runtime_report())

    assert "--order8-diagnostic-near-contact-base-pose" not in command
    assert "--order8-diagnostic-near-contact-joint-positions-json" not in command
    assert "--order8-diagnostic-near-contact-object-pose" not in command
    assert command[command.index("--order8-diagnostic-qclose-base-pose") + 1] == "0.5"
    assert "--order8-diagnostic-qclose-joint-positions-json" in command
    assert "--order8-diagnostic-qclose-state-json" in command
    assert "--order8-diagnostic-qclose-zero-velocities" in command


def test_qclose_source_rejects_missing_checkpoint() -> None:
    with pytest.raises(ValueError, match="q_close"):
        source_command_at_qclose(_source_command(), {})


def test_stable_grasp_step_count_includes_first_two_contact_grasp_sample() -> None:
    assert stable_grasp_step_count(_runtime_report()) == 61


def test_stable_grasp_step_count_rejects_absent_grasp() -> None:
    report = _runtime_report()
    report["order8_natural_contact_step_evidence"] = []

    with pytest.raises(ValueError, match="two-pad grasp"):
        stable_grasp_step_count(report)


def test_proxy_pad_spawn_command_runs_only_initialization_and_holds_kit() -> None:
    command = build_proxy_pad_spawn_command(
        _source_command(),
        steps=1,
        keep_open_s=120.0,
    )

    assert command[:4] == [
        "/home/leus/.local/bin/micromamba",
        "run",
        "-n",
        "isaaclab3",
    ]
    assert command[command.index("--steps") + 1] == "1"
    assert "--headless" not in command
    assert "--realtime-playback" not in command
    assert command[command.index("--viz") + 1] == "kit"
    assert command[command.index("--keep-open-after-smoke-s") + 1] == "120.0"
    assert "--order8-diagnostic-proxy-pad" in command


def test_proxy_pad_spawn_command_rejects_non_proxy_source() -> None:
    source = _source_command()
    source.remove("--order8-diagnostic-proxy-pad")

    with pytest.raises(ValueError, match="proxy-pad"):
        build_proxy_pad_spawn_command(source, steps=1, keep_open_s=120.0)


def test_legacy_proxy_config_migrates_only_the_new_compliance_contract() -> None:
    payload = Order8NaturalContactConfig().to_dict()
    payload["config_version"] = "order8_natural_contact_config_v10"
    payload.pop("selected_gripper_compliant_contact_stiffness_n_per_m")
    payload.pop("selected_gripper_compliant_contact_damping_n_s_per_m")
    command = _source_command() + [
        "--order8-config-json",
        __import__("json").dumps(payload),
    ]

    migrated = upgrade_legacy_order8_config(command)
    config_payload = __import__("json").loads(
        migrated[migrated.index("--order8-config-json") + 1]
    )

    assert config_payload["config_version"] == ORDER8_NATURAL_CONTACT_CONFIG_VERSION
    assert (
        config_payload["selected_gripper_compliant_contact_stiffness_n_per_m"]
        == pytest.approx(7500.0)
    )
    assert (
        config_payload["selected_gripper_compliant_contact_damping_n_s_per_m"]
        == pytest.approx(75.0)
    )


@pytest.mark.parametrize(
    ("steps", "keep_open_s"),
    ((0, 120.0), (1, 0.0), (1, float("nan"))),
)
def test_proxy_pad_spawn_command_rejects_invalid_duration(
    steps: int,
    keep_open_s: float,
) -> None:
    with pytest.raises(ValueError):
        build_proxy_pad_spawn_command(
            _source_command(),
            steps=steps,
            keep_open_s=keep_open_s,
        )
