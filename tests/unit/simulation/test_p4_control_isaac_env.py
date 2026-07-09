from __future__ import annotations

from amsrr.simulation import (
    IsaacLabBackend,
    IsaacLabBackendConfig,
    P4ControlIsaacEnv,
    P4ControlLowLevelEnvConfig,
    load_isaac_lab_backend_config,
    load_p4_control_low_level_env_config,
)


def test_p4_control_low_level_config_loader() -> None:
    backend_config, env_config = load_p4_control_low_level_env_config("configs/training/p4_control_low_level.yaml")

    assert backend_config.micromamba_env == "isaaclab3"
    assert backend_config.holon_urdf_path == "assets/robots/holon/holon.urdf"
    assert backend_config.generated_usd_path == "artifacts/isaac/robots/holon/holon/holon.usda"
    assert backend_config.rotor_force_application == "wrench_composer"
    assert env_config.position_error_threshold_m == 0.20
    assert env_config.fixed_morphology_module_count == 2


def test_isaac_backend_probe_and_conversion_command_are_config_driven() -> None:
    config = load_isaac_lab_backend_config("configs/env/isaac_lab.yaml")
    backend = IsaacLabBackend(config)

    availability = backend.availability()
    command = backend.urdf_conversion_command()

    assert availability.metadata["backend_version"] == "isaac_lab_backend_v1"
    assert availability.urdf_exists is True
    assert "convert_urdf.py" in command[2]
    assert command[-2].endswith("assets/robots/holon/holon.urdf")
    assert command[-1].endswith("artifacts/isaac/robots/holon")


def test_p4_control_smoke_scenarios_are_deterministic() -> None:
    env = P4ControlIsaacEnv(config=P4ControlLowLevelEnvConfig())

    scenarios = env.smoke_scenarios()

    assert [scenario.smoke_name for scenario in scenarios] == [
        "single_module_hover",
        "fixed_morphology_hover",
        "fixed_morphology_waypoint",
    ]
    assert scenarios[0].module_count == 1
    assert scenarios[1].module_count == 2
    assert scenarios[2].waypoint_tracking is True


def test_p4_control_dry_run_smokes_skip_without_completion_claim() -> None:
    env = P4ControlIsaacEnv(config=P4ControlLowLevelEnvConfig())

    results = env.run_smokes(dry_run=True)

    assert len(results) == 3
    assert all(result.skipped for result in results)
    assert all(not result.attempted for result in results)
    assert all(not result.isaac_backed for result in results)
    assert {result.skip_reason for result in results} == {"dry_run"}


def test_p4_control_real_smokes_skip_when_backend_missing() -> None:
    backend = IsaacLabBackend(
        IsaacLabBackendConfig(
            isaaclab_path="/tmp/amsrr_missing_isaaclab",
            holon_urdf_path="/tmp/amsrr_missing_holon.urdf",
        )
    )
    env = P4ControlIsaacEnv(config=P4ControlLowLevelEnvConfig(), backend=backend)

    results = env.run_smokes(dry_run=False)

    assert len(results) == 3
    assert all(result.skipped for result in results)
    assert all("isaaclab_path_missing" in str(result.skip_reason) for result in results)
