from __future__ import annotations

from amsrr.simulation import (
    IsaacLabBackend,
    IsaacLabBackendConfig,
    IsaacLabAvailability,
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
    assert env_config.hover_target_height_m == 0.5
    assert env_config.fixed_morphology_module_count == 2


def test_isaac_backend_probe_and_conversion_command_are_config_driven() -> None:
    config = load_isaac_lab_backend_config("configs/env/isaac_lab.yaml")
    backend = IsaacLabBackend(config)

    availability = backend.availability()
    command = backend.urdf_conversion_command()
    spawn_command = backend.holon_spawn_probe_command(
        config_path="configs/env/isaac_lab.yaml",
        generated_usd_dir="/tmp/amsrr_isaac_holon_spawn",
        generated_usd_path="/tmp/amsrr_isaac_holon_spawn/holon/holon.usda",
        steps=3,
    )
    command_probe = backend.holon_command_probe_command(
        config_path="configs/env/isaac_lab.yaml",
        generated_usd_dir="/tmp/amsrr_isaac_holon_spawn",
        generated_usd_path="/tmp/amsrr_isaac_holon_spawn/holon/holon.usda",
        steps=20,
        hover_force_scale=0.25,
        gimbal_target_rad=0.05,
    )
    controller_command_probe = backend.holon_controller_command_probe_command(
        config_path="configs/env/isaac_lab.yaml",
        generated_usd_dir="/tmp/amsrr_isaac_holon_spawn",
        generated_usd_path="/tmp/amsrr_isaac_holon_spawn/holon/holon.usda",
        steps=20,
    )
    single_module_hover_smoke = backend.holon_single_module_hover_smoke_command(
        config_path="configs/env/isaac_lab.yaml",
        generated_usd_dir="/tmp/amsrr_isaac_holon_spawn",
        generated_usd_path="/tmp/amsrr_isaac_holon_spawn/holon/holon.usda",
        steps=600,
        hover_target_height=0.5,
        position_tolerance_m=0.20,
        attitude_tolerance_rad=0.25,
        hold_duration_s=1.0,
    )
    force_convert_hover_smoke = backend.holon_single_module_hover_smoke_command(
        config_path="configs/env/isaac_lab.yaml",
        convert_if_missing=False,
        force_convert=True,
        steps=600,
    )

    assert availability.metadata["backend_version"] == "isaac_lab_backend_v1"
    assert availability.urdf_exists is True
    assert "convert_urdf.py" in command[2]
    assert command[-2].endswith("assets/robots/holon/holon.urdf")
    assert command[-1].endswith("artifacts/isaac/robots/holon")
    assert spawn_command[:2] == command[:2]
    assert spawn_command[2].endswith("scripts/p4_control_holon_spawn_probe.py")
    assert "--convert-if-missing" in spawn_command
    assert "--headless" not in spawn_command
    assert "/tmp/amsrr_isaac_holon_spawn/holon/holon.usda" in spawn_command
    assert command_probe[:3] == spawn_command[:3]
    assert "--hover-force-scale" in command_probe
    assert "0.25" in command_probe
    assert "--gimbal-target-rad" in command_probe
    assert "0.05" in command_probe
    assert controller_command_probe[:3] == spawn_command[:3]
    assert "--controller-command-smoke" in controller_command_probe
    assert single_module_hover_smoke[:3] == spawn_command[:3]
    assert "--single-module-hover-smoke" in single_module_hover_smoke
    assert "--hover-target-height" in single_module_hover_smoke
    assert "0.5" in single_module_hover_smoke
    assert "--hover-position-tolerance-m" in single_module_hover_smoke
    assert "0.2" in single_module_hover_smoke
    assert "--force-convert" in force_convert_hover_smoke
    assert "--convert-if-missing" not in force_convert_hover_smoke


def test_p4_control_smoke_scenarios_are_deterministic() -> None:
    env = P4ControlIsaacEnv(config=P4ControlLowLevelEnvConfig())

    scenarios = env.smoke_scenarios()

    assert [scenario.smoke_name for scenario in scenarios] == [
        "single_module_hover",
        "fixed_morphology_hover",
        "fixed_morphology_waypoint",
    ]
    assert scenarios[0].module_count == 1
    assert scenarios[0].target_pose_world[2] == 0.5
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


def test_p4_control_real_smokes_run_single_module_and_skip_fixed_cases() -> None:
    backend = _SingleModuleHoverBackend(
        {
            "isaac_backed": True,
            "spawn_passed": True,
            "command_probe_passed": True,
            "single_module_hover_smoke": True,
            "single_module_hover_smoke_passed": True,
            "single_module_hover_steps": 200,
            "single_module_hover_requested_steps": 600,
            "single_module_hover_duration_s": 1.0,
            "single_module_hover_hold_time_s": 1.0,
            "single_module_hover_final_position_error_m": 0.014,
            "single_module_hover_final_attitude_error_rad": 0.004,
            "single_module_hover_qp_infeasible_count": 0,
            "single_module_hover_clipped_target_count": 0,
        }
    )
    env = P4ControlIsaacEnv(config=P4ControlLowLevelEnvConfig(), backend=backend)  # type: ignore[arg-type]

    results = env.run_smokes(dry_run=False)

    assert [result.smoke_name for result in results] == [
        "single_module_hover",
        "fixed_morphology_hover",
        "fixed_morphology_waypoint",
    ]
    assert results[0].attempted is True
    assert results[0].passed is True
    assert results[0].isaac_backed is True
    assert results[0].metrics["single_module_hover_final_position_error_m"] == 0.014
    assert results[1].skipped is True
    assert results[1].skip_reason == "real_isaac_execution_not_implemented"
    assert results[2].skipped is True
    assert backend.calls[0]["force_convert"] is True
    assert backend.calls[0]["steps"] == 600
    assert backend.calls[0]["hover_target_height"] == 0.5


class _SingleModuleHoverBackend:
    def __init__(self, report: dict[str, object]) -> None:
        self.report = report
        self.calls: list[dict[str, object]] = []

    def availability(self) -> IsaacLabAvailability:
        return IsaacLabAvailability(
            available=True,
            isaaclab_path_exists=True,
            launch_script_exists=True,
            urdf_exists=True,
            generated_usd_exists=False,
            python_modules_available=True,
            missing_reasons=[],
        )

    def run_holon_single_module_hover_smoke(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return dict(self.report)
