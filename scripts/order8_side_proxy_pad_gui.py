from __future__ import annotations

"""Inspect surface-following Order-8 Dock-cone micro-pads on one module.

The parent process automatically relaunches this file in the configured Isaac
Lab micromamba environment.  The child authors many small collision-enabled
orange tiles under the two yaw Dock rigid links, initializes one module, and
then advances rendering only.  No grasp, contact rollout, or physics step is
executed.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BACKEND_CONFIG = "configs/env/isaac_lab.yaml"
DEFAULT_PAD_CONFIG = "configs/training/order8_side_proxy_pad_preview.yaml"
CHILD_FLAG = "--order8-side-proxy-pad-isaac-child"
PREVIEW_ROOT_PATH = "/World/Order8SideProxyPadPreview/Holon"
PAD_PRIM_PREFIX = "Order8SideMicroPad"


def _bootstrap_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend-config", default=DEFAULT_BACKEND_CONFIG)
    return parser


def build_isaac_child_command(
    argv: list[str],
    *,
    micromamba_executable: str,
    environment_name: str,
    python_executable: str = "python",
) -> list[str]:
    return [
        str(micromamba_executable),
        "run",
        "-n",
        str(environment_name),
        str(python_executable),
        str(Path(__file__).resolve()),
        CHILD_FLAG,
        *argv,
    ]


def ensure_gui_visualizer(argv: list[str]) -> list[str]:
    """Select Kit unless the caller explicitly chose a visualizer mode."""

    if "--viz" in argv or "--headless" in argv:
        return list(argv)
    return [*argv, "--viz", "kit"]


def _launch_isaac_child(argv: list[str]) -> int:
    from amsrr.simulation.isaac_lab_backend import load_isaac_lab_backend_config

    bootstrap, _ = _bootstrap_parser().parse_known_args(argv)
    backend = load_isaac_lab_backend_config(str(_repo_path(bootstrap.backend_config)))
    micromamba = shutil.which("micromamba")
    if micromamba is None:
        fallback = Path.home() / ".local" / "bin" / "micromamba"
        if not fallback.is_file():
            raise FileNotFoundError(
                "micromamba is required to launch the configured Isaac Lab environment"
            )
        micromamba = str(fallback)
    command = build_isaac_child_command(
        ensure_gui_visualizer(argv),
        micromamba_executable=str(Path(micromamba).resolve()),
        environment_name=str(backend.micromamba_env),
    )
    return int(subprocess.run(command, cwd=REPO_ROOT, check=False).returncode)


def _child_parser(AppLauncher) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Show the surface-following Order-8 Dock-cone micro-pads on one Holon "
            "module. The viewer does not advance physics."
        )
    )
    parser.add_argument(CHILD_FLAG, action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend-config", default=DEFAULT_BACKEND_CONFIG)
    parser.add_argument("--pad-config", default=DEFAULT_PAD_CONFIG)
    parser.add_argument(
        "--focus-link",
        choices=("overview", "yaw_dock_mech1", "yaw_dock_mech2"),
        default="overview",
        help="Initial camera target; the viewport remains freely orbitable.",
    )
    parser.add_argument(
        "--keep-open-s",
        type=float,
        default=0.0,
        help=(
            "Viewer duration in wall-clock seconds. Zero keeps it open until "
            "the Kit window is closed."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _run_isaac_child(argv: list[str]) -> int:
    from isaaclab.app import AppLauncher

    parser = _child_parser(AppLauncher)
    args = parser.parse_args(argv)
    if not args.order8_side_proxy_pad_isaac_child:
        parser.error(f"missing internal flag {CHILD_FLAG}")
    if args.keep_open_s < 0.0:
        parser.error("--keep-open-s must be non-negative")
    launcher = AppLauncher(args)
    simulation_app = launcher.app
    try:
        return _build_and_show_preview(args, simulation_app)
    finally:
        simulation_app.close()


def _build_and_show_preview(args: argparse.Namespace, simulation_app) -> int:
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    from amsrr.simulation.isaac_lab_backend import load_isaac_lab_backend_config
    from amsrr.simulation.order8_side_proxy_pad import (
        build_order8_side_proxy_pad_specs,
        load_order8_side_proxy_pad_preview_config,
    )

    backend = load_isaac_lab_backend_config(str(_repo_path(args.backend_config)))
    usd_path = _repo_path(backend.generated_usd_path)
    urdf_path = _repo_path(backend.holon_urdf_path)
    if not usd_path.is_file():
        raise FileNotFoundError(
            f"generated Holon USD is missing: {usd_path}; regenerate it first"
        )
    pad_config = load_order8_side_proxy_pad_preview_config(
        _repo_path(args.pad_config)
    )
    pad_specs = build_order8_side_proxy_pad_specs(
        urdf_path=urdf_path,
        config=pad_config,
    )

    sim_utils.create_new_stage()
    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=0.02, device=str(args.device))
    )
    dome = sim_utils.DomeLightCfg(intensity=1800.0, color=(1.0, 1.0, 1.0))
    dome.func("/World/Order8SideProxyPadPreview/DomeLight", dome)
    key = sim_utils.DistantLightCfg(
        intensity=2500.0,
        color=(1.0, 0.95, 0.90),
        angle=0.45,
    )
    key.func("/World/Order8SideProxyPadPreview/KeyLight", key)
    spawn = sim_utils.UsdFileCfg(
        usd_path=str(usd_path),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=0.1,
            enable_gyroscopic_forces=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.0,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    )
    spawn.func(PREVIEW_ROOT_PATH, spawn, translation=(0.0, 0.0, 0.0))

    body_paths = _resolve_link_body_paths(
        stage=sim.stage,
        root_path=PREVIEW_ROOT_PATH,
        link_ids=tuple(spec.link_id for spec in pad_specs),
        Usd=Usd,
        UsdPhysics=UsdPhysics,
    )
    material = _create_pad_material(
        stage=sim.stage,
        color=pad_config.display_color_rgb,
        opacity=pad_config.display_opacity,
        Gf=Gf,
        Sdf=Sdf,
        UsdShade=UsdShade,
    )
    pad_paths: dict[str, list[str]] = {
        link_id: [] for link_id in body_paths
    }
    for index, spec in enumerate(pad_specs):
        body_path = body_paths[spec.link_id]
        pad_path = f"{body_path}/{PAD_PRIM_PREFIX}_{index + 1}"
        cube = UsdGeom.Cube.Define(sim.stage, Sdf.Path(pad_path))
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*pad_config.display_color_rgb)])
        cube.CreateDisplayOpacityAttr([float(pad_config.display_opacity)])
        xformable = UsdGeom.Xformable(cube.GetPrim())
        xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*spec.center_local)
        )
        qx, qy, qz, qw = spec.orientation_local_xyzw
        xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(float(qw), Gf.Vec3d(float(qx), float(qy), float(qz)))
        )
        xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*spec.size_m)
        )
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="full",
        )
        if cube.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                f"side proxy pad must inherit rigid motion from {body_path}: {pad_path}"
            )
        pad_paths[spec.link_id].append(pad_path)

    _set_preview_camera(
        sim=sim,
        stage=sim.stage,
        focus_link=str(args.focus_link),
        pad_paths=pad_paths,
        Gf=Gf,
        UsdGeom=UsdGeom,
    )
    # Reset initializes the referenced articulation.  No sim.step() call is
    # made after this point; the hold loop updates rendering only.
    sim.reset()
    simulation_app.update()
    counts_by_link = {
        link_id: len(paths) for link_id, paths in sorted(pad_paths.items())
    }
    tangential_sizes = [value for spec in pad_specs for value in spec.size_m[:2]]
    summary = {
        "version": pad_config.version,
        "acceptance_eligible": False,
        "visual_approval_recorded": pad_config.visual_approval_recorded,
        "contact_runtime_enabled": pad_config.contact_runtime_enabled,
        "physics_steps_after_reset": 0,
        "module_count": 1,
        "pad_count": len(pad_specs),
        "pad_count_by_link": counts_by_link,
        "pad_tangential_size_range_mm": [
            1000.0 * min(tangential_sizes),
            1000.0 * max(tangential_sizes),
        ],
        "pad_thickness_mm": 1000.0 * pad_config.thickness_m,
        "mesh_clearance_mm": 1000.0 * pad_config.mesh_clearance_m,
        "maximum_local_surface_fit_gap_mm": 1000.0
        * max(spec.surface_fit_max_gap_m for spec in pad_specs),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(
        "[order8-side-proxy-pad-gui] Orange translucent micro-tiles follow only "
        "the authored Dock conical collision surface. Physics is paused; orbit/zoom "
        "the viewport and close Kit when inspection is complete.",
        flush=True,
    )
    started = time.monotonic()
    while simulation_app.is_running():
        if args.keep_open_s > 0.0 and time.monotonic() - started >= args.keep_open_s:
            break
        simulation_app.update()
        time.sleep(1.0 / 60.0)
    return 0


def _resolve_link_body_paths(*, stage, root_path, link_ids, Usd, UsdPhysics):
    wanted = set(link_ids)
    found: dict[str, list[str]] = {link_id: [] for link_id in wanted}
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        name = str(prim.GetName())
        if (
            name in wanted
            and path.startswith(str(root_path).rstrip("/") + "/")
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ):
            found[name].append(path)
    invalid = {key: value for key, value in found.items() if len(value) != 1}
    if invalid:
        raise RuntimeError(
            "single-module side-proxy preview could not resolve unique Dock "
            f"rigid bodies: {invalid}"
        )
    return {key: value[0] for key, value in found.items()}


def _create_pad_material(*, stage, color, opacity, Gf, Sdf, UsdShade):
    material_path = "/World/Order8SideProxyPadPreview/Looks/ProxyPadOrange"
    material = UsdShade.Material.Define(stage, Sdf.Path(material_path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{material_path}/Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    return material


def _set_preview_camera(*, sim, stage, focus_link, pad_paths, Gf, UsdGeom):
    if focus_link == "overview":
        sim.set_camera_view(eye=[0.65, 0.55, 0.42], target=[0.12, 0.0, 0.02])
        return
    paths = pad_paths[focus_link]
    if not paths:
        raise RuntimeError(f"no micro-pad paths were authored for {focus_link}")
    cache = UsdGeom.XformCache()
    centers = [
        cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation()
        for path in paths
    ]
    target = [
        sum(float(center[axis]) for center in centers) / len(centers)
        for axis in range(3)
    ]
    eye = [target[0] + 0.28, target[1] + 0.28, target[2] + 0.18]
    sim.set_camera_view(eye=eye, target=target)


def _repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if CHILD_FLAG in values:
        return _run_isaac_child(values)
    return _launch_isaac_child(values)


if __name__ == "__main__":
    raise SystemExit(main())
