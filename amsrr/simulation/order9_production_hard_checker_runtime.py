from __future__ import annotations

"""Owned production binding for the hybrid Order 9 ``C_H`` gate.

The lightweight QP is process-local.  The counterfactual rollout is owned by
one persistent Isaac process per immutable (topology, object geometry,
``pi_L`` checkpoint) bucket.  This module builds that bucket, authenticates its
descriptor before the first proposal, and closes it as one resource.
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from amsrr.feasibility.contact_wrench_trajectory import (
    ContactWrenchTrajectoryFeasibilityChecker,
)
from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.task_spec import GeometrySpec, GeometryType, ObjectSpec, TaskSpec
from amsrr.simulation.order9_shadow_runtime import ImmutableMainStateShadowBackend
from amsrr.simulation.order9_shadow_worker import (
    JsonLineSubprocessShadowTransport,
    Order9MainStateExporter,
    Order9ShadowWorkerTransport,
    PersistentIsaacShadowDriver,
)
from amsrr.training.order9_curriculum import (
    Order9LearningConfig,
    load_order9_learning_config,
)
from amsrr.training.order9_hard_checker import build_order9_production_hard_checker
from amsrr.training.order9_randomization import Order9RandomizationSample
from amsrr.utils.hashing import hash_file, stable_hash


ORDER9_PRODUCTION_HARD_CHECKER_RUNTIME_VERSION = (
    "order9_production_hard_checker_runtime_v1"
)


@dataclass(frozen=True)
class Order9ShadowBucketIdentity:
    task_id: str
    task_hash: str
    topology_structural_hash: str
    morphology_hash: str
    object_id: str
    object_geometry_type: str
    object_size_m: tuple[float, float, float]
    object_mass_kg: float
    object_inertia_body: tuple[float, float, float, float, float, float]
    object_friction: float
    selected_gripper_friction: float
    contact_stiffness_n_per_m: float
    contact_damping_n_s_per_m: float
    support_top_z_m: float

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "task_hash",
            "topology_structural_hash",
            "morphology_hash",
            "object_id",
            "object_geometry_type",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"Order9 shadow bucket {name} must be non-empty")
        for name in ("task_hash", "topology_structural_hash", "morphology_hash"):
            _require_sha256(str(getattr(self, name)), name)
        if self.object_geometry_type not in {"box", "sphere", "cylinder", "capsule"}:
            raise ValueError("Order9 shadow bucket geometry is unsupported")
        if len(self.object_size_m) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in self.object_size_m
        ):
            raise ValueError("Order9 shadow bucket object size must be positive")
        if len(self.object_inertia_body) != 6 or any(
            not math.isfinite(value) for value in self.object_inertia_body
        ):
            raise ValueError("Order9 shadow bucket inertia must contain six finite values")
        for name in (
            "object_mass_kg",
            "object_friction",
            "selected_gripper_friction",
            "contact_stiffness_n_per_m",
            "contact_damping_n_s_per_m",
            "support_top_z_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Order9 shadow bucket {name} must be positive")

    @property
    def bucket_hash(self) -> str:
        return stable_hash(self.__dict__)


class Order9ProductionHardCheckerRuntime:
    """One authenticated worker plus the checker that consumes its evidence."""

    def __init__(
        self,
        *,
        checker: ContactWrenchTrajectoryFeasibilityChecker,
        backend: ImmutableMainStateShadowBackend,
        driver: PersistentIsaacShadowDriver,
        transport: Order9ShadowWorkerTransport,
        bucket: Order9ShadowBucketIdentity,
        worker_descriptor: Mapping[str, Any],
    ) -> None:
        self.checker = checker
        self.backend = backend
        self.driver = driver
        self.transport = transport
        self.bucket = bucket
        self.worker_descriptor = dict(worker_descriptor)
        self._closed = False

    @property
    def runtime_version(self) -> str:
        return (
            f"{ORDER9_PRODUCTION_HARD_CHECKER_RUNTIME_VERSION}:"
            f"{self.backend.backend_version}"
        )

    def close(self) -> None:
        if self._closed:
            return
        close = getattr(self.transport, "close", None)
        if callable(close):
            close()
        self._closed = True

    def __enter__(self) -> "Order9ProductionHardCheckerRuntime":
        if self._closed:
            raise RuntimeError("Order9 production hard-checker runtime is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def order9_shadow_bucket_from_sample(
    sample: Order9RandomizationSample,
    morphology_graph: MorphologyGraph,
    *,
    support_top_z_m: float,
) -> Order9ShadowBucketIdentity:
    sample.validate()
    morphology_graph.validate()
    task = sample.task_spec
    target = _target_object(task)
    geometry = _geometry(task, target.geometry_id)
    if target.mass_kg is None or target.inertia_kgm2 is None or target.friction is None:
        raise SchemaValidationError(
            "Order9 production object requires mass, inertia, and friction"
        )
    if tuple(float(value) for value in target.inertia_kgm2) != tuple(
        float(value) for value in sample.true_mass_properties.inertia_kgm2
    ):
        raise SchemaValidationError(
            "Order9 randomized task inertia differs from its true mass properties"
        )
    return Order9ShadowBucketIdentity(
        task_id=task.task_id,
        task_hash=task.stable_hash(),
        topology_structural_hash=morphology_structural_hash(morphology_graph),
        morphology_hash=morphology_graph.stable_hash(),
        object_id=target.object_id,
        object_geometry_type=geometry.geometry_type.value,
        object_size_m=_worker_size(geometry),
        object_mass_kg=float(target.mass_kg),
        object_inertia_body=tuple(float(value) for value in target.inertia_kgm2),  # type: ignore[arg-type]
        object_friction=float(target.friction),
        selected_gripper_friction=float(sample.selected_gripper_friction),
        contact_stiffness_n_per_m=float(sample.contact_stiffness_n_per_m),
        contact_damping_n_s_per_m=float(sample.contact_damping_n_s_per_m),
        support_top_z_m=float(support_top_z_m),
    )


def write_order9_shadow_bucket_morphology(
    directory: str | Path,
    morphology_graph: MorphologyGraph,
) -> Path:
    """Persist an immutable graph input without replacing conflicting bytes."""

    morphology_graph.validate()
    structural_hash = morphology_structural_hash(morphology_graph)
    target = Path(directory).resolve() / f"morphology_{structural_hash}.json"
    payload = morphology_graph.to_json(indent=2) + "\n"
    if target.exists():
        existing = MorphologyGraph.from_json(target.read_text(encoding="utf-8"))
        if existing.stable_hash() != morphology_graph.stable_hash():
            raise FileExistsError(
                "Order9 topology bucket path already contains a different graph"
            )
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Refuse a race with a different producer instead of overwriting it.
        if target.exists():
            raise FileExistsError(f"Order9 topology bucket appeared concurrently: {target}")
        os.link(temporary_name, target)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return target


def build_order9_shadow_worker_command(
    *,
    repository_root: str | Path,
    config_path: str | Path,
    worker_script: str | Path,
    micromamba_environment: str,
    device: str,
    pi_l_checkpoint_path: str | Path,
    pi_l_checkpoint_sha256: str,
    robot_usd_path: str | Path,
    morphology_graph_path: str | Path,
    bucket: Order9ShadowBucketIdentity,
    control_dt_s: float,
    micromamba_executable: str | Path | None = None,
) -> list[str]:
    repository = Path(repository_root).resolve()
    executable = (
        str(Path(micromamba_executable).resolve())
        if micromamba_executable is not None
        else shutil.which("micromamba")
    )
    if not executable:
        raise RuntimeError("micromamba executable is unavailable")
    if not micromamba_environment:
        raise ValueError("Order9 micromamba environment must be non-empty")
    if not math.isfinite(float(control_dt_s)) or control_dt_s <= 0.0:
        raise ValueError("Order9 shadow worker control_dt_s must be positive")
    command = [
        executable,
        "run",
        "-n",
        str(micromamba_environment),
        "--",
        "python",
        str(_resolved_file(worker_script, repository)),
        "--viz",
        "none",
        "--device",
        str(device),
        "--config",
        str(_resolved_file(config_path, repository)),
        "--pi-l-checkpoint",
        str(_resolved_file(pi_l_checkpoint_path, repository)),
        "--pi-l-checkpoint-sha256",
        pi_l_checkpoint_sha256,
        "--robot-usd",
        str(_resolved_file(robot_usd_path, repository)),
        "--morphology-graph-json",
        str(_resolved_file(morphology_graph_path, repository)),
        "--object-id",
        bucket.object_id,
        "--object-geometry",
        bucket.object_geometry_type,
        "--object-size",
        *(format(value, ".17g") for value in bucket.object_size_m),
        "--object-mass-kg",
        format(bucket.object_mass_kg, ".17g"),
        "--object-friction",
        format(bucket.object_friction, ".17g"),
        "--selected-gripper-friction",
        format(bucket.selected_gripper_friction, ".17g"),
        "--contact-stiffness",
        format(bucket.contact_stiffness_n_per_m, ".17g"),
        "--contact-damping",
        format(bucket.contact_damping_n_s_per_m, ".17g"),
        "--support-top-z",
        format(bucket.support_top_z_m, ".17g"),
        "--dt",
        format(float(control_dt_s), ".17g"),
    ]
    return command


def bind_order9_production_hard_checker(
    *,
    config: Order9LearningConfig,
    state_exporter: Order9MainStateExporter,
    transport: Order9ShadowWorkerTransport,
    pi_l_checkpoint_sha256: str,
    bucket: Order9ShadowBucketIdentity,
) -> Order9ProductionHardCheckerRuntime:
    """Authenticate an already-created transport, then expose production ``C_H``."""

    config.validate()
    _require_sha256(pi_l_checkpoint_sha256, "pi_l_checkpoint_sha256")
    response = dict(transport.request("describe", {}))
    if response.get("operation") != "describe" or response.get("accepted") is not True:
        raise RuntimeError("Order9 shadow worker rejected identity handshake")
    if response.get("pi_l_checkpoint_sha256") != pi_l_checkpoint_sha256:
        raise RuntimeError("Order9 shadow worker checkpoint identity mismatch")
    worker_version = str(response.get("worker_version", ""))
    descriptor = response.get("descriptor")
    if not worker_version or not isinstance(descriptor, Mapping):
        raise RuntimeError("Order9 shadow worker descriptor is incomplete")
    _verify_worker_descriptor(
        descriptor,
        bucket=bucket,
        config=config,
        checkpoint_sha256=pi_l_checkpoint_sha256,
    )
    driver = PersistentIsaacShadowDriver(
        state_exporter=state_exporter,
        transport=transport,
        pi_l_checkpoint_sha256=pi_l_checkpoint_sha256,
        worker_version=worker_version,
    )
    backend = ImmutableMainStateShadowBackend(driver)
    checker = build_order9_production_hard_checker(
        backend,
        config=config.hard_checker,
    )
    return Order9ProductionHardCheckerRuntime(
        checker=checker,
        backend=backend,
        driver=driver,
        transport=transport,
        bucket=bucket,
        worker_descriptor=descriptor,
    )


def launch_order9_production_hard_checker(
    *,
    config: Order9LearningConfig,
    config_path: str | Path,
    state_exporter: Order9MainStateExporter,
    sample: Order9RandomizationSample,
    morphology_graph: MorphologyGraph,
    robot_usd_path: str | Path,
    pi_l_checkpoint_path: str | Path,
    pi_l_checkpoint_sha256: str,
    repository_root: str | Path,
    morphology_bucket_directory: str | Path,
    worker_script: str | Path = "scripts/order9_isaac_shadow_worker.py",
    micromamba_environment: str = "isaaclab3",
    micromamba_executable: str | Path | None = None,
    timeout_s: float = 120.0,
    support_top_z_m: float | None = None,
) -> Order9ProductionHardCheckerRuntime:
    """Launch and own the exact persistent worker required by one rollout bucket."""

    config.validate()
    repository = Path(repository_root).resolve()
    resolved_config_path = _resolved_file(config_path, repository)
    file_config = load_order9_learning_config(resolved_config_path)
    if file_config.stable_hash() != config.stable_hash():
        raise SchemaValidationError(
            "Order9 in-memory and worker-file curriculum configurations differ"
        )
    checkpoint = _resolved_file(pi_l_checkpoint_path, repository)
    actual_checkpoint_hash = hash_file(checkpoint)
    if actual_checkpoint_hash != pi_l_checkpoint_sha256:
        raise SchemaValidationError("Order9 pi_L checkpoint byte hash mismatch")
    graph_path = write_order9_shadow_bucket_morphology(
        morphology_bucket_directory,
        morphology_graph,
    )
    bucket = order9_shadow_bucket_from_sample(
        sample,
        morphology_graph,
        support_top_z_m=(
            config.randomization.support_top_z_m
            if support_top_z_m is None
            else support_top_z_m
        ),
    )
    command = build_order9_shadow_worker_command(
        repository_root=repository,
        config_path=resolved_config_path,
        worker_script=worker_script,
        micromamba_environment=micromamba_environment,
        device=config.production_runtime.device,
        pi_l_checkpoint_path=checkpoint,
        pi_l_checkpoint_sha256=pi_l_checkpoint_sha256,
        robot_usd_path=robot_usd_path,
        morphology_graph_path=graph_path,
        bucket=bucket,
        control_dt_s=config.hard_checker.shadow_control_dt_s,
        micromamba_executable=micromamba_executable,
    )
    transport = JsonLineSubprocessShadowTransport(
        command,
        cwd=repository,
        timeout_s=timeout_s,
        environment={"PYTHONPATH": str(repository)},
    )
    try:
        return bind_order9_production_hard_checker(
            config=config,
            state_exporter=state_exporter,
            transport=transport,
            pi_l_checkpoint_sha256=pi_l_checkpoint_sha256,
            bucket=bucket,
        )
    except BaseException:
        transport.close()
        raise


def _verify_worker_descriptor(
    descriptor: Mapping[str, Any],
    *,
    bucket: Order9ShadowBucketIdentity,
    config: Order9LearningConfig,
    checkpoint_sha256: str,
) -> None:
    exact = {
        "topology_structural_hash": bucket.topology_structural_hash,
        "pi_l_checkpoint_sha256": checkpoint_sha256,
    }
    for name, expected in exact.items():
        if descriptor.get(name) != expected:
            raise RuntimeError(f"Order9 shadow worker descriptor has wrong {name}")
    _require_close(
        descriptor.get("control_dt_s"),
        config.hard_checker.shadow_control_dt_s,
        "control_dt_s",
    )
    _require_close(
        descriptor.get("maximum_horizon_s"),
        config.hard_checker.shadow_rollout_horizon_s,
        "maximum_horizon_s",
    )
    scene = descriptor.get("scene")
    if not isinstance(scene, Mapping):
        raise RuntimeError("Order9 shadow worker descriptor has no scene identity")
    scene_exact = {
        "object_id": bucket.object_id,
        "object_geometry_type": bucket.object_geometry_type,
    }
    for name, expected in scene_exact.items():
        if scene.get(name) != expected:
            raise RuntimeError(f"Order9 shadow worker scene has wrong {name}")
    if not _same_float_sequence(scene.get("object_size_m"), bucket.object_size_m):
        raise RuntimeError("Order9 shadow worker scene has wrong object_size_m")
    if not _same_float_sequence(
        scene.get("object_inertia_body"), bucket.object_inertia_body
    ):
        raise RuntimeError("Order9 shadow worker scene has wrong object inertia")
    for name in (
        "object_mass_kg",
        "object_friction",
        "selected_gripper_friction",
        "contact_stiffness_n_per_m",
        "contact_damping_n_s_per_m",
    ):
        _require_close(scene.get(name), getattr(bucket, name), name)
    center = scene.get("support_center_world_m")
    half = scene.get("support_half_extents_m")
    if not (
        isinstance(center, Sequence)
        and not isinstance(center, (str, bytes))
        and len(center) == 3
        and isinstance(half, Sequence)
        and not isinstance(half, (str, bytes))
        and len(half) == 3
    ):
        raise RuntimeError("Order9 shadow worker support identity is incomplete")
    _require_close(
        float(center[2]) + float(half[2]),
        bucket.support_top_z_m,
        "support_top_z_m",
    )


def _target_object(task: TaskSpec) -> ObjectSpec:
    target_ids = {
        goal.target_entity_id
        for goal in task.goals
        if goal.goal_type == "object_pose" and goal.target_entity_id is not None
    }
    matches = [obj for obj in task.scene.objects if obj.object_id in target_ids]
    if len(matches) != 1:
        raise SchemaValidationError(
            "Order9 production task requires exactly one object_pose target object"
        )
    return matches[0]


def _geometry(task: TaskSpec, geometry_id: str) -> GeometrySpec:
    matches = [item for item in task.scene.geometry_library if item.geometry_id == geometry_id]
    if len(matches) != 1:
        raise SchemaValidationError("Order9 target object geometry identity is ambiguous")
    return matches[0]


def _worker_size(geometry: GeometrySpec) -> tuple[float, float, float]:
    params = geometry.primitive_params or {}
    sx, sy, sz = (float(value) for value in geometry.scale)
    if geometry.geometry_type == GeometryType.BOX:
        raw = params.get("size_m")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 3:
            raise SchemaValidationError("Order9 box geometry requires size_m")
        return tuple(float(value) * scale for value, scale in zip(raw, (sx, sy, sz)))  # type: ignore[return-value]
    radius = float(params.get("radius_m", 0.0))
    if radius <= 0.0:
        raise SchemaValidationError("Order9 round geometry requires radius_m")
    if geometry.geometry_type == GeometryType.SPHERE:
        if not math.isclose(sx, sy) or not math.isclose(sy, sz):
            raise SchemaValidationError(
                "Order9 Isaac worker requires isotropically scaled spheres"
            )
        diameter = 2.0 * radius * sx
        return (diameter, diameter, diameter)
    if geometry.geometry_type in {GeometryType.CYLINDER, GeometryType.CAPSULE}:
        if not math.isclose(sx, sy):
            raise SchemaValidationError(
                "Order9 Isaac worker requires round cylinder/capsule cross-sections"
            )
        height = float(params.get("height_m", 0.0))
        if height <= 0.0:
            raise SchemaValidationError("Order9 round geometry requires height_m")
        return (2.0 * radius * sx, 2.0 * radius * sy, height * sz)
    raise SchemaValidationError(
        "Order9 production shadow currently supports primitive object buckets only"
    )


def _resolved_file(path: str | Path, repository: Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = repository / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _same_float_sequence(value: object, expected: Sequence[float]) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    return len(value) == len(expected) and all(
        math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-12)
        for left, right in zip(value, expected)
    )


def _require_close(value: object, expected: float, name: str) -> None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Order9 shadow worker descriptor lacks {name}") from exc
    if not math.isclose(parsed, float(expected), rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise RuntimeError(f"Order9 shadow worker descriptor has wrong {name}")


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"Order9 {name} must be a SHA-256 digest")


__all__ = [
    "ORDER9_PRODUCTION_HARD_CHECKER_RUNTIME_VERSION",
    "Order9ProductionHardCheckerRuntime",
    "Order9ShadowBucketIdentity",
    "bind_order9_production_hard_checker",
    "build_order9_shadow_worker_command",
    "launch_order9_production_hard_checker",
    "order9_shadow_bucket_from_sample",
    "write_order9_shadow_bucket_morphology",
]
