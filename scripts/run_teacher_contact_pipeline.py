#!/usr/bin/env python3
from __future__ import annotations

"""Prepare a naive contact plan or execute it with the real Isaac teacher harness.

prepare generates immutable plan/job artifacts without Isaac. execute consumes
that exact plan, uses no learned actions, and retains independent physical task
and safety evaluation. The protected C3 harness supplies teacher phase timing;
this is NOT evidence of autonomous pi_H or deployable phase selection.
"""

import argparse
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amsrr.policies.naive_contact_planner import (
    NaiveContactPlanReference,
    plan_teacher_contact_trajectory,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_actuator_aware_nominal_preload import (
    Order9ActuatorAwareNominalPreloadConfig,
)
from amsrr.training.order9_c3_nominal_trajectory import (
    load_order9_c3_accepted_nominal_bundle,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_rollout_buckets import (
    load_order9_pi_l_rollout_bucket_manifest,
)
from amsrr.utils.hashing import hash_file, stable_hash

PROTECTED_HARNESS = ROOT / "scripts/order9_vectorized_isaac_rollout.py"
PROTECTED_SHA256 = "e813f950dde32818b7375949bfb58baa983dde760e58da22cfbb79cc33b764c2"


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def prepare(args) -> int:
    from amsrr.training.order9_pi_l_stage_runner import order9_pi_l_collector_command

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config_path = Path(args.config).resolve()
    config = load_order9_learning_config(config_path)
    runtime = config.production_runtime
    physical = build_physical_model_from_config(ROOT / runtime.robot_model_config_path)
    manifest_path = Path(args.bucket_manifest).resolve()
    manifest = load_order9_pi_l_rollout_bucket_manifest(manifest_path)
    matches = [b for b in manifest.buckets if b.bucket_id == args.bucket_id]
    if len(matches) != 1:
        raise ValueError("choose exactly one existing teacher bucket")
    bucket = matches[0]
    task_path = manifest_path.parent / bucket.task_spec_path
    if hash_file(task_path) != bucket.task_spec_sha256:
        raise ValueError("teacher task bytes changed")
    task = TaskSpec.from_json(task_path.read_text())
    binding = bucket.metadata["accepted_nominal_trajectory"]
    bundle = load_order9_c3_accepted_nominal_bundle(
        ROOT / binding["set_manifest_path"],
        bucket_id=bucket.bucket_id,
        repository_root=ROOT,
        expected_set_sha256=binding["set_manifest_sha256"],
        expected_artifact_sha256=binding["artifact_sha256"],
        expected_structural_hash=bucket.structural_hash,
        expected_task_spec_sha256=bucket.task_spec_sha256,
        expected_physical_model_hash=physical.stable_hash(),
    )
    print(f"PLANNING {bucket.bucket_id} modules={bucket.module_count}", flush=True)
    plan = plan_teacher_contact_trajectory(
        bundle=bundle,
        physical_model=physical,
        task=task,
        object_mass_kg=bucket.estimated_mass_kg,
        contact_friction=bucket.selected_gripper_friction,
        contact_stiffness_n_per_m=bucket.contact_stiffness_n_per_m,
        preload_config=Order9ActuatorAwareNominalPreloadConfig(
            minimum_inward_lead_m=runtime.c3_virtual_contact_inward_lead_m,
            maximum_inward_lead_m=runtime.c3_actuator_aware_nominal_preload_maximum_m,
            inward_lead_quantization_m=runtime.c3_actuator_aware_nominal_preload_quantization_m,
            support_safety_factor=runtime.c3_actuator_aware_nominal_preload_support_safety_factor,
            maximum_peak_effort_utilization=runtime.c3_actuator_aware_nominal_preload_maximum_peak_utilization,
            minimum_anchor_force_fraction=runtime.c3_actuator_aware_nominal_preload_minimum_anchor_fraction,
            model_error_margin_m=runtime.c3_actuator_aware_nominal_preload_model_error_margin_m,
        ),
        deadline_s=args.planning_timeout_s,
    )
    plan_path = output / "plan.json"
    _write_new(plan_path, plan.to_dict())
    command = order9_pi_l_collector_command(
        python_executable=sys.executable,
        repository_root=ROOT,
        config_path=config_path,
        stage_id="c3_pi_l_ppo_arbitrary_morphology",
        parent_checkpoint_path=checkpoint,
        parent_checkpoint_sha256=hash_file(checkpoint),
        c3_reset_bank_path=output / "teacher_reset_bank.pt",
        generation_id=f"naive_contact:{bucket.bucket_id}",
        output_raw_path=output / "rollout.pt",
        bucket=bucket,
        bucket_manifest_path=manifest_path,
        c3_action_contract="contact_space_projected_policy_command",
    )
    command += [
        "--num-envs",
        str(args.episodes),
        "--rollout-steps",
        str(args.rollout_steps),
        "--evaluation-jsonl",
        str(output / "episodes.jsonl"),
        "--evaluation-episode-count",
        str(args.episodes),
        "--formal-phase-zero-start",
        "--diagnostic-nominal-qpid-only",
        "--virtual-contact-lead-mm",
        "0",
        "--diagnostic-collision-evidence",
        "--no-tensorboard",
    ]
    _write_new(
        output / "job.json",
        {
            "version": "teacher_contact_isaac_job_v1",
            "bucket_id": bucket.bucket_id,
            "module_count": bucket.module_count,
            "plan_path": str(plan_path),
            "plan_sha256": hash_file(plan_path),
            "plan_id": plan.plan_id,
            "config_sha256": hash_file(config_path),
            "argv": command[2:],
            "source_hashes": {
                str(p.relative_to(ROOT)): hash_file(p)
                for p in (
                    ROOT / "amsrr/policies/naive_contact_planner.py",
                    ROOT / "amsrr/simulation/teacher_contact_execution.py",
                    Path(__file__),
                )
            },
            "teacher_phase_supervision": "privileged_physical_outcome",
            "learned_action_enabled": False,
            "autonomous_request_runtime": False,
        },
    )
    print(
        json.dumps(
            {
                "plan_id": plan.plan_id,
                "preload": plan.evidence["preload"],
                "checks": plan.evidence["checks"],
                "job": str(output / "job.json"),
            }
        ),
        flush=True,
    )
    return 0


def execute(args) -> int:
    from amsrr.simulation.teacher_contact_execution import TeacherContactController

    job_path = Path(args.job).resolve()
    job = json.loads(job_path.read_text())
    output = job_path.parent
    if (output / "result.json").exists() or (output / "rollout.pt").exists():
        raise FileExistsError(
            "execution evidence already exists; use a new prepared job"
        )
    for relative, digest in job["source_hashes"].items():
        if hash_file(ROOT / relative) != digest:
            raise ValueError(f"prepared job source changed: {relative}")
    if hash_file(PROTECTED_HARNESS) != PROTECTED_SHA256:
        raise ValueError("protected Isaac harness changed")
    path = Path(job["plan_path"])
    if hash_file(path) != job["plan_sha256"]:
        raise ValueError("compiled contact plan bytes changed")
    plan = json.loads(path.read_text())
    if stable_hash(plan) != job["plan_id"]:
        raise ValueError("compiled contact plan identity changed")
    arguments = job["argv"]
    config_path = Path(arguments[arguments.index("--config") + 1])
    if hash_file(config_path) != job["config_sha256"]:
        raise ValueError("prepared job configuration changed")
    phases = {
        k: ContactWrenchTrajectory.from_dict(v)
        for k, v in plan["phase_trajectories"].items()
    }
    # JSON keys are sorted for hashing; the runtime's explicit phase order is semantic.
    from amsrr.simulation.order9_object_task_runtime import ORDER9_OBJECT_TASK_PHASES

    phases = {p.value: phases[p.value] for p in ORDER9_OBJECT_TASK_PHASES}

    class CompiledReference(NaiveContactPlanReference):
        sample_calls = 0

        def _sample(self, *args, **kwargs):
            type(self).sample_calls += 1
            return super()._sample(*args, **kwargs)

        def __init__(self, *, phase_trajectories, provenance=None, **kwargs):
            incoming = stable_hash(
                {k: v.to_dict() for k, v in phase_trajectories.items()}
            )
            if incoming != plan["evidence"]["teacher_phase_hash"]:
                raise ValueError("Isaac teacher input differs from the compiled plan")
            super().__init__(
                phase_trajectories=phases,
                provenance={
                    **(provenance or {}),
                    "compiled_contact_plan_id": job["plan_id"],
                    "compiled_contact_plan_sha256": job["plan_sha256"],
                    "preload_baked_before_validation": True,
                },
                **kwargs,
            )

    # Reuse one stable, hash-bound simulation entry; preserve its protected file.
    # This is an isolated teacher adapter, never imported by production runtime.
    namespace = {
        "__file__": str(PROTECTED_HARNESS),
        "__name__": "teacher_contact_harness",
        "__package__": None,
    }
    old_argv = sys.argv
    started = time.monotonic()

    def expired(_signum, _frame):
        raise TimeoutError("teacher Isaac execution deadline")

    old_handler = signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.timeout_s)
    try:
        sys.argv = [str(PROTECTED_HARNESS), *arguments]
        source = PROTECTED_HARNESS.read_text()
        marker = "\n_exit_code = 1\n"
        if source.count(marker) != 1:
            raise ValueError("protected harness entry boundary changed")
        exec(
            compile(source.split(marker)[0], str(PROTECTED_HARNESS), "exec"), namespace
        )
        namespace["Order9C3NominalTensorReference"] = CompiledReference
        namespace["Order9TensorPiLRuntime"] = TeacherContactController
        result = namespace["main"]()
        records = [
            json.loads(line)
            for line in (output / "episodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
        expected_count = int(
            arguments[arguments.index("--evaluation-episode-count") + 1]
        )
        passed = len(records) == expected_count and all(
            r["task_success"]
            and r["no_fallback_success"]
            and not r["safety_failure"]
            and r["isaac_backed"]
            and r["full_mesh_evaluation"]
            for r in records
        )
        if (
            TeacherContactController.execution_calls < 1
            or CompiledReference.sample_calls < 1
        ):
            raise RuntimeError("new contact plan did not reach the controller")
        report = {
            "version": "teacher_contact_isaac_result_v1",
            "bucket_id": job["bucket_id"],
            "plan_id": job["plan_id"],
            "plan_sha256": job["plan_sha256"],
            "wall_time_s": time.monotonic() - started,
            "controller_calls": TeacherContactController.execution_calls,
            "compiled_reference_sample_calls": CompiledReference.sample_calls,
            "passed": passed,
            "successful_episodes": sum(bool(r["task_success"]) for r in records),
            "expected_episodes": expected_count,
            "teacher_phase_supervision": job["teacher_phase_supervision"],
            "learned_action_enabled": False,
            "autonomous_request_runtime": False,
            "episodes": records,
            "isaac_result": result,
        }
        _write_new(output / "result.json", report)
        print("TEACHER_CONTACT_RESULT=" + str(output / "result.json"), flush=True)
    except Exception as error:
        _write_new(
            output / "execution_failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "wall_time_s": time.monotonic() - started,
                "plan_id": job["plan_id"],
            },
        )
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        sys.argv = old_argv
        app = namespace.get("simulation_app")
        if app is not None:
            app.close()
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    p.add_argument("--bucket-manifest", required=True)
    p.add_argument("--bucket-id", required=True)
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Legacy decoder configuration; actor inference is disabled",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--rollout-steps", type=int, default=15000)
    p.add_argument("--planning-timeout-s", type=float, default=180.0)
    e = sub.add_parser("execute")
    e.add_argument("--job", required=True)
    e.add_argument("--timeout-s", type=int, default=600)
    args = parser.parse_args()
    if args.action == "prepare":
        if not 1 <= args.episodes <= 4 or args.rollout_steps < 1:
            raise ValueError(
                "teacher smoke supports 1--4 episodes and positive horizon"
            )
        return prepare(args)
    if args.timeout_s < 1:
        raise ValueError("Isaac timeout must be positive")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
