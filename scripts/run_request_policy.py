#!/usr/bin/env python3
"""Prepare a selected request, run causal Isaac execution, archive on-policy events."""

from pathlib import Path
import sys, argparse, json, time, signal, shutil
from copy import deepcopy
from types import SimpleNamespace
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from amsrr.policies.request_actor_critic import RequestActorCritic, ACTOR_CRITIC_VERSION
from amsrr.policies.request_contact_planner import (
    plan_request_geometry,
    complete_request_phases,
    reuse_reviewed_request_geometry,
)
from amsrr.policies.naive_contact_planner import (
    plan_teacher_contact_trajectory,
    NaiveContactPlanReference,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.training.request_imitation import (
    load_episode,
    decision_context,
    write_json,
    PHASES,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_actuator_aware_nominal_preload import (
    Order9ActuatorAwareNominalPreloadConfig,
)
from amsrr.training.order9_c3_nominal_trajectory import _join_nominal_phase_trajectories
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.task_spec import TaskSpec
from amsrr.utils.hashing import hash_file, stable_hash
from scripts.run_teacher_contact_pipeline import PROTECTED_HARNESS, PROTECTED_SHA256


def sources():
    names = [
        "amsrr/policies/request_actor_critic.py",
        "amsrr/policies/request_high_level_policy.py",
        "amsrr/policies/contact_group_geometry.py",
        "amsrr/policies/request_contact_planner.py",
        "amsrr/policies/naive_contact_planner.py",
        "amsrr/simulation/request_event_execution.py",
        "amsrr/simulation/teacher_contact_execution.py",
        "amsrr/training/request_ppo.py",
        "amsrr/policies/high_level_requests.py",
        "amsrr/training/request_imitation.py",
        "scripts/run_request_policy.py",
    ]
    return {n: hash_file(ROOT / n) for n in names}


def save_planning_rejection(output, decision, checkpoint, sampled, reason):
    """An admitted action rejected by the bounded planner is a terminal outcome.

    It must remain in PPO data, rather than selecting only successful actions.
    This uses a recorded deployable reset observation; no Isaac success is claimed.
    """
    from amsrr.training.request_ppo import serialize_encoding

    event = dict(
        encoding=serialize_encoding(decision["encoded"]),
        action=decision["action"],
        log_prob=decision["log_prob"],
        value=decision["value"],
        reward=-5.0,
        done=True,
        request=decision["request"].to_dict(),
        requests=decision["requests"],
        snapshot_hash=decision["snapshot_hash"],
        planning_rejected=True,
    )
    torch.save(
        dict(
            checkpoint_sha256=hash_file(checkpoint),
            sampling="categorical" if sampled else "greedy",
            teacher_phase_supervision=False,
            complete=True,
            diagnostic_only=False,
            isaac_backed=False,
            reset_source="recorded_deployable_initial_observation",
            episodes=[[event]],
        ),
        output / "request_rollout.pt",
    )
    write_json(
        output / "planning_rejection.json",
        dict(
            reason=reason,
            task_success=False,
            isaac_backed=False,
            source_hashes=sources(),
        ),
    )


def prepare(a):
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model = RequestActorCritic.load(a.checkpoint).eval()
    torch.set_num_threads(1)
    physical = build_physical_model_from_config(ROOT / "configs/robot/robot_model.yaml")
    episode = load_episode(dict(path=a.record, sha256=hash_file(a.record)), physical)
    context = decision_context(episode, 0, physical)
    result = model.decide(
        context, deterministic=a.greedy, generator=torch.Generator().manual_seed(a.seed)
    )
    request = result["request"]
    selected = episode["selected_group"].group_id
    # Matching reviewed geometry is available only AFTER actor inference. The
    # caller recomputes nominal preload and checks the executable trajectory;
    # neither the selected teacher posture nor its group is an actor input.
    seed = (
        episode["bundle"]
        .phase_trajectories["contact_acquisition"]
        .knots[-1]
        .posture_target.joint_pos_target
        if request.contact_group_id == selected
        else None
    )
    base_seed = None
    if seed is not None:
        from amsrr.feasibility.articulated_reachability import (
            base_pose_for_centroidal_target,
        )
        from amsrr.geometry.pose_math import compose_pose, inverse_pose

        knot = episode["bundle"].phase_trajectories["contact_acquisition"].knots[-1]
        base = base_pose_for_centroidal_target(
            context.scene.morphology_graph,
            physical,
            seed,
            knot.centroidal_target.com_pos_world,
            knot.centroidal_target.body_orientation_world,
        )
        delta = compose_pose(
            context.observation.object_states[0].pose_world,
            inverse_pose(episode["task"].scene.objects[0].pose_world),
        )
        base_seed = compose_pose(delta, base)
    write_json(
        out / "decision.json",
        dict(
            request=request.to_dict(),
            teacher_group=selected,
            log_prob=result["log_prob"],
            snapshot_hash=result["snapshot_hash"],
            sampling="greedy" if a.greedy else "categorical",
            joint_seed_used=seed is not None,
            actor_input_initial_only=True,
            planning_seed=(
                "cached_geometric_prefix"
                if a.geometry is not None
                else (
                    "reviewed_geometric_path"
                    if seed is not None and not a.ignore_reviewed_geometry
                    else "posture_seed" if seed is not None else "unseeded"
                )
            ),
        ),
    )
    try:
        if a.geometry is None:
            if seed is not None and not a.ignore_reviewed_geometry:
                generated = reuse_reviewed_request_geometry(
                    context, request, episode["bundle"]
                )
            else:
                generated = plan_request_geometry(
                    context,
                    request,
                    joint_seed=seed,
                    base_seed=base_seed,
                    deadline_s=a.timeout_s - 120,
                )
        else:
            cached = json.loads(Path(a.geometry).read_text())
            if (
                cached["provenance"]["observation_snapshot"]
                != context.catalog.snapshot_hash
                or cached["provenance"]["request"] != request.to_dict()
            ):
                raise ValueError(
                    "cached geometric prefix belongs to a different request/initial observation"
                )
            task = TaskSpec.from_dict(cached["task"])
            grasp = {
                k: ContactWrenchTrajectory.from_dict(cached["phases"][k])
                for k in ("approach", "contact_acquisition")
            }
            generated = SimpleNamespace(
                task=task,
                morphology=MorphologyGraph.from_dict(cached["morphology"]),
                contact_candidate_set=ContactCandidateSet.from_dict(
                    cached["candidates"]
                ),
                phase_trajectories=complete_request_phases(grasp, task),
                provenance={
                    **cached["provenance"],
                    "reused_geometric_prefix": {
                        "path": str(Path(a.geometry).resolve()),
                        "sha256": hash_file(a.geometry),
                    },
                    "tail_recomputed_and_full_trajectory_rechecked": True,
                },
            )
        write_json(
            out / "geometry.json",
            dict(
                task=generated.task.to_dict(),
                morphology=generated.morphology.to_dict(),
                candidates=generated.contact_candidate_set.to_dict(),
                provenance=generated.provenance,
                phases={
                    k: v.to_dict() for k, v in generated.phase_trajectories.items()
                },
                source_hashes=sources(),
            ),
        )
        runtime = load_order9_learning_config(
            ROOT / "configs/training/order9_learning_curriculum.yaml"
        ).production_runtime
        case = episode["case"]
        plan = plan_teacher_contact_trajectory(
            bundle=generated,
            physical_model=physical,
            task=generated.task,
            object_mass_kg=case["estimated_mass_kg"],
            contact_friction=case["selected_gripper_friction"],
            contact_stiffness_n_per_m=case["contact_stiffness_n_per_m"],
            preload_config=Order9ActuatorAwareNominalPreloadConfig(
                minimum_inward_lead_m=runtime.c3_virtual_contact_inward_lead_m,
                maximum_inward_lead_m=runtime.c3_actuator_aware_nominal_preload_maximum_m,
                inward_lead_quantization_m=runtime.c3_actuator_aware_nominal_preload_quantization_m,
                support_safety_factor=runtime.c3_actuator_aware_nominal_preload_support_safety_factor,
                maximum_peak_effort_utilization=runtime.c3_actuator_aware_nominal_preload_maximum_peak_utilization,
                minimum_anchor_force_fraction=runtime.c3_actuator_aware_nominal_preload_minimum_anchor_fraction,
                model_error_margin_m=runtime.c3_actuator_aware_nominal_preload_model_error_margin_m,
            ),
            deadline_s=max(1.0, a.timeout_s - (time.monotonic() - started)),
        )
    except (ValueError, TimeoutError) as error:
        ordinary_rejection = isinstance(error, TimeoutError) or str(error).startswith(
            (
                "articulated trajectory teacher found no joint-reachable candidate group",
                "contact preload infeasible",
                "contact posture did not achieve",
                "contact posture has excessive tangential error",
                "contact plan collision",
                "request planner exhausted",
                "joint limit along",
                "joint speed along",
            )
        )
        if not ordinary_rejection:
            raise
        save_planning_rejection(out, result, a.checkpoint, not a.greedy, str(error))
        print("REQUEST_REJECTED=" + str(out / "planning_rejection.json"), flush=True)
        return 2
    write_json(out / "plan.json", plan.to_dict())
    write_json(
        out / "scene.json",
        dict(
            task=generated.task.to_dict(),
            morphology=generated.morphology.to_dict(),
            candidates=generated.contact_candidate_set.to_dict(),
            case=case,
        ),
    )
    write_json(
        out / "job.json",
        dict(
            version="request_isaac_job_v1",
            plan_id=plan.plan_id,
            plan_sha256=hash_file(out / "plan.json"),
            scene_sha256=hash_file(out / "scene.json"),
            checkpoint_path=str(Path(a.checkpoint).resolve()),
            checkpoint_sha256=hash_file(a.checkpoint),
            group_id=request.contact_group_id,
            seed=a.seed,
            sample=not a.greedy,
            source_hashes=sources(),
            preparation_seconds=time.monotonic() - started,
        ),
    )
    for name in sources():
        destination = out / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    print("REQUEST_JOB=" + str(out / "job.json"), flush=True)
    return 0


def execute(a):
    from amsrr.simulation.teacher_contact_execution import TeacherContactController
    from amsrr.simulation.request_event_execution import RequestEventSupervisor

    out = Path(a.job).resolve().parent
    job = json.loads(Path(a.job).read_text())
    if (out / "result.json").exists() or (out / "rollout.pt").exists():
        raise FileExistsError("execution already exists")
    if job["source_hashes"] != sources():
        raise ValueError("request job source changed")
    if hash_file(PROTECTED_HARNESS) != PROTECTED_SHA256:
        raise ValueError("protected C3 harness changed")
    for name in ("plan", "scene"):
        if hash_file(out / f"{name}.json") != job[f"{name}_sha256"]:
            raise ValueError("job data changed")
    if hash_file(job["checkpoint_path"]) != job["checkpoint_sha256"]:
        raise ValueError("actor changed")
    data = json.loads((out / "scene.json").read_text())
    plan = json.loads((out / "plan.json").read_text())
    if stable_hash(plan) != job["plan_id"]:
        raise ValueError("plan identity changed")
    phases = {
        k: ContactWrenchTrajectory.from_dict(plan["phase_trajectories"][k])
        for k in PHASES
    }
    morphology = MorphologyGraph.from_dict(data["morphology"])
    task = TaskSpec.from_dict(data["task"])
    candidates = ContactCandidateSet.from_dict(data["candidates"])
    case = data["case"]
    write_json(out / "execution_task.json", task.to_dict())
    physical = build_physical_model_from_config(ROOT / "configs/robot/robot_model.yaml")
    bundle = SimpleNamespace(
        morphology=morphology,
        contact_candidate_set=candidates,
        phase_trajectories=phases,
        trajectory=_join_nominal_phase_trajectories(None, phases),
        provenance=dict(
            execution_semantics="request_selected_checked_naive_contact_plan",
            compiled_plan_id=job["plan_id"],
            selected_surface_port_ids=sorted(
                {
                    next(
                        c.anchor_id
                        for c in candidates.candidates
                        if c.candidate_id == cid
                    )
                    for g in candidates.group_proposals
                    if g.group_id == job["group_id"]
                    for cid in g.candidate_ids
                }
            ),
        ),
    )
    supervisor = RequestEventSupervisor(
        model=RequestActorCritic.load(job["checkpoint_path"]),
        bundle=bundle,
        task=task,
        physical=physical,
        expected_group=job["group_id"],
        seed=job["seed"],
        sample=job["sample"],
        diagnostic_phase=(
            None if a.diagnostic_phase is None else PHASES.index(a.diagnostic_phase)
        ),
    )

    class RequestController(TeacherContactController):
        def compute_nominal_qpid_hold(self, **kwargs):
            supervisor.begin(self, kwargs["state"], kwargs["task_target"])
            return super().compute_nominal_qpid_hold(**kwargs)

    def requested_bundle(*args, **kwargs):
        # The original manifest is still validated for asset/task identity; the
        # new actor-selected, independently checked bundle replaces its motion.
        original_loader(
            *args,
            **{**kwargs, "expected_task_spec_sha256": case["task_spec"]["sha256"]},
        )
        if kwargs["expected_physical_model_hash"] != physical.stable_hash():
            raise ValueError("physical model mismatch")
        return bundle

    argv = [
        "--config",
        str(ROOT / "configs/training/order9_learning_curriculum.yaml"),
        "--stage",
        "c3_pi_l_ppo_arbitrary_morphology",
        "--pi-l-checkpoint",
        str(ROOT / case["c3_checkpoint"]["path"]),
        "--pi-l-checkpoint-sha256",
        case["c3_checkpoint"]["sha256"],
        "--generation-id",
        "request:" + case["candidate_id"],
        "--output-raw",
        str(out / "rollout.pt"),
        "--split",
        "validation",
        "--seed",
        str(case["seed"]),
        "--task-spec-json",
        str(out / "execution_task.json"),
        "--morphology-graph-json",
        str(ROOT / case["morphology_graph"]["path"]),
        "--robot-usd",
        str(ROOT / case["robot_usd"]["path"]),
        "--selected-gripper-friction",
        str(case["selected_gripper_friction"]),
        "--contact-stiffness",
        str(case["contact_stiffness_n_per_m"]),
        "--contact-damping",
        str(case["contact_damping_n_s_per_m"]),
        "--estimated-mass-kg",
        str(case["estimated_mass_kg"]),
        "--estimated-inertia-body",
        *map(str, case["estimated_inertia_body"]),
        "--estimated-com-object",
        *map(str, case["estimated_com_object"]),
        "--c3-nominal-set-manifest",
        str(ROOT / case["nominal_set"]["path"]),
        "--c3-nominal-set-sha256",
        case["nominal_set"]["sha256"],
        "--c3-nominal-artifact-sha256",
        case["nominal_artifact"]["sha256"],
        "--c3-reset-bank",
        str(out / "reset_bank.pt"),
        "--c3-action-contract",
        "contact_space_projected_policy_command",
        "--num-envs",
        "1",
        "--rollout-steps",
        str(a.rollout_steps),
        "--evaluation-jsonl",
        str(out / "episodes.jsonl"),
        "--evaluation-episode-count",
        "1",
        "--formal-phase-zero-start",
        "--diagnostic-nominal-qpid-only",
        "--virtual-contact-lead-mm",
        "0",
        "--diagnostic-collision-evidence",
        "--no-tensorboard",
    ]
    if a.diagnostic_phase is not None:
        argv.remove("--formal-phase-zero-start")
        argv += [
            "--diagnostic-initial-phase-index",
            str(PHASES.index(a.diagnostic_phase)),
            "--diagnostic-initial-phase-stratum-index",
            "0",
        ]
    namespace = {
        "__file__": str(PROTECTED_HARNESS),
        "__name__": "request_policy_harness",
        "__package__": None,
    }
    old_argv = sys.argv
    started = time.monotonic()
    try:
        sys.argv = [str(PROTECTED_HARNESS), *argv]
        source = PROTECTED_HARNESS.read_text()
        marker = "\n_exit_code = 1\n"
        if source.count(marker) != 1:
            raise ValueError("harness boundary changed")
        source = source.split(marker)[0]
        old = """        phase_supervision_success = (
            reward.phase_success
            if c3_privileged_phase_supervision
            else deployable_gate.phase_success
        )"""
        new = """        phase_supervision_success = request_supervisor.step(
            phase_index=pre_phase, phase_elapsed_s=pre_elapsed + float(args_cli.dt),
            target=pre_target, state=state, control_model=post_control,
            surface_distance=signed_surface_distance, relative_speed=selected_relative_speed_mps,
            motor_load=(hardware_joint_load_nm[:, None] * deployable_anchor_joint_owner_mask[None]).sum(dim=(-1,-2)),
            qp_feasible=allocation.feasible, privileged_reward=reward)
"""
        if source.count(old) != 1:
            raise ValueError("harness phase hook changed")
        source = source.replace(old, new)
        # Metadata must describe the actual authority, not inherited C3 timing.
        source = source.replace(
            "c3_privileged_phase_supervision = stage.stage_id == ORDER9_C3_STAGE_ID",
            "c3_privileged_phase_supervision = False",
        )
        exec(compile(source, str(PROTECTED_HARNESS), "exec"), namespace)
        original_loader = namespace["load_order9_c3_accepted_nominal_bundle"]
        namespace["load_order9_c3_accepted_nominal_bundle"] = requested_bundle
        namespace["Order9C3NominalTensorReference"] = NaiveContactPlanReference
        namespace["Order9TensorPiLRuntime"] = RequestController
        namespace["request_supervisor"] = supervisor
        code = namespace["main"]()
        records = [
            json.loads(line)
            for line in (out / "episodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
        for record in records:
            # The untouched harness archive describes its legacy controller.
            # This policy result must describe the actual request authority.
            record["high_level_decision_count"] = len(supervisor.events)
            metadata = record["metadata"]
            metadata.pop("c3_privileged_phase_supervision_contract", None)
            metadata.pop("release_joint_position_tolerance_rad", None)
            metadata.update(
                request_actor_critic_version=ACTOR_CRITIC_VERSION,
                deterministic_policy=not job["sample"],
                teacher_phase_supervision=False,
                phase_supervision_contract="request_actor_and_measured_state_guard_v1",
                contact_acquisition_success_contract="measured_geometry_motor_load_qp_dwell_v1",
                release_success_contract="measured_separation_object_pose_dwell_v1",
                contact_wrench_estimate_used_for_guard=False,
                physical_telemetry_metadata_source=str(out / "episodes.jsonl"),
            )
        successful = bool(
            len(records) == 1
            and records[0]["task_success"]
            and not records[0]["safety_failure"]
            and records[0]["no_fallback_success"]
            and records[0]["isaac_backed"]
            and records[0]["full_mesh_evaluation"]
            and (
                a.diagnostic_phase is not None
                or (
                    supervisor.maximum_lift >= 0.048
                    and supervisor.maximum_transport >= 0.15
                )
            )
        )
        motion = supervisor.finish(successful)
        torch.save(
            dict(
                checkpoint_sha256=job["checkpoint_sha256"],
                sampling="categorical" if job["sample"] else "greedy",
                teacher_phase_supervision=False,
                complete=True,
                diagnostic_only=a.diagnostic_phase is not None,
                episodes=[supervisor.events],
            ),
            out / "request_rollout.pt",
        )
        write_json(
            out / "result.json",
            dict(
                passed=successful,
                episodes=records,
                motion=motion,
                teacher_phase_supervision=False,
                actor_controls_contact_and_transitions=a.diagnostic_phase is None,
                physical_acceptance_eligible=a.diagnostic_phase is None,
                diagnostic_phase=a.diagnostic_phase,
                controller_calls=RequestController.execution_calls,
                source_hashes=sources(),
                seconds=time.monotonic() - started,
                isaac_result=code,
            ),
        )
        write_json(out / "execution_trace.json", supervisor.trace)
        print("REQUEST_RESULT=" + str(out / "result.json"), flush=True)
        return 0 if successful else 2
    except Exception as e:
        if supervisor.events:
            torch.save(
                dict(
                    checkpoint_sha256=job["checkpoint_sha256"],
                    sampling="categorical" if job["sample"] else "greedy",
                    teacher_phase_supervision=False,
                    complete=False,
                    diagnostic_only=True,
                    episodes=[supervisor.events],
                ),
                out / "incomplete_request_events.pt",
            )
        write_json(
            out / "execution_failure.json",
            dict(
                error=repr(e),
                seconds=time.monotonic() - started,
                trace=supervisor.trace,
                initial_runtime_check=supervisor.initial_runtime_check,
            ),
        )
        raise
    finally:
        sys.argv = old_argv
        if namespace.get("simulation_app") is not None:
            namespace["simulation_app"].close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="mode", required=True)
    a = s.add_parser("prepare")
    a.add_argument("--record", required=True)
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--seed", type=int, default=17)
    a.add_argument("--greedy", action="store_true")
    a.add_argument(
        "--ignore-reviewed-geometry",
        action="store_true",
        help="Generate a fresh route even when a reviewed route for the selected binding is available",
    )
    a.add_argument(
        "--geometry",
        help="Reuse only an exact snapshot/request geometric prefix; recompute and check the complete tail",
    )
    a.add_argument("--timeout-s", type=int, default=590)
    a = s.add_parser("execute")
    a.add_argument("--job", required=True)
    a.add_argument("--timeout-s", type=int, default=900)
    a.add_argument("--rollout-steps", type=int, default=20000)
    a.add_argument(
        "--diagnostic-phase",
        choices=PHASES,
        help="Phase-isolated debug only; never physical acceptance or PPO data",
    )
    a = p.parse_args()
    signal.signal(
        signal.SIGALRM,
        lambda *_: (_ for _ in ()).throw(TimeoutError("request job deadline")),
    )
    signal.alarm(a.timeout_s)
    torch.set_num_threads(1)
    if a.mode == "prepare":
        return prepare(a)
    return execute(a)


if __name__ == "__main__":
    raise SystemExit(main())
