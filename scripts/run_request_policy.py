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
from amsrr.policies.high_level_requests import REQUEST_ADMISSION_CONTRACT
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
from amsrr.training.request_ppo import request_runtime_contracts
from scripts.run_teacher_contact_pipeline import PROTECTED_HARNESS, PROTECTED_SHA256


def sources():
    names = [
        "amsrr/policies/request_actor_critic.py",
        "amsrr/policies/request_morphology_features.py",
        "amsrr/policies/request_high_level_policy.py",
        "amsrr/policies/contact_group_geometry.py",
        "amsrr/policies/request_contact_planner.py",
        "amsrr/feasibility/articulated_reachability.py",
        "amsrr/policies/naive_contact_planner.py",
        "amsrr/simulation/request_event_execution.py",
        "amsrr/simulation/teacher_contact_execution.py",
        "amsrr/simulation/order9_tensor_isaac_io.py",
        "amsrr/controllers/batched_rigid_body_model.py",
        "amsrr/controllers/batched_virtual_thrust_qp.py",
        "amsrr/controllers/batched_qpid_controller.py",
        "amsrr/policies/order9_tensor_command_decoder.py",
        "amsrr/feasibility/order9_native_posture_ik.py",
        "amsrr/feasibility/order9_posture_collision.py",
        "amsrr/feasibility/native/order9_posture_native.cpp",
        "amsrr/controllers/grasp_slip_compensation.py",
        "amsrr/training/request_ppo.py",
        "amsrr/training/order9_tensor_reward.py",
        "amsrr/training/order9_anchor_normal_force_estimator.py",
        "amsrr/training/order9_deployable_phase_gate.py",
        "amsrr/utils/tensor_dataclass_graph.py",
        "amsrr/simulation/placement_support.py",
        "amsrr/geometry/convex_clearance.py",
        "amsrr/policies/high_level_requests.py",
        "amsrr/policies/assignment_feasibility.py",
        "amsrr/training/request_imitation.py",
        "scripts/run_request_policy.py",
    ]
    return {n: hash_file(ROOT / n) for n in names}


def restore_cached_request_geometry(cached, context, request, reviewed_bundle):
    """Rebuild through the same planner branch as a fresh selected request."""
    if cached["provenance"].get("contact_goal_initialization_retry", {}).get("version") == "bounded_selected_contact_goal_v1":
        raise ValueError("obsolete free-tilt contact initialization geometry")
    if (cached["provenance"]["observation_snapshot"] != context.catalog.snapshot_hash
            or cached["provenance"]["request"] != request.to_dict()):
        raise ValueError("cached geometric prefix belongs to a different request/initial observation")
    if cached["provenance"].get("teacher_geometry_reused", False):
        # A reviewed route has authored durations and a complete tail. Passing
        # its prefix through the generic planner silently changed 3 s to 30 s.
        return reuse_reviewed_request_geometry(context, request, reviewed_bundle)
    task = TaskSpec.from_dict(cached["task"])
    grasp = {k: ContactWrenchTrajectory.from_dict(cached["phases"][k])
             for k in ("approach", "contact_acquisition")}
    return SimpleNamespace(
        task=task, morphology=MorphologyGraph.from_dict(cached["morphology"]),
        contact_candidate_set=ContactCandidateSet.from_dict(cached["candidates"]),
        phase_trajectories=complete_request_phases(grasp, task),
        provenance=deepcopy(cached["provenance"]),
    )


def _with_contact_point_velocities(source):
    """Adapt the pinned harness to use matching link-origin pose and twist.

    Isaac's body_pose_w is at the link origin, whereas body_lin_vel_w is at
    its COM. The existing feedback helper shifts twist to the grasp point,
    so it must receive body_link_vel_w. Its resulting relative linear twist
    is also the speed for contact maintenance, in an orthonormal contact frame.
    The protected historical C3 harness remains unchanged on disk.
    """
    from textwrap import dedent, indent

    twist = dedent("""\
        selected_body_twist = torch.cat(
            (
                _torch(robot.data.body_lin_vel_w).index_select(
                    1, selected_anchor_body_indices_tensor
                ),
                _torch(robot.data.body_ang_vel_w).index_select(
                    1, selected_anchor_body_indices_tensor
                ),
            ),
            dim=-1,
        )""")
    for prefix, spaces in (("pre_", 16), ("", 8)):
        old = indent(twist.replace("selected_body_twist", prefix + "selected_body_twist"), " " * spaces)
        new = indent(
            prefix + "selected_body_twist = _torch(robot.data.body_link_vel_w).index_select(\n"
            "    1, selected_anchor_body_indices_tensor\n)", " " * spaces,
        )
        if source.count(old) != 1:
            raise ValueError("harness selected-body velocity boundary changed")
        source = source.replace(old, new)
    old = """        selected_relative_speed_mps = torch.linalg.vector_norm(
            contact.selected_link_twist_world[..., :3]
            - state.object_twist_world[:, None, :3],
            dim=-1,
        )"""
    new = """        selected_relative_speed_mps = torch.linalg.vector_norm(
            relative_twist_contact[..., :3], dim=-1
        )"""
    if source.count(old) != 1:
        raise ValueError("harness contact-speed boundary changed")
    return source.replace(old, new)


def _with_refreshed_contact_buffers(source):
    """Read each sensor's data container after the mandatory scene update.

    PhysX ContactSensor.data currently invokes its getter even for up-to-date
    rows. The pinned harness already forces both sensors to update every step.
    Keep the container (not a tensor snapshot), including across in-place reset.
    """
    begin = "    for rollout_index in range(int(args_cli.rollout_steps)):\n"
    end = "    rollout_elapsed = time.perf_counter() - rollout_started\n"
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError("harness sensor loop boundary changed")
    start, stop = source.index(begin), source.index(end)
    loop = source[start:stop]
    for old, new, count in (("robot_sensor.data", "robot_contact_data", 1),
                            ("object_sensor.data", "object_contact_data", 2)):
        if loop.count(old) != count:
            raise ValueError("harness contact data access boundary changed")
        loop = loop.replace(old, new)
    update = "        scene.update(float(args_cli.dt))\n"
    if loop.count(update) != 1:
        raise ValueError("harness sensor refresh boundary changed")
    loop = loop.replace(update, update + """        if robot_sensor._data is not robot_contact_data or object_sensor._data is not object_contact_data:
            raise RuntimeError("contact sensor was reinitialized during rollout")
""")
    setup = """    if scene.cfg.lazy_sensor_update:
        raise RuntimeError("request contact data reuse requires eager sensor refresh")
    robot_contact_data = robot_sensor.data
    object_contact_data = object_sensor.data
"""
    return source[:start] + setup + loop + source[stop:]


def _with_lazy_collision_details(source):
    """Collect detailed collision diagnostics only for newly colliding rows.

    Contact reduction and every safety decision still run on every step.
    The diagnostic body poses/forces have no consumer when no event occurred.
    """
    begin = "            environment_forces = (\n"
    end = "            diagnostic_collision_reported |= newly_observed\n"
    loop = """            for environment in (
                torch.nonzero(newly_observed, as_tuple=False).flatten().tolist()
            ):
"""
    if source.count(begin) != 1 or source.count(end) != 1 or source.count(loop) != 1:
        raise ValueError("harness collision diagnostic boundary changed")
    start, stop = source.index(begin), source.index(end)
    block = source[start:stop].replace(loop, "            for environment in new_collision_ids:\n")
    from textwrap import indent
    guarded = """            new_collision_ids = torch.nonzero(newly_observed, as_tuple=False).flatten().tolist()
            if new_collision_ids:
""" + indent(block, "    ")
    return source[:start] + guarded + source[stop:]


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
        time_s=0.0,
        end_time_s=0.0,
        reward_timing="observed_reward_time_v1",
        reward_events=[(0.0, -5.0)],
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
            runtime_contracts=request_runtime_contracts(),
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


def _with_independent_request_environments(source):
    """Retire each evaluation row once, preserving its terminal sample."""
    replacements = {
        "    for rollout_index in range(int(args_cli.rollout_steps)):\n":
        "    for rollout_index in range(int(args_cli.rollout_steps)):\n"
        "        request_evaluation_active = evaluation_active.clone()\n",
        "                valid=torch.ones_like(terminal),":
        "                valid=request_evaluation_active,",
        "        reset_ids = torch.nonzero(terminal, as_tuple=False).flatten()":
        "        reset_ids = torch.nonzero(terminal & evaluation_active, as_tuple=False).flatten()",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("harness independent-environment boundary changed")
        source = source.replace(old, new)
    return source


def _with_reset_twist_reference_point(source):
    """The measured body twist is at the assembled COM; Isaac resets its root."""
    old = "    robot.write_root_velocity_to_sim_index(root_velocity=body_twist, env_ids=env_ids)"
    new = """    root_twist = body_twist.clone()
    root_twist[:, :3] -= torch.cross(
        body_twist[:, 3:], desired_body_world[:, :3] - root_pose_world[:, :3], dim=-1
    )
    robot.write_root_velocity_to_sim_index(root_velocity=root_twist, env_ids=env_ids)"""
    if source.count(old) != 1:
        raise ValueError("harness reset twist boundary changed")
    return source.replace(old, new)


def _with_colocated_request_environments(source):
    """Allow a zero-spacing evaluation batch only with explicit collision isolation."""
    replacements = {
        "args_cli.env_spacing <= 0.0": "args_cli.env_spacing < 0.0",
        "    scene_cfg.replicate_physics = True":
        "    scene_cfg.replicate_physics = True\n    scene_cfg.filter_collisions = True",
        "    scene = InteractiveScene(scene_cfg)":
        "    scene = InteractiveScene(scene_cfg)\n"
        "    if float(args_cli.env_spacing) == 0.0:\n"
        "        if not scene.stage.GetPrimAtPath('/World/collisions'):\n"
        "            raise RuntimeError('co-located environments require collision filtering')\n"
        "        if not torch.equal(scene.env_origins, torch.zeros_like(scene.env_origins)):\n"
        "            raise RuntimeError('co-located environment origins differ')",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("harness co-located environment boundary changed")
        source = source.replace(old, new)
    return source


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
    from amsrr.training.request_ppo import serialize_encoding
    torch.save(dict(encoding=serialize_encoding(result["encoded"]),
                    requests=result["requests"]), out / "prepared_initial_encoding.pt")
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
            generated = restore_cached_request_geometry(
                cached, context, request, episode["bundle"]
            )
            generated.provenance = {
                    **generated.provenance,
                    "reused_geometric_prefix": {
                        "path": str(Path(a.geometry).resolve()),
                        "sha256": hash_file(a.geometry),
                    },
                    "tail_rebuilt_by_original_planner_branch": True,
                    "nominal_preload_recomputed_and_full_trajectory_rechecked": True,
                }
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
            initial_task=context.task_spec.to_dict(),
            initial_observation=context.scene.runtime_observation.to_dict(),
            initial_candidates=episode["initial_scene"].contact_candidate_set.to_dict(),
        ),
    )
    write_json(
        out / "job.json",
        dict(
            version="request_isaac_job_v3_fixed_predecision_catalog",
            request_admission_contract=REQUEST_ADMISSION_CONTRACT,
            runtime_contracts=request_runtime_contracts(),
            prepared_initial_encoding_sha256=hash_file(out / "prepared_initial_encoding.pt"),
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
    from amsrr.simulation.request_event_execution import (
        BatchedRequestEventSupervisor,
        RequestEventSupervisor,
        apply_object_goal_outcomes,
        object_pose_goal_deadlines,
        validate_initial_encoding,
    )

    out = Path(a.job).resolve().parent
    job = json.loads(Path(a.job).read_text())
    if a.num_envs < 1:
        raise ValueError("num-envs must be positive")
    if a.env_spacing < 0:
        raise ValueError("env-spacing must be nonnegative")
    environment_seeds = a.environment_seeds or [job["seed"]] * a.num_envs
    if len(environment_seeds) != a.num_envs:
        raise ValueError("one policy seed is required per environment")
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
    evaluation_checkpoints = a.evaluation_checkpoints
    if evaluation_checkpoints is not None and (
        len(evaluation_checkpoints) != a.num_envs
    ):
        raise ValueError("evaluation requires one checkpoint per environment")
    checkpoint_paths = evaluation_checkpoints or [job["checkpoint_path"]] * a.num_envs
    checkpoint_hashes = [hash_file(path) for path in checkpoint_paths]
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
    from amsrr.schemas.runtime import RuntimeObservation
    initial_task = TaskSpec.from_dict(data["initial_task"])
    initial_observation = RuntimeObservation.from_dict(data["initial_observation"])
    initial_candidates = ContactCandidateSet.from_dict(data["initial_candidates"])
    if hash_file(out / "prepared_initial_encoding.pt") != job["prepared_initial_encoding_sha256"]:
        raise ValueError("prepared initial encoding changed")
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
    models = {path: RequestActorCritic.load(path) for path in set(checkpoint_paths)}
    supervisor = BatchedRequestEventSupervisor([RequestEventSupervisor(
        model=models[path],
        bundle=bundle,
        task=task,
        initial_task=initial_task,
        initial_candidates=initial_candidates,
        physical=physical,
        expected_group=job["group_id"],
        seed=seed,
        sample=job["sample"],
        diagnostic_phase=(
            None if a.diagnostic_phase is None else PHASES.index(a.diagnostic_phase)
        ),
    ) for seed, path in zip(environment_seeds, checkpoint_paths)])
    from amsrr.training.request_ppo import serialize_encoding
    expected_initial = [individual.model.decide(
        individual.context(initial_observation, 0, initial=True),
        deterministic=not job["sample"], generator=torch.Generator().manual_seed(seed)
    ) for individual, seed in zip(supervisor.supervisors, environment_seeds)]
    prepared = torch.load(out / "prepared_initial_encoding.pt", weights_only=True)
    validate_initial_encoding(prepared["encoding"], serialize_encoding(expected_initial[0]["encoded"]))
    if prepared["requests"] != expected_initial[0]["requests"]:
        raise ValueError("prepared initial catalog changed")

    from amsrr.controllers.grasp_slip_compensation import CheckedGraspSlipController
    slip_controller = CheckedGraspSlipController(
        plan=plan, bundle=bundle, physical_model=physical, task=task,
        count=a.num_envs,
    )

    class RequestController(TeacherContactController):
        def compute_nominal_qpid_hold(self, **kwargs):
            was_started = supervisor.supervisors[0].started
            supervisor.begin(self, kwargs["state"], kwargs["task_target"])
            if not was_started and a.diagnostic_phase is None:
                if tuple(f"module_{m}:{j}" for m in self.builder.module_ids
                         for j in self.decoder.local_joint_ids) != tuple(slip_controller.ids):
                    raise ValueError("grasp closure joint ordering differs from controller")
                for individual, expected in zip(supervisor.supervisors, expected_initial):
                    actual = individual.events[0]
                    validate_initial_encoding(serialize_encoding(expected["encoded"]), actual["encoding"])
                    if expected["request"].to_dict() != actual["request"]:
                        raise ValueError("prepared request differs from live initial request")
            kwargs["task_target"] = slip_controller.apply(
                kwargs["task_target"], kwargs["state"], supervisor.supervisors
            )
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
        str(a.num_envs),
        "--env-spacing",
        str(a.env_spacing),
        "--rollout-steps",
        str(a.rollout_steps),
        "--evaluation-jsonl",
        str(out / "episodes.jsonl"),
        "--evaluation-episode-count",
        str(a.num_envs),
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
    old_linalg_backend = torch.backends.cuda.preferred_linalg_library()
    started = time.monotonic()
    try:
        # MAGMA's batched cholesky_solve allocates outside CUDA Graph capture.
        # cuSOLVER supports the identical solve for parallel environments.
        torch.backends.cuda.preferred_linalg_library("cusolver")
        sys.argv = [str(PROTECTED_HARNESS), *argv]
        source = PROTECTED_HARNESS.read_text()
        marker = "\n_exit_code = 1\n"
        if source.count(marker) != 1:
            raise ValueError("harness boundary changed")
        source = source.split(marker)[0]
        source = _with_contact_point_velocities(source)
        source = _with_independent_request_environments(source)
        source = _with_reset_twist_reference_point(source)
        # Outcome-only evidence, sampled with the existing post-physics contact
        # matrix. Never routed to the actor or the deployable supervisor.
        hook = "                prohibited_collision=contact.prohibited_collision,"
        if source.count(hook) != 1:
            raise ValueError("harness privileged reward input boundary changed")
        source = source.replace(hook, hook + """
                supported_placement=placement_support.sample(
                    phase_index=pre_phase, object_pose_world=state.object_pose_world,
                    time_s=pre_time.double() + float(args_cli.dt),
                    scene_origins=scene.env_origins,
                    net_force_world=_torch(object_sensor.data.net_forces_w)[:, 0, :],
                    robot_contact_force_world=object_matrix[:, 0, :, :]),""")
        source = _with_refreshed_contact_buffers(source)
        source = _with_lazy_collision_details(source)
        if a.env_spacing == 0:
            source = _with_colocated_request_environments(source)
        old = """        phase_supervision_success = (
            reward.phase_success
            if c3_privileged_phase_supervision
            else deployable_gate.phase_success
        )"""
        new = """        phase_supervision_success = request_supervisor.step(
            active=request_evaluation_active,
            phase_index=pre_phase, phase_elapsed_s=pre_elapsed + float(args_cli.dt),
            target=pre_target, state=state, control_model=post_control,
            surface_distance=signed_surface_distance, relative_speed=selected_relative_speed_mps,
            grasp_position_world=grasp_position,
            motor_load=(hardware_joint_load_nm[:, None] * deployable_anchor_joint_owner_mask[None]).sum(dim=(-1,-2)),
            qp_feasible=allocation.feasible, privileged_reward=reward)
        reward = request_supervisor.apply_deadlines(
            reward, time_s=pre_time.double() + float(args_cli.dt),
            object_pose_world=state.object_pose_world, active=request_evaluation_active)
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
        for name in ("Order9TensorRewardEngine", "Order9AnchorNormalForceEstimator", "Order9DeployablePhaseGate"):
            original = namespace[name]
            def graph_enabled(*args, _class=original, **kwargs):
                value = _class(*args, **kwargs)
                value._use_cuda_graph = True
                return value
            namespace[name] = graph_enabled
        original_loader = namespace["load_order9_c3_accepted_nominal_bundle"]
        namespace["load_order9_c3_accepted_nominal_bundle"] = requested_bundle
        def observed_reset_reference(**kwargs):
            return NaiveContactPlanReference(
                initial_observation=initial_observation, physical_model=physical, **kwargs
            )
        namespace["Order9C3NominalTensorReference"] = observed_reset_reference
        namespace["Order9TensorPiLRuntime"] = RequestController
        namespace["request_supervisor"] = supervisor
        from amsrr.simulation.placement_support import PlacementSupportEvidence
        from amsrr.training.order9_tensor_reward import Order9TensorRewardGateConfig
        support_evidence = PlacementSupportEvidence(
            task.to_dict(), force_threshold_n=Order9TensorRewardGateConfig().contact_force_threshold_n)
        namespace["placement_support"] = support_evidence
        code = namespace["main"]()
        write_json(out / "placement_support_trace.json", support_evidence.trace)
        records = [
            json.loads(line)
            for line in (out / "episodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
        physical_rollout = torch.load(
            out / "rollout.pt", map_location="cpu", weights_only=True
        )
        telemetry = physical_rollout["tensors"]
        if len(records) != a.num_envs:
            raise ValueError("missing or duplicate environment outcomes")
        motions, all_deadlines, all_termination_times, successes = [], [], [], []
        for index, (record, individual, deadline_monitor) in enumerate(
            zip(records, supervisor.supervisors, supervisor.deadlines)
        ):
            valid = telemetry["valid"][:, index].bool()
            poses = telemetry["post_object_pose_world"][valid, index].double().clone()
            poses[:, :3] -= supervisor.origins[index].double()
            deadline_outcomes = object_pose_goal_deadlines(
                task,
                telemetry["time_s"][valid, index].double()
                + float(physical_rollout["metadata"]["control_dt_s"]),
                poses,
            )
            # The untouched harness archive describes its legacy controller.
            # This policy result must describe the actual request authority.
            record["high_level_decision_count"] = len(individual.events)
            metadata = record["metadata"]
            metadata.pop("c3_privileged_phase_supervision_contract", None)
            metadata.pop("release_joint_position_tolerance_rad", None)
            metadata.update(
                grasp_slip_contract=request_runtime_contracts()["grasp_slip"],
                grasp_slip_reserve_m=slip_controller.reserve["span_m"],
                request_actor_critic_version=individual.model.version,
                deterministic_policy=not job["sample"],
                teacher_phase_supervision=False,
                phase_supervision_contract="request_actor_and_measured_state_guard_v1",
                contact_acquisition_success_contract="measured_geometry_motor_load_qp_dwell_v1",
                release_success_contract="measured_separation_object_pose_dwell_v1",
                contact_wrench_estimate_used_for_guard=False,
                contact_velocity_contract=request_runtime_contracts()["contact_velocity"],
                request_admission_contract=REQUEST_ADMISSION_CONTRACT,
                physical_telemetry_metadata_source=str(out / "episodes.jsonl"),
                object_pose_goal_deadlines=deadline_outcomes,
                object_pose_success_contract="authored_final_pose_and_deadline_v1",
                object_position_tolerance_m=deadline_outcomes[0]["tolerance_pos_m"],
                legacy_harness_task_success=record["task_success"],
                object_goal_deadline_terminal=deadline_monitor.deadline_missed,
                request_environment_index=index,
                request_policy_seed=environment_seeds[index],
                request_checkpoint_path=str(Path(checkpoint_paths[index]).resolve()),
                request_checkpoint_sha256=checkpoint_hashes[index],
            )
            apply_object_goal_outcomes(
                record,
                deadline_outcomes,
                deadline_terminated=deadline_monitor.deadline_missed,
            )
            successful = bool(
                record["task_success"]
                and not record["safety_failure"]
                and record["no_fallback_success"]
                and record["isaac_backed"]
                and record["full_mesh_evaluation"]
                and (
                    a.diagnostic_phase is not None
                    or (
                        individual.maximum_lift >= 0.048
                        and individual.maximum_transport >= 0.15
                    )
                )
            )
            successes.append(successful)
            motions.append(individual.finish(successful))
            all_deadlines.append(deadline_outcomes)
            all_termination_times.append(deadline_monitor.termination_time_s)
        successful = all(successes)
        torch.save(
            dict(
                checkpoint_sha256=job["checkpoint_sha256"],
                sampling="categorical" if job["sample"] else "greedy",
                teacher_phase_supervision=False,
                complete=True,
                diagnostic_only=a.diagnostic_phase is not None or a.benchmark or evaluation_checkpoints is not None,
                evaluation_only=evaluation_checkpoints is not None,
                environment_checkpoint_sha256=checkpoint_hashes,
                episodes=supervisor.episodes,
                runtime_contracts=request_runtime_contracts(),
                environment_seeds=environment_seeds,
            ),
            out / "request_rollout.pt",
        )
        write_json(
            out / "result.json",
            dict(
                passed=successful,
                episodes=records,
                motion=motions[0] if a.num_envs == 1 else motions,
                environment_count=a.num_envs,
                environment_seeds=environment_seeds,
                environment_origins_world=supervisor.origins.tolist(),
                environment_checkpoint_paths=[str(Path(p).resolve()) for p in checkpoint_paths],
                environment_checkpoint_sha256=checkpoint_hashes,
                success_count=sum(successes),
                benchmark=a.benchmark,
                teacher_phase_supervision=False,
                actor_controls_contact_and_transitions=a.diagnostic_phase is None,
                physical_acceptance_eligible=a.diagnostic_phase is None and not a.benchmark,
                diagnostic_phase=a.diagnostic_phase,
                object_pose_goal_deadlines=all_deadlines[0] if a.num_envs == 1 else all_deadlines,
                deadline_termination_time_s=all_termination_times[0] if a.num_envs == 1 else all_termination_times,
                controller_calls=RequestController.execution_calls,
                source_hashes=sources(),
                seconds=time.monotonic() - started,
                isaac_result=code,
            ),
        )
        write_json(out / "execution_trace.json", supervisor.supervisors[0].trace if a.num_envs == 1 else supervisor.trace)
        write_json(out / "grasp_slip_trace.json", slip_controller.trace)
        print("REQUEST_RESULT=" + str(out / "result.json"), flush=True)
        return 0 if successful else 2
    except Exception as e:
        write_json(out / "grasp_slip_trace.json", slip_controller.trace)
        if any(supervisor.episodes):
            torch.save(
                dict(
                    checkpoint_sha256=job["checkpoint_sha256"],
                    sampling="categorical" if job["sample"] else "greedy",
                    teacher_phase_supervision=False,
                    complete=False,
                    diagnostic_only=True,
                    episodes=supervisor.episodes,
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
        torch.backends.cuda.preferred_linalg_library(old_linalg_backend)
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
    a.add_argument("--num-envs", type=int, default=1,
                   help="Independent Isaac replicas of this checked scene/plan")
    a.add_argument("--environment-seeds", type=int, nargs="+",
                   help="Per-environment policy RNG seeds; initial selected group must match this plan")
    a.add_argument("--env-spacing", type=float, default=3.0,
                   help="Replica spacing in metres; zero uses collision-isolated coincident coordinates")
    a.add_argument("--evaluation-checkpoints", nargs="+",
                   help="Evaluation only: one saved actor per environment, sharing this checked plan; sampling follows the prepared job")
    a.add_argument("--benchmark", action="store_true",
                   help="Exclude this timing run from PPO and formal physical acceptance")
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
