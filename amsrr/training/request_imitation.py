"""Causal request imitation from accepted, immutable R1 teacher executions.

The eight-phase IRG below describes this teacher's declared plan, not a new
universal task state machine. This is offline imitation, not an autonomous
controller evaluation. Raw contact/success/next-phase values are label-only.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import fields, replace
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from amsrr.encoders.morphology_graph_encoder import MorphologyGraphBatch
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.policies.high_level_requests import (
    RequestCatalogBuilder,
    contact_surface_hash,
)
from amsrr.policies.request_high_level_policy import (
    REQUEST_FEATURE_NAMES,
    REQUEST_FEATURE_VERSION,
    RequestHighLevelPolicy,
    RequestHighLevelPolicyConfig,
)
from amsrr.policies.contact_group_geometry import contact_geometry, CONTACT_FEATURE_DIM
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.high_level import ActiveExecutionState, RequestRankingLabel
from amsrr.schemas.irg import IRGEdge, IRGEdgeType, IRGNode, IRGNodeType, PhaseType
from amsrr.schemas.runtime import TaskProgressState
from amsrr.schemas.policies import ObjectTarget
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
    ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME,
)
from amsrr.training.order9_r1_pi_h_dataset_preflight import (
    _load_case_context,
    _read_json,
    _runtime_observation,
    _validate_accepted_record,
    _validate_raw_rollout,
    _validated_binding,
    select_order9_r1_pi_h_decision_rows,
)
from amsrr.training.order9_teacher import compile_high_level_context
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.utils.hashing import hash_file, stable_hash

VERSION = "r1_teacher_request_imitation_v1"
PHASES = tuple(p.value for p in ORDER9_OBJECT_TASK_PHASES)
ACTOR_TO_PHASE = dict(zip(ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME, PHASES))
CATEGORIES = ("initial_group", "contact_group", "continuation", "transition", "forced")
ROOT = Path(__file__).resolve().parents[2]
SOURCE_SUMMARY = (
    ROOT
    / "artifacts/p4_full/order9/r1_teacher/formal_collection_v19/collection_summary_identity_v2.json"
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def teacher_request_scene(
    bundle, task, *, vertical_scale: float = 1.0, plan_committed: bool = False
):
    """Declare the executed plan's phases/goals before observing any labels.

    Preload belongs to contact acquisition. Original force/contact/constraint
    nodes are retained. The initial decision uses only TaskSpec-derived goals;
    a previously committed plan may supply execution goals/durations later.
    The old seven-phase IRG and source teacher files remain unchanged.
    """
    candidates = (
        bundle.contact_candidate_set
        if plan_committed
        else uncommitted_teacher_candidates(bundle.contact_candidate_set)
    )
    scene = compile_high_level_context(task, bundle.morphology, candidates)
    graph = scene.irg
    lift_clearance = next(
        n.feature["tolerance"]["height_margin_m"]
        for n in graph.nodes
        if n.node_type == IRGNodeType.STATE_TARGET
        and "height_margin_m" in n.feature.get("tolerance", {})
    )
    old = {
        n.feature["phase_label"]: n
        for n in graph.nodes
        if n.node_type == IRGNodeType.PHASE
    }
    names = dict(
        zip(
            PHASES[:6],
            (
                "approach_object",
                "establish_object_contacts",
                "lift_object",
                "transport_object",
                "place_object",
                "release_contacts",
            ),
        )
    )
    nodes = {phase: old[label] for phase, label in names.items()}
    removed = {old["apply_grasp_wrench"].node_id}
    removed.update(
        n.node_id for n in graph.nodes if n.node_type == IRGNodeType.STATE_TARGET
    )
    preload_id, contact_id = (
        old["apply_grasp_wrench"].node_id,
        nodes["contact_acquisition"].node_id,
    )
    graph.nodes = [n for n in graph.nodes if n.node_id not in removed]
    edges = []
    for edge in graph.edges:
        if edge.edge_type == IRGEdgeType.TEMPORAL_NEXT:
            continue
        if edge.src_id == preload_id:
            edge.src_id = contact_id
        if edge.dst_id == preload_id:
            continue  # task containment is already present for contact acquisition
        if edge.src_id not in removed and edge.dst_id not in removed:
            edges.append(edge)
    graph.edges = edges
    next_id = max(n.node_id for n in graph.nodes) + 1
    task_id = next(n.node_id for n in graph.nodes if n.node_type == IRGNodeType.TASK)
    for phase in PHASES[6:]:
        node = IRGNode(
            next_id,
            IRGNodeType.PHASE,
            phase,
            1.0,
            True,
            None,
            {"phase_type": PhaseType.FREE_MOTION.value},
        )
        next_id += 1
        graph.nodes.append(node)
        graph.edges.append(IRGEdge(task_id, node.node_id, IRGEdgeType.CONTAINS))
        nodes[phase] = node
    lift_start = (
        (bundle.phase_trajectories["lift"].knots[0].object_targets[0].pose_target_world)
        if plan_committed
        else None
    )
    for index, phase in enumerate(PHASES):
        node = nodes[phase]
        trajectory = bundle.phase_trajectories[phase] if plan_committed else None
        node.feature.update(
            phase_label=phase,
            phase_index=index,
            nominal_duration_s=trajectory.horizon_s if trajectory is not None else None,
        )
        if index:
            graph.edges.append(
                IRGEdge(
                    nodes[PHASES[index - 1]].node_id,
                    node.node_id,
                    IRGEdgeType.TEMPORAL_NEXT,
                )
            )
        if trajectory is not None:
            targets = trajectory.knots[-1].object_targets
        else:
            targets = []
            for goal in task.goals:
                if goal.goal_type != "object_pose":
                    continue
                obj = next(
                    o
                    for o in task.scene.objects
                    if o.object_id == goal.target_entity_id
                )
                pose = list(
                    obj.pose_world
                    if phase in {"approach", "contact_acquisition", "lift"}
                    else goal.target_pose_world
                )
                if phase in {"lift", "transport"}:
                    pose[2] += lift_clearance
                targets.append(ObjectTarget(obj.object_id, tuple(pose)))
        for target in targets:
            if target.pose_target_world is None:
                continue
            pose = list(target.pose_target_world)
            if plan_committed and phase in {"lift", "transport", "place"}:
                pose[2] = lift_start[2] + vertical_scale * (pose[2] - lift_start[2])
            goal = next(
                g
                for g in task.goals
                if g.target_entity_id == target.object_id
                and g.goal_type == "object_pose"
            )
            target_node = IRGNode(
                next_id,
                IRGNodeType.STATE_TARGET,
                f"{phase}:planned_object_goal",
                1.0,
                True,
                node.node_id,
                {
                    "target_type": "object_pose",
                    "target_entity_id": target.object_id,
                    "pose_target_world": pose,
                    "tolerance": {
                        "pos_m": goal.tolerance_pos_m,
                        "rot_rad": goal.tolerance_rot_rad,
                    },
                },
            )
            next_id += 1
            graph.nodes.append(target_node)
            graph.edges.append(
                IRGEdge(node.node_id, target_node.node_id, IRGEdgeType.REQUIRES)
            )
    graph.metadata["teacher_request_schedule"] = VERSION
    graph.metadata["plan_committed_before_decision"] = plan_committed
    graph.validate()
    scene = replace(
        scene, interaction_envelope=InteractionEnvelopeExtractor().extract(graph)
    )
    return scene, {phase: node.node_id for phase, node in nodes.items()}


def uncommitted_teacher_candidates(candidate_set):
    """Recover R1 proposals before the teacher's chosen-group refinement.

    The R1 importer handles whole-scene planar transfers, which preserve the
    world vertical used by the recorded clearance shift. Undo that shift before
    RequestCatalogBuilder re-grounds proposals in the measured object frame.
    The source execution catalog and its accepted trajectory remain untouched.
    Other legacy refinements require their own provenance/frame validation;
    they cannot silently become pre-decision actor input.
    """
    candidates = deepcopy(candidate_set)
    for candidate in candidates.candidates:
        markers = {key for key in candidate.candidate_scores if key.startswith("r1_")}
        if markers - {"r1_vertical_clearance_shift_m"}:
            raise ValueError("unsupported post-selection teacher candidate refinement")
        shift = candidate.candidate_scores.pop("r1_vertical_clearance_shift_m", 0.0)
        if not math.isfinite(shift):
            raise ValueError("nonfinite post-selection teacher clearance shift")
        if shift:
            for name in ("contact_pose_world", "contact_frame_world"):
                pose = list(getattr(candidate, name))
                pose[2] -= shift
                setattr(candidate, name, tuple(pose))
    candidates.assignment_feasibility_cache = {}
    candidates.validate()
    return candidates


def load_episode(binding, physical_model):
    record_path = _validated_binding(binding, repository=ROOT, label="accepted_record")
    accepted = _read_json(record_path)
    _validate_accepted_record(accepted)
    case_path = _validated_binding(
        accepted["case_manifest"], repository=ROOT, label="case"
    )
    raw_path = _validated_binding(accepted["raw_rollout"], repository=ROOT, label="raw")
    case = _read_json(case_path)
    bundle, task = _load_case_context(
        case,
        episode_id=accepted["episode_id"],
        task_id=accepted["task_id"],
        split=DatasetSplit(accepted["dataset_split"]),
        module_count=accepted["module_count"],
        repository=ROOT,
    )
    payload = torch.load(raw_path, map_location="cpu", weights_only=False)
    metadata, tensors = _validate_raw_rollout(
        payload, task=task, bundle=bundle, accepted=accepted
    )
    if metadata["physical_model_hash"] != physical_model.stable_hash():
        raise ValueError("teacher physical model differs")
    if any(
        bool(tensors[key].abs().max() > 1e-7)
        for key in (
            "applied_global_action",
            "joint_action",
            "contact_space_residual_action",
        )
    ):
        raise ValueError("teacher used learned corrections")
    scene, phase_ids = teacher_request_scene(
        bundle,
        task,
        vertical_scale=float(
            metadata.get("r1_diagnostic_maintain_vertical_scale") or 1.0
        ),
        plan_committed=True,
    )
    initial_scene, initial_phase_ids = teacher_request_scene(
        bundle, task, plan_committed=False
    )
    if initial_phase_ids != phase_ids:
        raise ValueError("initial/execution phase identity differs")
    selected = {
        a.candidate_id
        for a in bundle.phase_trajectories["contact_acquisition"]
        .knots[-1]
        .contact_assignments
    }
    matching = [
        g
        for g in bundle.contact_candidate_set.group_proposals
        if set(g.candidate_ids) == selected
    ]
    if len(matching) != 1:
        raise ValueError(
            "executed teacher assignment does not identify one contact group"
        )
    return dict(
        accepted=accepted,
        case=case,
        metadata=metadata,
        tensors=tensors,
        bundle=bundle,
        task=task,
        scene=scene,
        initial_scene=initial_scene,
        phase_ids=phase_ids,
        selected_group=matching[0],
        source_binding=binding,
    )


def decision_context(episode, raw_index: int, physical_model, *, contact_only=False):
    """Only observations at t and execution/choice history strictly before t.

    The logged phase at t is the expert's choice and is deliberately not read
    for the active phase. task_progress from the legacy loader is scrubbed.
    """
    x, metadata, bundle = episode["tensors"], episode["metadata"], episode["bundle"]
    observation = _runtime_observation(
        metadata=metadata, tensors=x, raw_index=raw_index, morphology=bundle.morphology
    )
    observation.task_progress = TaskProgressState()
    if contact_only:
        if raw_index != 0:
            raise ValueError("initial contact imitation cannot use post-choice observations")
        scene = replace(
            deepcopy(episode["initial_scene"]), runtime_observation=observation
        )
        execution = ActiveExecutionState(
            episode["phase_ids"]["approach"], observation.time_s
        )
        return RequestCatalogBuilder().build(
            scene,
            task_spec=episode["task"],
            physical_model=physical_model,
            execution_state=execution,
        )
    scene = replace(
        deepcopy(episode["scene"] if raw_index > 0 else episode["initial_scene"]),
        runtime_observation=observation,
    )
    history = x["phase_index"][:raw_index, 0]
    previous_actor_phase = (
        int(history[-1])
        if len(history)
        else ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME[0]
    )
    previous_phase = ACTOR_TO_PHASE[previous_actor_phase]
    previous_start = (
        int(torch.nonzero(history == previous_actor_phase)[0]) if len(history) else 0
    )
    group = episode["selected_group"]
    group_id = (
        group.group_id
        if raw_index > 0 and previous_phase not in {"retreat", "settle"}
        else None
    )
    execution = ActiveExecutionState(
        episode["phase_ids"][previous_phase],
        float(x["time_s"][previous_start, 0]),
        contact_group_id=group_id,
        plan_id=(
            stable_hash(episode["case"]["nominal_artifact"]) if raw_index > 0 else None
        ),
    )
    if previous_phase in {
        "contact_acquisition",
        "lift",
        "transport",
        "place",
        "release",
    }:
        source_observation = deepcopy(observation)
        for obj in source_observation.object_states:
            obj.pose_world = next(
                o.pose_world
                for o in episode["task"].scene.objects
                if o.object_id == obj.object_id
            )
        for candidate in bundle.contact_candidate_set.candidates:
            if candidate.candidate_id in group.candidate_ids:
                execution.contact_bindings[candidate.candidate_id] = (
                    candidate.slot_id,
                    candidate.anchor_id,
                    candidate.target_entity_id,
                )
                execution.contact_surface_hashes[candidate.candidate_id] = (
                    contact_surface_hash(candidate, source_observation)
                )
    return RequestCatalogBuilder().build(
        scene,
        task_spec=episode["task"],
        physical_model=physical_model,
        execution_state=execution,
    )


def teacher_label(episode, raw_index, context, *, contact_only=False):
    phase = ACTOR_TO_PHASE[int(episode["tensors"]["phase_index"][raw_index, 0])]
    selected_group = episode["selected_group"].group_id
    choices = [
        i
        for i, e in enumerate(context.catalog.entries)
        if e.phase_id
        == (
            context.execution_state.phase_id
            if contact_only
            else episode["phase_ids"][phase]
        )
        and e.request.contact_group_id in (selected_group, None)
    ]
    if len(choices) != 1:
        raise ValueError(
            f"teacher request not uniquely represented: {phase}, {len(choices)}"
        )
    index = choices[0]
    entry = context.catalog.entries[index]
    # Binding includes the applied-reference record and declared execution transform.
    checked_hash = stable_hash(
        {
            "nominal": episode["case"]["nominal_artifact"],
            "execution_case": episode["accepted"]["case_manifest"],
            "observed_execution": episode["accepted"]["raw_rollout"],
        }
    )
    label = RequestRankingLabel(
        entry.request,
        context.catalog.snapshot_hash,
        checked_hash,
        "first_choice_success",
    )
    label.validate()
    category = (
        "initial_group"
        if raw_index == 0
        else (
            "contact_group"
            if contact_only
            else "transition" if entry.transition is not None else "continuation"
        )
    )
    if len(context.catalog.entries) == 1:
        category = "forced"
    return index, label, category, phase


def _graph_dict(batch):
    return {
        f.name: getattr(batch, f.name).cpu()
        for f in fields(batch)
        if isinstance(getattr(batch, f.name), torch.Tensor)
    }


def _pad_graphs(graphs):
    result = {}
    for name in graphs[0]:
        values = [g[name] for g in graphs]
        shape = [
            len(values),
            *[max(v.shape[i] for v in values) for i in range(1, values[0].ndim)],
        ]
        fill = -1 if name in {"edge_index", "module_ids", "edge_ids"} else 0
        padded = torch.full(shape, fill, dtype=values[0].dtype)
        for i, value in enumerate(values):
            slices = (i, *[slice(0, size) for size in value.shape[1:]])
            padded[slices] = value[0]
        result[name] = padded
    return result


def graph_batch(tensors, indices, device):
    values = {k: v[indices].to(device) for k, v in tensors.items()}
    return MorphologyGraphBatch(
        **values, graph_ids=tuple("request" for _ in range(len(indices)))
    )


def prepare_dataset(
    output: Path,
    *,
    limit_episodes: int | None = None,
    timeout_s: float = 1200,
    config_path: Path = ROOT / "configs/training/order9_learning_curriculum.yaml",
    selection_scope: str = "requests",
    event_sampling: bool = False,
):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    config = load_order9_learning_config(config_path)
    physical = build_physical_model_from_config(
        ROOT / config.production_runtime.robot_model_config_path
    )
    policy = RequestHighLevelPolicy(
        RequestHighLevelPolicyConfig(selection_scope=selection_scope)
    ).eval()
    summary = _read_json(SOURCE_SUMMARY)
    bindings = summary["collection_records"]
    if summary["status"] != "complete" or len(bindings) != 140:
        raise ValueError("incomplete formal teacher collection")
    if limit_episodes is not None:
        bindings = bindings[:limit_episodes]
    examples, masks, graphs, rows, source_rows = [], [], [], [], []
    candidate_rows, memberships = [], []
    maximum_runtime_equivalence_error = 0.0
    for binding in bindings:
        episode = load_episode(binding, physical)
        x = episode["tensors"]
        selected = set(
            select_order9_r1_pi_h_decision_rows(
                valid=x["valid"][:, 0],
                time_s=x["time_s"][:, 0],
                phase_index=x["phase_index"][:, 0],
            )
        )
        boundaries = (
            torch.nonzero(x["phase_index"][1:, 0] != x["phase_index"][:-1, 0]).flatten()
            + 1
        ).tolist()
        selected.update(i - 1 for i in boundaries)
        if selection_scope == "contact_group":
            selected = {0}
        elif event_sampling:
            # Learn initial choices, actual boundary requests and ordinary
            # continuation. Never disguise a chosen/posture state as an initial
            # group query. Avoid teaching the exact privileged dwell clock.
            starts = [0, *boundaries]
            ends = [*boundaries, len(x["phase_index"])]
            selected = {0, *boundaries}
            selected.update((a + b) // 2 for a, b in zip(starts, ends) if b - a > 2)
        for raw_index in sorted(selected):
            if time.monotonic() - started > timeout_s:
                raise TimeoutError("request dataset conversion deadline")
            contact_only = selection_scope == "contact_group"
            context = decision_context(
                episode, raw_index, physical, contact_only=contact_only
            )
            # Freeze both tensors before inspecting the current label.
            features = policy.request_features(context).detach().cpu()
            eligible = policy.selection_mask(context).cpu()
            candidate_features, membership = contact_geometry(context)
            graph = policy.morphology_encoder.tensorizer.tensorize(
                [context.scene.morphology_graph],
                runtime_observations=[context.scene.runtime_observation],
            )
            index, label, category, phase = teacher_label(
                episode, raw_index, context, contact_only=contact_only
            )
            if not eligible[index]:
                raise ValueError("teacher choice excluded by selection scope")
            if raw_index == 0 or raw_index in boundaries:
                original = int(x["phase_index"][raw_index, 0])
                original_success = bool(x["actor_task_success"][raw_index, 0])
                try:
                    x["phase_index"][raw_index, 0] = (
                        ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME[-1]
                    )
                    x["actor_task_success"][raw_index, 0] = not original_success
                    alternative = decision_context(
                        episode, raw_index, physical, contact_only=contact_only
                    )
                    torch.testing.assert_close(
                        policy.request_features(alternative), features, rtol=0, atol=0
                    )
                    assert (
                        alternative.catalog.snapshot_hash
                        == context.catalog.snapshot_hash
                    )
                    alternate_features, alternate_members = contact_geometry(
                        alternative
                    )
                    torch.testing.assert_close(
                        alternate_features, candidate_features, rtol=0, atol=0
                    )
                    assert torch.equal(alternate_members, membership)
                finally:
                    x["phase_index"][raw_index, 0] = original
                    x["actor_task_success"][raw_index, 0] = original_success
            if raw_index == 0:
                with torch.no_grad():
                    expected = policy(context)
                    encoded = policy.forward_encoded(
                        features[None],
                        graph,
                        eligible[None],
                        candidates=candidate_features[None],
                        membership=membership[None],
                    )[0]
                error = float((expected[eligible] - encoded[eligible]).abs().max())
                maximum_runtime_equivalence_error = max(
                    maximum_runtime_equivalence_error, error
                )
                if error > 1e-6:
                    raise ValueError("training tensor path differs from runtime")
            if not len(features) or not torch.isfinite(features).all():
                raise ValueError("empty/nonfinite request feature tensor")
            rows.append(
                {
                    "episode_id": episode["accepted"]["episode_id"],
                    "split": episode["accepted"]["dataset_split"],
                    "raw_index": raw_index,
                    "initial": context.execution_state.plan_id is None,
                    "committed_group_id": context.execution_state.contact_group_id,
                    "time_s": context.observation.time_s,
                    "category": category,
                    "phase": phase,
                    "label": index,
                    "label_record": label.to_dict(),
                    "requests": [e.request.to_dict() for e in context.catalog.entries],
                    "phase_ids": [e.phase_id for e in context.catalog.entries],
                    "heuristic_index": int(torch.nonzero(eligible)[0]),
                    "module_count": episode["accepted"]["module_count"],
                }
            )
            examples.append(features)
            masks.append(eligible)
            candidate_rows.append(candidate_features)
            memberships.append(membership)
            graphs.append(_graph_dict(graph))
        source_rows.append(
            {
                "binding": binding,
                "episode_id": episode["accepted"]["episode_id"],
                "split": episode["accepted"]["dataset_split"],
                "structural_hash": episode["case"]["structural_hash"],
                "pose_condition_hash": episode["accepted"]["pose_condition_hash"],
                "raw": episode["accepted"]["raw_rollout"],
                "initial_candidate_set_hash": stable_hash(
                    episode["initial_scene"].contact_candidate_set.to_dict()
                ),
                "execution_candidate_set_hash": stable_hash(
                    episode["bundle"].contact_candidate_set.to_dict()
                ),
                "removed_postselection_shifts": {
                    str(c.candidate_id): c.candidate_scores[
                        "r1_vertical_clearance_shift_m"
                    ]
                    for c in episode["bundle"].contact_candidate_set.candidates
                    if "r1_vertical_clearance_shift_m" in c.candidate_scores
                },
            }
        )
        print(
            json.dumps(
                {
                    "prepared_episodes": len(source_rows),
                    "rows": len(rows),
                    "wall_time_s": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    width = max(len(x) for x in examples)
    features = torch.zeros((len(rows), width, len(REQUEST_FEATURE_NAMES)))
    mask = torch.zeros(features.shape[:2], dtype=torch.bool)
    for i, (values, eligible) in enumerate(zip(examples, masks)):
        features[i, : len(values)] = values
        mask[i, : len(values)] = eligible
    candidate_width = max(len(c) for c in candidate_rows)
    candidate_tensor = torch.zeros((len(rows), candidate_width, CONTACT_FEATURE_DIM))
    membership_tensor = torch.zeros(
        (len(rows), width, candidate_width), dtype=torch.bool
    )
    for i, (candidates, member) in enumerate(zip(candidate_rows, memberships)):
        candidate_tensor[i, : len(candidates)] = candidates
        membership_tensor[i, : member.shape[0], : member.shape[1]] = member
    payload = {
        "version": VERSION,
        "feature_version": REQUEST_FEATURE_VERSION,
        "selection_scope": selection_scope,
        "event_sampling": event_sampling,
        "features": features,
        "mask": mask,
        "graphs": _pad_graphs(graphs),
        "candidate_features": candidate_tensor,
        "membership": membership_tensor,
        "rows": rows,
        "sources": source_rows,
    }
    torch.save(payload, output / "dataset.pt")
    counts = Counter(r["split"] for r in rows)
    morphologies = {
        s: {r["structural_hash"] for r in source_rows if r["split"] == s}
        for s in counts
    }
    if any(
        morphologies.get("train", set()) & morphologies.get(s, set())
        for s in ("validation", "held_out")
    ):
        raise ValueError("training morphology leaked into evaluation split")
    if len({r["pose_condition_hash"] for r in source_rows}) != len(source_rows):
        raise ValueError("duplicate scene across teacher episodes")
    report = {
        "version": VERSION,
        "feature_version": REQUEST_FEATURE_VERSION,
        "selection_scope": selection_scope,
        "episode_count": len(source_rows),
        "row_count": len(rows),
        "split_rows": dict(counts),
        "categories": dict(Counter(r["category"] for r in rows)),
        "phase_counts": dict(Counter(r["phase"] for r in rows)),
        "catalog_width": width,
        "runtime_equivalence_max_error": maximum_runtime_equivalence_error,
        "counterfactual_causality_passed": True,
        "current_teacher_phase_in_input": False,
        "postselection_teacher_refinements_in_initial_input": False,
        "episodes_with_restored_candidate_geometry": sum(
            bool(source["removed_postselection_shifts"]) for source in source_rows
        ),
        "raw_contact_in_input": False,
        "dataset_path": str(output / "dataset.pt"),
        "dataset_sha256": hash_file(output / "dataset.pt"),
        "source_summary": {
            "path": str(SOURCE_SUMMARY),
            "sha256": hash_file(SOURCE_SUMMARY),
        },
        "source_hashes": source_hashes(),
        "preparation_wall_time_s": time.monotonic() - started,
        "config": {"path": str(config_path), "sha256": hash_file(config_path)},
        "physical_model_hash": physical.stable_hash(),
        "scope": "historical successful teacher requests; not new C_H acceptance or autonomous execution",
    }
    write_json(output / "dataset_manifest.json", report)
    return report


def source_hashes():
    paths = (
        Path(__file__),
        ROOT / "amsrr/policies/request_high_level_policy.py",
        ROOT / "amsrr/policies/high_level_requests.py",
        ROOT / "amsrr/schemas/high_level.py",
        ROOT / "amsrr/encoders/morphology_graph_encoder.py",
        ROOT / "amsrr/policies/contact_group_geometry.py",
        ROOT / "amsrr/robot_model/urdf_transforms.py",
    )
    return {str(p.relative_to(ROOT)): hash_file(p) for p in paths}


def score_dataset(policy, data, indices, *, device, batch_size=256):
    scores = []
    policy.eval()
    with torch.no_grad():
        for batch in indices.split(batch_size):
            values = policy.forward_encoded(
                data["features"][batch].to(device),
                graph_batch(data["graphs"], batch, device),
                data["mask"][batch].to(device),
                candidates=data["candidate_features"][batch].to(device),
                membership=data["membership"][batch].to(device),
            )
            scores.append(values.cpu())
    return torch.cat(scores)


def ranking_metrics(scores, rows):
    predictions = scores.argmax(1).tolist()
    labels = [r["label"] for r in rows]
    result = {
        "count": len(rows),
        "top1_accuracy": sum(p == y for p, y in zip(predictions, labels)) / len(rows),
    }
    by_category = {}
    for category in CATEGORIES:
        pairs = [
            (p, y)
            for p, y, r in zip(predictions, labels, rows)
            if r["category"] == category
        ]
        if pairs:
            by_category[category] = {
                "count": len(pairs),
                "accuracy": sum(p == y for p, y in pairs) / len(pairs),
            }
    result["categories"] = by_category
    result["balanced_accuracy"] = sum(
        v["accuracy"] for k, v in by_category.items() if k != "forced"
    ) / sum(k != "forced" for k in by_category)
    result["per_output_accuracy"] = {
        field: sum(
            r["requests"][p][field] == r["requests"][y][field]
            for p, y, r in zip(predictions, labels, rows)
        )
        / len(rows)
        for field in ("contact_group_id", "transition_id", "subgoal_id")
    }
    result["heuristic_top1_accuracy"] = sum(
        r["heuristic_index"] == r["label"] for r in rows
    ) / len(rows)
    return result


def validation_selection_accuracy(metrics, selection_scope):
    """Select contact warmup by independent initial choices, not teacher states."""
    if selection_scope == "contact_group":
        initial = metrics["categories"].get("initial_group")
        if initial is None or initial["count"] < 1:
            raise ValueError("contact selection requires initial validation examples")
        return initial["accuracy"]
    return metrics["balanced_accuracy"]


def validate_contact_group_dataset(data):
    """Only actual initial choices can supervise initial-contact warmup."""
    seen = set()
    excluded_features = [
        REQUEST_FEATURE_NAMES.index(name)
        for name in (
            "same_contact_group",
            "elapsed_phase_s",
            "nominal_phase_duration_s",
            "nominal_phase_progress",
        )
    ]
    for index, row in enumerate(data["rows"]):
        if (
            row["raw_index"] != 0
            or row["category"] != "initial_group"
            or (row["episode_id"], row["raw_index"]) in seen
        ):
            raise ValueError(
                "contact-only dataset requires unique observed-state queries"
            )
        seen.add((row["episode_id"], row["raw_index"]))
        mask = data["mask"][index]
        requests = row["requests"]
        expected = torch.zeros_like(mask)
        for j, request in enumerate(requests):
            expected[j] = (
                request["contact_group_id"] is not None
                and request["transition_id"] is None
            )
        if not torch.equal(mask, expected) or not bool(mask[row["label"]]):
            raise ValueError(
                "contact-only mask or teacher label includes a temporal choice"
            )
        eligible = [r for j, r in enumerate(requests) if mask[j]]
        groups = [r["contact_group_id"] for r in eligible]
        if (
            not groups
            or len(groups) != len(set(groups))
            or len({r["subgoal_id"] for r in eligible}) != 1
        ):
            raise ValueError("contact-only candidates must vary only in contact group")
        if bool(data["features"][index, mask][:, excluded_features].ne(0).any()):
            raise ValueError("contact-only initial input contains execution history")


def train_policy(
    dataset_manifest: Path,
    output: Path,
    *,
    epochs=100,
    patience=15,
    batch_size=256,
    learning_rate=0.001,
    seed=7,
    timeout_s=1200,
    device="cuda",
    evaluate_held_out=False,
    stopping_rule="validation",
):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    if stopping_rule not in {"validation", "train_fit"}:
        raise ValueError("unknown training stopping rule")
    started = time.monotonic()
    manifest = _read_json(dataset_manifest)
    if manifest["source_hashes"] != source_hashes():
        raise ValueError("request dataset source changed")
    path = Path(manifest["dataset_path"])
    if hash_file(path) != manifest["dataset_sha256"]:
        raise ValueError("request dataset bytes changed")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["feature_version"] != REQUEST_FEATURE_VERSION:
        raise ValueError("request feature version differs")
    torch.manual_seed(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    selection_scope = manifest["selection_scope"]
    if data["selection_scope"] != selection_scope:
        raise ValueError("request dataset selection scope differs")
    policy = RequestHighLevelPolicy(
        RequestHighLevelPolicyConfig(selection_scope=selection_scope)
    ).to(device)
    splits = {
        s: torch.tensor(
            [i for i, r in enumerate(data["rows"]) if r["split"] == s], dtype=torch.long
        )
        for s in ("train", "validation", "held_out")
    }
    if any(not len(i) for i in splits.values()):
        raise ValueError("all three dataset splits are required")
    train = splits["train"]
    if selection_scope == "contact_group":
        validate_contact_group_dataset(data)
    usable = train[
        torch.tensor([data["rows"][i]["category"] != "forced" for i in train])
    ]
    raw = data["features"][train][data["mask"][train]]
    with torch.no_grad():
        policy.feature_mean.copy_(raw.mean(0).to(device))
        scale = raw.std(0)
        scale = torch.where(scale > 1e-5, scale, torch.ones_like(scale))
        policy.feature_scale.copy_(scale.to(device))
        used_candidates = (
            data["membership"][train] & data["mask"][train, :, None]
        ).any(1)
        local = data["candidate_features"][train][used_candidates]
        policy.contact_mean.copy_(local.mean(0).to(device))
        policy.contact_scale.copy_(local.std(0).clamp_min(0.05).to(device))
    category_counts = Counter(data["rows"][i]["category"] for i in usable.tolist())
    sample_weights = torch.tensor(
        [1.0 / category_counts[data["rows"][i]["category"]] for i in usable.tolist()]
    )
    labels = torch.tensor([r["label"] for r in data["rows"]], dtype=torch.long)
    weight_decay = 0.01 if selection_scope == "contact_group" else 1e-4
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    config = {
        "epochs": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "sampling": (
            "all_train_rows_shuffled"
            if selection_scope == "contact_group"
            else "category_balanced"
        ),
        "seed": seed,
        "device": device,
        "timeout_s": timeout_s,
        "selection_metric": "validation balanced accuracy over initial_group/continuation/transition",
        "selection_scope": selection_scope,
        "held_out_evaluation_requested": evaluate_held_out,
        "stopping_rule": stopping_rule,
        "train_target_accuracy": 1.0 if stopping_rule == "train_fit" else None,
        "source_hashes": source_hashes(),
        "dataset_manifest_sha256": hash_file(dataset_manifest),
    }
    if selection_scope == "contact_group":
        config["selection_metric"] = "validation initial-group accuracy"
    config["validation_selection_metric"] = (
        "validation initial-group accuracy"
        if selection_scope == "contact_group"
        else "validation top1 request accuracy"
    )
    if stopping_rule == "train_fit":
        config["selection_metric"] = (
            "train exact agreement, tie-break train cross-entropy"
        )
    write_json(output / "training_config.json", config)
    best, best_epoch, stale, history = -1.0, 0, 0, []
    best_loss, validation_best, validation_best_epoch = float("inf"), -1.0, 0
    stop_reason = "epoch_limit"
    training_rows = [data["rows"][i] for i in train.tolist()]
    validation_rows = [data["rows"][i] for i in splits["validation"].tolist()]
    baseline = ranking_metrics(
        score_dataset(policy, data, splits["validation"], device=device),
        validation_rows,
    )
    validation_selection_accuracy(baseline, selection_scope)
    write_json(output / "untrained_validation.json", baseline)
    for epoch in range(1, epochs + 1):
        policy.train()
        sampled = usable[
            (
                torch.randperm(len(usable))
                if selection_scope == "contact_group"
                else torch.multinomial(sample_weights, len(usable), replacement=True)
            )
        ]
        losses = []
        for batch in sampled.split(batch_size):
            if time.monotonic() - started > timeout_s:
                raise TimeoutError("request imitation training deadline")
            optimizer.zero_grad(set_to_none=True)
            scores = policy.forward_encoded(
                data["features"][batch].to(device),
                graph_batch(data["graphs"], batch, device),
                data["mask"][batch].to(device),
                candidates=data["candidate_features"][batch].to(device),
                membership=data["membership"][batch].to(device),
            )
            loss = F.cross_entropy(scores, labels[batch].to(device))
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite request imitation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        train_scores = score_dataset(policy, data, train, device=device)
        training_metrics = ranking_metrics(train_scores, training_rows)
        training_loss = float(F.cross_entropy(train_scores, labels[train]))
        validation = ranking_metrics(
            score_dataset(policy, data, splits["validation"], device=device),
            validation_rows,
        )
        item = {
            "epoch": epoch,
            "loss": sum(losses) / len(losses),
            "validation": validation,
            "train": training_metrics,
            "train_evaluation_loss": training_loss,
            "wall_time_s": time.monotonic() - started,
        }
        history.append(item)
        print(json.dumps(item), flush=True)
        with (output / "epochs.jsonl").open("a") as stream:
            stream.write(json.dumps(item) + "\n")
        validation_metric = validation_selection_accuracy(validation, selection_scope)
        auxiliary_metric = (
            validation_metric
            if selection_scope == "contact_group"
            else validation["top1_accuracy"]
        )
        if auxiliary_metric > validation_best + 1e-6:
            validation_best, validation_best_epoch = auxiliary_metric, epoch
            torch.save(policy.checkpoint(), output / "validation_best.pt")
        metric = (
            training_metrics["top1_accuracy"]
            if stopping_rule == "train_fit"
            else validation_metric
        )
        if metric > best + 1e-6 or (
            stopping_rule == "train_fit"
            and metric == best
            and training_loss < best_loss
        ):
            best, best_epoch, stale, best_loss = metric, epoch, 0, training_loss
            torch.save(policy.checkpoint(), output / "checkpoint.pt")
        else:
            stale += 1
        if stopping_rule == "train_fit" and training_metrics["top1_accuracy"] == 1.0:
            stop_reason = "train_exact_match"
            break
        if stopping_rule == "validation" and stale >= patience:
            stop_reason = "validation_patience"
            break
    torch.save(policy.checkpoint(), output / "last.pt")
    # Held-out evaluation is opt-in and always follows completed model selection.
    policy = RequestHighLevelPolicy.load(output / "checkpoint.pt").to(device)
    metrics = {}
    for split, indices in splits.items():
        if split == "held_out" and not evaluate_held_out:
            continue
        scores = score_dataset(policy, data, indices, device=device)
        metrics[split] = ranking_metrics(
            scores, [data["rows"][i] for i in indices.tolist()]
        )
        if split == "held_out":
            torch.save(
                {"indices": indices, "scores": scores, "predictions": scores.argmax(1)},
                output / "held_out_predictions.pt",
            )
    report = {
        "version": VERSION,
        "status": "training_complete",
        "selection_scope": selection_scope,
        "best_epoch": best_epoch,
        "completed_epochs": len(history),
        "stopping_rule": stopping_rule,
        "stop_reason": stop_reason,
        "validation_best_epoch": validation_best_epoch,
        "validation_best_accuracy": validation_best,
        "validation_selection_metric": config["validation_selection_metric"],
        "final_epoch_train": history[-1]["train"],
        "final_epoch_validation": history[-1]["validation"],
        "metrics": metrics,
        "checkpoint_path": str(output / "checkpoint.pt"),
        "checkpoint_sha256": hash_file(output / "checkpoint.pt"),
        "wall_time_s": time.monotonic() - started,
        "source_hashes": source_hashes(),
        "dataset_manifest_sha256": hash_file(dataset_manifest),
        "held_out_used_for_selection": False,
        "held_out_evaluated": evaluate_held_out,
        "autonomous_isaac_task_success_evaluated": False,
    }
    write_json(output / "training_result.json", report)
    return report
