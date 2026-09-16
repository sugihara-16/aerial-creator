from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from tests.unit.policies.test_high_level_requests import request_scene, decision
from amsrr.policies.request_high_level_policy import (
    REQUEST_FEATURE_NAMES,
    RequestHighLevelPolicy,
    RequestHighLevelPolicyConfig,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.policies.contact_group_geometry import contact_geometry
from amsrr.schemas.irg import IRGEdgeType, IRGNodeType
from amsrr.schemas.policies import (
    ContactWrenchTrajectory,
    InteractionKnot,
    ObjectTarget,
)
from amsrr.training.request_imitation import (
    PHASES,
    _graph_dict,
    _pad_graphs,
    graph_batch,
    ranking_metrics,
    teacher_request_scene,
    uncommitted_teacher_candidates,
    validate_contact_group_dataset,
    validation_selection_accuracy,
)


def test_batched_training_is_identical_to_runtime_and_updates_graph_encoder(
    request_scene,
):
    torch.manual_seed(2)
    context = decision(request_scene)
    model = RequestHighLevelPolicy()
    with torch.no_grad():
        model.feature_mean[:] = 0.2
        model.feature_scale[:] = 2.0
    features = model.request_features(context)
    tensorized = model.morphology_encoder.tensorizer.tensorize(
        [context.scene.morphology_graph],
        runtime_observations=[context.scene.runtime_observation],
    )
    graphs = _pad_graphs([_graph_dict(tensorized), _graph_dict(tensorized)])
    mask = torch.ones((2, len(features)), dtype=torch.bool)
    mask[1, -1] = False
    candidates, membership = contact_geometry(context)
    result = model.forward_encoded(
        features[None].repeat(2, 1, 1),
        graph_batch(graphs, torch.arange(2), "cpu"),
        mask,
        candidates=candidates[None].repeat(2, 1, 1),
        membership=membership[None].repeat(2, 1, 1),
    )
    torch.testing.assert_close(result[0], model(context), rtol=1e-5, atol=1e-6)
    assert torch.isneginf(result[1, -1])
    loss = torch.nn.functional.cross_entropy(result, torch.zeros(2, dtype=torch.long))
    loss.backward()
    gradient = model.morphology_encoder.node_projection[0].weight.grad
    assert (
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient.abs().max() > 0
    )
    restored = RequestHighLevelPolicy.from_checkpoint(model.checkpoint())
    torch.testing.assert_close(restored(context), model(context))


def test_owner_geometry_reaches_features_without_changing_group_average(request_scene):
    model = RequestHighLevelPolicy()
    before = model.request_features(decision(request_scene))
    scene, task, physical, execution = deepcopy(request_scene)
    for state in scene.runtime_observation.module_states:
        state.pose_world = (state.pose_world[0] + 0.13, *state.pose_world[1:])
    after = model.request_features(decision((scene, task, physical, execution)))
    group = [
        i
        for i, name in enumerate(REQUEST_FEATURE_NAMES)
        if name.startswith("contact_group.")
    ]
    owner = [
        i
        for i, name in enumerate(REQUEST_FEATURE_NAMES)
        if name.startswith("contact_owner.")
    ]
    torch.testing.assert_close(before[:, group], after[:, group])
    assert not torch.allclose(before[:, owner], after[:, owner])


def test_invalid_normalizer_is_rejected():
    model = RequestHighLevelPolicy()
    checkpoint = model.checkpoint()
    checkpoint["state_dict"]["feature_scale"][0] = 0.0
    with pytest.raises(SchemaValidationError, match="normalization"):
        RequestHighLevelPolicy.from_checkpoint(checkpoint)


def test_teacher_schedule_has_all_executed_phases_without_mutating_inputs(
    request_scene,
):
    scene, task, physical, _ = request_scene
    obj = task.scene.objects[0]
    paths = {}
    for index, phase in enumerate(PHASES):
        pose = (obj.pose_world[0] + 0.01 * index, *obj.pose_world[1:])
        paths[phase] = ContactWrenchTrajectory(
            1.0,
            1.0,
            [
                InteractionKnot(
                    t, [], object_targets=[ObjectTarget(obj.object_id, pose)]
                )
                for t in (0.0, 1.0)
            ],
        )
    bundle = SimpleNamespace(
        morphology=scene.morphology_graph,
        contact_candidate_set=scene.contact_candidate_set,
        phase_trajectories=paths,
    )
    before = {k: v.to_dict() for k, v in paths.items()}
    adapted, ids = teacher_request_scene(bundle, task, plan_committed=True)
    assert tuple(ids) == PHASES
    temporal = [
        e for e in adapted.irg.edges if e.edge_type == IRGEdgeType.TEMPORAL_NEXT
    ]
    assert [(e.src_id, e.dst_id) for e in temporal] == list(
        zip(list(ids.values())[:-1], list(ids.values())[1:])
    )
    assert len([n for n in adapted.irg.nodes if n.node_type == IRGNodeType.PHASE]) == 8
    assert before == {k: v.to_dict() for k, v in paths.items()}
    initial, _ = teacher_request_scene(bundle, task)
    # No chosen teacher trajectory may affect the first request's input.
    missing_plan = SimpleNamespace(
        morphology=bundle.morphology,
        contact_candidate_set=bundle.contact_candidate_set,
        phase_trajectories={},
    )
    alternative, _ = teacher_request_scene(missing_plan, task, vertical_scale=0.123)
    assert initial.irg.to_dict() == alternative.irg.to_dict()
    assert (
        initial.interaction_envelope.to_dict()
        == alternative.interaction_envelope.to_dict()
    )
    assert all(
        n.feature.get("nominal_duration_s") is None
        for n in initial.irg.nodes
        if n.node_type == IRGNodeType.PHASE
    )


def test_balanced_metric_exposes_always_continue_failure():
    requests = [
        {"contact_group_id": "g", "transition_id": None, "subgoal_id": "hold"},
        {"contact_group_id": "g", "transition_id": "next", "subgoal_id": "move"},
    ]
    rows = [
        {
            "label": 0,
            "category": "continuation",
            "requests": requests,
            "heuristic_index": 0,
        }
        for _ in range(99)
    ]
    rows.append(
        {
            "label": 1,
            "category": "transition",
            "requests": requests,
            "heuristic_index": 0,
        }
    )
    metrics = ranking_metrics(torch.tensor([[1.0, 0.0]] * 100), rows)
    assert metrics["top1_accuracy"] == 0.99
    assert metrics["balanced_accuracy"] == 0.5


def test_teacher_selected_group_refinement_cannot_change_initial_geometry(
    request_scene,
):
    from amsrr.training.order9_r1_support_clearance_teacher_v12 import (
        _shift_contact_candidate_set,
    )

    scene, task, physical, execution = request_scene
    source = scene.contact_candidate_set
    baseline, _ = contact_geometry(decision(request_scene))
    # Use the actual legacy producer: it refines only the teacher's chosen group.
    for index, group in enumerate(source.group_proposals[:2]):
        refined = _shift_contact_candidate_set(
            source,
            selected_candidate_ids=frozenset(group.candidate_ids),
            shift_m=0.015 * (index + 1),
        )
        before = refined.to_dict()
        bundle = SimpleNamespace(
            morphology=scene.morphology_graph,
            contact_candidate_set=refined,
            phase_trajectories={},
        )
        initial, _ = teacher_request_scene(bundle, task)
        runtime_scene = replace(
            deepcopy(scene), contact_candidate_set=initial.contact_candidate_set
        )
        actual, _ = contact_geometry(
            decision((runtime_scene, task, physical, execution))
        )
        torch.testing.assert_close(actual, baseline, rtol=0, atol=1e-6)
        assert refined.to_dict() == before
        assert all(
            not any(k.startswith("r1_") for k in c.candidate_scores)
            for c in initial.contact_candidate_set.candidates
        )


def test_runtime_rejects_refined_teacher_candidates_before_commit(request_scene):
    scene, _, _, execution = request_scene
    candidate = scene.contact_candidate_set.candidates[0]
    candidate.candidate_scores["r1_vertical_clearance_shift_m"] = 0.015
    with pytest.raises(SchemaValidationError, match="post-selection"):
        RequestHighLevelPolicy().rank(decision(request_scene))
    # A previously committed plan may legitimately expose its refined geometry.
    execution.plan_id = "committed-plan"
    features, _ = contact_geometry(decision(request_scene))
    assert torch.isfinite(features).all()


@pytest.mark.parametrize(
    "markers, message",
    [
        ({"r1_vertical_clearance_shift_m": float("nan")}, "nonfinite"),
        ({"r1_planar_contact_frame_shift_world_x_m": 0.002}, "unsupported"),
    ],
)
def test_unproven_refinement_cannot_silently_enter_initial_inputs(
    request_scene, markers, message
):
    candidates = deepcopy(request_scene[0].contact_candidate_set)
    candidates.candidates[0].candidate_scores.update(markers)
    with pytest.raises(ValueError, match=message):
        uncommitted_teacher_candidates(candidates)


@pytest.mark.parametrize(
    "old_version",
    [
        "request_measured_gripper_member_geometry_v3",
        "request_preselection_gripper_member_geometry_v4",
    ],
)
def test_checkpoint_from_incorrect_geometry_requires_retraining(old_version):
    checkpoint = RequestHighLevelPolicy().checkpoint()
    checkpoint["feature_version"] = old_version
    with pytest.raises(SchemaValidationError, match="compatible"):
        RequestHighLevelPolicy.from_checkpoint(checkpoint)


def test_contact_only_scope_masks_transitions_and_survives_reload(request_scene):
    model = RequestHighLevelPolicy(
        RequestHighLevelPolicyConfig(selection_scope="contact_group")
    )
    context = decision(request_scene)
    allowed = model.selection_mask(context)
    assert allowed.any() and (~allowed).any()
    scores = model(context)
    scores.retain_grad()
    target = int(torch.nonzero(allowed)[0])
    torch.nn.functional.cross_entropy(scores[None], torch.tensor([target])).backward()
    assert torch.count_nonzero(scores.grad[~allowed]) == 0
    assert torch.isneginf(scores[~allowed]).all()
    ranked = model.rank(context)
    assert ranked and all(r.transition_id is None for r in ranked)
    assert len({r.subgoal_id for r in ranked}) == 1
    restored = RequestHighLevelPolicy.from_checkpoint(model.checkpoint())
    assert restored.config.selection_scope == "contact_group"
    assert restored.rank(context) == ranked
    request_scene[3].plan_id = "previously-committed-plan"
    with pytest.raises(SchemaValidationError, match="uncommitted"):
        restored.rank(decision(request_scene))


def test_contact_only_dataset_rejects_later_history_and_temporal_labels(request_scene):
    model = RequestHighLevelPolicy(
        RequestHighLevelPolicyConfig(selection_scope="contact_group")
    )
    context = decision(request_scene)
    mask = model.selection_mask(context)
    features = model.request_features(context)
    data = {
        "rows": [
            {
                "episode_id": "e0",
                "raw_index": 0,
                "category": "initial_group",
                "label": int(torch.nonzero(mask)[0]),
                "requests": [e.request.to_dict() for e in context.catalog.entries],
            }
        ],
        "mask": mask[None],
        "features": features[None],
    }
    validate_contact_group_dataset(data)
    bad = deepcopy(data)
    bad["rows"][0]["raw_index"] = 1
    with pytest.raises(ValueError, match="unique observed"):
        validate_contact_group_dataset(bad)
    bad = deepcopy(data)
    bad["rows"][0]["label"] = int(torch.nonzero(~mask)[0])
    with pytest.raises(ValueError, match="temporal choice"):
        validate_contact_group_dataset(bad)
    bad = deepcopy(data)
    bad["features"][0, mask, REQUEST_FEATURE_NAMES.index("same_contact_group")] = 1
    with pytest.raises(ValueError, match="execution history"):
        validate_contact_group_dataset(bad)


def test_contact_checkpoint_selection_ignores_better_teacher_state_accuracy():
    # First model gets every initial choice right but only 40% of later states.
    # Second gets 99% of later states, but fails half the independent choices.
    initial_best = {
        "top1_accuracy": 0.46,
        "balanced_accuracy": 0.7,
        "categories": {"initial_group": {"count": 10, "accuracy": 1.0}},
    }
    later_best = {
        "top1_accuracy": 0.941,
        "balanced_accuracy": 0.745,
        "categories": {"initial_group": {"count": 10, "accuracy": 0.5}},
    }
    assert validation_selection_accuracy(initial_best, "contact_group") == 1.0
    assert validation_selection_accuracy(later_best, "contact_group") == 0.5
    assert validation_selection_accuracy(initial_best, "requests") == 0.7
    with pytest.raises(ValueError, match="initial validation"):
        validation_selection_accuracy({"categories": {}}, "contact_group")


@pytest.mark.parametrize("stopping_rule", ["validation", "train_fit"])
def test_contact_stage_training_never_scores_held_out_by_default(
    request_scene, tmp_path, monkeypatch, stopping_rule
):
    from amsrr.training import request_imitation as training
    from amsrr.utils.hashing import hash_file

    model = RequestHighLevelPolicy(
        RequestHighLevelPolicyConfig(selection_scope="contact_group")
    )
    context = decision(request_scene)
    mask = model.selection_mask(context)
    candidates, membership = contact_geometry(context)
    graph = model.morphology_encoder.tensorizer.tensorize(
        [context.scene.morphology_graph],
        runtime_observations=[context.scene.runtime_observation],
    )
    data = {
        "feature_version": model.config.feature_version,
        "selection_scope": "contact_group",
        "features": model.request_features(context)[None].repeat(3, 1, 1),
        "mask": mask[None].repeat(3, 1),
        "candidate_features": candidates[None].repeat(3, 1, 1),
        "membership": membership[None].repeat(3, 1, 1),
        "graphs": _pad_graphs([_graph_dict(graph)] * 3),
        "rows": [
            {
                "episode_id": split,
                "split": split,
                "raw_index": 0,
                "category": "initial_group",
                "label": int(torch.nonzero(mask)[0]),
                "heuristic_index": int(torch.nonzero(mask)[0]),
                "requests": [e.request.to_dict() for e in context.catalog.entries],
            }
            for split in ("train", "validation", "held_out")
        ],
    }
    dataset = tmp_path / "dataset.pt"
    torch.save(data, dataset)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_hashes": training.source_hashes(),
                "dataset_path": str(dataset),
                "dataset_sha256": hash_file(dataset),
                "selection_scope": "contact_group",
            }
        )
    )
    score = training.score_dataset
    scored_indices = []

    def checked_score(policy, payload, indices, **kwargs):
        assert 2 not in indices.tolist(), "held-out inference is prohibited in warmup"
        scored_indices.extend(indices.tolist())
        result = score(policy, payload, indices, **kwargs)
        if stopping_rule == "train_fit":
            # Simulate a measured plateau below exact fit: patience must not stop it.
            target = payload["rows"][int(indices[0])]["label"]
            result[:, target] = result[:, mask].min() - 1
        return result

    monkeypatch.setattr(training, "score_dataset", checked_score)
    report = training.train_policy(
        manifest,
        tmp_path / "training",
        epochs=3,
        patience=1,
        batch_size=1,
        device="cpu",
        stopping_rule=stopping_rule,
    )
    assert set(scored_indices) == {0, 1}
    assert set(report["metrics"]) == {"train", "validation"}
    assert not report["held_out_evaluated"]
    assert report["validation_selection_metric"] == "validation initial-group accuracy"
    assert not (tmp_path / "training/held_out_predictions.pt").exists()
    assert (tmp_path / "training/last.pt").exists()
    if stopping_rule == "train_fit":
        assert report["completed_epochs"] == 3
        assert report["stop_reason"] == "epoch_limit"
        history = [
            json.loads(line)
            for line in (tmp_path / "training/epochs.jsonl").read_text().splitlines()
        ]
        assert all(row["train"]["count"] == 1 for row in history)
    restored = RequestHighLevelPolicy.load(report["checkpoint_path"])
    assert all(r.transition_id is None for r in restored.rank(context))


def test_contact_features_use_observed_joints_and_require_ancestor_feedback(
    request_scene,
):
    before, _ = contact_geometry(decision(request_scene))
    changed = deepcopy(request_scene)
    for module in changed[0].runtime_observation.module_states:
        module.joint_positions = {k: 0.2 for k in module.joint_positions}
    after, _ = contact_geometry(decision(changed))
    # Module origins and targets have not moved; only motor feedback changed.
    assert not torch.allclose(before, after)
    for module in changed[0].runtime_observation.module_states:
        module.joint_positions = {}
    with pytest.raises(SchemaValidationError, match="ancestor joint missing"):
        contact_geometry(decision(changed))


@pytest.mark.parametrize("joint_angle", [0.0, 0.23])
def test_observed_anchor_fk_uses_module_frame_not_urdf_root(request_scene, joint_angle):
    from amsrr.feasibility.articulated_reachability import (
        resolve_mesh_backed_anchor_references,
    )
    from amsrr.geometry.pose_math import transform_from_pose
    from amsrr.policies.contact_group_geometry import observed_anchor_poses
    from amsrr.robot_model.whole_structure_kinematics import (
        WholeStructureKinematics,
        ordered_global_dock_joint_ids,
    )

    scene, task, physical, execution = deepcopy(request_scene)
    graph = scene.morphology_graph
    references = resolve_mesh_backed_anchor_references(
        graph, physical, [a.anchor_id for a in graph.robot_anchors]
    )
    positions = {
        name: joint_angle * (-1 if index % 2 else 1)
        for index, name in enumerate(ordered_global_dock_joint_ids(graph, physical))
    }
    # Independent PhysicalModel FK already uses the controller/Isaac module
    # frame. Nontrivial body rotation also catches world-axis offset patches.
    reference = WholeStructureKinematics().forward(
        graph, physical, positions, (1.2, -0.8, 0.9, 0.5, 0.5, 0.5, 0.5), references
    )
    for state in scene.runtime_observation.module_states:
        state.pose_world = reference.module_root_poses_world[state.module_id]
        for name in state.joint_positions:
            state.joint_positions[name] = positions.get(
                f"module_{state.module_id}:{name}", 0.0
            )
    actual = observed_anchor_poses(decision((scene, task, physical, execution)))
    for anchor_id, expected in reference.anchor_poses_world.items():
        torch.testing.assert_close(
            torch.tensor(actual[anchor_id][:3], dtype=torch.float64),
            torch.tensor(expected[:3], dtype=torch.float64),
            rtol=0,
            atol=1e-9,
        )
        torch.testing.assert_close(
            torch.tensor(transform_from_pose(actual[anchor_id]).rotation),
            torch.tensor(transform_from_pose(expected).rotation),
            rtol=0,
            atol=1e-6,
        )


def test_contact_selector_ignores_old_world_features_and_candidate_order(request_scene):
    model = RequestHighLevelPolicy(
        RequestHighLevelPolicyConfig(selection_scope="contact_group")
    ).eval()
    context = decision(request_scene)
    candidates, membership = contact_geometry(context)
    features = model.request_features(context)[None]
    mask = model.selection_mask(context)[None]
    # None graph is intentional: the local selector must not read it.
    expected = model.forward_encoded(
        features, None, mask, candidates=candidates[None], membership=membership[None]
    )
    perm = torch.arange(len(candidates) - 1, -1, -1)
    actual = model.forward_encoded(
        features + 100,
        None,
        mask,
        candidates=candidates[perm][None],
        membership=membership[:, perm][None],
    )
    torch.testing.assert_close(actual, expected)
    actual[mask].sum().backward()
    assert model.contact_member[0].weight.grad.abs().max() > 0
    assert model.morphology_encoder.node_projection[0].weight.grad is None


def test_contact_features_invariant_to_world_xy_translation(request_scene):
    before, membership = contact_geometry(decision(request_scene))
    changed = deepcopy(request_scene)
    scene = changed[0]

    def translate(p):
        return (p[0] + 4.2, p[1] - 2.3, *p[2:])

    for state in (
        *scene.runtime_observation.module_states,
        *scene.runtime_observation.object_states,
    ):
        state.pose_world = translate(state.pose_world)
    for candidate in scene.contact_candidate_set.candidates:
        candidate.contact_pose_world = translate(candidate.contact_pose_world)
        candidate.contact_frame_world = translate(candidate.contact_frame_world)
    for obj in changed[1].scene.objects:
        obj.pose_world = translate(obj.pose_world)
    after, members = contact_geometry(decision(changed))
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
    assert torch.equal(membership, members)
