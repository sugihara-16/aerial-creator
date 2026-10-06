"""Measured contact features matching the legacy v32 local selector.

IDs index catalog membership; arbitrary candidate/group/entity names are not
numeric inputs. FK uses observed motor positions, never a teacher target pose.
"""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import math

import numpy as np
import torch

from amsrr.geometry.pose_math import compose_pose, inverse_pose, transform_from_pose
from amsrr.policies.contact_candidate_encoder import ContactCandidateEncoder
from amsrr.robot_model.urdf_loader import load_urdf
from amsrr.robot_model.urdf_transforms import (
    link_poses_at_joint_positions,
    module_base_link_name,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.utils.hashing import hash_file

PORTS = (
    "pitch_connect_point_1",
    "pitch_connect_point_2",
    "yaw_connect_point_1",
    "yaw_connect_point_2",
)
# Exactly the columns consumed by the successful legacy v32 local selector.
LOCAL_COLUMNS = (
    2,
    5,
    *range(6, 19),
    *range(21, 31),
    *range(33, 41),
    *range(48, 57),
    *range(75, 88),
)
CONTACT_FEATURE_DIM = len(LOCAL_COLUMNS)
CONTACT_NUMERIC_FEATURE_CONTRACT = "physical_scores_without_ids_or_catalog_counts_v1"


def normalize_contact_features(candidates, mean, scale):
    """Keep checkpoint width while excluding proposal enumeration statistics.

    Catalog membership/coverage counts depend on the sampling budget, not the
    quality of a physical contact. Mask after normalization so old checkpoint
    means cannot reintroduce these nonphysical signals.
    """
    local = (candidates - mean) / scale
    local = local.clone()
    local[..., [LOCAL_COLUMNS.index(17), LOCAL_COLUMNS.index(18)]] = 0
    return local


def relative_pose_features(parent, child):
    p, c = transform_from_pose(parent), transform_from_pose(child)
    rp, rc = np.asarray(p.rotation), np.asarray(c.rotation)
    position = rp.T @ (np.asarray(c.translation) - np.asarray(p.translation))
    rotation = rp.T @ rc
    return [*position, *rotation[:, 0], *rotation[:, 1]]


@lru_cache(maxsize=8)
def _urdf(path, sha256):
    if hash_file(path) != sha256:
        raise SchemaValidationError("contact geometry URDF identity changed")
    model = load_urdf(path)
    if len(model.root_links) != 1:
        raise SchemaValidationError("contact geometry requires a tree URDF")
    return model


@lru_cache(maxsize=4096)
def _link_poses(path, sha256, positions):
    return link_poses_at_joint_positions(_urdf(path, sha256), dict(positions))


def observed_anchor_poses(context):
    model = context.physical_model
    path = str(Path(model.urdf_path).resolve())
    sha = model.metadata["urdf_hash"]
    urdf = _urdf(path, sha)
    base_link = module_base_link_name(urdf)
    parents = {j.child_link: j for j in urdf.joints}
    modules = {m.module_id: m for m in context.observation.module_states}
    result = {}
    for anchor in context.scene.morphology_graph.robot_anchors:
        module = modules[anchor.module_id]
        for link in (anchor.link_id, base_link):
            while link in parents:
                joint = parents[link]
                if (
                    joint.joint_type != "fixed"
                    and joint.name not in module.joint_positions
                ):
                    raise SchemaValidationError(
                        "observed anchor ancestor joint missing"
                    )
                link = joint.parent_link
        poses = _link_poses(path, sha, tuple(sorted(module.joint_positions.items())))
        # Runtime module poses refer to the configured baselink (Holon: fc).
        # URDF FK returns root-link transforms; convert before composing them.
        anchor_in_module = compose_pose(
            inverse_pose(poses[base_link]),
            compose_pose(poses[anchor.link_id], anchor.local_pose),
        )
        result[anchor.anchor_id] = compose_pose(module.pose_world, anchor_in_module)
    return result


def contact_geometry(context):
    """Return [candidate,55] local features and [request,candidate] membership."""
    context.validate_snapshot()
    return _contact_geometry(context)


def _contact_geometry(context):
    """Calculate geometry for an already validated, unchanged context."""
    if context.execution_state.plan_id is None and any(
        key.startswith("r1_")
        for candidate in context.scene.contact_candidate_set.candidates
        for key in candidate.candidate_scores
    ):
        raise SchemaValidationError(
            "post-selection teacher candidate refinement in uncommitted actor input"
        )
    if len(context.observation.object_states) != 1:
        raise SchemaValidationError("legacy local profile requires one rigid object")
    obj = context.observation.object_states[0]
    if obj.generalized_q is not None:
        raise SchemaValidationError("legacy local profile requires a rigid object")
    rotation = np.asarray(transform_from_pose(obj.pose_world).rotation)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    frame = (*obj.pose_world[:2], 0.0, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    transform = inverse_pose(frame)
    rot = np.asarray(transform_from_pose(transform).rotation)
    candidates = deepcopy(context.scene.contact_candidate_set)
    # Same passive planar-frame change as legacy v29, retaining height/gravity.
    for candidate in candidates.candidates:
        candidate.contact_pose_world = compose_pose(
            transform, candidate.contact_pose_world
        )
        candidate.contact_frame_world = compose_pose(
            transform, candidate.contact_frame_world
        )
        candidate.normal_world = tuple(rot @ candidate.normal_world)
        candidate.tangent_basis_world = tuple(
            (np.asarray(candidate.tangent_basis_world).reshape(2, 3) @ rot.T).reshape(
                -1
            )
        )
        for key in list(candidate.candidate_scores):
            if key.endswith("_world_x_m"):
                prefix = key[:-3]
                keys = [prefix + axis + "_m" for axis in "xyz"]
                if not all(k in candidate.candidate_scores for k in keys):
                    raise SchemaValidationError(
                        "incomplete world vector in candidate scores"
                    )
                values = rot @ [candidate.candidate_scores[k] for k in keys]
                candidate.candidate_scores.update(zip(keys, map(float, values)))
    raw = np.asarray(
        ContactCandidateEncoder(d_model=48).encode(candidates).candidate_tokens()
    )
    anchors = {a.anchor_id: a for a in context.scene.morphology_graph.robot_anchors}
    modules = {m.module_id: m for m in context.observation.module_states}
    base = modules[context.scene.morphology_graph.base_module_id].pose_world
    measured = observed_anchor_poses(context)
    rows = []
    for i, c in enumerate(context.scene.contact_candidate_set.candidates):
        anchor = anchors[c.anchor_id]
        module = modules[anchor.module_id]
        port = anchor.capability.get("dock_port_local_id")
        if port not in PORTS:
            raise SchemaValidationError(
                "contact feature requires physical gripper port"
            )
        raw[i, 31:48] = [
            0.0,
            0.0,
            c.slot_id / 8.0,
            *anchor.local_pose,
            *compose_pose(transform, module.pose_world),
        ]
        raw[i, [19, 20]] = 0.0
        full = [
            *raw[i],
            *relative_pose_features(measured[c.anchor_id], c.contact_pose_world),
            *relative_pose_features(obj.pose_world, measured[c.anchor_id]),
            *relative_pose_features(obj.pose_world, c.contact_pose_world),
            *relative_pose_features(base, module.pose_world),
            *[float(port == name) for name in PORTS],
        ]
        rows.append([full[j] for j in LOCAL_COLUMNS])
    features = torch.tensor(rows, dtype=torch.float32)
    if not torch.isfinite(features).all():
        raise SchemaValidationError("nonfinite measured contact geometry")
    ids = {
        c.candidate_id: i
        for i, c in enumerate(context.scene.contact_candidate_set.candidates)
    }
    membership = torch.zeros((len(context.catalog.entries), len(ids)), dtype=torch.bool)
    for i, entry in enumerate(context.catalog.entries):
        membership[i, [ids[c] for c in entry.candidate_ids]] = True
    return features, membership
