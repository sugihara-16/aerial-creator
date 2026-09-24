"""Canonical observed morphology and candidate ownership for request ranking.

One graph encoding is shared by all candidates. No IK, route lookup, object
shape identifier, scene identifier, or teacher result is used at inference.
"""
from dataclasses import replace

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from amsrr.encoders.morphology_graph_encoder import MORPHOLOGY_NODE_FEATURE_NAMES


def canonical_morphology_batch(batch):
    """Remove arbitrary planar world-frame origin/yaw, preserving gravity/height."""
    features = batch.node_features.clone()
    names = MORPHOLOGY_NODE_FEATURE_NAMES
    base_column = names.index("module.is_base")
    runtime = names.index("runtime.pose.x")
    twist = names.index("runtime.twist.vx")
    for b in range(len(features)):
        valid = batch.node_mask[b]
        bases = torch.nonzero(valid & (features[b, :, base_column] > .5)).flatten()
        if len(bases) != 1:
            raise ValueError("canonical request graph requires exactly one base")
        base = int(bases[0])
        for start in (0, runtime):
            pose = features[b, valid, start:start + 7].detach().cpu().numpy().copy()
            reference = features[b, base, start:start + 7].detach().cpu().numpy()
            rotation = Rotation.from_quat(reference[3:]).as_matrix()
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
            inverse = Rotation.from_euler("z", -yaw)
            pose[:, :3] = inverse.apply(pose[:, :3] - [reference[0], reference[1], 0.])
            pose[:, 3:] = (inverse * Rotation.from_quat(pose[:, 3:])).as_quat()
            features[b, valid, start:start + 7] = torch.as_tensor(pose, device=features.device, dtype=features.dtype)
            if start == runtime:
                for offset in (twist, twist + 3):
                    velocity = features[b, valid, offset:offset + 3].detach().cpu().numpy()
                    features[b, valid, offset:offset + 3] = torch.as_tensor(
                        inverse.apply(velocity), device=features.device, dtype=features.dtype)
    return replace(batch, node_features=features)


def candidate_owner_indices(context, batch):
    anchors = {a.anchor_id: a for a in context.scene.morphology_graph.robot_anchors}
    modules = batch.module_ids[0].tolist()
    return torch.tensor([modules.index(anchors[c.anchor_id].module_id)
                         for c in context.scene.contact_candidate_set.candidates], dtype=torch.long)
