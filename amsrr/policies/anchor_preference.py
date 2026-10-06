"""Fixed upstream-anchor preference, shared by collection, PPO and inference.

Distances belong to the upstream morphology, never to a teacher trajectory or
an outcome. They are separate from learned candidate quality features.
"""

from dataclasses import dataclass, asdict
import math

import torch

ANCHOR_PREFERENCE_CONTRACT = "group_assigned_boundary_steps_prior_v1"


@dataclass(frozen=True)
class AnchorPreference:
    distance_decay: float = math.log(2.)
    kl_coefficient: float = .05
    contract: str = ANCHOR_PREFERENCE_CONTRACT

    def __post_init__(self):
        if self.contract != ANCHOR_PREFERENCE_CONTRACT:
            raise ValueError("unsupported anchor preference contract")
        if any(not math.isfinite(x) or x <= 0 for x in
               (self.distance_decay, self.kl_coefficient)):
            raise ValueError("anchor preference requires positive finite coefficients")

    def to_dict(self):
        return asdict(self)

    def logits(self, distances, mask):
        if distances.shape != mask.shape or mask.dtype != torch.bool:
            raise ValueError("anchor distances must match request support")
        if (not mask.any(-1).all() or not torch.isfinite(distances[mask]).all()
                or (distances[mask] < 0).any()):
            raise ValueError("invalid anchor distances or support")
        return (-self.distance_decay * distances).masked_fill(~mask, -torch.inf)


def assigned_anchor_distance(anchor_ids, distances_by_anchor):
    """Minimum total steps with one distinct selected dock per upstream group.

    The assignment is polynomial in contact count, not an enumeration of
    permutations. A physical pair shared by two groups is counted once.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    groups = sorted({group for distances in distances_by_anchor.values() for group in distances})
    if not groups or len(anchor_ids) != len(groups) or len(set(anchor_ids)) != len(anchor_ids):
        raise ValueError("anchor preference needs one distinct contact per upstream group")
    costs = np.full((len(groups), len(anchor_ids)), np.inf)
    for j, anchor in enumerate(anchor_ids):
        if anchor not in distances_by_anchor:
            raise ValueError("selected anchor has no upstream distance metadata")
        for i, group in enumerate(groups):
            value = distances_by_anchor[anchor].get(group)
            if value is not None:
                if isinstance(value, bool) or not math.isfinite(value) or value < 0 or int(value) != value:
                    raise ValueError("boundary distance must be a nonnegative integer")
                costs[i, j] = value
    try:
        row, col = linear_sum_assignment(costs)
    except ValueError as error:
        raise ValueError("selected docks cannot cover the upstream anchor groups") from error
    return float(costs[row, col].sum())


def request_anchor_distances(context, mask):
    """Encode only current initial choices; temporal decisions have no prior."""
    result = torch.zeros(len(context.catalog.entries), dtype=torch.float32)
    if context.execution_state.plan_id is not None:
        return result
    morphology = context.scene.morphology_graph
    distances = {a.anchor_id: a.capability['grasp_neighborhood_distances']
                 for a in morphology.robot_anchors
                 if 'grasp_neighborhood_distances' in a.capability}
    if not distances:
        raise ValueError("anchor-preference policy requires upstream neighborhood distances")
    candidates = {c.candidate_id: c for c in context.scene.contact_candidate_set.candidates}
    cached = {}
    for index, entry in enumerate(context.catalog.entries):
        if not mask[index]:
            continue
        anchors = tuple(sorted(candidates[c].anchor_id for c in entry.candidate_ids))
        if anchors not in cached:
            cached[anchors] = assigned_anchor_distance(anchors, distances)
        result[index] = cached[anchors]
    return result
