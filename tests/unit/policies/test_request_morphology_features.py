from dataclasses import replace
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tests.unit.policies.test_high_level_requests import request_scene,decision
from amsrr.policies.request_actor_critic import RequestActorCritic
from amsrr.policies.request_morphology_features import canonical_morphology_batch
from amsrr.encoders.morphology_graph_encoder import MORPHOLOGY_NODE_FEATURE_NAMES


def test_common_planar_frame_change_does_not_change_actor_graph(request_scene):
    graph=RequestActorCritic(morphology_aware=True).encode(decision(request_scene))["graph"]
    values=graph.node_features.clone();valid=graph.node_mask[0]
    runtime=MORPHOLOGY_NODE_FEATURE_NAMES.index("runtime.pose.x")
    twist=MORPHOLOGY_NODE_FEATURE_NAMES.index("runtime.twist.vx")
    delta=Rotation.from_euler("z",.73)
    for start in (0,runtime):
        pose=values[0,valid,start:start+7].numpy().copy()
        pose[:,:3]=delta.apply(pose[:,:3])+[2.3,-1.4,0.]
        pose[:,3:]=(delta*Rotation.from_quat(pose[:,3:])).as_quat()
        values[0,valid,start:start+7]=torch.tensor(pose)
    for start in (twist,twist+3):
        values[0,valid,start:start+3]=torch.tensor(delta.apply(values[0,valid,start:start+3].numpy()),dtype=values.dtype)
    restored=canonical_morphology_batch(replace(graph,node_features=values))
    torch.testing.assert_close(graph.node_features,restored.node_features,rtol=1e-5,atol=1e-6)
