from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from amsrr.encoders.morphology_graph_encoder import (
    MORPHOLOGY_EDGE_FEATURE_NAMES,
    MORPHOLOGY_GRAPH_ENCODER_VERSION,
    MORPHOLOGY_GRAPH_TENSORIZER_VERSION,
    MORPHOLOGY_NODE_FEATURE_NAMES,
    MorphologyGraphEncoder,
    MorphologyGraphTensorizer,
)
from amsrr.encoders.workspace_builder import workspace_token_group_from_encoder_output
from amsrr.geometry.pose_math import inverse_pose
from amsrr.morphology.random_connected import (
    RandomConnectedMorphologyDistribution,
    morphology_structural_hash,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.runtime import ContactState, ModuleRuntimeState, RuntimeObservation, TaskProgressState
from amsrr.schemas.policies import ControllerStatus


@pytest.fixture(scope="module")
def morphology_distribution() -> RandomConnectedMorphologyDistribution:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    return RandomConnectedMorphologyDistribution(physical_model)


def test_tensorizer_builds_padded_bidirectional_module_graph_batch(
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph_two = morphology_distribution.sample(seed=2, module_count=2)
    graph_eight = morphology_distribution.sample(seed=8, module_count=8)
    batch = MorphologyGraphTensorizer().tensorize(
        [graph_two, graph_eight],
        runtime_observations=[_runtime(graph_two), _runtime(graph_eight)],
    )

    assert batch.node_features.shape == (2, 8, len(MORPHOLOGY_NODE_FEATURE_NAMES))
    assert batch.node_mask.sum(dim=1).tolist() == [2, 8]
    assert batch.module_ids[0, :2].tolist() == [0, 1]
    assert batch.module_ids[0, 2:].tolist() == [-1] * 6
    assert batch.edge_features.shape == (2, 14, len(MORPHOLOGY_EDGE_FEATURE_NAMES))
    assert batch.edge_mask.sum(dim=1).tolist() == [2, 14]
    assert batch.edge_ids[0, :2].tolist() == [0, 0]
    assert batch.edge_ids[0, 2:].tolist() == [-1] * 12
    assert batch.metadata["tensorizer_version"] == MORPHOLOGY_GRAPH_TENSORIZER_VERSION
    assert batch.metadata["schema_ids_are_numeric_features"] is False
    assert batch.metadata["privileged_contact_wrench_features"] is False

    source, destination = batch.edge_index[0, :, :2].tolist()
    assert source == [0, 1]
    assert destination == [1, 0]


def test_encoder_global_embedding_is_module_id_and_list_order_invariant_and_nodes_are_equivariant(
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=37, module_count=6)
    runtime = _runtime(graph)
    renamed_graph, module_id_map = _rename_reorder_and_reverse_edges(graph)
    renamed_runtime = _remap_runtime(runtime, renamed_graph, module_id_map)
    torch.manual_seed(1234)
    encoder = MorphologyGraphEncoder(d_model=24, message_passing_steps=3)
    encoder.eval()

    original = encoder([graph], runtime_observations=[runtime])
    renamed = encoder([renamed_graph], runtime_observations=[renamed_runtime])

    assert torch.allclose(original.graph_embeddings, renamed.graph_embeddings, atol=1.0e-6)
    original_nodes = _node_embeddings_by_source_id(original)
    renamed_nodes = _node_embeddings_by_source_id(renamed)
    for old_id, new_id in module_id_map.items():
        assert torch.allclose(original_nodes[old_id], renamed_nodes[new_id], atol=1.0e-6)
    assert renamed.metadata["encoder_version"] == MORPHOLOGY_GRAPH_ENCODER_VERSION


def test_encoder_changes_embedding_for_distinct_dock_topology(
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    left = morphology_distribution.sample(seed=3, module_count=6)
    right = morphology_distribution.sample(seed=17, module_count=6)
    assert morphology_structural_hash(left) != morphology_structural_hash(right)
    torch.manual_seed(99)
    encoder = MorphologyGraphEncoder(d_model=16, message_passing_steps=2)
    encoder.eval()

    output = encoder([left, right])

    assert not torch.allclose(output.graph_embeddings[0], output.graph_embeddings[1])


def test_runtime_states_join_by_module_id_and_privileged_contact_wrench_is_excluded(
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=12, module_count=4)
    first = _runtime(graph)
    second = _runtime(graph)
    second.module_states = list(reversed(second.module_states))
    first.contact_states = [
        ContactState(
            contact_id="contact-0",
            entity_a="robot",
            entity_b="floor",
            wrench_world=[1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
        )
    ]
    second.contact_states = [
        ContactState(
            contact_id="contact-0",
            entity_a="robot",
            entity_b="floor",
            wrench_world=[100.0, -200.0, 300.0, 10.0, -20.0, 30.0],
        )
    ]
    tensorizer = MorphologyGraphTensorizer()

    first_batch = tensorizer.tensorize([graph], runtime_observations=[first])
    second_batch = tensorizer.tensorize([graph], runtime_observations=[second])

    assert torch.equal(first_batch.module_ids, second_batch.module_ids)
    assert torch.allclose(first_batch.node_features, second_batch.node_features)

    missing = _runtime(graph)
    missing.module_states.pop()
    with pytest.raises(SchemaValidationError, match="exactly match"):
        tensorizer.tensorize([graph], runtime_observations=[missing])


def test_padding_does_not_change_valid_node_or_global_embeddings(
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    small = morphology_distribution.sample(seed=21, module_count=2)
    large = morphology_distribution.sample(seed=22, module_count=8)
    torch.manual_seed(7)
    encoder = MorphologyGraphEncoder(d_model=20, message_passing_steps=2)
    encoder.eval()

    alone = encoder([small], runtime_observations=[_runtime(small)])
    batched = encoder(
        [small, large],
        runtime_observations=[_runtime(small), _runtime(large)],
    )

    assert torch.allclose(alone.tokens[0, :2], batched.tokens[0, :2], atol=1.0e-7)
    assert torch.allclose(alone.graph_embeddings[0], batched.graph_embeddings[0], atol=1.0e-7)
    assert batched.mask[0].tolist() == [True, True, False, False, False, False, False, False]
    assert torch.count_nonzero(batched.tokens[0, 2:]).item() == 0
    assert batched.source_ids[0, 2:].tolist() == [-1] * 6


def test_encoder_is_differentiable_and_adapts_to_morphology_workspace_group(
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=44, module_count=5)
    torch.manual_seed(31)
    encoder = MorphologyGraphEncoder(d_model=12, message_passing_steps=2)

    output = encoder([graph], runtime_observations=[_runtime(graph)])
    valid_tokens = output.tokens * output.mask.unsqueeze(-1)
    loss = valid_tokens.square().mean() + output.graph_embeddings.square().mean()
    loss.backward()

    assert encoder.node_projection[0].weight.grad is not None
    assert torch.count_nonzero(encoder.node_projection[0].weight.grad).item() > 0
    assert encoder.layers[0].message[0].weight.grad is not None
    assert torch.count_nonzero(encoder.layers[0].message[0].weight.grad).item() > 0

    group = workspace_token_group_from_encoder_output("morphology", output)
    assert group.tokens is output.tokens
    assert group.mask is output.mask
    assert group.source_ids is output.source_ids


def test_tensorizer_fails_closed_on_invalid_edge_reference(
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=9, module_count=3)
    bad_edge = replace(graph.dock_edges[0], src_port_id=999_999)
    malformed = replace(graph, dock_edges=[bad_edge, *graph.dock_edges[1:]])

    with pytest.raises(SchemaValidationError, match="unknown port"):
        MorphologyGraphTensorizer().tensorize([malformed])


def _runtime(graph: MorphologyGraph) -> RuntimeObservation:
    morphology = graph
    return RuntimeObservation(
        time_s=0.25,
        morphology_graph=morphology,
        module_states=[
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=(
                    1.0 + float(module.pose_in_design_frame[0]),
                    -0.5 + float(module.pose_in_design_frame[1]),
                    1.2 + float(module.pose_in_design_frame[2]),
                    *module.pose_in_design_frame[3:],
                ),
                twist_world=[
                    0.01 * module.module_id,
                    -0.02 * module.module_id,
                    0.0,
                    0.0,
                    0.0,
                    0.03,
                ],
                joint_positions={"dock_joint": 0.01 * module.module_id, "gimbal": 0.0},
                joint_velocities={"dock_joint": -0.02 * module.module_id, "gimbal": 0.0},
                health=0.95,
            )
            for module in morphology.modules
        ],
        object_states=[],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(progress_ratio=0.2),
    )


def _rename_reorder_and_reverse_edges(graph):
    old_module_ids = sorted(module.module_id for module in graph.modules)
    new_ids = list(reversed([100 + index * 7 for index in range(len(old_module_ids))]))
    module_id_map = dict(zip(old_module_ids, new_ids, strict=True))
    old_ports = list(reversed(graph.ports))
    port_id_map = {
        port.port_global_id: 10_000 + index * 11
        for index, port in enumerate(old_ports)
    }
    modules = [
        replace(module, module_id=module_id_map[module.module_id])
        for module in reversed(graph.modules)
    ]
    ports = [
        replace(
            port,
            port_global_id=port_id_map[port.port_global_id],
            module_id=module_id_map[port.module_id],
        )
        for port in old_ports
    ]
    edges = [
        replace(
            edge,
            src_module_id=module_id_map[edge.dst_module_id],
            src_port_id=port_id_map[edge.dst_port_id],
            dst_module_id=module_id_map[edge.src_module_id],
            dst_port_id=port_id_map[edge.src_port_id],
            relative_pose_src_to_dst=inverse_pose(edge.relative_pose_src_to_dst),
        )
        for edge in reversed(graph.dock_edges)
    ]
    anchors = [
        replace(anchor, module_id=module_id_map[anchor.module_id])
        for anchor in reversed(graph.robot_anchors)
    ]
    control_groups = [
        replace(
            group,
            module_ids=[module_id_map[module_id] for module_id in reversed(group.module_ids)],
        )
        for group in reversed(graph.control_groups)
    ]
    return (
        replace(
            graph,
            graph_id=f"{graph.graph_id}:renamed",
            modules=modules,
            ports=ports,
            dock_edges=edges,
            robot_anchors=anchors,
            control_groups=control_groups,
            base_module_id=module_id_map[graph.base_module_id],
        ),
        module_id_map,
    )


def _remap_runtime(
    runtime: RuntimeObservation,
    renamed_graph,
    module_id_map: dict[int, int],
) -> RuntimeObservation:
    return replace(
        runtime,
        morphology_graph=renamed_graph,
        module_states=[
            replace(state, module_id=module_id_map[state.module_id])
            for state in reversed(runtime.module_states)
        ],
    )


def _node_embeddings_by_source_id(output) -> dict[int, torch.Tensor]:
    return {
        int(output.source_ids[0, index].item()): output.tokens[0, index]
        for index, valid in enumerate(output.mask[0].tolist())
        if valid
    }
