from __future__ import annotations

from pathlib import Path

from amsrr.feasibility.checker import FeasibilityChecker
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.random_connected import (
    RandomConnectedMorphologyDistribution,
    morphology_structural_hash,
)
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.feasibility import FeasibilityResult
from amsrr.schemas.order3 import (
    ORDER3_POOL_VERSION,
    Order3MorphologyPoolEntry,
    Order3MorphologyPoolManifest,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_dataset import (
    load_order9_dataset,
    validate_order9_dataset_for_stage,
)
from amsrr.training.order9_design_teacher_dataset import (
    ORDER9_PI_D_TEACHER_VERSION,
    Order9PiDTeacherDatasetConfig,
    _order9_base_task,
    build_order9_pi_d_teacher_dataset,
    build_order9_pi_d_teacher_record,
)
from amsrr.training.order9_pipeline import order9_stage_by_id
from amsrr.training.order9_randomization import Order9ExpandedObjectRandomizer
from amsrr.training.p2_inspection_context import default_grasp_carry_task_spec


def test_pi_d_teacher_replays_a_structural_target_through_exact_masks() -> None:
    model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    target = RandomConnectedMorphologyDistribution(model).sample(
        seed=72, module_count=4
    )
    base = _order9_base_task(
        default_grasp_carry_task_spec(), Order9PiDTeacherDatasetConfig()
    )
    task = Order9ExpandedObjectRandomizer().sample(
        base, seed=9701, sample_index=1
    ).task_spec
    irg = IRGBuilder().build(task)
    envelope = InteractionEnvelopeExtractor().extract(irg)

    record = build_order9_pi_d_teacher_record(
        DesignPolicyContext(task, irg, model, envelope),
        target,
        episode_id="order9-pi-d-teacher-unit",
        split=DatasetSplit.TRAIN,
    )

    assert record.task_success
    assert record.feasibility_result is not None
    assert record.feasibility_result.feasible
    assert record.steps[-1].terminal
    assert record.steps[-1].candidates[
        record.steps[-1].selected_candidate_index
    ].action.action_type.value == "stop"
    assert record.trajectory_provenance.source_version == ORDER9_PI_D_TEACHER_VERSION
    assert morphology_structural_hash(record.design_output.target_morphology) == (
        morphology_structural_hash(target)
    )


def test_pi_d_teacher_dataset_is_task_and_structure_split_safe(
    tmp_path: Path,
) -> None:
    model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    distribution = RandomConnectedMorphologyDistribution(model)
    graphs = []
    hashes: set[str] = set()
    seed = 100
    while len(graphs) < 3:
        graph = distribution.sample(seed=seed, module_count=3)
        structural_hash = morphology_structural_hash(graph)
        if structural_hash not in hashes:
            graphs.append(graph)
            hashes.add(structural_hash)
        seed += 1
    splits = (DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.HELD_OUT)
    entries = [
        Order3MorphologyPoolEntry(
            split=split,
            module_count=3,
            structural_hash=morphology_structural_hash(graph),
            requested_seed=index,
            accepted_proposal_seed=index,
            morphology_graph=graph,
            feasibility_result=FeasibilityResult(
                feasible=True,
                hard_violations=[],
                soft_violations=[],
                margins={},
                proxy_scores={},
                checker_version="unit_test_source_pool",
            ),
        )
        for index, (split, graph) in enumerate(zip(splits, graphs))
    ]
    pool = Order3MorphologyPoolManifest(
        pool_version=ORDER3_POOL_VERSION,
        master_seed=100,
        physical_model_hash=model.stable_hash(),
        config_hash="unit-test",
        entries=entries,
        split_counts={split.value: 1 for split in DatasetSplit},
        module_count_counts={
            str(module_count): 3 if module_count == 3 else 0
            for module_count in range(2, 9)
        },
    )
    config = Order9PiDTeacherDatasetConfig(
        train_record_count=1,
        validation_record_count=1,
        held_out_record_count=1,
        min_modules=3,
        max_modules=3,
    )

    manifest = build_order9_pi_d_teacher_dataset(
        pool,
        tmp_path,
        config=config,
        physical_model=model,
    )
    bundle = load_order9_dataset(tmp_path / "manifest.json")
    stage = order9_stage_by_id(
        load_order9_learning_config(), "c7_pi_d_structured_bc"
    )
    validation = validate_order9_dataset_for_stage(bundle, stage)

    assert manifest.record_counts["design_action_trajectory"] == 3
    assert len(bundle.sequential_design_records) == 3
    assert validation.valid, validation.failures
    assert len(
        {
            morphology_structural_hash(record.design_output.target_morphology)
            for record in bundle.sequential_design_records
        }
    ) == 3
