from __future__ import annotations

"""Production deterministic-teacher dataset for sequential Order 9 ``pi_D``.

The source morphology pool supplies only a split-safe structural target.  The
same grammar used by the learned runtime emits every graph edit, task-specific
anchor binding, control group, mask, and final STOP.  The deterministic hard
checker remains the authority for admitting a record.
"""

import gzip
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from amsrr.feasibility.checker import FeasibilityChecker
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.policies.design_candidate_generator import DesignActionCandidate
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.order9_design_grammar import (
    ORDER9_DESIGN_GRAMMAR_VERSION,
    Order9DesignGrammar,
    Order9DesignTeacherStep,
    Order9PartialDesignState,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.datasets import (
    P4_3_DATASET_SCHEMA_VERSION,
    DatasetKind,
    DatasetShard,
    DatasetSplit,
    DesignActionCandidateRecord,
    P4_3DatasetManifest,
    PolicyBehaviorTrace,
    SequentialDesignStepRecord,
    SequentialDesignTrajectoryRecord,
    StageDecisionMasks,
    TrajectoryProvenance,
    TrajectorySourceKind,
)
from amsrr.schemas.feasibility import FeasibilityResult
from amsrr.schemas.morphology import (
    DesignActionType,
    DesignOutput,
    MorphologyGraph,
)
from amsrr.schemas.order3 import Order3MorphologyPoolEntry, Order3MorphologyPoolManifest
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_randomization import (
    ORDER9_EXPANDED_RANDOMIZATION_VERSION,
    Order9ExpandedObjectRandomizer,
)
from amsrr.training.p2_inspection_context import default_grasp_carry_task_spec
from amsrr.utils.hashing import hash_file, stable_hash


ORDER9_PI_D_TEACHER_VERSION = "order9_structural_pool_grammar_teacher_v1"
ORDER9_PI_D_DATASET_BUILDER_VERSION = "order9_pi_d_teacher_dataset_builder_v1"


@dataclass(frozen=True)
class Order9PiDTeacherDatasetConfig:
    seed: int = 9700
    train_record_count: int = 400
    validation_record_count: int = 50
    held_out_record_count: int = 50
    min_modules: int = 2
    max_modules: int = 8

    @property
    def record_count(self) -> int:
        return (
            self.train_record_count
            + self.validation_record_count
            + self.held_out_record_count
        )

    def validate(self) -> None:
        if self.seed < 0:
            raise SchemaValidationError("Order9 pi_D teacher seed must be non-negative")
        if min(
            self.train_record_count,
            self.validation_record_count,
            self.held_out_record_count,
        ) < 1:
            raise SchemaValidationError(
                "Order9 pi_D teacher requires non-empty train/validation/held-out splits"
            )
        if not 2 <= self.min_modules <= self.max_modules <= 8:
            raise SchemaValidationError(
                "Order9 pi_D teacher module range must lie within [2, 8]"
            )


def build_order9_pi_d_teacher_record(
    context: DesignPolicyContext,
    structural_target: MorphologyGraph,
    *,
    episode_id: str,
    split: DatasetSplit,
    source_metadata: dict[str, object] | None = None,
    checker: FeasibilityChecker | None = None,
) -> SequentialDesignTrajectoryRecord:
    """Replay one split-owned structural target through the production grammar."""

    if not episode_id:
        raise SchemaValidationError("Order9 pi_D teacher episode_id must be non-empty")
    trace, design, feasibility = build_order9_task_conditioned_design_teacher(
        context,
        structural_target,
        checker=checker,
    )
    source_hash = morphology_structural_hash(structural_target)
    generated_hash = morphology_structural_hash(design.target_morphology)
    if generated_hash != source_hash:
        raise SchemaValidationError(
            "Order9 pi_D grammar changed the split-defining structural hash"
        )
    steps: list[SequentialDesignStepRecord] = []
    for index, item in enumerate(trace):
        candidates = item.candidate_step.candidates
        selected_indices = [
            candidate_index
            for candidate_index, candidate in enumerate(candidates)
            if candidate.action.to_dict()
            == item.candidate_step.selected_action.to_dict()
        ]
        if len(selected_indices) != 1:
            raise SchemaValidationError(
                "Order9 pi_D teacher action does not identify one grammar candidate"
            )
        selected_index = selected_indices[0]
        terminal = index == len(trace) - 1
        steps.append(
            SequentialDesignStepRecord(
                step_index=index,
                partial_action_history=list(item.state.action_history),
                candidates=[
                    DesignActionCandidateRecord(
                        candidate_index=candidate_index,
                        action=candidate.action,
                        valid=candidate.valid,
                        reason_code=candidate.reason_code,
                        score_prior=candidate.score_prior,
                    )
                    for candidate_index, candidate in enumerate(candidates)
                ],
                selected_candidate_index=selected_index,
                reward=1.0 if terminal else 0.0,
                terminal=terminal,
                truncated=False,
                behavior_trace=PolicyBehaviorTrace(
                    policy_family="pi_d",
                    policy_version=ORDER9_PI_D_TEACHER_VERSION,
                    action_semantics="masked_grammar_candidate_index",
                    action_payload={
                        "selected_action": item.candidate_step.selected_action.to_dict(),
                        "selected_candidate_index": selected_index,
                    },
                    stochastic=False,
                ),
            )
        )
    record = SequentialDesignTrajectoryRecord(
        record_id=f"{episode_id}:pi_d",
        episode_id=episode_id,
        task_id=context.task_spec.task_id,
        split=split,
        task_spec=context.task_spec,
        irg=context.irg,
        interaction_envelope=context.interaction_envelope,
        physical_model_hash=context.physical_model.stable_hash(),
        steps=steps,
        design_output=design,
        feasibility_result=feasibility,
        episode_return=1.0,
        task_success=True,
        failure_reason=None,
        stage_masks=StageDecisionMasks(design_decision_mask=True),
        trajectory_provenance=TrajectoryProvenance(
            source_kind=TrajectorySourceKind.DETERMINISTIC_TEACHER,
            source_version=ORDER9_PI_D_TEACHER_VERSION,
            metadata={
                "grammar_version": ORDER9_DESIGN_GRAMMAR_VERSION,
                "source_graph_id": structural_target.graph_id,
                "source_structural_hash": source_hash,
                "generated_structural_hash": generated_hash,
                "module_count": len(structural_target.modules),
                **dict(source_metadata or {}),
            },
        ),
    )
    record.validate()
    return record


def build_order9_task_conditioned_design_teacher(
    context: DesignPolicyContext,
    structural_target: MorphologyGraph,
    *,
    checker: FeasibilityChecker | None = None,
) -> tuple[list[Order9DesignTeacherStep], DesignOutput, FeasibilityResult]:
    """Add task-owned anchors/bindings to a split-owned structural topology.

    C3--C6 precede a deployable learned ``pi_D``.  They therefore draw a
    structural graph from the immutable morphology pool and replay that graph
    through the same grammar used by C7/C8.  This is the one deterministic
    topology-to-task adapter; callers must not invent phase-specific anchor
    heuristics outside the grammar.
    """

    grammar = Order9DesignGrammar(context, checker=checker)
    trace, design = _structural_teacher_trace(grammar, structural_target)
    feasibility = grammar.checker.check_design(
        design,
        task_spec=context.task_spec,
        irg=context.irg,
        physical_model=context.physical_model,
    )
    if not feasibility.feasible:
        codes = sorted({item.code for item in feasibility.hard_violations})
        raise SchemaValidationError(
            "Order9 pi_D deterministic teacher produced an infeasible design: "
            + ",".join(codes)
        )
    if (
        morphology_structural_hash(design.target_morphology)
        != morphology_structural_hash(structural_target)
    ):
        raise SchemaValidationError(
            "Order9 task-conditioned design changed the split-owned structure"
        )
    return trace, design, feasibility


def build_order9_pi_d_teacher_dataset(
    morphology_pool: Order3MorphologyPoolManifest | str | Path,
    output_dir: str | Path,
    *,
    config: Order9PiDTeacherDatasetConfig | None = None,
    base_task_spec: TaskSpec | None = None,
    physical_model: PhysicalModel | None = None,
    robot_model_config_path: str = "configs/robot/robot_model.yaml",
) -> P4_3DatasetManifest:
    """Build the 500-record C7 dataset with task and structural split isolation."""

    cfg = config or Order9PiDTeacherDatasetConfig()
    cfg.validate()
    pool, pool_reference, pool_digest = _load_pool(morphology_pool)
    model = physical_model or build_physical_model_from_config(
        robot_model_config_path
    )
    if pool.physical_model_hash != model.stable_hash():
        raise SchemaValidationError(
            "Order9 pi_D source pool physical-model hash is stale"
        )
    base = _order9_base_task(base_task_spec or default_grasp_carry_task_spec(), cfg)
    randomizer = Order9ExpandedObjectRandomizer()
    records: list[SequentialDesignTrajectoryRecord] = []
    random_seeds: list[int] = []
    global_index = 0
    split_counts = (
        (DatasetSplit.TRAIN, cfg.train_record_count),
        (DatasetSplit.VALIDATION, cfg.validation_record_count),
        (DatasetSplit.HELD_OUT, cfg.held_out_record_count),
    )
    for split, count in split_counts:
        entries_by_module = _pool_entries_by_module(pool.entries, split, cfg)
        for split_index in range(count):
            module_count = cfg.min_modules + split_index % (
                cfg.max_modules - cfg.min_modules + 1
            )
            candidates = entries_by_module[module_count]
            source_entry = candidates[
                (split_index // (cfg.max_modules - cfg.min_modules + 1))
                % len(candidates)
            ]
            random_seed = cfg.seed + global_index
            sample = randomizer.sample(
                base,
                seed=random_seed,
                sample_index=global_index,
                held_out=split == DatasetSplit.HELD_OUT,
            )
            task = sample.task_spec
            irg = IRGBuilder().build(task)
            envelope = InteractionEnvelopeExtractor().extract(irg)
            context = DesignPolicyContext(task, irg, model, envelope)
            episode_id = f"order9-pi-d-{split.value}-{global_index:06d}"
            records.append(
                build_order9_pi_d_teacher_record(
                    context,
                    source_entry.morphology_graph,
                    episode_id=episode_id,
                    split=split,
                    source_metadata={
                        "pool_structural_hash": source_entry.structural_hash,
                        "pool_requested_seed": source_entry.requested_seed,
                        "pool_accepted_proposal_seed": (
                            source_entry.accepted_proposal_seed
                        ),
                        "object_randomization_version": (
                            sample.randomization_version
                        ),
                        "object_randomization_seed": random_seed,
                        "object_randomization_sample_index": global_index,
                    },
                )
            )
            random_seeds.append(random_seed)
            global_index += 1
    _require_output_split_isolation(records)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    shards: list[DatasetShard] = []
    for split, _ in split_counts:
        selected = [record for record in records if record.split == split]
        shard_path = target / f"design_action_trajectory_{split.value}.jsonl.gz"
        _atomic_write_jsonl_gzip(shard_path, selected)
        shards.append(
            DatasetShard(
                dataset_kind=DatasetKind.DESIGN_ACTION_TRAJECTORY,
                split=split,
                path=shard_path.name,
                record_count=len(selected),
                sha256=hash_file(shard_path),
            )
        )
    split_task_ids = {
        split: sorted(record.task_id for record in records if record.split == split)
        for split in DatasetSplit
    }
    task_hashes = {
        record.task_id: record.task_spec.stable_hash() for record in records
    }
    geometry_hashes = {
        f"{record.task_id}:{geometry.geometry_id}": geometry.stable_hash()
        for record in records
        for geometry in record.task_spec.scene.geometry_library
    }
    record_counts = {kind.value: 0 for kind in DatasetKind}
    record_counts[DatasetKind.DESIGN_ACTION_TRAJECTORY.value] = len(records)
    thrust_hash = str(model.metadata.get("thrust_model_hash", ""))
    if not thrust_hash:
        raise SchemaValidationError(
            "Order9 pi_D teacher PhysicalModel lacks thrust-model provenance"
        )
    simulator_identity = {
        "kind": "not_applicable_deterministic_design_teacher",
        "teacher_version": ORDER9_PI_D_TEACHER_VERSION,
        "grammar_version": ORDER9_DESIGN_GRAMMAR_VERSION,
    }
    manifest = P4_3DatasetManifest(
        dataset_id=(
            "order9-pi-d-teacher-"
            + stable_hash(
                {
                    "pool": pool_digest,
                    "records": [record.stable_hash() for record in records],
                    "shards": [shard.sha256 for shard in shards],
                }
            )[:16]
        ),
        schema_version=P4_3_DATASET_SCHEMA_VERSION,
        source_archive_paths=[pool_reference],
        source_episode_ids=[record.episode_id for record in records],
        train_task_ids=split_task_ids[DatasetSplit.TRAIN],
        validation_task_ids=split_task_ids[DatasetSplit.VALIDATION],
        held_out_task_ids=split_task_ids[DatasetSplit.HELD_OUT],
        shards=shards,
        record_counts=record_counts,
        source_hash=pool_digest,
        config_hash=stable_hash(
            {
                "builder": asdict(cfg),
                "randomizer": randomizer.config.to_dict(),
                "base_task": base.stable_hash(),
            }
        ),
        robot_model_hash=model.stable_hash(),
        urdf_hash=hash_file(model.urdf_path),
        thrust_model_hash=thrust_hash,
        task_hashes=task_hashes,
        geometry_hashes=geometry_hashes,
        random_seeds=random_seeds,
        simulator_version=str(simulator_identity["kind"]),
        simulator_hash=stable_hash(simulator_identity),
        metadata={
            "builder_version": ORDER9_PI_D_DATASET_BUILDER_VERSION,
            "teacher_version": ORDER9_PI_D_TEACHER_VERSION,
            "grammar_version": ORDER9_DESIGN_GRAMMAR_VERSION,
            "source_pool_version": pool.pool_version,
            "source_pool_sha256": pool_digest,
            "task_disjoint_splits": True,
            "structural_hash_disjoint_splits": True,
            "module_count_balanced_within_each_split": True,
            "module_count_min": cfg.min_modules,
            "module_count_max": cfg.max_modules,
            "expanded_object_randomization_version": (
                ORDER9_EXPANDED_RANDOMIZATION_VERSION
            ),
            "held_out_object_families_used_only_for_held_out_split": True,
            "hard_feasibility_required": True,
            "simulator_execution_required_for_bc_teacher": False,
            "dynamic_assembly_policy_learned": False,
            "gzip_shards": True,
        },
    )
    manifest.validate()
    _atomic_write_text(target / "manifest.json", manifest.to_json(indent=2) + "\n")
    return manifest


def _structural_teacher_trace(
    grammar: Order9DesignGrammar,
    structural_target: MorphologyGraph,
    *,
    maximum_steps: int = 256,
) -> tuple[list[Order9DesignTeacherStep], DesignOutput]:
    structural_target.validate()
    target_module_ids = sorted(module.module_id for module in structural_target.modules)
    if target_module_ids != list(range(len(target_module_ids))):
        raise SchemaValidationError(
            "Order9 pi_D structural teacher requires contiguous module IDs from zero"
        )
    target_pairs = {
        frozenset((edge.src_port_id, edge.dst_port_id))
        for edge in structural_target.dock_edges
    }
    target_roles = {
        module.module_id: module.role_id for module in structural_target.modules
    }
    state = grammar.initial_state()
    trace: list[Order9DesignTeacherStep] = []
    while not state.stopped:
        if len(trace) >= maximum_steps:
            raise SchemaValidationError(
                "Order9 pi_D structural teacher exceeded maximum steps"
            )
        candidates = grammar.candidates(state)
        selected = _select_structural_candidate(
            state,
            candidates,
            target_module_count=len(target_module_ids),
            target_base_module_id=structural_target.base_module_id,
            target_pairs=target_pairs,
            target_roles=target_roles,
            target_graph=structural_target,
        )
        from amsrr.policies.design_candidate_generator import DesignCandidateStep

        trace.append(
            Order9DesignTeacherStep(
                state=state,
                candidate_step=DesignCandidateStep(
                    step_index=len(trace),
                    selected_action=selected.action,
                    candidates=candidates,
                ),
            )
        )
        state = grammar.apply(state, selected)
    return trace, grammar.build_design_output(state)


def _select_structural_candidate(
    state: Order9PartialDesignState,
    candidates: Sequence[DesignActionCandidate],
    *,
    target_module_count: int,
    target_base_module_id: int | None,
    target_pairs: set[frozenset[int]],
    target_roles: dict[int, str],
    target_graph: MorphologyGraph,
) -> DesignActionCandidate:
    valid = [candidate for candidate in candidates if candidate.valid]
    by_type = {
        action_type: [
            candidate
            for candidate in valid
            if candidate.action.action_type == action_type
        ]
        for action_type in DesignActionType
    }
    if len(state.module_ids) < target_module_count and by_type[DesignActionType.ADD_MODULE]:
        return by_type[DesignActionType.ADD_MODULE][0]
    if state.base_module_id is None:
        matches = [
            candidate
            for candidate in by_type[DesignActionType.SET_BASE_MODULE]
            if int(candidate.action.params["module_id"]) == target_base_module_id
        ]
        if matches:
            return matches[0]
    if by_type[DesignActionType.CONNECT_PORT]:
        used = {frozenset(pair) for pair in state.connected_port_pairs}
        matches = []
        for candidate in by_type[DesignActionType.CONNECT_PORT]:
            pair = frozenset(
                (
                    int(candidate.action.params["src_port_id"]),
                    int(candidate.action.params["dst_port_id"]),
                )
            )
            if pair in target_pairs and pair not in used:
                matches.append(candidate)
        if matches:
            return matches[0]
    if by_type[DesignActionType.ASSIGN_ROLE]:
        module_id = int(
            by_type[DesignActionType.ASSIGN_ROLE][0].action.params["module_id"]
        )
        desired = target_roles[module_id]
        matches = [
            candidate
            for candidate in by_type[DesignActionType.ASSIGN_ROLE]
            if str(candidate.action.params["role_id"]) == desired
        ]
        if matches:
            return matches[0]
        raise SchemaValidationError(
            f"Order9 pi_D grammar cannot preserve structural role {desired!r}"
        )
    if by_type[DesignActionType.BIND_ANCHOR_TO_SLOT]:
        return by_type[DesignActionType.BIND_ANCHOR_TO_SLOT][0]
    if by_type[DesignActionType.CREATE_ANCHOR]:
        return max(
            by_type[DesignActionType.CREATE_ANCHOR],
            key=lambda candidate: _anchor_candidate_score(
                candidate, state, target_graph
            ),
        )
    if by_type[DesignActionType.SET_CONTROL_GROUP]:
        return by_type[DesignActionType.SET_CONTROL_GROUP][0]
    if by_type[DesignActionType.STOP]:
        return by_type[DesignActionType.STOP][0]
    reasons = ",".join(candidate.reason_code for candidate in candidates)
    raise SchemaValidationError(
        "Order9 pi_D structural target cannot reach a valid grammar action: " + reasons
    )


def _anchor_candidate_score(
    candidate: DesignActionCandidate,
    state: Order9PartialDesignState,
    target: MorphologyGraph,
) -> tuple[int, int, int, int]:
    module_id = int(candidate.action.params["module_id"])
    existing = {anchor.module_id for anchor in state.anchors}
    distances = _graph_distances(target, module_id)
    separation = min(
        (distances.get(other, 0) for other in existing),
        default=max(distances.values(), default=0),
    )
    degree = sum(
        module_id in {edge.src_module_id, edge.dst_module_id}
        for edge in target.dock_edges
    )
    return (
        1 if module_id not in existing else 0,
        separation,
        -degree,
        -int(candidate.action.params["surface_port_id"]),
    )


def _graph_distances(graph: MorphologyGraph, source: int) -> dict[int, int]:
    adjacency = {module.module_id: set() for module in graph.modules}
    for edge in graph.dock_edges:
        adjacency[edge.src_module_id].add(edge.dst_module_id)
        adjacency[edge.dst_module_id].add(edge.src_module_id)
    distances = {source: 0}
    pending = [source]
    while pending:
        current = pending.pop(0)
        for neighbor in sorted(adjacency[current]):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            pending.append(neighbor)
    return distances


def _pool_entries_by_module(
    entries: Sequence[Order3MorphologyPoolEntry],
    split: DatasetSplit,
    config: Order9PiDTeacherDatasetConfig,
) -> dict[int, list[Order3MorphologyPoolEntry]]:
    grouped = {
        module_count: sorted(
            (
                entry
                for entry in entries
                if entry.split == split and entry.module_count == module_count
            ),
            key=lambda entry: entry.structural_hash,
        )
        for module_count in range(config.min_modules, config.max_modules + 1)
    }
    missing = [count for count, values in grouped.items() if not values]
    if missing:
        raise SchemaValidationError(
            f"Order9 pi_D morphology pool lacks {split.value} entries for {missing}"
        )
    return grouped


def _order9_base_task(
    source: TaskSpec, config: Order9PiDTeacherDatasetConfig
) -> TaskSpec:
    data = source.to_dict()
    constraints = dict(data["robot_constraints"])
    constraints["min_modules"] = config.min_modules
    constraints["max_modules"] = config.max_modules
    data["robot_constraints"] = constraints
    metadata = dict(data.get("metadata", {}) or {})
    metadata.update(
        {
            "order9_pi_d_teacher_module_min": config.min_modules,
            "order9_pi_d_teacher_module_max": config.max_modules,
        }
    )
    data["metadata"] = metadata
    return TaskSpec.from_dict(data)


def _load_pool(
    value: Order3MorphologyPoolManifest | str | Path,
) -> tuple[Order3MorphologyPoolManifest, str, str]:
    if isinstance(value, Order3MorphologyPoolManifest):
        value.validate()
        return value, "in_memory:order3_morphology_pool", stable_hash(value.to_dict())
    path = Path(value)
    pool = Order3MorphologyPoolManifest.from_json(path.read_text(encoding="utf-8"))
    return pool, str(path), hash_file(path)


def _require_output_split_isolation(
    records: Iterable[SequentialDesignTrajectoryRecord],
) -> None:
    owners: dict[str, DatasetSplit] = {}
    for record in records:
        if record.design_output is None:
            raise SchemaValidationError("Order9 pi_D teacher record lacks a design")
        structural_hash = morphology_structural_hash(
            record.design_output.target_morphology
        )
        previous = owners.setdefault(structural_hash, record.split)
        if previous != record.split:
            raise SchemaValidationError(
                "Order9 pi_D generated structural hash crosses dataset splits"
            )


def _atomic_write_jsonl_gzip(path: Path, records: Sequence[SchemaBase]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        with gzip.open(temporary_name, "wt", encoding="utf-8") as handle:
            for record in records:
                record.validate()
                handle.write(record.to_json())
                handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_text(path: Path, value: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "ORDER9_PI_D_DATASET_BUILDER_VERSION",
    "ORDER9_PI_D_TEACHER_VERSION",
    "Order9PiDTeacherDatasetConfig",
    "build_order9_pi_d_teacher_dataset",
    "build_order9_pi_d_teacher_record",
]
