"""Ground finite request tuples from IRG and a pre-decision observation snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math

from amsrr.geometry.pose_math import compose_pose, inverse_pose, transform_from_pose
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.high_level import (
    ActiveExecutionState,
    ContactLoadEstimate,
    ExecutionGuardSample,
    HighLevelObservation,
    HighLevelRequest,
    HighLevelRequestCatalog,
    RequestCatalogEntry,
    PreviousRequestOutcome,
)
from amsrr.schemas.irg import IRGEdgeType, IRGNodeType, PhaseType
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import ControllerStatus
from amsrr.schemas.runtime import RuntimeObservation, TaskProgressState
from amsrr.schemas.task_spec import TaskSpec
from amsrr.utils.hashing import stable_hash


def contact_surface_hash(candidate, observation) -> str:
    """Material contact point identity, independent of object rigid motion."""
    objects = {o.object_id: o for o in observation.object_states}
    inverse = (
        inverse_pose(objects[candidate.target_entity_id].pose_world)
        if candidate.target_entity_id in objects
        else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    )

    def local(pose):
        values = list(compose_pose(inverse, pose))
        if values[6] < 0:
            values[3:] = [-x for x in values[3:]]
        return [round(x, 8) + 0.0 for x in values]

    return stable_hash(
        {
            "slot": candidate.slot_id,
            "anchor": candidate.anchor_id,
            "entity": candidate.target_entity_id,
            "mode": candidate.contact_mode,
            "pose": local(candidate.contact_pose_world),
            "frame": local(candidate.contact_frame_world),
            "friction": candidate.friction,
            "area": candidate.patch_area_m2,
        }
    )


@dataclass(frozen=True)
class HighLevelDecisionContext:
    """Detached inputs; validate_snapshot detects mutation after catalog creation."""

    scene: HighLevelPolicyContext
    task_spec: TaskSpec
    physical_model: PhysicalModel
    observation: HighLevelObservation
    execution_state: ActiveExecutionState
    catalog: HighLevelRequestCatalog
    catalog_hash: str = ""
    previous_outcome: PreviousRequestOutcome | None = None

    def compute_snapshot_hash(self) -> str:
        return stable_hash(
            {
                "irg": self.scene.irg.to_dict(),
                "envelope": self.scene.interaction_envelope.to_dict(),
                "morphology": self.scene.morphology_graph.to_dict(),
                "candidates": self.scene.contact_candidate_set.to_dict(),
                "task": self.task_spec.to_dict(),
                "model": self.physical_model.to_dict(),
                "observation": self.observation.to_dict(),
                "execution": self.execution_state.to_dict(),
                "encoder_runtime": self.scene.runtime_observation.to_dict(),
                "previous_outcome": (
                    None
                    if self.previous_outcome is None
                    else self.previous_outcome.to_dict()
                ),
            }
        )

    def validate_snapshot(self) -> None:
        self.observation.validate()
        self.execution_state.validate()
        self.catalog.validate()
        if self.previous_outcome is not None:
            self.previous_outcome.validate()
            if self.previous_outcome.observed_time_s > self.observation.time_s:
                raise SchemaValidationError("future outcome in decision input")
        if self.catalog.snapshot_hash != self.compute_snapshot_hash():
            raise SchemaValidationError("high-level decision snapshot was modified")
        if self.catalog.stable_hash() != self.catalog_hash:
            raise SchemaValidationError("high-level request catalog was modified")


class RequestCatalogBuilder:
    """No teacher phase, arbitrary candidate subsets, or learned continuous targets."""

    def __init__(self, *, maximum_entries: int = 256) -> None:
        if isinstance(maximum_entries, bool) or maximum_entries < 1:
            raise ValueError("maximum_entries must be positive")
        self.maximum_entries = maximum_entries

    def build(
        self,
        scene: HighLevelPolicyContext,
        *,
        task_spec: TaskSpec,
        physical_model: PhysicalModel,
        execution_state: ActiveExecutionState,
        contact_estimates: list[ContactLoadEstimate] | None = None,
        guard_samples: list[ExecutionGuardSample] | None = None,
        previous_outcome: PreviousRequestOutcome | None = None,
    ) -> HighLevelDecisionContext:
        raw = scene.runtime_observation
        if raw is None:
            raise SchemaValidationError(
                "request selection requires a current observation"
            )
        if (
            scene.irg.task_id != task_spec.task_id
            or scene.contact_candidate_set.task_id != task_spec.task_id
        ):
            raise SchemaValidationError("request task identity mismatch")
        if (
            scene.contact_candidate_set.morphology_graph_id
            != scene.morphology_graph.graph_id
        ):
            raise SchemaValidationError("request morphology identity mismatch")
        if raw.morphology_graph.stable_hash() != scene.morphology_graph.stable_hash():
            raise SchemaValidationError("observation/morphology mismatch")
        observation = HighLevelObservation.from_dict(
            HighLevelObservation(
                time_s=raw.time_s,
                module_states=raw.module_states,
                object_states=raw.object_states,
                controller_status=ControllerStatus(
                    raw.controller_status.status, raw.controller_status.qp_feasible
                ),
                contact_estimates=contact_estimates or [],
                guard_samples=guard_samples or [],
            ).to_dict()
        )
        execution = ActiveExecutionState.from_dict(execution_state.to_dict())
        if execution.phase_started_s > observation.time_s:
            raise SchemaValidationError("execution phase starts in the future")
        # Drop raw contact truth AND all teacher-owned task_progress, including metrics.
        detached = deepcopy(scene)
        # Runtime phase authority is the executor state, not a teacher-mutated IRG.
        detached.irg.active_phase_id = None
        clean_runtime = RuntimeObservation(
            time_s=observation.time_s,
            morphology_graph=detached.morphology_graph,
            module_states=deepcopy(observation.module_states),
            object_states=deepcopy(observation.object_states),
            contact_states=[],
            controller_status=deepcopy(observation.controller_status),
            task_progress=TaskProgressState(),
        )
        candidates = deepcopy(scene.contact_candidate_set)
        candidates.assignment_feasibility_cache = {}
        # Re-ground source candidates in the observed object frame. Static supports stay fixed.
        authored = {obj.object_id: obj.pose_world for obj in task_spec.scene.objects}
        observed = {obj.object_id: obj.pose_world for obj in observation.object_states}
        for candidate in candidates.candidates:
            if candidate.target_entity_id not in authored:
                continue
            entity = candidate.target_entity_id
            if entity not in observed:
                raise SchemaValidationError(f"missing object observation: {entity}")
            delta = compose_pose(observed[entity], inverse_pose(authored[entity]))
            candidate.contact_pose_world = compose_pose(
                delta, candidate.contact_pose_world
            )
            candidate.contact_frame_world = compose_pose(
                delta, candidate.contact_frame_world
            )
            rotation = transform_from_pose(delta).rotation

            def rotate(v):
                return tuple(
                    sum(rotation[i][j] * v[j] for j in range(3)) for i in range(3)
                )

            candidate.normal_world = rotate(candidate.normal_world)
            candidate.tangent_basis_world = list(
                rotate(candidate.tangent_basis_world[:3])
                + rotate(candidate.tangent_basis_world[3:])
            )
        detached = HighLevelPolicyContext(
            detached.irg,
            detached.interaction_envelope,
            detached.morphology_graph,
            candidates,
            clean_runtime,
        )
        context = HighLevelDecisionContext(
            detached,
            deepcopy(task_spec),
            deepcopy(physical_model),
            observation,
            execution,
            HighLevelRequestCatalog("pending", []),
            previous_outcome=deepcopy(previous_outcome),
        )
        entries = self._entries(context)
        catalog = HighLevelRequestCatalog(context.compute_snapshot_hash(), entries)
        return HighLevelDecisionContext(
            detached,
            context.task_spec,
            context.physical_model,
            observation,
            execution,
            catalog,
            catalog.stable_hash(),
            context.previous_outcome,
        )

    def _entries(self, context: HighLevelDecisionContext) -> list[RequestCatalogEntry]:
        irg = context.scene.irg
        phases = {n.node_id: n for n in irg.nodes if n.node_type == IRGNodeType.PHASE}
        state = context.execution_state
        if state.phase_id not in phases:
            raise SchemaValidationError("active execution phase is not in this IRG")
        candidate_set = context.scene.contact_candidate_set
        candidate_set.validate()
        by_id = {c.candidate_id: (i, c) for i, c in enumerate(candidate_set.candidates)}
        if len(by_id) != len(candidate_set.candidates):
            raise SchemaValidationError("duplicate candidate ID")
        groups = {g.group_id: g for g in candidate_set.group_proposals}
        if len(groups) != len(candidate_set.group_proposals):
            raise SchemaValidationError("duplicate contact group ID")
        for candidate_id, binding in state.contact_bindings.items():
            if candidate_id not in by_id:
                raise SchemaValidationError(
                    "active contact disappeared during resampling"
                )
            c = by_id[candidate_id][1]
            if (c.slot_id, c.anchor_id, c.target_entity_id) != binding:
                raise SchemaValidationError("active contact semantic binding changed")
            if state.contact_surface_hashes.get(candidate_id) != contact_surface_hash(
                c, context.observation
            ):
                raise SchemaValidationError(
                    "active contact surface identity missing or changed"
                )
        transitions = [
            e
            for e in irg.edges
            if e.src_id == state.phase_id
            and e.edge_type
            in (
                IRGEdgeType.TEMPORAL_NEXT,
                IRGEdgeType.GUARD_TRANSITION,
                IRGEdgeType.FALLBACK,
            )
        ]
        entries = []
        for edge in [None, *transitions]:
            phase_id = state.phase_id if edge is None else edge.dst_id
            if phase_id not in phases:
                raise SchemaValidationError("transition does not reference a phase")
            phase = phases[phase_id]
            kind = PhaseType(phase.feature["phase_type"])
            transition_id = (
                None
                if edge is None
                else "transition:"
                + stable_hash({"irg": irg.stable_hash(), "edge": edge.to_dict()})
            )
            guard_ids = (
                []
                if edge is None
                else self._guard_ids(phases[state.phase_id], phase, edge)
            )
            targets = self._targets(irg, phase_id)
            # A bundle preserves simultaneous requirements; the phase completion goal is grounded too.
            subgoal_id = f"phase:{phase_id}:" + stable_hash(
                [n.to_dict() for n in targets]
            )
            eligible = list(groups.values())
            if state.contact_bindings:
                eligible = [
                    g
                    for g in eligible
                    if g.group_id == state.contact_group_id
                    and set(g.candidate_ids) == set(state.contact_bindings)
                ]
            if not state.contact_bindings and kind in (
                PhaseType.FREE_MOTION,
                PhaseType.RECOVERY,
            ):
                eligible = [None]
            for group in eligible:
                if group is not None:
                    ids = group.candidate_ids
                    if (
                        not ids
                        or len(ids) != len(set(ids))
                        or group.group_violation_codes
                        or not math.isfinite(group.group_score)
                        or any(cid not in by_id for cid in ids)
                    ):
                        continue
                    indices = [by_id[cid][0] for cid in ids]
                    if any(
                        not candidate_set.candidate_mask[i]
                        or not candidate_set.candidates[i].unary_valid
                        for i in indices
                    ):
                        continue
                    if any(
                        candidate_set.pairwise_conflict_matrix[i][j]
                        for i in indices
                        for j in indices
                        if i != j
                    ):
                        continue
                    if len({by_id[cid][1].anchor_id for cid in ids}) != len(ids):
                        continue
                    if not self._covers_slots(irg, [by_id[cid][1] for cid in ids]):
                        continue
                entries.append(
                    RequestCatalogEntry(
                        HighLevelRequest(
                            None if group is None else group.group_id,
                            transition_id,
                            subgoal_id,
                        ),
                        phase_id,
                        [] if group is None else list(group.candidate_ids),
                        deepcopy(targets),
                        deepcopy(edge),
                        kind.value,
                        guard_ids,
                        (0.0 if group is None else group.group_score)
                        + (1.0 if edge else 0.0),
                    )
                )
        entries.sort(key=lambda e: (-e.heuristic_score, e.request.stable_hash()))
        return entries[: self.maximum_entries]

    @staticmethod
    def _guard_ids(source, target, edge) -> list[str]:
        def identifier(condition, default):
            if condition is None:
                return default
            if (
                not isinstance(condition, dict)
                or set(condition) != {"guard_id"}
                or not isinstance(condition["guard_id"], str)
            ):
                raise SchemaValidationError(
                    "new runtime guard requires a declared guard_id"
                )
            return condition["guard_id"]

        guards = [
            identifier(
                source.feature.get("exit_condition"), f"phase:{source.node_id}:complete"
            )
        ]
        if target.feature.get("entry_condition") is not None:
            guards.append(identifier(target.feature["entry_condition"], ""))
        if edge.condition is not None:
            guards.append(identifier(edge.condition, ""))
        return list(dict.fromkeys(guards))

    @staticmethod
    def _targets(irg, phase_id):
        nodes = {n.node_id: n for n in irg.nodes}
        visited, pending = set(), [phase_id]
        while pending:
            nid = pending.pop()
            if nid in visited:
                continue
            visited.add(nid)
            pending.extend(
                e.dst_id
                for e in irg.edges
                if e.src_id == nid
                and e.edge_type
                in (IRGEdgeType.REQUIRES, IRGEdgeType.SUPPORTS, IRGEdgeType.ACTIVATES)
            )
        return [
            nodes[nid]
            for nid in sorted(visited)
            if nodes[nid].node_type == IRGNodeType.STATE_TARGET
        ]

    @staticmethod
    def _covers_slots(irg, candidates):
        for node in irg.nodes:
            if node.node_type != IRGNodeType.CONTACT_SLOT:
                continue
            f = node.feature
            count = sum(c.slot_id == f["slot_id"] for c in candidates)
            if f.get("required", False) and count < f.get("min_count_group", 0):
                return False
            if count > f.get("max_count_group", len(candidates)):
                return False
        return True


class HeuristicHighLevelPolicy:
    policy_version = "heuristic_request_ranking_v1"

    def rank(self, context: HighLevelDecisionContext) -> list[HighLevelRequest]:
        context.validate_snapshot()
        return [
            deepcopy(e.request)
            for e in sorted(
                context.catalog.entries,
                key=lambda e: (-e.heuristic_score, e.request.stable_hash()),
            )
        ]

    def rank_with_scores(self, context: HighLevelDecisionContext):
        ranked = self.rank(context)
        return ranked, [context.catalog.resolve(r).heuristic_score for r in ranked]
