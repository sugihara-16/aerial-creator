"""New request-planning entry point, with a killable bounded planning worker.

Call begin/poll outside the controller tick. The spawn factory must construct
the robot-specific planning model and independent shadow C_H in the child;
live simulator/CUDA objects must not be forked or pickled. No legacy teacher,
post-check posture resolver, pi_L, or unchecked hold is used as fallback.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
from time import monotonic
from uuid import uuid4
from typing import Callable

from amsrr.policies.high_level_requests import (
    HighLevelDecisionContext,
    HeuristicHighLevelPolicy,
)
from amsrr.policies.request_execution import RequestPlanExecutor, guards_ready
from amsrr.policies.request_trajectory_planner import (
    ConstrainedRequestPlanner,
    RequestPlanningResult,
)
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.high_level import (
    HighLevelDecisionRecord,
    HighLevelRequest,
    GeneratedExecutionPlan,
    PlanCheckRecord,
)
from amsrr.utils.hashing import stable_hash


@dataclass
class RequestAttemptRecord(SchemaBase):
    rank: int
    request: HighLevelRequest
    planning_status: str
    reason: str
    planning_elapsed_s: float
    plan: GeneratedExecutionPlan | None = None
    check: PlanCheckRecord | None = None
    checking_elapsed_s: float = 0.0

    def validate(self):
        if self.rank < 0 or self.planning_status not in {
            "solution_found",
            "no_solution_found",
            "timeout",
            "invalid_request",
        }:
            raise SchemaValidationError("invalid planning attempt result")
        if any(
            not math.isfinite(x) or x < 0
            for x in (self.planning_elapsed_s, self.checking_elapsed_s)
        ):
            raise SchemaValidationError("invalid planning/checking duration")
        if (self.plan is not None) != (self.planning_status == "solution_found"):
            raise SchemaValidationError("planning status and generated plan disagree")
        if self.check is not None and (
            self.plan is None or self.check.plan_hash != self.plan.stable_hash()
        ):
            raise SchemaValidationError("check does not belong to this generated plan")


@dataclass
class RequestRuntimeResult(SchemaBase):
    decision_id: str
    status: str
    attempts: list[RequestAttemptRecord] = field(default_factory=list)
    accepted_rank: int | None = None
    reason: str = ""
    # Acceptance and execution/task outcomes are deliberately separate.
    execution_success: bool | None = None


@dataclass(frozen=True)
class RequestRuntimeConfig:
    wall_budget_s: float = 2.0
    maximum_candidates: int = 3

    def __post_init__(self):
        if not math.isfinite(self.wall_budget_s) or self.wall_budget_s <= 0:
            raise ValueError("planning wall budget must be finite and positive")
        if isinstance(self.maximum_candidates, bool) or self.maximum_candidates < 1:
            raise ValueError("candidate budget must be positive")


def _planning_worker(connection, factory, context, ranked, deadline):
    # Isolate only this worker and descendants; never kill other user processes.
    os.setsid()

    def expire(signum, frame):
        os.killpg(os.getpid(), signal.SIGKILL)

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - monotonic()))
    try:
        from amsrr.feasibility.request_plan import RequestPlanChecker

        planner, checker = factory(context)
        if not isinstance(planner, ConstrainedRequestPlanner) or not isinstance(
            checker, RequestPlanChecker
        ):
            raise TypeError(
                "factory must build the constrained planner and independent request C_H"
            )
        for rank, request in ranked:
            connection.send(("started", rank, request.to_dict()))
            result: RequestPlanningResult = planner.plan(
                context, request, deadline=deadline
            )
            attempt = RequestAttemptRecord(
                rank,
                request,
                result.status,
                result.reason,
                result.elapsed_s,
                result.plan,
            )
            if result.plan is not None:
                plan_hash = result.plan.stable_hash()
                start = monotonic()
                attempt.check = checker.check(result.plan, context, deadline=deadline)
                attempt.checking_elapsed_s = monotonic() - start
                if (
                    plan_hash != result.plan.stable_hash()
                    or attempt.check.plan_hash != plan_hash
                ):
                    raise SchemaValidationError("checker mutated generated plan")
            connection.send(("attempt", attempt.to_dict()))
            if attempt.check is not None and attempt.check.accepted:
                break
            if monotonic() >= deadline:
                break
        connection.send(("done",))
    except Exception as error:
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()
        signal.setitimer(signal.ITIMER_REAL, 0)


class HighLevelRequestRuntime:
    """Heuristic by default; a learned ranker is an optional injected replacement."""

    def __init__(
        self,
        *,
        worker_factory: Callable,
        executor: RequestPlanExecutor,
        archive_directory: str | Path,
        policy=None,
        config: RequestRuntimeConfig | None = None,
    ):
        self.worker_factory = worker_factory
        self.executor = executor
        self.policy = policy or HeuristicHighLevelPolicy()
        self.config = config or RequestRuntimeConfig()
        self.archive_directory = Path(archive_directory)
        self._process = None
        self._connection = None
        self._origin = None
        self._result = None
        self._started = None
        self._deadline = 0.0

    def begin(self, context: HighLevelDecisionContext) -> str:
        if self._process is not None:
            raise RuntimeError("a request is already being planned")
        context.validate_snapshot()
        if self.executor.state.stable_hash() != context.execution_state.stable_hash():
            raise SchemaValidationError(
                "decision execution state is not executor-owned"
            )
        scorer = getattr(self.policy, "rank_with_scores", None)
        ranked, scores = (
            scorer(context) if scorer else (self.policy.rank(context), None)
        )
        context.validate_snapshot()
        decision = HighLevelDecisionRecord(
            deepcopy(context.observation),
            deepcopy(context.execution_state),
            deepcopy(context.catalog),
            ranked,
            self.policy.policy_version,
            ranking_scores=scores,
        )
        identity = {
            "decision": decision.to_dict(),
            "runtime_config": self.config.__dict__,
            "previous_outcome": (
                None
                if context.previous_outcome is None
                else context.previous_outcome.to_dict()
            ),
            "execution_config": self.executor.config.__dict__,
            "model_hash": context.physical_model.stable_hash(),
            "task_hash": context.task_spec.stable_hash(),
            "irg_hash": context.scene.irg.stable_hash(),
            "catalog_hash": context.catalog.stable_hash(),
        }
        identity["input_hash"] = stable_hash(identity)
        decision_id = uuid4().hex
        self._archive(decision_id, "request", identity)
        self._origin = deepcopy(context)
        self._result = RequestRuntimeResult(decision_id, "planning")
        eligible = []
        for rank, request in enumerate(ranked):
            if len(eligible) >= self.config.maximum_candidates:
                break
            if guards_ready(
                context.catalog.resolve(request),
                context.observation,
                self.executor.config,
            ):
                eligible.append((rank, request))
            else:
                self._result.attempts.append(
                    RequestAttemptRecord(
                        rank,
                        request,
                        "invalid_request",
                        "transition_guard_not_ready",
                        0.0,
                    )
                )
        if not eligible:
            self._finish(
                "recovery_required", "empty catalog or no observed transition guard"
            )
            return decision_id
        spawn = mp.get_context("spawn")
        receiver, sender = spawn.Pipe(duplex=False)
        self._deadline = monotonic() + self.config.wall_budget_s
        self._connection = receiver
        self._process = spawn.Process(
            target=_planning_worker,
            args=(
                sender,
                self.worker_factory,
                deepcopy(context),
                eligible,
                self._deadline,
            ),
        )
        self._started = None
        try:
            self._process.start()
        except Exception as error:
            self._process = None
            receiver.close()
            self._connection = None
            self._finish("invalid_request", f"worker construction failed: {error}")
        finally:
            sender.close()
        return decision_id

    def poll(self, current: HighLevelDecisionContext) -> RequestRuntimeResult | None:
        if self._result is None:
            return None
        if self._process is None:
            return deepcopy(self._result)
        while self._connection.poll():
            try:
                message = self._connection.recv()
            except EOFError:
                break
            if message[0] == "started":
                self._started = (message[1], HighLevelRequest.from_dict(message[2]))
            elif message[0] == "attempt":
                attempt = RequestAttemptRecord.from_dict(message[1])
                self._result.attempts.append(attempt)
                self._archive(
                    self._result.decision_id,
                    f"candidate-{attempt.rank}",
                    attempt.to_dict(),
                )
                self._started = None
                if attempt.check is not None and attempt.check.accepted:
                    if monotonic() > self._deadline:
                        return self._finish(
                            "timeout", "accepted result arrived after deadline"
                        )
                    try:
                        self.executor.install(
                            attempt.plan, attempt.check, self._origin, current
                        )
                    except (SchemaValidationError, ValueError, KeyError) as error:
                        return self._finish("stale_or_unsafe", str(error))
                    self._result.accepted_rank = attempt.rank
                    return self._finish(
                        (
                            "first_choice_accepted"
                            if attempt.rank == 0
                            else "searched_candidate_accepted"
                        ),
                        "",
                    )
            elif message[0] == "error":
                return self._finish("invalid_request", message[1])
            elif message[0] == "done":
                timed_out = any(
                    a.planning_status == "timeout" for a in self._result.attempts
                )
                return self._finish(
                    "timeout" if timed_out else "recovery_required",
                    "no checked plan found within candidate budget",
                )
        if monotonic() >= self._deadline:
            if self._started is not None:
                rank, request = self._started
                self._result.attempts.append(
                    RequestAttemptRecord(
                        rank,
                        request,
                        "timeout",
                        "planning/validation worker deadline",
                        self.config.wall_budget_s,
                    )
                )
            return self._finish("timeout", "planning/validation worker deadline")
        if not self._process.is_alive():
            return self._finish(
                "worker_failed", f"planning worker exited: {self._process.exitcode}"
            )
        return None

    def _archive(self, identity, suffix, payload):
        self.archive_directory.mkdir(parents=True, exist_ok=True)
        path = self.archive_directory / f"{identity}.{suffix}.json"
        data = (
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2
            )
            + "\n"
        )
        # Identical decisions can be inspected again; different evidence is never overwritten.
        if path.exists():
            if path.read_text() != data:
                raise SchemaValidationError(
                    "refusing to overwrite archived decision evidence"
                )
            return
        with path.open("x") as stream:
            stream.write(data)

    def _finish(self, status, reason):
        self.close()
        self._result.status, self._result.reason = status, reason
        self._archive(self._result.decision_id, "outcome", self._result.to_dict())
        return deepcopy(self._result)

    def close(self):
        if self._process is not None:
            pid = self._process.pid
            if pid:
                try:
                    if os.getpgid(pid) == pid:
                        os.killpg(pid, signal.SIGKILL)
                    elif self._process.is_alive():
                        self._process.kill()
                except ProcessLookupError:
                    pass
                self._process.join(timeout=0.1)
            self._process.close()
            self._process = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
