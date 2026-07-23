from __future__ import annotations

"""Fail-closed import of the promoted Order 9 v2 prefix into v3.

The import does not rewrite or relabel a legacy checkpoint.  It proves that the
successor preserved the completed stage definitions and allows the exact
promoted v2 C2 pi_L checkpoint to initialize only the first v3 stage.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from amsrr.schemas.common import (
    SchemaBase,
    SchemaValidationError,
    require_non_empty,
)
from amsrr.schemas.order9 import (
    Order9PolicyFamily,
    Order9StageRunManifest,
    Order9StageRunStatus,
)
from amsrr.training.order9_checkpoints import (
    LoadedOrder9PolicyCheckpoint,
    load_order9_policy_checkpoint,
)
from amsrr.training.order9_curriculum import (
    ORDER9_CURRICULUM_VERSION,
    ORDER9_LEGACY_CURRICULUM_VERSION,
    Order9CurriculumStage,
    Order9LearningConfig,
    load_order9_learning_config,
)
from amsrr.utils.config import load_config
from amsrr.utils.hashing import hash_file, stable_hash


ORDER9_CURRICULUM_LINEAGE_IMPORT_VERSION = (
    "order9_curriculum_lineage_import_v1"
)
DEFAULT_ORDER9_CURRICULUM_LINEAGE_IMPORT_PATH = Path(
    "configs/training/order9_curriculum_lineage_v3.yaml"
)


@dataclass
class Order9CurriculumLineageImport(SchemaBase):
    import_version: str
    source_config_path: str
    successor_config_path: str
    source_schedule_version: str
    successor_schedule_version: str
    source_schedule_hash: str
    successor_schedule_hash: str
    imported_through_stage_id: str
    imported_through_stage_index: int
    successor_first_stage_id: str
    promoted_stage_manifest_path: str
    promoted_stage_manifest_sha256: str
    policy_checkpoint_path: str
    policy_checkpoint_sha256: str
    policy_family: Order9PolicyFamily

    def validate(self) -> None:
        if self.import_version != ORDER9_CURRICULUM_LINEAGE_IMPORT_VERSION:
            raise SchemaValidationError(
                "Order9 curriculum lineage import version mismatch"
            )
        for name in (
            "source_config_path",
            "successor_config_path",
            "source_schedule_version",
            "successor_schedule_version",
            "imported_through_stage_id",
            "successor_first_stage_id",
            "promoted_stage_manifest_path",
            "policy_checkpoint_path",
        ):
            require_non_empty(
                str(getattr(self, name)),
                f"Order9CurriculumLineageImport.{name}",
            )
        for name in (
            "source_schedule_hash",
            "successor_schedule_hash",
            "promoted_stage_manifest_sha256",
            "policy_checkpoint_sha256",
        ):
            _require_sha256(
                str(getattr(self, name)),
                f"Order9CurriculumLineageImport.{name}",
            )
        if self.source_schedule_version != ORDER9_LEGACY_CURRICULUM_VERSION:
            raise SchemaValidationError(
                "Order9 lineage source must be the historical v2 schedule"
            )
        if self.successor_schedule_version != ORDER9_CURRICULUM_VERSION:
            raise SchemaValidationError(
                "Order9 lineage successor must be the progressive v3 schedule"
            )
        if self.imported_through_stage_index != 2:
            raise SchemaValidationError(
                "Order9 v3 imports exactly the completed C0--C2 prefix"
            )
        if self.policy_family != Order9PolicyFamily.PI_L:
            raise SchemaValidationError(
                "Order9 v2 C2 lineage import must contain the pi_L checkpoint"
            )


@dataclass(frozen=True)
class ValidatedOrder9CurriculumLineage:
    specification: Order9CurriculumLineageImport
    specification_path: Path
    source_config_path: Path
    successor_config_path: Path
    promoted_stage_manifest_path: Path
    policy_checkpoint_path: Path
    promoted_stage_manifest: Order9StageRunManifest


def load_order9_curriculum_lineage_import(
    path: str | Path = DEFAULT_ORDER9_CURRICULUM_LINEAGE_IMPORT_PATH,
) -> Order9CurriculumLineageImport:
    return Order9CurriculumLineageImport.from_dict(load_config(path))


def validate_order9_curriculum_lineage_import(
    path: str | Path = DEFAULT_ORDER9_CURRICULUM_LINEAGE_IMPORT_PATH,
    *,
    successor_config: Order9LearningConfig | None = None,
    load_checkpoint: bool = True,
    device: str | torch.device = "cpu",
) -> ValidatedOrder9CurriculumLineage:
    specification_path = Path(path).resolve()
    specification = load_order9_curriculum_lineage_import(specification_path)
    source_path = _resolve_repository_path(
        specification.source_config_path, specification_path
    )
    successor_path = _resolve_repository_path(
        specification.successor_config_path, specification_path
    )
    manifest_path = _resolve_repository_path(
        specification.promoted_stage_manifest_path, specification_path
    )
    checkpoint_path = _resolve_repository_path(
        specification.policy_checkpoint_path, specification_path
    )

    source = load_order9_learning_config(source_path)
    successor = successor_config or load_order9_learning_config(successor_path)
    source_hash = stable_hash(source.curriculum.to_dict())
    successor_hash = stable_hash(successor.curriculum.to_dict())
    if source.curriculum.schedule_version != specification.source_schedule_version:
        raise SchemaValidationError(
            "Order9 lineage source schedule version does not match its config"
        )
    if (
        successor.curriculum.schedule_version
        != specification.successor_schedule_version
    ):
        raise SchemaValidationError(
            "Order9 lineage successor schedule version does not match its config"
        )
    if source_hash != specification.source_schedule_hash:
        raise SchemaValidationError(
            "Order9 lineage source schedule hash mismatch"
        )
    if successor_hash != specification.successor_schedule_hash:
        raise SchemaValidationError(
            "Order9 lineage successor schedule hash mismatch"
        )

    imported_index = specification.imported_through_stage_index
    if len(source.curriculum.stages) <= imported_index:
        raise SchemaValidationError(
            "Order9 lineage source omits the imported stage"
        )
    if len(successor.curriculum.stages) <= imported_index + 1:
        raise SchemaValidationError(
            "Order9 lineage successor omits its first new stage"
        )
    source_imported = source.curriculum.stages[imported_index]
    successor_imported = successor.curriculum.stages[imported_index]
    successor_first = successor.curriculum.stages[imported_index + 1]
    if (
        source_imported.stage_id != specification.imported_through_stage_id
        or successor_imported.stage_id != specification.imported_through_stage_id
        or successor_first.stage_id != specification.successor_first_stage_id
    ):
        raise SchemaValidationError(
            "Order9 lineage stage boundary identity mismatch"
        )
    source_prefix = [
        stage.to_dict()
        for stage in source.curriculum.stages[: imported_index + 1]
    ]
    successor_prefix = [
        stage.to_dict()
        for stage in successor.curriculum.stages[: imported_index + 1]
    ]
    if source_prefix != successor_prefix:
        raise SchemaValidationError(
            "Order9 v3 changed a completed C0--C2 stage definition"
        )

    if hash_file(manifest_path) != specification.promoted_stage_manifest_sha256:
        raise SchemaValidationError(
            "Order9 lineage promoted-stage manifest hash mismatch"
        )
    manifest = Order9StageRunManifest.from_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if (
        not manifest.promoted
        or manifest.status != Order9StageRunStatus.PROMOTED
        or manifest.stage_id != specification.imported_through_stage_id
        or manifest.stage_index != imported_index
        or manifest.schedule_hash != specification.source_schedule_hash
    ):
        raise SchemaValidationError(
            "Order9 lineage source manifest is not the promoted v2 C2 boundary"
        )
    manifest_checkpoint = manifest.policy_checkpoint_sha256_by_family.get(
        Order9PolicyFamily.PI_L.value
    )
    if manifest_checkpoint != specification.policy_checkpoint_sha256:
        raise SchemaValidationError(
            "Order9 lineage manifest and pi_L checkpoint hash disagree"
        )
    if hash_file(checkpoint_path) != specification.policy_checkpoint_sha256:
        raise SchemaValidationError(
            "Order9 lineage pi_L checkpoint hash mismatch"
        )
    if load_checkpoint:
        checkpoint = load_order9_policy_checkpoint(
            checkpoint_path,
            device=device,
            expected_sha256=specification.policy_checkpoint_sha256,
            expected_family=specification.policy_family,
            expected_schedule_hash=specification.source_schedule_hash,
        )
        _validate_imported_checkpoint_metadata(
            checkpoint,
            source_stage=source_imported,
        )
    return ValidatedOrder9CurriculumLineage(
        specification=specification,
        specification_path=specification_path,
        source_config_path=source_path,
        successor_config_path=successor_path,
        promoted_stage_manifest_path=manifest_path,
        policy_checkpoint_path=checkpoint_path,
        promoted_stage_manifest=manifest,
    )


def load_order9_stage_parent_checkpoint(
    config: Order9LearningConfig,
    stage: Order9CurriculumStage,
    checkpoint_path: str | Path,
    *,
    expected_family: Order9PolicyFamily | str,
    update_index: int | None = None,
    lineage_import_path: str | Path = (
        DEFAULT_ORDER9_CURRICULUM_LINEAGE_IMPORT_PATH
    ),
    device: str | torch.device = "cpu",
) -> LoadedOrder9PolicyCheckpoint:
    """Load a current-schedule parent; legacy C2 requires explicit migration."""

    checkpoint = Path(checkpoint_path)
    checkpoint_sha256 = hash_file(checkpoint)
    current_schedule_hash = stable_hash(config.curriculum.to_dict())
    if config.curriculum.schedule_version != ORDER9_CURRICULUM_VERSION:
        return load_order9_policy_checkpoint(
            checkpoint,
            device=device,
            expected_family=expected_family,
            expected_schedule_hash=current_schedule_hash,
        )
    specification = load_order9_curriculum_lineage_import(lineage_import_path)
    is_import_candidate = (
        config.curriculum.schedule_version == ORDER9_CURRICULUM_VERSION
        and stage.stage_id == specification.successor_first_stage_id
        and checkpoint_sha256 == specification.policy_checkpoint_sha256
    )
    if not is_import_candidate:
        return load_order9_policy_checkpoint(
            checkpoint,
            device=device,
            expected_family=expected_family,
            expected_schedule_hash=current_schedule_hash,
        )
    validate_order9_curriculum_lineage_import(
        lineage_import_path,
        successor_config=config,
        load_checkpoint=True,
        device=device,
    )
    raise SchemaValidationError(
        "Order9 legacy C2 checkpoint cannot execute C3 directly; create and "
        "use the provenance-bound active-knot C3 initializer"
    )


def _validate_imported_checkpoint_metadata(
    checkpoint: LoadedOrder9PolicyCheckpoint,
    *,
    source_stage: Order9CurriculumStage,
) -> None:
    if (
        checkpoint.metadata.curriculum_stage_id != source_stage.stage_id
        or checkpoint.metadata.curriculum_stage_index != source_stage.stage_index
    ):
        raise SchemaValidationError(
            "Order9 imported checkpoint metadata is not the v2 C2 stage"
        )


def _resolve_repository_path(raw: str, specification_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    working = (Path.cwd() / path).resolve()
    if working.exists():
        return working
    repository_relative = (
        specification_path.parent.parent.parent / path
    ).resolve()
    return repository_relative


def _require_sha256(value: str, path: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaValidationError(f"{path} must be a lowercase SHA-256 digest")
