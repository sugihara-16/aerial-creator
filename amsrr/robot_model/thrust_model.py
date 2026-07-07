from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty, require_non_negative
from amsrr.utils.config import load_config


@dataclass
class ThrustModelEntry(SchemaBase):
    rotor_id: str
    thrust_min_n: float
    thrust_max_n: float
    reaction_torque_coeff_nm_per_n: float

    def validate(self) -> None:
        require_non_empty(self.rotor_id, "ThrustModelEntry.rotor_id")
        require_non_negative(self.thrust_min_n, "ThrustModelEntry.thrust_min_n")
        require_non_negative(self.thrust_max_n, "ThrustModelEntry.thrust_max_n")
        if self.thrust_max_n < self.thrust_min_n:
            raise SchemaValidationError("ThrustModelEntry.thrust_max_n must be >= thrust_min_n")


@dataclass
class ThrustModel(SchemaBase):
    rotors: list[ThrustModelEntry]

    def validate(self) -> None:
        ids = [entry.rotor_id for entry in self.rotors]
        if len(ids) != len(set(ids)):
            raise SchemaValidationError("ThrustModel.rotors contains duplicate rotor_id values")

    def by_rotor_id(self) -> dict[str, ThrustModelEntry]:
        return {entry.rotor_id: entry for entry in self.rotors}


def normalize_rotor_id(rotor_id: str) -> str:
    return rotor_id.replace("_", "")


def load_thrust_model(path: str | Path) -> ThrustModel:
    data = load_config(path)
    return ThrustModel.from_dict(data)

