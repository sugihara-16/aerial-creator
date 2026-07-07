from __future__ import annotations

import pytest

from amsrr.robot_model.thrust_model import ThrustModel, load_thrust_model, normalize_rotor_id
from amsrr.schemas.common import SchemaValidationError


def test_thrust_model_loads_config() -> None:
    thrust_model = load_thrust_model("configs/robot/thrust_model.yaml")

    assert [entry.rotor_id for entry in thrust_model.rotors] == ["thrust_1", "thrust_2", "thrust_3", "thrust_4"]
    assert all(entry.thrust_max_n == 20.0 for entry in thrust_model.rotors)
    assert normalize_rotor_id("thrust_1") == "thrust1"


def test_thrust_model_rejects_duplicate_rotor_ids() -> None:
    with pytest.raises(SchemaValidationError, match="duplicate"):
        ThrustModel.from_dict(
            {
                "rotors": [
                    {
                        "rotor_id": "thrust_1",
                        "thrust_min_n": 0.0,
                        "thrust_max_n": 20.0,
                        "reaction_torque_coeff_nm_per_n": 0.0,
                    },
                    {
                        "rotor_id": "thrust_1",
                        "thrust_min_n": 0.0,
                        "thrust_max_n": 20.0,
                        "reaction_torque_coeff_nm_per_n": 0.0,
                    },
                ]
            }
        )

