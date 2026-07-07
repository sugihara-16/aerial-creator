from __future__ import annotations

from amsrr.utils.config import load_config
from amsrr.utils.hashing import stable_hash


def test_config_loading_and_hashing() -> None:
    robot_config = load_config("configs/robot/robot_model.yaml")
    thrust_config = load_config("configs/robot/thrust_model.yaml")
    p0_config = load_config("configs/training/p0_schema_tests.yaml")

    assert robot_config["robot_model"]["module_type"] == "holon"
    assert len(thrust_config["rotors"]) == 4
    assert p0_config["phase"] == "P0"

    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})

