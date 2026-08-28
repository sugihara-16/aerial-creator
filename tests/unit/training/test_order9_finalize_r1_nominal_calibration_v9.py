from __future__ import annotations

from scripts.order9_finalize_r1_nominal_calibration_v9 import _validate_protocol


def test_order9_r1_v9_protocol_and_approval_are_hash_bound() -> None:
    protocol, approval = _validate_protocol()

    assert protocol["maximum_candidates_per_isaac_scene"] == 4
    assert protocol["candidate_set_digest_required_in_restart_manifest_name"]
    assert protocol["expected_legacy_v7_evidence_reuse_count"] == 254
    assert not protocol["pi_l_actor_command_applied"]
    assert protocol["qpid_qp_applied"]
    assert protocol["local_servo_applied"]
    assert approval["decision"] == "approved"
