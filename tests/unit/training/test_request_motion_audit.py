import json

import pytest
import torch

from amsrr.training.request_motion_audit import audit_all


def test_batch_audit_reads_once_and_checks_each_physical_trajectory(tmp_path, monkeypatch):
    import amsrr.training.request_motion_audit as audit
    initial = torch.tensor([0., 0., 0., 1., 0., 0., 0.])
    poses = initial.repeat(2, 2, 1)
    poses[-1, 0, :3] = torch.tensor([.2, 0., .05])
    raw = dict(metadata=dict(control_dt_s=.1), tensors=dict(
        valid=torch.ones(2, 2, dtype=torch.bool),
        post_object_pose_world=poses, object_pose_world=initial.repeat(2, 2, 1),
        post_object_twist_world=torch.zeros(2, 2, 6),
        time_s=torch.tensor([[0., 0.], [.1, .1]]),
        applied_global_action=torch.zeros(2, 2, 1),
        contact_space_residual_action=torch.zeros(2, 2, 1),
        prohibited_collision=torch.zeros(2, 2, dtype=torch.bool)))
    torch.save(raw, tmp_path/'rollout.pt')
    result = dict(episodes=[dict(task_success=True), dict(task_success=False)],
                  environment_origins_world=[[0., 0., 0.]] * 2)
    (tmp_path/'result.json').write_text(json.dumps(result))
    scene = dict(task=dict(goals=[dict(goal_type='object_pose',
        target_pose_world=[.2, 0., .05, 1., 0., 0., 0.],
        tolerance_pos_m=.01, tolerance_rot_rad=.1, time_limit_s=1.)]))
    (tmp_path/'scene.json').write_text(json.dumps(scene))
    reads, hashes = [], []
    load, hash_file = torch.load, audit.hash_file
    def counted_load(*args, **kwargs):
        reads.append(args[0])
        return load(*args, **kwargs)
    def counted_hash(path):
        hashes.append(path)
        return hash_file(path)
    monkeypatch.setattr(torch, 'load', counted_load)
    monkeypatch.setattr(audit, 'hash_file', counted_hash)
    reports = audit_all(tmp_path)
    assert len(reads) == len(hashes) == 1
    assert [r['physical_goal_and_motion_passed'] for r in reports] == [True, False]
    assert reports[0]['physical_rollout_sha256'] == reports[1]['physical_rollout_sha256']
    result['episodes'][1]['task_success'] = True
    (tmp_path/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='runtime success disagrees'):
        audit_all(tmp_path)
