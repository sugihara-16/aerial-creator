"""Failed or relabeled physical records must not become imitation targets."""
import json

import pytest

from amsrr.training.request_demonstrations import prepare_demonstration_dataset
from amsrr.utils.hashing import hash_file


@pytest.mark.parametrize('corruption', ['job_identity', 'split', 'failed_execution'])
def test_demonstrations_reject_invalid_provenance_and_outcome(tmp_path, corruption):
    root=tmp_path/'job'
    root.mkdir()
    def write(path,value):
        path.write_text(json.dumps(value))
    record=root/'record.json'
    write(record,dict(dataset_split='train',episode_id='original_episode'))
    write(root/'scene.json',dict(initial_task={},morphology={},initial_observation={}))
    write(root/'plan.json',{})
    write(root/'result.json',dict(passed=False,physical_acceptance_eligible=True))
    job=dict(source_dataset_split='train',source_record_path=str(record),
        source_record_sha256=hash_file(record),scene_sha256=hash_file(root/'scene.json'),
        plan_sha256=hash_file(root/'plan.json'))
    write(root/'job.json',job)
    binding=dict(job_path='job/job.json',job_sha256=hash_file(root/'job.json'),split='train')
    if corruption=='job_identity':
        write(root/'job.json',{**job,'source_dataset_split':'validation'})
        expected='job changed'
    elif corruption=='split':
        binding['split']='validation'
        expected='split differs'
    else:
        expected='physical task success'
    manifest=tmp_path/'manifest.json'
    write(manifest,dict(version='request_demonstrations_v1',demonstrations=[binding]))
    with pytest.raises(ValueError,match=expected):
        prepare_demonstration_dataset(manifest,tmp_path/'dataset')
    assert not (tmp_path/'dataset').exists()
