from collections import Counter
from pathlib import Path
import json

import pytest
from scripts.prepare_request_condition_dataset import AXES, bounds, panels, prepare, read, samples, sha, write

DISTRIBUTION = dict(volume_factor=[.8,1.2], x_over_y=[.65,.9], z_over_y=[.3,.5],
                    mass_factor=[.8,1.2], true_com_fraction=[-.05,.05])


def test_samples_reproducible_in_range_and_every_face_covered():
    a = samples(14, DISTRIBUTION, 123)
    assert a == samples(14, DISTRIBUTION, 123)
    assert a != samples(14, DISTRIBUTION, 124)
    assert Counter(x[2] for x in a) == dict(interior=56, single_boundary=28, multiple_boundaries=28)
    faces = Counter()
    for _, variant, kind, values, com in a:
        point = [values[k] for k in AXES[:4]]+com
        assert all(lo <= x <= hi for x,(lo,hi) in zip(point,bounds(DISTRIBUTION)))
        edge = [(axis,side) for axis,(x,ends) in enumerate(zip(point,bounds(DISTRIBUTION)))
                for side,end in enumerate(ends) if x == end]
        assert len(edge) == (0 if kind == 'interior' else 1 if kind == 'single_boundary' else 2 if variant == 6 else 3)
        if kind == 'single_boundary':faces.update(edge)
    assert len(faces) == 14 and set(faces.values()) == {2}


def test_panels_cover_pool_once_without_repeating_donor_within_module():
    cases = [dict(episode_id=f'{m}_{j}_{v}', module_count=m, record=f'{m}_{j}')
             for m in (2,3) for j in range(14) for v in range(9 if j<4 else 8)]
    result = panels(cases,4,7)
    assert len(result) == 29
    assert Counter(c['episode_id'] for panel in result for c in panel) == Counter(c['episode_id'] for c in cases)
    for panel in result:
        assert Counter(c['module_count'] for c in panel) == {2:4,3:4}
        assert len({c['record'] for c in panel}) == 8


@pytest.fixture
def source(tmp_path):
    records=tmp_path/'records';records.mkdir()
    task=tmp_path/'task.json';write(task,{})
    binding=dict(path=str(task),sha256=sha(task))
    cases=[];held=[]
    for module in (2,3):
        for index in range(15):
            split='train' if index<14 else 'validation'
            name=f'm{module}_{index}'
            manifest=tmp_path/f'{name}_manifest.json'
            write(manifest,dict(split=split,task_spec=binding,morphology_graph=binding,structural_hash=f's{module}_{index%2}'))
            record=records/f'{name}.json'
            write(record,dict(dataset_split=split,status='accepted',training_eligible=split=='train',
                episode_id=name,module_count=module,pose_condition_hash=name,
                case_manifest=dict(path=str(manifest),sha256=sha(manifest))))
            if index<4 or split!='train':
                condition=tmp_path/f'{name}_condition.json'
                write(condition,dict(volume_factor=.99,x_over_y=.72,z_over_y=.4,mass_factor=1.,com_fraction=[0,0,0]))
                case=dict(episode_id=name,bucket_id=name,split=split,record=str(record),record_sha256=sha(record),
                          module_count=module,condition=str(condition),condition_sha256=sha(condition))
                (cases if split=='train' else held).append(case)
    parent=dict(maximum_grasp_contacts=2,training_draws_per_case=16,training_cases=cases,evaluation_cases=held,
        untouched_test_cases=[],distribution=DISTRIBUTION,configuration_hashes={},config_sha256='config',ppo={},
        training_seed_base=7,evaluation_seed_base=17,evaluation_draws_per_case=1,
        prepare_wallclock_timeout_s=360,execute_wallclock_timeout_s=1200,
        selection_rule=[],validation_interval=3,stale_validations=3,minimum_additional_updates=9,reward_tolerance=.01)
    path=tmp_path/'parent.json';write(path,parent)
    checkpoint=tmp_path/'update_16/checkpoint.pt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'model')
    (checkpoint.parent/'training_state.pt').write_bytes(b'state')
    return path,records,checkpoint,parent


def test_dataset_preserves_held_out_and_old_train_and_writes_usable_panels(tmp_path,source):
    path,records,checkpoint,parent=source
    before={c['condition']:sha(c['condition']) for c in parent['evaluation_cases']}
    output=tmp_path/'output'
    data,coverage=prepare(path,records,output,checkpoint,100)
    assert coverage['condition_count']==232 and coverage['original_train_scenes']==28
    assert data['training_cases'][:8]==parent['training_cases']
    assert data['evaluation_cases']==parent['evaluation_cases']
    assert {p:sha(p) for p in before}==before
    for p in data['panels']:
        protocol=read(p['path'])
        assert sha(p['path'])==p['sha256']
        assert len(protocol['training_cases'])==8 and protocol['max_grasp_contacts']==2
        assert protocol['evaluation_cases']==parent['evaluation_cases']
        assert protocol['ppo']==parent['ppo']
        assert protocol['initial_optimizer_updates']==16
    assert set(c['record'] for c in data['training_cases']).isdisjoint(c['record'] for c in data['evaluation_cases'])
    with pytest.raises(FileExistsError):prepare(path,records,output,checkpoint,100)


def test_reject_training_scene_alias_of_held_out(tmp_path,source):
    path,records,checkpoint,parent=source
    donor=records/'m2_0.json';record=read(donor);record['pose_condition_hash']='m2_14';write(donor,record)
    with pytest.raises(ValueError,match='scene overlaps'):
        prepare(path,records,tmp_path/'bad',checkpoint,100)
    assert not (tmp_path/'bad').exists()


def test_reject_protected_condition_tampering(tmp_path,source):
    path,records,checkpoint,parent=source
    Path(parent['evaluation_cases'][0]['condition']).write_text('{}')
    with pytest.raises(ValueError,match='protected evaluation'):
        prepare(path,records,tmp_path/'bad',checkpoint,100)
