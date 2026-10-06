from pathlib import Path
import hashlib
import subprocess

import pytest
from scripts.run_request_training import (archive_raw, better, plateau, read,
    select_best, stage_command, validate_summary, load_training_panels, write, sha,
    validate_optimizer_recovery)


def metric(k, successes=20, reward=1., safety=0):
    return dict(update=k, successes=successes, mean_reward=reward, safety_failures=safety)


def test_selection_prioritizes_success_then_reward_and_earlier_tie():
    assert better(metric(9, 21, -5), metric(8, 20, 9), .01)
    assert select_best([metric(20, reward=1.001), metric(17)], .01)['update'] == 17
    assert select_best([metric(20, reward=1.1), metric(17)], .01)['update'] == 20
    assert better(metric(18, safety=0), metric(17, safety=1), .01)


def test_neighbor_improvement_requires_new_plateau():
    c = dict(resume_update=17, validation_interval=3, minimum_new_updates=9, stale_validations=3)
    assert plateau(26, metric(17), c)
    assert not plateau(23, metric(14), c)
    assert not plateau(26, metric(21), c)  # Only23,26 follow the newly found best21.
    assert plateau(29, metric(21), c)


def test_stage_command_keeps_guard_and_evaluation_boundary(tmp_path):
    template = ['python', 'scripts/run_guarded_request.py', '--output', 'oldguard', '--',
                'python', 'scripts/run_request_collection.py', '--output', 'oldcollection',
                '--checkpoint', 'oldcheckpoint', '--workers', '20']
    evaluation = stage_command(template, tmp_path, Path('update_17/checkpoint.pt'), 17, False)
    training = stage_command(template, tmp_path, Path('update_17/checkpoint.pt'), 17, True)
    assert '--update-index' not in evaluation and '--ppo-output' not in evaluation
    assert training[-4:] == ['--update-index', '17', '--ppo-output', str(tmp_path/'update_18')]
    assert template[3] == 'oldguard'
    assert training[3] == str(tmp_path/'guards/train_17')


def test_panel_rejects_duplicate_missing_or_wrong_checkpoint():
    cases = [dict(episode_id='a'), dict(episode_id='b')]
    rows = [dict(case=c['episode_id'], seed=17, checkpoint_sha256='hash', reward=1.,
                 passed=True, safety_failure=False) for c in cases]
    values = dict(rows=rows, count=2, successes=2, safety_failures=0, mean_reward=1.)
    assert validate_summary(values, cases, 1, 'hash', None, 99)['successes'] == 2
    with pytest.raises(ValueError, match='checkpoint'):
        validate_summary(values, cases, 1, 'other', None, 99)
    with pytest.raises(ValueError, match='wrong-seed'):
        validate_summary(values, cases, 1, 'hash', 0, 99)
    rows[1] = rows[0]
    with pytest.raises(ValueError, match='duplicate'):
        validate_summary(values, cases, 1, 'hash', None, 99)


def test_raw_archive_roundtrip_and_does_not_remove_training_input(tmp_path):
    group = tmp_path/'panel/case/group_0'
    group.mkdir(parents=True)
    raw = group/'rollout.pt'
    original = bytes(range(256))*1024
    raw.write_bytes(original)
    training = group/'training_rollout.pt'
    training.write_bytes(b'keep')
    archive_raw(tmp_path)
    assert not raw.exists() and training.read_bytes() == b'keep'
    row = read(tmp_path/'raw_log_archives.json')['files'][0]
    restored = subprocess.check_output(['zstd', '-d', '--stdout', '--quiet', row['archive']])
    assert restored == original
    assert row['sha256'] == hashlib.sha256(original).hexdigest()
    archive_raw(tmp_path)
    assert len(read(tmp_path/'raw_log_archives.json')['files']) == 1


def test_full_schedule_refines_historical_neighbor_and_stops_after_nine_updates(tmp_path):
    from scripts.run_request_training import Schedule
    runner = object.__new__(Schedule)
    runner.root = tmp_path
    runner.contract = dict(resume_update=17, validation_interval=3, minimum_new_updates=9,
                           stale_validations=3, baseline_updates=[14, 17], reward_tolerance=.01,
                           neighbor_radius=2)
    runner.checkpoints = {k: Path(f'update_{k}/checkpoint.pt') for k in (14, 15, 16, 17)}
    runner.validations, runner.curve = [], []
    calls = []

    def stage(k, training=False):
        calls.append((k, training))
        if training:
            runner.curve.append(dict(count=448))
            runner.checkpoints[k+1] = Path(f'update_{k+1}/checkpoint.pt')
        else:
            runner.validations.append(metric(k, successes=21 if k == 16 else 20))
    runner.stage = stage
    runner.run()
    result = read(tmp_path/'completed.json')
    assert result['best']['update'] == 16
    assert result['last_update'] == 26
    assert result['training_episodes'] == 9*448
    assert set(range(14, 19)) <= set(result['evaluated_updates'])
    assert [k for k, train in calls if train] == list(range(17, 26))


def test_signal_unwinds_wait_before_reaping_child():
    import signal
    from types import SimpleNamespace
    from scripts.run_request_training import Schedule
    calls = []
    runner = object.__new__(Schedule)
    runner.child = SimpleNamespace(send_signal=lambda s: calls.append(('signal', s)),
                                   wait=lambda timeout: calls.append(('wait', timeout)))
    with pytest.raises(InterruptedError):
        runner.interrupt(signal.SIGTERM, None)
    assert calls == []  # Signal handler must not recursively enter Popen.wait's lock.
    runner.stop_child()
    assert calls == [('signal', signal.SIGTERM), ('wait', 60)]


def test_real_interrupted_child_wait_finishes_without_recursive_wait_deadlock():
    import sys
    code = '''
import signal, subprocess, sys
from scripts.run_request_training import Schedule
r = object.__new__(Schedule)
r.child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
signal.signal(signal.SIGALRM, r.interrupt)
signal.setitimer(signal.ITIMER_REAL, .05)
try:
    r.child.wait()
    raise AssertionError('interrupt did not arrive')
except InterruptedError:
    r.stop_child()
assert r.child.poll() is not None
'''
    subprocess.run([sys.executable, '-c', code], check=True, timeout=3)


def test_rotating_pool_preserves_fixed_settings_coverage_and_file_identity(tmp_path):
    base = dict(evaluation_cases=[{'episode_id': 'held'}], untouched_test_cases=[], ppo={'epochs': 8},
                distribution={'mass': [.8, 1.2]}, training_draws_per_case=16,
                training_seed_base=17, configuration_hashes={})
    cases = [dict(episode_id=str(i), split='train') for i in range(4)]
    entries = []
    for i in range(2):
        path = tmp_path/f'panel_{i}.json'
        write(path, dict(base, training_cases=cases[i*2:i*2+2]))
        entries.append(dict(path=str(path), sha256=sha(path), cases=[str(i*2), str(i*2+1)], episodes=32))
    dataset = tmp_path/'dataset.json'
    write(dataset, dict(panels=entries, training_cases=cases))
    assert len(load_training_panels(dataset, base)) == 2
    with pytest.raises(ValueError, match='fixed setting'):
        load_training_panels(dataset, dict(base, ppo={'epochs': 9}))
    write(dataset, dict(panels=entries*2, training_cases=cases))
    with pytest.raises(ValueError, match='exactly once'):
        load_training_panels(dataset, base)
    write(dataset, dict(panels=entries, training_cases=cases))
    Path(entries[0]['path']).write_text('{}')
    with pytest.raises(ValueError, match='panel changed'):
        load_training_panels(dataset, base)


def test_full_pool_is_used_before_plateau_can_stop():
    c = dict(resume_update=16, validation_interval=3, minimum_new_updates=30, stale_validations=3)
    assert not plateau(43, metric(16), c)
    assert plateau(46, metric(16), c)
    assert not plateau(46, metric(43), c)
    resumed = dict(c, resume_update=26, validation_anchor_update=16, minimum_new_updates=20)
    assert not plateau(43, metric(16), resumed)
    assert plateau(46, metric(16), resumed)
    assert not plateau(46, metric(43), resumed)


def test_optimizer_recovery_binds_existing_collection_and_allows_only_device_change(tmp_path):
    stage = tmp_path/'stage';stage.mkdir()
    ck = tmp_path/'checkpoint.pt';ck.write_bytes(b'checkpoint')
    state = tmp_path/'training_state.pt';state.write_bytes(b'optimizer')
    rollout = tmp_path/'rollout.pt';rollout.write_bytes(b'complete rollout')
    manifest = tmp_path/'groups.json';write(manifest, {})
    output = tmp_path/'updated'
    write(output/'result.json', dict(reload_exact=True, optimization=dict(completed_epochs=8, optimizer_updates_resumed=22)))
    original = ['python', 'train.py', '--checkpoint', str(ck), '--rollouts', str(rollout),
                '--output', str(output), '--contact-group-manifest', str(manifest), '--device', 'cuda:0']
    write(stage/'optimizer_command.json', dict(argv=original))
    write(stage/'collection_binding.json', dict(collection_policy_update=22))
    write(stage/'panel/summary.json', {})
    write(stage/'panel/completed.json', {})
    failed = tmp_path/'failed_guard.json';write(failed, dict(exit_code=1))
    recovered = [*original[:-1], 'cpu']
    guard = tmp_path/'recovery_guard.json';write(guard, dict(exit_code=0, error=None, seconds=3, command=['env', *recovered]))
    paths = [ck, state, rollout, manifest, stage/'optimizer_command.json', stage/'collection_binding.json',
             stage/'panel/summary.json', stage/'panel/completed.json']
    evidence = dict(failed_guard_sha256=sha(failed), input_hashes={str(p):sha(p) for p in paths},
                    optimizer_argv=recovered, device='cpu', guard_result=str(guard), guard_result_sha256=sha(guard))
    write(stage/'optimizer_recovery.json', evidence)
    assert validate_optimizer_recovery(stage, failed) == (3, 'cpu')
    del evidence['input_hashes'][str(rollout)]
    write(stage/'optimizer_recovery.json', evidence)
    with pytest.raises(ValueError, match='omits'):
        validate_optimizer_recovery(stage, failed)
    evidence['input_hashes'][str(rollout)] = sha(rollout)
    write(stage/'optimizer_recovery.json', evidence)
    rollout.write_bytes(b'changed')
    with pytest.raises(ValueError, match='input changed'):
        validate_optimizer_recovery(stage, failed)
