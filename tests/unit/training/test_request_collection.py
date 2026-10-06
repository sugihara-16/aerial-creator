from pathlib import Path
from types import SimpleNamespace
import json

import torch
import pytest
from concurrent.futures import ThreadPoolExecutor

from tests.unit.policies.test_high_level_requests import request_scene, decision
from amsrr.policies.request_actor_critic import RequestActorCritic
from amsrr.training.request_collection import execute_panel
from amsrr.utils.hashing import hash_file


def test_replay_rechecks_batch_rounding_but_rejects_real_probability_mismatch(monkeypatch, request_scene):
    import amsrr.training.request_collection as collector
    from amsrr.training.request_ppo import serialize_encoding
    actor = RequestActorCritic().eval()
    selected = actor.decide(decision(request_scene), deterministic=True)
    event = dict(encoding=serialize_encoding(selected['encoded']),
                 action=selected['action'], log_prob=selected['log_prob'])
    original = collector.evaluate_batch
    def rounded_batch(model, data, indices, device):
        scores, value = original(model, data, indices, device)
        if len(indices) > 1:
            scores[:, event['action']] += .001
        return scores, value
    monkeypatch.setattr(collector, 'evaluate_batch', rounded_batch)
    errors = collector.replay_log_probability_errors(actor, [event, event])
    assert (errors < 2e-5).all()
    corrupted = dict(event, log_prob=event['log_prob'] + .1)
    errors = collector.replay_log_probability_errors(actor, [event, corrupted])
    assert errors[0] < 2e-5 and errors[1] > .09


def test_affinity_release_only_changes_owned_live_process(tmp_path, monkeypatch):
    import os
    import threading
    from scripts.run_request_collection import release_worker_affinity
    worker = SimpleNamespace(_lifecycle_lock=threading.Lock(), _closed=False,
        cpu_affinity=(0,), process=None)
    calls = []
    monkeypatch.setattr(os, 'sched_setaffinity', lambda pid, cpus: calls.append((pid, set(cpus))))
    release_worker_affinity(worker, {1, 2})
    assert worker.cpu_affinity is None and calls == []
    worker.process = SimpleNamespace(pid=987654321, poll=lambda: 0)
    release_worker_affinity(worker, {1, 2})
    assert calls == []  # A reused PID must never be touched after owned exit.
    worker._closed = True
    worker.process = SimpleNamespace(pid=987654321, poll=lambda: None)
    release_worker_affinity(worker, {1, 2})
    assert calls == []


def test_affinity_release_widens_live_child_without_restarting_it():
    import os
    import subprocess
    import sys
    import threading
    from scripts.run_request_collection import release_worker_affinity
    allowed = sorted(os.sched_getaffinity(0))[:2]
    if len(allowed) < 2:
        pytest.skip('needs two allowed CPUs')
    code = ('import os,sys,json; os.sched_setaffinity(0,{' + str(allowed[0]) + '}); '
            'print(json.dumps(sorted(os.sched_getaffinity(0))),flush=True); '
            'sys.stdin.readline(); print(json.dumps(sorted(os.sched_getaffinity(0))),flush=True)')
    child = subprocess.Popen([sys.executable, '-u', '-c', code],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    worker = SimpleNamespace(_lifecycle_lock=threading.Lock(), _closed=False,
                             cpu_affinity=(allowed[0],), process=child)
    try:
        assert json.loads(child.stdout.readline()) == allowed[:1]
        release_worker_affinity(worker, set(allowed))
        child.stdin.write('\n'); child.stdin.flush()
        assert json.loads(child.stdout.readline()) == allowed
        assert child.wait(timeout=5) == 0
    finally:
        if child.poll() is None:
            child.terminate(); child.wait(timeout=5)


@pytest.mark.parametrize("workers", [1,3])
@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("anchor_source", ['defined', 'all_free', 'neighborhood'])
def test_grouping_preserves_every_independent_draw_and_rejected_training_signal(tmp_path, monkeypatch, request_scene, workers, staged, anchor_source):
    import amsrr.training.request_collection as collector
    import amsrr.training.request_object_conditions as conditions
    from scripts.run_request_policy import save_planning_rejection
    context = decision(request_scene)
    actor = RequestActorCritic().eval()
    model = tmp_path/'model'
    model.mkdir()
    checkpoint = model/'checkpoint.pt'
    torch.save(actor.checkpoint(), checkpoint)
    record = tmp_path/'record.json'
    record.write_text(json.dumps(dict(dataset_split='train')))
    case = dict(episode_id='case', split='train', record=str(record), record_sha256=hash_file(record))
    monkeypatch.setattr(collector, 'build_physical_model_from_config', lambda path: None)
    monkeypatch.setattr(conditions, 'expand_episode_grasp_contacts', lambda *a, **kw: (None, context))
    def write(path, data):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data))
    def command(argv, log, timeout, allowed):
        assert 'prepare' in argv  # The fake planner rejects; Isaac must never run.
        assert ('--max-grasp-contacts' in argv) == (anchor_source != 'defined')
        assert ('--grasp-anchor-source' in argv) == (anchor_source == 'neighborhood')
        out = Path(argv[argv.index('--output')+1])
        out.mkdir()
        seed = int(argv[argv.index('--seed')+1])
        with torch.no_grad():
            selected = actor.decide(context, generator=torch.Generator().manual_seed(seed))
        write(out/'decision.json', dict(request=selected['request'].to_dict()))
        save_planning_rejection(out, selected, checkpoint, True, 'synthetic unreachable plan')
    reports = []
    for batch_limit in [1, 16]:
        root = tmp_path/str(batch_limit)
        root.mkdir()
        protocol = dict(training_cases=[case], training_draws_per_case=17,
            training_seed_base=123, maximum_training_environments=batch_limit,
            grasp_anchor_source=anchor_source)
        runtime = SimpleNamespace(root=root, repo=tmp_path, python='unused', protocol=protocol,
            load_episode=lambda *a: None, geometry_index=lambda:{}, sha=hash_file,
            write=write, command=command, verify=lambda:None, emit=lambda value:None,
            stage_physics=staged,
            run_physics_batch=lambda *a: pytest.fail('rejected plans must not run physics'))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            if workers > 1:
                runtime.group_executor = executor
            report, paths = execute_panel(runtime, 'train', [case], [checkpoint], update_index=2)
        rows = report['model']['rows']
        assert len(rows) == 17 and len({r['seed'] for r in rows}) == 17
        assert report['model']['mean_reward'] == -5 and report['model']['successes'] == 0
        events = []
        for path in paths:
            archive = torch.load(path, weights_only=True)
            assert not archive['evaluation_only'] and not archive['diagnostic_only']
            events += archive['episodes']
        assert len(events) == 17 and all(e[0]['done'] and e[0]['reward'] == -5 for e in events)
        manifest = json.loads((root/'train/contact_groups.json').read_text())
        assert manifest['group_size'] == 17
        assert {r['path'] for r in manifest['rollouts']} == set(paths)
        assert all(hash_file(r['path']) == r['sha256'] for r in manifest['rollouts'])
        reports.append({r['seed']:(r['selected_group'], r['reward'], r['replay_error']) for r in rows})
    assert reports[0] == reports[1]


def test_stop_signal_interrupts_worker_poll_instead_of_being_swallowed():
    import os
    import signal
    import threading
    from multiprocessing import Pipe
    import pytest
    from scripts.run_request_collection import interrupt_collection, CollectionInterrupted
    parent, child = Pipe()
    previous = signal.signal(signal.SIGINT, interrupt_collection)
    timer = threading.Timer(.05, lambda: os.kill(os.getpid(), signal.SIGINT))
    try:
        timer.start()
        with pytest.raises(CollectionInterrupted):
            parent.poll(2)
    finally:
        timer.cancel()
        timer.join()
        signal.signal(signal.SIGINT, previous)
        parent.close()
        child.close()


@pytest.mark.parametrize('event_count', [1, 3])
def test_existing_optimizer_keeps_protocol_settings_and_all_rejection_handling(tmp_path, event_count):
    from amsrr.training.request_collection import ppo_update_command
    archive = tmp_path/'rollout.pt'
    torch.save(dict(complete=True, diagnostic_only=False, evaluation_only=False,
        episodes=[[dict(done=i == event_count-1) for i in range(event_count)] for _ in range(4)]), archive)
    protocol = dict(training_cases=[{}, {}], training_draws_per_case=2,
        ppo=dict(learning_rate=.0007, epochs=4, contact_loss_weight=.5,
                 contact_target_kl=.09, temporal_target_kl=.1))
    command = ppo_update_command(python='python', checkpoint='checkpoint.pt', archives=[archive],
        output='update', manifest='manifest.json', protocol=protocol)
    assert command[command.index('--learning-rate')+1] == '0.0007'
    assert command[command.index('--epochs')+1] == '4'
    assert ('--contact-target-kl' in command) == (event_count > 1)
    assert ('--temporal-target-kl' in command) == (event_count > 1)


@pytest.mark.parametrize('flag', ['diagnostic_only', 'evaluation_only', 'teacher_phase_supervision', 'teacher_contact_supervision'])
def test_optimizer_bridge_rejects_non_training_artifacts(tmp_path, flag):
    from amsrr.training.request_collection import ppo_update_command
    archive = tmp_path/'rollout.pt'
    torch.save(dict(complete=True, episodes=[[{}]], **{flag: True}), archive)
    with pytest.raises(ValueError, match='without demonstrations or diagnostics'):
        ppo_update_command(python='python', checkpoint='checkpoint.pt', archives=[archive],
            output='update', manifest='manifest.json', protocol={})


def test_optimizer_bridge_declares_value_baseline_once_without_modifying_adam(tmp_path):
    from amsrr.training.request_collection import ppo_update_command
    archive = tmp_path/'rollout.pt'
    torch.save(dict(complete=True, episodes=[[dict(done=False), dict(done=True)]]), archive)
    state = tmp_path/'training_state.pt'
    protocol = dict(training_cases=[{}], training_draws_per_case=1,
                    ppo=dict(contact_baseline='value_v1', gae_lambda=.95))
    for previous in ['condition_loo_v1', 'value_v1']:
        torch.save(dict(contact_baseline_kind=previous, actor_optimizer={'step': 72}), state)
        before = state.read_bytes()
        command = ppo_update_command(python='python', checkpoint=tmp_path/'checkpoint.pt',
            archives=[archive], output=tmp_path/'next', manifest='manifest.json', protocol=protocol)
        assert '--contact-group-manifest' not in command
        assert '--contact-baseline' not in command
        assert command[command.index('--gae-lambda')+1] == '0.95'
        assert ('--contact-baseline-transition-from' in command) == (previous != 'value_v1')
        assert state.read_bytes() == before


def test_optimizer_bridge_rejects_reduced_panel(tmp_path):
    from amsrr.training.request_collection import ppo_update_command
    archive = tmp_path/'rollout.pt'
    torch.save(dict(complete=True, episodes=[[{}]]), archive)
    with pytest.raises(ValueError, match='expected 32'):
        ppo_update_command(python='python', checkpoint='checkpoint.pt', archives=[archive],
            output='update', manifest='manifest.json',
            protocol=dict(training_cases=[{}, {}], training_draws_per_case=16))


@pytest.mark.parametrize('audit_submit_failure', [False, True])
def test_staged_physics_waits_for_all_plans_and_retains_mixed_draw_order(tmp_path, monkeypatch, request_scene, audit_submit_failure):
    """Accepted jobs cross the phase barrier; rejected draws retain training data."""
    from copy import deepcopy
    import amsrr.training.request_collection as collector
    import amsrr.training.request_object_conditions as conditions
    from scripts.run_request_policy import save_planning_rejection
    context = decision(request_scene)
    actor = RequestActorCritic().eval()
    checkpoint = tmp_path/'model/checkpoint.pt'
    checkpoint.parent.mkdir()
    torch.save(actor.checkpoint(), checkpoint)
    record = tmp_path/'record.json'
    record.write_text(json.dumps(dict(dataset_split='train')))
    cases = [dict(episode_id=f'case{i}', split='train', record=str(record),
                  record_sha256=hash_file(record)) for i in range(2)]
    monkeypatch.setattr(collector, 'build_physical_model_from_config', lambda _: None)
    monkeypatch.setattr(conditions, 'expand_episode_grasp_contacts', lambda *a, **kw: (None, context))
    prepared, executed, closed = [], [], []
    def write(path, data):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data))
    def command(argv, *_):
        assert 'prepare' in argv
        out = Path(argv[argv.index('--output')+1]); out.mkdir()
        seed = int(argv[argv.index('--seed')+1])
        selected = actor.decide(context, generator=torch.Generator().manual_seed(seed))
        write(out/'decision.json', dict(request=selected['request'].to_dict()))
        save_planning_rejection(out, selected, checkpoint, True, 'synthetic rejection')
        if out.parent.name == 'case0':
            (out/'planning_rejection.json').unlink()
        prepared.append(out)
    def physics(commands, folder):
        assert closed == [True]
        assert {p.parent.name for p in prepared} == {'case0', 'case1'}
        for argv, *_ in reversed(commands):  # Completion may differ from protocol order.
            out = Path(argv[argv.index('--job')+1]).parent
            assert out.parent.name == 'case0'
            seeds = list(map(int, argv[argv.index('--environment-seeds')+1:argv.index('--rollout-steps')]))
            archive = torch.load(out/'request_rollout.pt', weights_only=True)
            original = archive['episodes'][0]
            episodes = []
            for seed in seeds:
                selected = actor.decide(context, generator=torch.Generator().manual_seed(seed))
                events = deepcopy(original)
                events[0].update(action=selected['action'], log_prob=selected['log_prob'])
                episodes.append(events)
            archive.update(evaluation_only=False, episodes=episodes, environment_seeds=seeds,
                environment_checkpoint_sha256=[hash_file(checkpoint)]*len(seeds))
            torch.save(archive, out/'request_rollout.pt')
            write(out/'result.json', dict(physical_acceptance_eligible=True,
                teacher_phase_supervision=False, environment_origins_world=[[0.,0.,0.]]*len(seeds),
                episodes=[dict(fallback_decision_count=0, task_success=True, safety_failure=False,
                               no_fallback_success=True, failure_reason=None)]*len(seeds)))
            executed.append(out)
    runtime = SimpleNamespace(root=tmp_path, repo=tmp_path, python='unused',
        protocol=dict(training_cases=cases, training_draws_per_case=4, training_seed_base=123,
                      maximum_training_environments=2), load_episode=lambda *a: None,
        geometry_index=lambda: {}, sha=hash_file, write=write, command=command,
        verify=lambda: None, emit=lambda _: None, stage_physics=True,
        close_workers=lambda: closed.append(True), run_physics_batch=physics,
        audit=lambda *a, **kw: dict(physical_goal_and_motion_passed=True))
    with ThreadPoolExecutor(max_workers=2) as executor:
        if audit_submit_failure:
            from threading import Event, Timer
            started, release, finished = Event(), Event(), Event()
            def blocking_audit(*args, **kwargs):
                started.set()
                assert release.wait(5)
                finished.set()
                return dict(physical_goal_and_motion_passed=True)
            runtime.audit = blocking_audit
            count = 0
            def submit(fn, *args):
                nonlocal count
                if fn.__name__ == 'finish_group':
                    count += 1
                    if count == 2:
                        assert started.wait(5)
                        Timer(0.05, release.set).start()
                        raise RuntimeError('synthetic audit submit failure')
                return executor.submit(fn, *args)
            runtime.group_executor = SimpleNamespace(submit=submit)
            with pytest.raises(RuntimeError, match='synthetic audit submit failure'):
                execute_panel(runtime, 'train', cases, [checkpoint], update_index=2)
            assert finished.is_set()
            return
        runtime.group_executor = executor
        report, paths = execute_panel(runtime, 'train', cases, [checkpoint], update_index=2)
    rows = report['model']['rows']
    assert len(rows) == 8 and report['model']['successes'] == 4
    assert [r['case'] for r in rows] == ['case0']*4+['case1']*4
    assert all(not r['planning_rejected'] for r in rows[:4])
    assert all(r['planning_rejected'] for r in rows[4:])
    assert len(executed) > 0 and sum(len(torch.load(p, weights_only=True)['episodes']) for p in paths) == 8


@pytest.mark.parametrize('message', [
    'joint/centroidal preload IK did not realize bounded normal lead',
    'contact posture has excessive carried-goal clearance or normal error',
    'selected contact goal IK did not converge: position=0.2',
])
def test_bounded_preload_failures_are_rejected_without_hiding_input_errors(message):
    from scripts.run_request_policy import is_ordinary_planning_rejection
    assert is_ordinary_planning_rejection(ValueError(message))
    assert is_ordinary_planning_rejection(TimeoutError('planner deadline'))
    assert not is_ordinary_planning_rejection(ValueError('nominal preload model contains non-finite values'))
    assert not is_ordinary_planning_rejection(ValueError('joint discontinuity entering grasp'))
    assert not is_ordinary_planning_rejection(RuntimeError(message))
