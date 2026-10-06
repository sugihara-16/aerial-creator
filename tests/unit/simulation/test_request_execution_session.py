from types import SimpleNamespace

import pytest

from scripts.run_request_policy import RequestIsaacSession, _with_request_session


def test_shared_reset_aligns_settling_drift_without_moving_reference_or_offsets():
    import torch
    from scripts.run_request_policy import synchronize_copied_object_reset
    origins = torch.tensor([[0.,0.,0.], [3.,0.,0.]])
    local = torch.tensor([[1.,2.,.5,0.,0.,0.,1.], [1.00002,2.,.5,0.,0.,0.,1.]])
    world = local.clone(); world[:,:3] += origins
    authored = local[:1].repeat(2,1)
    data = SimpleNamespace(default_root_state=authored, root_pose_w=world.clone())
    calls = []
    def apply(root_pose, env_ids):
        calls.append('write');data.root_pose_w[env_ids] = root_pose
    obj = SimpleNamespace(data=data, write_root_pose_to_sim_index=apply)
    class Scene(dict):
        num_envs=2;device='cpu';env_origins=origins
        def update(self, dt):calls.append(('update',dt))
    scene=Scene(object=obj);sim=SimpleNamespace(forward=lambda:calls.append('forward'))
    result=synchronize_copied_object_reset(scene,sim)
    assert result['aligned']
    assert torch.equal(data.root_pose_w[0],world[0])
    torch.testing.assert_close(data.root_pose_w[1,:3],world[0,:3]+origins[1],rtol=0,atol=0)
    assert calls == ['write','forward',('update',0.)]
    calls.clear()
    assert not synchronize_copied_object_reset(scene,sim)['aligned']
    assert not calls
    data.default_root_state[1,0] += .01
    with pytest.raises(ValueError,match='identical authored'):
        synchronize_copied_object_reset(scene,sim)


def test_shared_reset_hook_precedes_original_strict_check_and_fails_closed():
    from scripts.run_request_policy import _with_synchronized_object_reset
    source='            c3_phase_reset_expected = _install_c3_nominal_phase_bank(scene)\n'
    changed=_with_synchronized_object_reset(source)
    assert changed.index('synchronize_request_reset') < changed.index('_install_c3_nominal_phase_bank')
    assert source in changed
    with pytest.raises(ValueError,match='boundary'):
        _with_synchronized_object_reset(source*2)


def test_application_reused_but_arguments_are_resolved_each_job():
    calls = []
    closed = []
    def factory(args):
        calls.append(args)
        args.device = 'cuda:0'
        return SimpleNamespace(app=SimpleNamespace(close=lambda **kwargs: closed.append(kwargs['exit_code'])))
    session = RequestIsaacSession()
    first = SimpleNamespace(device='cuda', headless=True, enable_cameras=False, livestream=0)
    launcher = session.acquire(factory, first)
    second = SimpleNamespace(device='cuda', headless=True, enable_cameras=False, livestream=0)
    assert session.acquire(factory, second) is launcher
    assert second.device == 'cuda:0' and len(calls) == 1
    with pytest.raises(ValueError, match='settings changed'):
        session.acquire(factory, SimpleNamespace(device='cpu', headless=True, enable_cameras=False, livestream=0))
    session.close()
    session.close()
    assert closed == [0]


def test_application_hook_fails_closed_if_boundary_changes():
    source = 'app_launcher = AppLauncher(args_cli)\nsimulation_app = app_launcher.app\n'
    assert 'request_isaac_session.acquire' in _with_request_session(source)
    with pytest.raises(ValueError, match='boundary changed'):
        _with_request_session(source + source)


def test_batch_records_every_job_and_never_overwrites_results(tmp_path, monkeypatch):
    import json
    import scripts.run_request_policy as runner
    from amsrr.utils.hashing import hash_file
    monkeypatch.setattr(runner, 'sources', lambda: {'runtime': 'fixed'})
    jobs = []
    for index in range(2):
        directory = tmp_path / str(index)
        directory.mkdir()
        for name in ('plan', 'scene'):
            (directory / (name + '.json')).write_text('{}')
        job = dict(source_hashes=runner.sources(), **{name + '_sha256': hash_file(directory / (name + '.json')) for name in ('plan', 'scene')})
        (directory / 'job.json').write_text(json.dumps(job))
        jobs.append(['--job', str(directory / 'job.json')])
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(jobs))
    sessions = []
    def execute(args, *, session):
        sessions.append(session)
        runner.write_json(__import__('pathlib').Path(args.job).parent / 'result.json', dict(passed=True))
        return 0
    monkeypatch.setattr(runner, 'execute', execute)
    assert runner.execute_batch(SimpleNamespace(manifest=str(manifest), timeout_s=30)) == 0
    assert sessions[0] is sessions[1]
    assert len(json.loads(manifest.with_suffix('.results.json').read_text())) == 2
    assert len(manifest.with_suffix('.progress.jsonl').read_text().splitlines()) == 2


def test_cpu_session_bounds_both_carb_and_physics_workers(monkeypatch):
    monkeypatch.setenv('AMSRR_ISAAC_THREADS', '1')
    arguments = SimpleNamespace(device='cpu', headless=True, enable_cameras=False, livestream=0)
    session = RequestIsaacSession()
    session.acquire(lambda args: SimpleNamespace(app=None), arguments)
    assert '--/plugins/carb.tasking.plugin/threadCount=1' in arguments.kit_args
    assert '--/persistent/physics/numThreads=2' in arguments.kit_args
    assert session.requested['physics_worker_threads'] == 2


def test_cpu_session_uses_calling_thread_and_rejects_stalled_dispatcher(monkeypatch):
    monkeypatch.setenv('AMSRR_ISAAC_THREADS', '1')
    monkeypatch.setenv('AMSRR_ISAAC_PHYSX_THREADS', '0')
    args = SimpleNamespace(device='cpu', headless=True, enable_cameras=False, livestream=0)
    session = RequestIsaacSession()
    session.acquire(lambda args: SimpleNamespace(app=None), args)
    assert '--/persistent/physics/numThreads=0' in args.kit_args
    assert session.requested['physics_worker_threads'] == 0
    monkeypatch.setenv('AMSRR_ISAAC_PHYSX_THREADS', '1')
    with pytest.raises(ValueError, match='thread counts'):
        RequestIsaacSession().acquire(lambda args: None, args)


def test_rejected_episode_commands_are_not_applied_and_other_rows_are_unchanged():
    from dataclasses import dataclass
    import torch
    from scripts.run_request_policy import mask_rejected_request_commands, terminate_rejected_request_rows
    @dataclass
    class Allocation:
        rotor_thrusts_n: object
        vectoring_joint_targets_rad: object
    @dataclass
    class Result:
        allocation: object
    @dataclass
    class Command:
        local_joint_ids: object
        joint_position_targets_rad: object
        joint_velocity_targets_radps: object
        joint_torque_bias_nm: object
    @dataclass
    class Reward:
        terminal_failure: object
        phase_success: object
    step=SimpleNamespace(controller_result=Result(Allocation(torch.full((2,2),7.),torch.full((2,1),8.))),
        policy_command=Command(('b','a'),torch.full((2,1,2),9.),torch.full((2,1,2),10.),torch.full((2,1,2),11.)))
    robot=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.tensor([[.1,.2,.3],[.4,.5,.6]])))
    io=SimpleNamespace(local_joint_ids=('a','b'),local_joint_indices=((0,2),),module_ids=(0,),
        _constants=lambda device,dtype: {'vectoring':torch.tensor([1])})
    assert mask_rejected_request_commands(step,robot,io,{}) is step
    changed=mask_rejected_request_commands(step,robot,io,{1:'unsafe'})
    assert changed.controller_result.allocation.rotor_thrusts_n.tolist()==[[7.,7.],[0.,0.]]
    torch.testing.assert_close(changed.controller_result.allocation.vectoring_joint_targets_rad,torch.tensor([[8.],[.5]]))
    torch.testing.assert_close(changed.policy_command.joint_position_targets_rad,torch.tensor([[[9.,9.]],[[.6,.4]]]))
    assert changed.policy_command.joint_velocity_targets_radps.tolist()==[[[10.,10.]],[[0.,0.]]]
    assert changed.policy_command.joint_torque_bias_nm.tolist()==[[[11.,11.]],[[0.,0.]]]
    assert step.controller_result.allocation.rotor_thrusts_n.tolist()==[[7.,7.],[7.,7.]]
    reward=Reward(torch.tensor([False,False]),torch.tensor([True,True]));phase=torch.tensor([True,True])
    terminated,next_phase=terminate_rejected_request_rows(reward,phase,torch.tensor([True,True]),{1:'unsafe'})
    assert terminated.terminal_failure.tolist()==[False,True]
    assert next_phase.tolist()==[True,False]
    assert reward.terminal_failure.tolist()==[False,False]


def test_rejection_actuator_hook_is_unique_in_real_harness():
    from pathlib import Path
    from scripts.run_request_policy import PROTECTED_HARNESS
    source = Path(PROTECTED_HARNESS).read_text()
    boundary = '        io.apply(\n            robot=robot,\n            policy_command=policy_step.policy_command,'
    assert source.count(boundary) == 1
