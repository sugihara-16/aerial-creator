from dataclasses import asdict
from types import SimpleNamespace
import pytest
import amsrr.policies.request_contact_planner as p


def context():
    morphology=SimpleNamespace(base_module_id=9)
    scene=SimpleNamespace(morphology_graph=morphology,runtime_observation=SimpleNamespace(module_states=[SimpleNamespace(module_id=1,pose_world=('wrong',)),SimpleNamespace(module_id=9,pose_world=(0,0,1,0,0,0,1))]))
    return SimpleNamespace(scene=scene,physical_model=object())


def test_accepted_primary_path_is_returned_without_retry(monkeypatch):
    expected=object();calls=[]
    monkeypatch.setattr(p,'_plan_request_geometry',lambda *a,**kw:(calls.append(kw),expected)[1])
    assert p.plan_request_geometry(object(),object()) is expected
    assert len(calls)==1


@pytest.mark.parametrize('fail_second',[False,True])
def test_retry_keeps_binding_constraints_and_propagates_seed(monkeypatch,fail_second):
    calls=[];request=object();ctx=context();seeds=({'j':-.5},{'j':.5})
    monkeypatch.setattr(p,'ordered_global_dock_joint_ids',lambda *a:['j'])
    monkeypatch.setattr(p,'_global_joint_limits',lambda *a:{'j':(-1,1)})
    monkeypatch.setattr(p,'_joint_limit_branch_seeds',lambda *a:seeds)
    def plan(c,r,**kw):
        assert c is ctx and r is request
        calls.append(kw)
        if len(calls)==1:raise p._ContactGoalInitializationError('did not converge')
        if fail_second and len(calls)==2:raise ValueError('collision remains')
        return SimpleNamespace(provenance={})
    monkeypatch.setattr(p,'_plan_request_geometry',plan)
    result=p.plan_request_geometry(ctx,request,deadline_s=20)
    assert len(calls)==(3 if fail_second else 2)
    for call,seed in zip(calls[1:],seeds):
        assert call['joint_seed'] is seed
        assert call['base_seed']==(0,0,1,0,0,0,1)
        assert 0<call['deadline_s']<=20
        settings=asdict(call['initial_ik_config']);assert settings.pop('maximum_iterations')==240
        assert settings.pop('preserve_base_tilt') is True
        default=asdict(p.ArticulatedIKConfig());default.pop('maximum_iterations');default.pop('preserve_base_tilt');assert settings==default
    assert result.provenance['contact_goal_initialization_retry']['maximum_branches']==2


def test_later_collision_failure_does_not_trigger_contact_retry(monkeypatch):
    calls=[]
    def plan(*a,**kw):calls.append(kw);raise ValueError('later collision')
    monkeypatch.setattr(p,'_plan_request_geometry',plan)
    with pytest.raises(ValueError,match='later collision'):p.plan_request_geometry(object(),object())
    assert len(calls)==1


def test_all_failed_branches_stop_without_changing_contact_group(monkeypatch):
    ctx=context();request=object();calls=[]
    monkeypatch.setattr(p,'ordered_global_dock_joint_ids',lambda *a:['j'])
    monkeypatch.setattr(p,'_global_joint_limits',lambda *a:{'j':(-1,1)})
    monkeypatch.setattr(p,'_joint_limit_branch_seeds',lambda *a:({'j':-.5},{'j':.5}))
    def plan(c,r,**kw):
        assert r is request
        calls.append(kw)
        raise p._ContactGoalInitializationError('no solution')
    monkeypatch.setattr(p,'_plan_request_geometry',plan)
    with pytest.raises(ValueError,match='no solution'):p.plan_request_geometry(ctx,request)
    assert len(calls)==3


def test_base_tilt_preservation_is_explicit_and_boolean():
    assert p.ArticulatedIKConfig().preserve_base_tilt is False
    assert p.ArticulatedIKConfig(preserve_base_tilt=True).preserve_base_tilt is True
    with pytest.raises(ValueError,match="preserve_base_tilt"):
        p.ArticulatedIKConfig(preserve_base_tilt="yes")


@pytest.mark.parametrize('failures',[1,2])
def test_unreachable_pregrasp_shortens_intermediate_clearance_only(monkeypatch,failures):
    calls=[];ctx=object();request=object();seed={'joint':.3};base=(0,0,1,0,0,0,1)
    def plan(c,r,**kw):
        assert c is ctx and r is request
        assert kw['joint_seed'] is seed and kw['base_seed'] is base
        assert 'initial_ik_config' not in kw
        calls.append(kw)
        if len(calls)<=failures:
            raise p._PregraspGoalInitializationError('unreachable open grasp')
        return SimpleNamespace(provenance={})
    monkeypatch.setattr(p,'_plan_request_geometry',plan)
    result=p.plan_request_geometry(ctx,request,joint_seed=seed,base_seed=base,deadline_s=20)
    assert [c.get('pregrasp_clearance_m',.08) for c in calls]==[.08,.04,.02][:failures+1]
    assert all(0<c['deadline_s']<=20 for c in calls)
    assert result.provenance['pregrasp_clearance_retry']['collision_margin_unchanged']


def test_pregrasp_search_never_accepts_collision_failure(monkeypatch):
    calls=[]
    def plan(*args,**kwargs):
        calls.append(kwargs)
        if len(calls)==1:
            raise p._PregraspGoalInitializationError('unreachable open grasp')
        raise ValueError('contact plan collision')
    monkeypatch.setattr(p,'_plan_request_geometry',plan)
    with pytest.raises(ValueError,match='contact plan collision'):
        p.plan_request_geometry(object(),object())
    assert len(calls)==2


def test_pregrasp_search_has_finite_shared_budget(monkeypatch):
    calls=[];now=[0.]
    monkeypatch.setattr(p.time,'monotonic',lambda:now[0])
    def plan(*args,**kwargs):
        calls.append(kwargs);now[0]+=21
        raise p._PregraspGoalInitializationError('unreachable open grasp')
    monkeypatch.setattr(p,'_plan_request_geometry',plan)
    with pytest.raises(TimeoutError,match='planning deadline'):
        p.plan_request_geometry(object(),object(),deadline_s=20)
    assert len(calls)==1


def test_all_pregrasp_candidates_reject_without_unbounded_retry(monkeypatch):
    calls=[]
    def plan(*args,**kwargs):
        calls.append(kwargs)
        raise p._PregraspGoalInitializationError('unreachable open grasp')
    monkeypatch.setattr(p,'_plan_request_geometry',plan)
    with pytest.raises(ValueError,match='unreachable open grasp'):
        p.plan_request_geometry(object(),object())
    assert len(calls)==3
