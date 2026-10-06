"""Outcome ranking, deployable duration decisions, and explicit model migration."""
from copy import deepcopy
from dataclasses import replace
import torch
import pytest
from tests.unit.policies.test_high_level_requests import request_scene,decision
from amsrr.policies.request_actor_critic import RequestActorCritic
from amsrr.policies.request_ranking import action_statistics
from amsrr.training.request_ppo import (serialize_encoding,collate_transitions,evaluate_batch,
    configure_learning_signals,observed_contact_return_pairs,contact_return_ranking_loss)
from amsrr.utils.hashing import hash_file


def committed(parts):
    scene,task,physical,state=parts
    group=scene.contact_candidate_set.group_proposals[0]
    state=replace(state,plan_id='current',contact_group_id=group.group_id)
    return decision((scene,task,physical,state))


def test_duration_sampling_is_learnable_and_replays_the_executed_option(request_scene):
    model=RequestActorCritic().eval();model.enable_temporal_options()
    context=committed(request_scene)
    ds=model.decide_many(context,generators=[torch.Generator().manual_seed(i) for i in range(256)])
    enc=ds[0]['encoded'];p=ds[0]['logits'].softmax(-1);hold=enc['hold_indices']>0
    assert float(p[hold].sum())==pytest.approx(.2,abs=1e-6)
    assert {d['hold_duration_s'] for d in ds}=={0.,.5,1.,2.,4.,8.}
    events=[]
    for d in ds:
        assert d['request'].to_dict()==d['requests'][d['action']]
        assert (d['request'].transition_id is None)==(d['hold_duration_s']>0)
        events.append(dict(encoding=serialize_encoding(d['encoded']),action=d['action']))
    data=collate_transitions(events);scores,_=evaluate_batch(model,data,torch.arange(len(ds)),'cpu')
    lp=action_statistics(scores,events)[0]
    torch.testing.assert_close(lp,torch.tensor([d['log_prob'] for d in ds]),atol=2e-6,rtol=0)
    # One duration receives positive credit: the policy can learn it, rather
    # than a sampler secretly overriding a different network action.
    old=float(p[enc['hold_indices']==5].sum())
    model.zero_grad();scores,_=model.evaluate_encoding(enc)
    loss=-scores.log_softmax(-1)[0,torch.where(enc['hold_indices']==5)[0][0]]
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:param.add_(param.grad,alpha=-.05)
    new=model.evaluate_encoding(enc)[0].softmax(-1)[0]
    assert float(new[enc['hold_indices']==5].sum().detach())>old
    assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
    restored=RequestActorCritic.from_checkpoint(model.checkpoint())
    torch.testing.assert_close(model.evaluate_encoding(enc),restored.evaluate_encoding(enc),atol=0,rtol=0)


def test_duration_expansion_preserves_initial_choices_and_critic_pool(request_scene):
    model=RequestActorCritic().eval();initial=decision(request_scene);temporal=committed(request_scene)
    before=model.evaluate_encoding(model.encode(initial));v=model.evaluate_encoding(model.encode(temporal))[1]
    model.enable_temporal_options()
    torch.testing.assert_close(before,model.evaluate_encoding(model.encode(initial)),atol=0,rtol=0)
    torch.testing.assert_close(v,model.evaluate_encoding(model.encode(temporal))[1],atol=2e-7,rtol=0)
    e=serialize_encoding(model.encode(initial));old=deepcopy(e);old.pop('hold_indices');old.pop('request_indices')
    with pytest.raises(ValueError,match='mixed duration'):
        collate_transitions([dict(encoding=e),dict(encoding=old)])


@pytest.mark.parametrize('enable_correction', [True, False])
def test_migration_preserves_old_weights_and_optimizer_moments_by_name(tmp_path, enable_correction):
    model=RequestActorCritic(morphology_aware=True).eval()
    actor=torch.optim.Adam(model.ranker.parameters(),lr=1e-4)
    for p in model.ranker.parameters():p.grad=torch.ones_like(p)*.01
    actor.step()
    value=torch.optim.Adam(model.value_head.parameters(),lr=1e-4)
    checkpoint=tmp_path/'checkpoint.pt';torch.save(model.checkpoint(),checkpoint)
    saved=dict(checkpoint_sha256=hash_file(checkpoint),learning_rate=1e-4,
        completed_updates=10,profile='timed_value_v1',contact_baseline_kind='condition_loo_v1',
        actor_optimizer=actor.state_dict(),value_optimizer=value.state_dict())
    torch.save(saved,tmp_path/'training_state.pt')
    out=tmp_path/'migrated';configure_learning_signals(checkpoint,out,enable_return_correction=enable_correction)
    new=RequestActorCritic.load(out/'checkpoint.pt');state=torch.load(out/'training_state.pt',weights_only=True)
    for name,p in model.state_dict().items():torch.testing.assert_close(p,new.state_dict()[name],atol=0,rtol=0)
    opt=torch.optim.Adam(new.ranker.parameters(),lr=1e-4);opt.load_state_dict(state['actor_optimizer'])
    newparams=dict(new.ranker.named_parameters())
    for name,p in model.ranker.named_parameters():
        torch.testing.assert_close(actor.state[p],opt.state[newparams[name]],atol=0,rtol=0)
    assert not opt.state[new.ranker.hold_head[0].weight]
    assert state['completed_updates']==10 and new.object_condition_aware
    assert (new.contact_return_ranking['weight']==1. if enable_correction else new.contact_return_ranking is None)


def test_return_pairs_ignore_unexecuted_actions_and_remove_prefix_cost():
    events=[dict(action=a,planning_cost=c,planning_rejected=rej,episode_success=a==0) for a,c,rej in
            [(0,1.,False),(0,1.,False),(1,.25,False),(1,.25,False),(2,2.,True),(2,2.,True)]]
    target=torch.tensor([4.,4.,-1.25,-1.25,-7.,-7.])
    base=dict(event_indices=list(range(6)),conditions=['same']*6)
    pairs=observed_contact_return_pairs(events,target,base)
    assert len(pairs)==1 and pairs[0]['winner']==0 and pairs[0]['loser']==1
    assert pairs[0]['return_gap']==6.
    logits=torch.tensor([[0.,1.,4.]],requires_grad=True)
    pairs[0]['index']=0
    loss=contact_return_ranking_loss(logits,logits.detach().clone(),pairs);loss.backward()
    assert logits.grad[0,0]<0 and logits.grad[0,1]>0 and logits.grad[0,2]==0
    # Once the requested improvement is met the auxiliary term no longer
    # drives the distribution toward deterministic overconfidence.
    x=torch.tensor([[2.,0.,4.]],requires_grad=True)
    loss=contact_return_ranking_loss(x,logits.detach(),pairs);loss.backward()
    assert loss==0 and (x.grad==0).all()


def test_unknown_and_single_draw_outcomes_do_not_invent_pairs():
    events=[dict(action=0,episode_success=True),dict(action=0,episode_success=True),dict(action=1,episode_success=False)]
    pairs=observed_contact_return_pairs(events,torch.tensor([1.,1.,-1.]),dict(event_indices=[0,1,2],conditions=['x']*3))
    assert pairs==[]
    scores=torch.tensor([[1.,-torch.inf]],requires_grad=True)
    loss=contact_return_ranking_loss(scores,scores.detach(),pairs);loss.backward()
    assert torch.isfinite(scores.grad).all() and (scores.grad==0).all()


@pytest.mark.parametrize('correction', ['checkpoint', 'disabled'])
def test_ppo_uses_recorded_outcomes_for_within_condition_preferences(request_scene,tmp_path,correction):
    """A sampled bandit with a known winner exercises the complete update path."""
    import json
    from amsrr.training.request_ppo import ppo_update, request_runtime_contracts
    from amsrr.utils.hashing import stable_hash
    torch.manual_seed(712)
    model=RequestActorCritic().eval()
    model.contact_return_ranking=dict(version='condition_success_return_pairs_v1',weight=1.,min_samples=2)
    with torch.no_grad():
        model.ranker.contact_head[3].weight.zero_()
        model.ranker.contact_head[3].bias.zero_()
    encoded=model.encode(decision(request_scene))
    scores,value=model.evaluate_encoding(encoded)
    desired=int(torch.where(encoded['mask'])[0][0])
    distribution=torch.distributions.Categorical(logits=scores[0].detach())
    episodes=[]
    for action in distribution.sample((96,)):
        success=int(action)==desired
        reward=5. if success else -5.
        episodes.append([dict(encoding=serialize_encoding(encoded),action=int(action),
            value=float(value[0].detach()),log_prob=float(distribution.log_prob(action)),
            time_s=0.,end_time_s=1.,reward=reward,done=True,task_success=success,
            reward_timing='observed_reward_time_v1',reward_events=[(1.,reward)])])
    checkpoint=tmp_path/'checkpoint.pt';torch.save(model.checkpoint(),checkpoint)
    rollout=tmp_path/'rollout.pt'
    torch.save(dict(checkpoint_sha256=hash_file(checkpoint),sampling='categorical',
        teacher_phase_supervision=False,complete=True,runtime_contracts=request_runtime_contracts(),
        episodes=episodes,environment_seeds=list(range(96))),rollout)
    record=tmp_path/'record.json';record.write_text(json.dumps(dict(dataset_split='train')))
    identity=dict(record_sha256=hash_file(record),snapshot_hash='synthetic_initial')
    manifest=tmp_path/'groups.json'
    manifest.write_text(json.dumps(dict(version='same_initial_condition_rollout_groups_v1',
        group_size=96,rollouts=[dict(path=str(rollout),sha256=hash_file(rollout),
        record_path=str(record),**identity,condition_hash=stable_hash(identity))])))
    result=ppo_update(checkpoint,[rollout],tmp_path/'updated',training_profile='timed_value_v1',
        contact_group_manifest=manifest,epochs=4,value_epochs=2,learning_rate=.001,
        contact_return_correction=correction)
    comparison=result['contact_return_ranking']
    if correction == 'checkpoint':
        assert comparison['observations'] and comparison['metrics']['regressed']==0
        assert comparison['metrics']['improved']==comparison['metrics']['pairs']
    else:
        assert comparison['config'] is None and not comparison['observations']
    assert result['reload_exact']
    restored=RequestActorCritic.load(tmp_path/'updated/checkpoint.pt')
    assert (restored.contact_return_ranking is None) == (correction == 'disabled')
    assert float(restored.evaluate_encoding(encoded)[0].softmax(-1)[0,desired].detach())>float(distribution.probs[desired])


def test_preference_projection_corrects_bad_step_without_freezing_other_outputs(monkeypatch):
    from types import SimpleNamespace
    import amsrr.training.request_ppo as ppo
    model=SimpleNamespace(ranker=torch.nn.Linear(1,3,bias=False))
    with torch.no_grad():model.ranker.weight.zero_()
    previous={'ranker.weight':model.ranker.weight.detach().clone()}
    with torch.no_grad():model.ranker.weight.copy_(torch.tensor([[-.2],[.3],[.4]]))
    monkeypatch.setattr(ppo,'evaluate_batch',lambda m,d,ids,device:(m.ranker(torch.ones(len(ids),1)),None))
    pair=dict(index=0,winner=0,loser=1)
    result=ppo.preserve_observed_contact_preferences(model,{},torch.zeros(1,3),[pair],previous,'cpu')
    assert result['accepted'] and result['projected']
    x=model.ranker.weight.detach().flatten()
    assert x[0]>=x[1] and x[2]==pytest.approx(.4)
    assert (x!=0).any()


def test_mixed_and_success_only_outcomes_are_not_hard_ranking_labels():
    events=[dict(action=a,episode_success=s) for a,s in [(0,True),(0,True),(1,True),(1,True),(2,True),(2,False)]]
    result=observed_contact_return_pairs(events,torch.tensor([9.,9.,3.,3.,8.,-2.]),
        dict(event_indices=list(range(6)),conditions=['x']*6))
    assert result==[]


def test_nonlinear_projection_preserves_independent_temporal_update(monkeypatch):
    from types import SimpleNamespace
    import amsrr.training.request_ppo as ppo
    ranker=torch.nn.Module()
    ranker.contact=torch.nn.Parameter(torch.zeros(1))
    ranker.request_head=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad():ranker.request_head.weight.zero_()
    model=SimpleNamespace(ranker=ranker)
    previous={'ranker.'+k:v.clone() for k,v in ranker.state_dict().items()}
    with torch.no_grad():
        ranker.contact.fill_(-.1);ranker.request_head.weight.fill_(.4)
    def evaluate(m,d,ids,device):
        x=m.ranker.contact
        # A narrow nonlinear feasible interval requires repeated linearization.
        gap=x-10000*x.square()
        return torch.stack((gap,gap*0),-1),None
    monkeypatch.setattr(ppo,'evaluate_batch',evaluate)
    result=ppo.preserve_observed_contact_preferences(model,{},torch.zeros(1,2),
        [dict(index=0,winner=0,loser=1)],previous,'cpu')
    assert result['accepted'] and result['scale']==1
    assert len(result['nonlinear_iterations'])>1
    assert float(evaluate(model,None,None,None)[0][0,0].detach())>=-1e-6
    assert float(ranker.request_head.weight.detach())==pytest.approx(.4)


def test_temporal_heads_cannot_change_initial_contact_logits(request_scene):
    model=RequestActorCritic().eval();model.enable_temporal_options()
    encoding=model.encode(decision(request_scene))
    with torch.no_grad():
        before=model.evaluate_encoding(encoding)[0]
        for head in (model.ranker.request_head,model.ranker.hold_head):
            for p in head.parameters():p.add_(torch.randn_like(p))
        after=model.evaluate_encoding(encoding)[0]
    torch.testing.assert_close(before,after,rtol=0,atol=0)


def test_rejected_ranking_retains_zero_hold_and_failure_outcome(request_scene,tmp_path):
    from scripts.run_request_policy import save_planning_rejection
    model=RequestActorCritic().eval();model.enable_temporal_options()
    d=model.decide(decision(request_scene),generator=torch.Generator().manual_seed(2))
    checkpoint=tmp_path/'checkpoint.pt';torch.save(model.checkpoint(),checkpoint)
    output=tmp_path/'rejected';output.mkdir()
    save_planning_rejection(output,d,checkpoint,True,'rejected')
    arc=torch.load(output/'request_rollout.pt',weights_only=True)
    e=arc['episodes'][0][0]
    assert e['hold_duration_s']==0. and e['task_success'] is False
    assert e['planning_rejected'] and e['done']


def test_varied_object_training_cannot_silently_use_legacy_actor(request_scene):
    from amsrr.policies.request_actor_critic import require_training_object_inputs
    task=deepcopy(request_scene[1]);task.metadata['object_condition']={'mass_factor':1.2,'com_fraction':[0.,0.,0.]}
    legacy=RequestActorCritic(morphology_aware=True)
    with pytest.raises(ValueError,match='requires mass/COM inputs'):
        require_training_object_inputs(legacy,task)
    legacy.enable_object_conditions()
    require_training_object_inputs(legacy,task)
    legacy.enable_temporal_options()
    with pytest.raises(ValueError,match='observed motion feedback'):
        require_training_object_inputs(legacy,task)
    legacy.enable_motion_feedback()
    require_training_object_inputs(legacy,task)


def test_multiple_keep_requests_share_one_total_initial_wait_probability(request_scene):
    model=RequestActorCritic().eval();model.enable_temporal_options()
    e=model.encode(committed(request_scene));holds=e['hold_indices']>0
    indices=torch.cat([torch.arange(len(holds)),torch.where(holds)[0]])
    # Two possible keep subgoals must not double the exploration probability.
    for key in ['features','mask','membership','hold_indices','request_indices']:
        e[key]=e[key][indices]
    with torch.no_grad():p=model.evaluate_encoding(e)[0].softmax(-1)[0]
    assert float(p[e['hold_indices']>0].sum())==pytest.approx(.2,abs=1e-6)


@pytest.mark.parametrize('include_object_conditions', [False, True])
def test_motion_migration_preserves_outputs_and_pads_optimizer_state(request_scene,tmp_path,include_object_conditions):
    from amsrr.training.request_ppo import configure_motion_feedback
    model=RequestActorCritic(morphology_aware=include_object_conditions,
        object_condition_aware=include_object_conditions).eval();model.enable_temporal_options()
    if include_object_conditions:
        model.enable_motion_feedback()
        request_scene[1].metadata.update(estimated_mass_kg=.2,estimated_com_object=[.01,0.,0.])
    optimizers=[torch.optim.Adam(model.ranker.parameters(),lr=1e-4),
                torch.optim.Adam(model.value_head.parameters(),lr=1e-4)]
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for p in group['params']:p.grad=torch.ones_like(p)*.01
        optimizer.step()
    checkpoint=tmp_path/'checkpoint.pt';torch.save(model.checkpoint(),checkpoint)
    torch.save(dict(checkpoint_sha256=hash_file(checkpoint),learning_rate=1e-4,
        completed_updates=10,actor_optimizer=optimizers[0].state_dict(),
        value_optimizer=optimizers[1].state_dict()),tmp_path/'training_state.pt')
    out=tmp_path/'motion';configure_motion_feedback(checkpoint,out,
        include_object_conditions=include_object_conditions)
    new=RequestActorCritic.load(out/'checkpoint.pt')
    for context in [decision(request_scene),committed(request_scene)]:
        torch.testing.assert_close(model.evaluate_encoding(model.encode(context)),
            new.evaluate_encoding(new.encode(context)),rtol=1e-6,atol=2e-6)
    state=torch.load(out/'training_state.pt',weights_only=True)
    assert state['completed_updates']==10
    for key,module,original in [('actor_optimizer',new.ranker,optimizers[0]),
                                ('value_optimizer',new.value_head,optimizers[1])]:
        optimizer=torch.optim.Adam(module.parameters(),lr=1e-4);optimizer.load_state_dict(state[key])
        for oldp,newp in zip(original.param_groups[0]['params'],optimizer.param_groups[0]['params']):
            for name,oldvalue in original.state[oldp].items():
                newvalue=optimizer.state[newp][name]
                if oldvalue.ndim==2 and oldvalue.shape!=newvalue.shape:
                    torch.testing.assert_close(oldvalue,newvalue[:,:oldvalue.shape[1]],rtol=0,atol=0)
                    assert not newvalue[:,oldvalue.shape[1]:].count_nonzero()
                else:torch.testing.assert_close(oldvalue,newvalue,rtol=0,atol=0)
            newp.grad=torch.ones_like(newp)*.01
        optimizer.step()  # A padded moment must work on the next real update.


def test_temporal_mass_com_changes_waiting_and_value_without_truth(request_scene):
    request_scene[1].metadata.update(estimated_mass_kg=.2,estimated_com_object=[.01,0.,0.])
    model=RequestActorCritic(morphology_aware=True,object_condition_aware=True).eval()
    model.enable_temporal_options();model.enable_motion_feedback()
    encoded=model.encode(committed(request_scene));changed=deepcopy(encoded)
    changed['object_features']=encoded['object_features']+torch.tensor([.1,.1,0.,0.])
    with torch.no_grad():
        original=model.evaluate_encoding(encoded)
        torch.testing.assert_close(original,model.evaluate_encoding(changed),rtol=0,atol=0)
    model.enable_temporal_object_conditions()
    torch.testing.assert_close(original,model.evaluate_encoding(encoded),rtol=1e-6,atol=2e-6)
    # Exercise actual gradients from both returns and waiting decisions.
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    for _ in range(2):
        optimizer.zero_grad();scores,value=model.evaluate_encoding(encoded)
        wait=torch.where(encoded['hold_indices']==5)[0][0]
        loss=-scores.log_softmax(-1)[0,wait]+(value-2.).square().mean()
        loss.backward();optimizer.step()
    with torch.no_grad():
        first=model.evaluate_encoding(encoded);second=model.evaluate_encoding(changed)
        assert not torch.allclose(first[0].softmax(-1),second[0].softmax(-1))
        assert not torch.allclose(first[1],second[1])
        restored=RequestActorCritic.from_checkpoint(model.checkpoint())
        torch.testing.assert_close(first,restored.evaluate_encoding(encoded),rtol=0,atol=0)
        for p in model.ranker.parameters():p.requires_grad_(False)
        vf=model.frozen_value_features(encoded['features'][None],encoded['graph'],
            encoded['mask'][None],encoded['hold_indices'][None],encoded['motion_features'][None],
            encoded['object_features'][None])
        torch.testing.assert_close(first[1],model.value_head(vf).squeeze(-1),rtol=0,atol=0)


def test_motion_feedback_is_causal_typed_and_replays(request_scene):
    from amsrr.schemas.high_level import ContactMotionEstimate
    from amsrr.schemas.common import SchemaValidationError
    from amsrr.policies.high_level_requests import RequestCatalogBuilder
    base=committed(request_scene)
    ids=sorted({cid for e in base.catalog.entries if e.request.contact_group_id==base.execution_state.contact_group_id for cid in e.candidate_ids})
    motion=[ContactMotionEstimate(cid,base.observation.time_s,.003,.02,.17,True) for cid in ids]
    context=RequestCatalogBuilder().build(base.scene,task_spec=base.task_spec,physical_model=base.physical_model,
        execution_state=base.execution_state,contact_motion=motion)
    model=RequestActorCritic().eval();model.enable_temporal_options();model.enable_motion_feedback()
    e=model.encode(context);f=e['motion_features']
    assert f.tolist()==pytest.approx([.3,.3,.4,1.7,1.7,1.,1.,0.,0.])
    assert model.encode(base)['motion_features'][5]==0
    with torch.no_grad():
        model.ranker.hold_head[0].weight[:,-9:].fill_(.2)
        model.ranker.hold_head[-1].weight[0].fill_(.1)
    without=deepcopy(e);without['motion_features'].zero_()
    assert not torch.equal(model.evaluate_encoding(without)[0],model.evaluate_encoding(e)[0])
    d=model.decide(context,generator=torch.Generator().manual_seed(9))
    event=dict(encoding=serialize_encoding(d['encoded']),action=d['action'])
    data=collate_transitions([event]);scores,_=evaluate_batch(model,data,torch.tensor([0]),'cpu')
    assert float(action_statistics(scores,[event])[0].detach())==pytest.approx(d['log_prob'],abs=2e-6)
    future=deepcopy(motion);future[0].time_s+=1.
    with pytest.raises(SchemaValidationError,match='future'):
        RequestCatalogBuilder().build(base.scene,task_spec=base.task_spec,physical_model=base.physical_model,
            execution_state=base.execution_state,contact_motion=future)
    invalid=deepcopy(motion);invalid[0].motor_load_nm=float('nan')
    with pytest.raises(SchemaValidationError):
        RequestCatalogBuilder().build(base.scene,task_spec=base.task_spec,physical_model=base.physical_model,
            execution_state=base.execution_state,contact_motion=invalid)
