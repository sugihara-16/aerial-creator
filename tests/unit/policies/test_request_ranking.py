import math
import torch
import pytest
from amsrr.policies.request_ranking import prefix_log_probability, action_statistics, record_ranking
from amsrr.policies.request_actor_critic import RequestActorCritic
from tests.unit.policies.test_high_level_requests import request_scene, decision


def test_stopped_prefix_probability_normalizes_and_never_credits_untried_suffix():
    logits = torch.tensor([math.log(.2), math.log(.3), math.log(.5)], dtype=torch.float64, requires_grad=True)
    # Only candidate 2 is feasible: the stopped prefixes partition all outcomes.
    prefixes = {(2,), (0,2), (1,2), (0,1,2), (1,0,2)}
    probabilities = [prefix_log_probability(logits,p).exp() for p in prefixes]
    assert float(torch.stack(probabilities).sum().detach()) == pytest.approx(1)
    assert float(prefix_log_probability(logits,[0,2]).exp().detach()) == pytest.approx(.2*.5/.8)
    prefix_log_probability(logits,[0,2]).backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError): prefix_log_probability(logits,[0,0])


def test_prefix_replay_gradient_and_conditional_kl():
    z = torch.tensor([[0.,1.,2.],[0.,1.,2.]],requires_grad=True)
    events=[dict(action=2,ranking_prefix=[0,2],encoding={'initial':True}),dict(action=1)]
    log,entropy,kl=action_statistics(z,events,z.detach())
    assert log[0].item()==pytest.approx(-2.407605964-0.313261688)
    assert log[1].item()==pytest.approx(-1.407605964)
    assert torch.all(kl==0)
    (log.sum()+.02*entropy.sum()).backward()
    assert torch.isfinite(z.grad).all()
    changed=z.detach().clone();changed[0,0]+=1
    assert action_statistics(changed,events,z.detach())[2][0]>0


def test_full_ranking_rng_is_independent_of_accepted_prefix(request_scene):
    model=RequestActorCritic().eval()
    model.ranked_contact={'version':'first_feasible_plackett_luce_v1','attempt_cost':.25}
    context=decision(request_scene)
    generator=torch.Generator().manual_seed(37)
    first=model.decide(context,generator=generator)
    state=generator.get_state()
    model.ranking_prefix_length=len(first['ranking_order'])
    generator=torch.Generator().manual_seed(37)
    last=model.decide(context,generator=generator)
    assert torch.equal(state,generator.get_state())
    assert first['ranking_order']==last['ranking_order']
    assert last['ranking_prefix'][-1]==last['action']
    restored=RequestActorCritic.from_checkpoint(model.checkpoint())
    assert restored.ranked_contact==model.ranked_contact
    for key,value in model.state_dict().items(): assert torch.equal(value,restored.state_dict()[key])
    event=dict(reward=1.,reward_events=[(2.,1.)],time_s=0.)
    record_ranking(event,last)
    assert event['reward']==1-.25*len(last['ranking_prefix'])
    assert sum(v for _,v in event['reward_events'])==event['reward']


def test_collector_splits_distinct_failed_prefixes_without_losing_draws(tmp_path, monkeypatch, request_scene):
    import json
    from types import SimpleNamespace
    from pathlib import Path
    import amsrr.training.request_collection as collector
    import amsrr.training.request_object_conditions as conditions
    from scripts.run_request_policy import save_planning_rejection
    from amsrr.utils.hashing import hash_file
    context=decision(request_scene)
    actor=RequestActorCritic().eval()
    actor.ranked_contact={'version':'first_feasible_plackett_luce_v1','attempt_cost':.25}
    checkpoint=tmp_path/'update_8/checkpoint.pt';checkpoint.parent.mkdir()
    torch.save(actor.checkpoint(),checkpoint)
    record=tmp_path/'record.json';record.write_text('{"dataset_split":"train"}')
    case=dict(episode_id='case',split='train',record=str(record),record_sha256=hash_file(record))
    monkeypatch.setattr(collector,'build_physical_model_from_config',lambda _:None)
    monkeypatch.setattr(conditions,'expand_episode_grasp_contacts',lambda *a,**k:(None,context))
    def write(path,data):
        Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(data))
    def command(argv,*args):
        assert 'prepare' in argv
        out=Path(argv[argv.index('--output')+1]);out.mkdir()
        seed=int(argv[argv.index('--seed')+1])
        actor.ranking_prefix_length=int(actor.encode(context)['mask'].sum())
        result=actor.decide(context,generator=torch.Generator().manual_seed(seed))
        write(out/'decision.json',dict(request=result['request'].to_dict(),action=result['action'],
              ranking_order=result['ranking_order'],ranking_prefix=result['ranking_prefix'],log_prob=result['log_prob']))
        save_planning_rejection(out,result,checkpoint,True,'all candidates rejected')
    protocol=dict(training_cases=[case],training_draws_per_case=16,training_seed_base=123,grasp_anchor_source='defined')
    runtime=SimpleNamespace(root=tmp_path/'collection',repo=tmp_path,python='unused',protocol=protocol,
       load_episode=lambda *a:None,geometry_index=lambda:{},sha=hash_file,write=write,command=command,
       verify=lambda:None,emit=lambda _:None,stage_physics=True,
       run_physics_batch=lambda *a:pytest.fail('no accepted plans'))
    report,archives=collector.execute_panel(runtime,'train',[case],[checkpoint],update_index=8)
    assert report['update_8']['count']==16
    events=[e[0] for p in archives for e in torch.load(p,weights_only=True)['episodes']]
    assert len(events)==16
    assert all(e['reward']==-5-.25*len(e['ranking_prefix']) for e in events)
    assert all(e['action']==e['ranking_prefix'][-1] for e in events)

    # Complete receipts retain prefix partitions on a same-source resume.
    runtime.command=lambda *a:pytest.fail('completed groups must not execute again')
    repeated, replay_paths=collector.execute_panel(runtime,'train',[case],[checkpoint],update_index=8)
    assert repeated==report and replay_paths==archives
