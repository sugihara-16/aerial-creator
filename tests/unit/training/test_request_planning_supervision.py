from copy import deepcopy
import pytest
import torch

from amsrr.training.request_planning_supervision import (
    structure_split, initial_action, observed_loss, classification_metrics, choose_planning_checkpoint, planning_ranking_loss,
)
from tests.unit.policies.test_high_level_requests import request_scene, decision
from amsrr.policies.request_actor_critic import RequestActorCritic, object_condition_features
from amsrr.training.request_ppo import categorical_kl_from_logits
from amsrr.training.request_ppo import serialize_encoding, collate_transitions, evaluate_batch
from amsrr.policies.high_level_requests import RequestCatalogBuilder


def test_structure_and_donor_boundaries_are_disjoint_and_reproducible():
    cases=[dict(structural_hash=f'{n}-{i}',module_count=n,original_episode_id=f'donor{n}-{i}',split='train')
           for n in (2,3) for i in range(5)]
    result=structure_split(cases,17,1)
    assert result==structure_split(list(reversed(cases)),17,1)
    assert set(result.values())=={'train','dev','holdout'}
    for n in (2,3):assert any(result[c['structural_hash']]=='train' for c in cases if c['module_count']==n)
    crossed=deepcopy(cases)
    a=next(c for c in crossed if result[c['structural_hash']]=='train')
    b=next(c for c in crossed if result[c['structural_hash']]=='holdout')
    b['original_episode_id']=a['original_episode_id']
    with pytest.raises(ValueError,match='donor crosses'):structure_split(crossed,17,1)
    cases[0]['split']='validation'
    with pytest.raises(ValueError,match='training conditions'):structure_split(cases,17,1)


def test_unobserved_actions_are_not_negative_labels_and_receive_no_gradient():
    scores=torch.tensor([[1.,9.,-torch.inf],[-1.,2.,-torch.inf]],requires_grad=True)
    labels=torch.tensor([[1.,-1.,-1.],[0.,1.,-1.]])
    loss=observed_loss(scores,labels)
    expected=(torch.nn.functional.softplus(torch.tensor(-1.))+
              (torch.nn.functional.softplus(torch.tensor(-1.))+torch.nn.functional.softplus(torch.tensor(-2.)))/2)/2
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert torch.isfinite(scores.grad).all()
    assert scores.grad[0,1]==0 and (scores.grad[:,2]==0).all()


def test_logged_candidate_accuracy_does_not_count_unobserved_choice_as_success():
    logits=torch.tensor([[1.,8.,-torch.inf],[-1.,2.,-torch.inf]])
    labels=torch.tensor([[1.,-1.,-1.],[0.,1.,-1.]])
    m=classification_metrics(logits,labels)
    assert m['observed_only_top1_successes']==2
    assert m['full_catalog_known_successes']==1
    assert m['full_catalog_unknown']==1


def test_labels_are_resolved_from_initial_catalog_not_result_pose():
    prepared=dict(encoding=dict(initial=torch.tensor(True),candidates=torch.zeros(2,3),mask=torch.tensor([True,True])),
                  requests=[dict(contact_group_id='A',transition_id=None),dict(contact_group_id='B',transition_id=None)])
    assert initial_action(prepared,'B')==1
    assert initial_action(prepared,'A')==0
    prepared['encoding']['initial']=torch.tensor(False)
    with pytest.raises(ValueError,match='pre-action'):initial_action(prepared,'A')


def test_initial_group_must_be_unique_and_active():
    prepared=dict(encoding=dict(initial=torch.tensor(True),candidates=torch.zeros(1,3),mask=torch.tensor([False])),
                  requests=[dict(contact_group_id='A',transition_id=None)])
    with pytest.raises(ValueError,match='one valid'):initial_action(prepared,'A')


def test_selection_requires_baseline_and_actual_full_catalog_outcomes():
    baseline=dict(epoch=0,accepted=4,conditions=6,evaluated=6,unknown=0)
    equal=dict(baseline,epoch=1)
    worse=dict(baseline,epoch=2,accepted=3)
    assert choose_planning_checkpoint([equal,worse,baseline])==baseline
    with pytest.raises(ValueError,match='baseline'):choose_planning_checkpoint([equal])
    with pytest.raises(ValueError,match='complete'):
        choose_planning_checkpoint([baseline,dict(equal,evaluated=5,unknown=1)])
    improved=dict(equal,accepted=5)
    assert choose_planning_checkpoint([baseline,improved])==improved


def test_auxiliary_gradient_reaches_shared_features_without_classifying_actor_logits(request_scene):
    torch.manual_seed(5)
    actor=RequestActorCritic(morphology_aware=True).eval()
    auxiliary=torch.nn.Linear(actor.ranker.contact_head[0].in_features,1)
    captured=[]
    hook=actor.ranker.contact_head.register_forward_pre_hook(lambda _,args:captured.append(args[0]))
    encoded=actor.encode(decision(request_scene))
    logits,_=actor.evaluate_encoding(encoded)
    labels=torch.full_like(logits,-1.)
    active=torch.nonzero(encoded['mask']).flatten()
    labels[0,active]=(torch.arange(len(active))%2).float()
    observed_loss(auxiliary(captured.pop()).squeeze(-1),labels).backward()
    assert all(p.grad is None for p in actor.ranker.contact_head.parameters())
    assert all(p.grad is None for p in actor.ranker.request_head.parameters())
    assert sum(float(p.grad.abs().sum()) for p in actor.ranker.contact_member.parameters() if p.grad is not None)>0
    assert sum(float(p.grad.abs().sum()) for p in actor.ranker.morphology_encoder.parameters() if p.grad is not None)>0
    assert auxiliary.weight.grad.abs().sum()>0
    hook.remove()


def test_policy_preservation_uses_unknown_valid_candidates_too():
    old=torch.tensor([[0.,-2.,-torch.inf]])
    new=torch.tensor([[0.,3.,-torch.inf]],requires_grad=True)
    loss=categorical_kl_from_logits(old,new).mean()
    loss.backward()
    assert loss>1 and new.grad[0,1]>0 and new.grad[0,2]==0


def test_ranking_is_shift_invariant_and_moves_accepted_above_rejected_only():
    scores=torch.tensor([[0.,2.,9.,-torch.inf],[1.,-1.,0.,-torch.inf]],requires_grad=True)
    labels=torch.tensor([[1.,0.,-1.,-1.],[1.,0.,0.,-1.]])
    loss=planning_ranking_loss(scores,labels)
    torch.testing.assert_close(loss,planning_ranking_loss(scores+37.,labels))
    loss.backward()
    assert scores.grad[0,0]<0 and scores.grad[0,1]>0
    assert scores.grad[0,2]==0 and (scores.grad[:,3]==0).all()
    assert torch.isfinite(scores.grad).all()
    positive_only=torch.tensor([[1.,2.]],requires_grad=True)
    zero=planning_ranking_loss(positive_only,torch.tensor([[1.,-1.]]))
    zero.backward()
    assert zero==0 and (positive_only.grad==0).all()


def _object_context(request_scene, mass=1., com=(0.,0.,0.)):
    context=decision(request_scene)
    task=deepcopy(context.task_spec)
    task.metadata.update(estimated_mass_kg=mass,estimated_com_object=list(com))
    return RequestCatalogBuilder().build(context.scene,task_spec=task,
        physical_model=context.physical_model,execution_state=context.execution_state)


def test_object_estimate_migration_is_exact_and_survives_ppo_encoding(request_scene):
    torch.manual_seed(5)
    actor=RequestActorCritic(morphology_aware=True).eval()
    context=_object_context(request_scene)
    old=actor.evaluate_encoding(actor.encode(context))
    actor.enable_object_conditions()
    encoded=actor.encode(context)
    current=actor.evaluate_encoding(encoded)
    torch.testing.assert_close(old,current,atol=0,rtol=0)
    assert encoded['object_features'].tolist()==[1.,0.,0.,0.]
    data=collate_transitions([dict(encoding=serialize_encoding(encoded))])
    replay=evaluate_batch(RequestActorCritic.from_checkpoint(actor.checkpoint()).eval(),data,torch.tensor([0]),'cpu')
    torch.testing.assert_close(current,replay,atol=0,rtol=0)
    missing=dict(encoded);missing.pop('object_features')
    with pytest.raises(ValueError,match='requires valid estimated'):actor.evaluate_encoding(missing)


def test_mass_and_com_are_available_to_initial_contact_head(request_scene):
    torch.manual_seed(5)
    actor=RequestActorCritic(morphology_aware=True,object_condition_aware=True).eval()
    base=actor.encode(_object_context(request_scene))
    mass=actor.encode(_object_context(request_scene,mass=1.5))
    com=actor.encode(_object_context(request_scene,com=(.02,0.,0.)))
    assert mass['object_features'][0]==1.5 and com['object_features'][1]==.2
    # A nonzero learned readout can depend on each estimate without changing geometry.
    with torch.no_grad():actor.ranker.object_condition_head[-1].weight.fill_(.01)
    a=actor.evaluate_encoding(base)[0];b=actor.evaluate_encoding(mass)[0];c=actor.evaluate_encoding(com)[0]
    active=torch.isfinite(a)
    assert not torch.equal(a[active],b[active]) and not torch.equal(a[active],c[active])
    raw=actor.encode(decision(request_scene)) if 'estimated_mass_kg' in decision(request_scene).task_spec.metadata else None
    if raw is None:
        with pytest.raises(ValueError,match='requires declared'):actor.encode(decision(request_scene))


def test_object_features_never_fall_back_to_simulated_true_properties(request_scene):
    context=_object_context(request_scene,mass=.8,com=(.01,0.,0.))
    before=object_condition_features(context.task_spec)
    task=deepcopy(context.task_spec)
    task.scene.objects[0].mass_kg=10.
    task.scene.objects[0].center_of_mass_object=[.1,0.,0.]
    torch.testing.assert_close(before,object_condition_features(task),atol=0,rtol=0)
    del task.metadata['estimated_mass_kg']
    with pytest.raises(ValueError,match='requires declared'):object_condition_features(task)
