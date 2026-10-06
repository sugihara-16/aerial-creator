"""Training-only planner prediction with full-support policy preservation.

This is an offline auxiliary-loss probe, not PPO and not a feasibility oracle.
Only archived pre-action encodings are inputs. Unobserved actions are unlabeled.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from amsrr.policies.request_actor_critic import RequestActorCritic, object_condition_features, OBJECT_FEATURE_VERSION
from amsrr.training.request_ppo import collate_transitions, evaluate_batch, categorical_kl_from_logits
from amsrr.utils.hashing import hash_file


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def structure_split(cases, seed, heldout_per_split=3):
    """Keep a training structure per module count; never split a donor."""
    groups = {}
    for c in cases:
        if c['split'] != 'train':
            raise ValueError('auxiliary labels must come from training conditions')
        groups.setdefault(c['structural_hash'], c['module_count'])
    key = lambda s: hashlib.sha256(f'{seed}:{s}'.encode()).hexdigest()
    reserved = {min((s for s,n in groups.items() if n == count), key=key)
                for count in set(groups.values())}
    available = sorted(set(groups)-reserved, key=key)
    if len(available) < 2 * heldout_per_split:
        raise ValueError('not enough disjoint structures for dev and holdout')
    splits = {s: 'train' for s in groups}
    for name, subset in [('dev', available[:heldout_per_split]),
                         ('holdout', available[heldout_per_split:2*heldout_per_split])]:
        splits.update({s: name for s in subset})
    donors = {}
    for c in cases:
        name = splits[c['structural_hash']]
        donor = c['original_episode_id']
        if donor in donors and donors[donor] != name:
            raise ValueError('donor crosses auxiliary split')
        donors[donor] = name
    return splits


def initial_action(prepared, group):
    enc = prepared['encoding']
    if not bool(enc['initial']) or not torch.isfinite(enc['candidates']).all():
        raise ValueError('planning supervision requires finite pre-action inputs')
    choices = [i for i,r in enumerate(prepared['requests'])
               if r['contact_group_id'] == group and r['transition_id'] is None
               and bool(enc['mask'][i])]
    if len(choices) != 1:
        raise ValueError('selected group must map to one valid initial action')
    return choices[0]


def observed_loss(logits, labels):
    """Equal weight per condition; absent labels cannot create negatives."""
    known = labels >= 0
    if not known.any(1).all():
        raise ValueError('each condition needs at least one observed planning label')
    # Replace inactive -inf scores before BCE; multiplying NaN by zero is unsafe.
    terms = F.binary_cross_entropy_with_logits(
        logits.masked_fill(~known, 0.), labels.clamp_min(0), reduction='none')
    return ((terms * known).sum(1) / known.sum(1)).mean()


def planning_ranking_loss(logits, labels):
    """Prefer observed accepted plans to rejected plans, with no absolute logit target.

    Unknown actions are not negative examples. Each mixed-label condition has
    equal weight, independent of its sampled action count; KL covers all actions.
    """
    positive=labels==1;negative=labels==0
    pairs=positive[:,:,None] & negative[:,None,:]
    usable=pairs.any((1,2))
    safe=logits.masked_fill(labels<0,0.)
    losses=F.softplus(safe[:,None,:]-safe[:,:,None])
    means=(losses*pairs).sum((1,2))/pairs.sum((1,2)).clamp_min(1)
    return means[usable].mean() if usable.any() else safe.sum()*0.


def prepare_labels(dataset_path, curve_path, output, seed=20260930):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    dataset = json.loads(Path(dataset_path).read_text())
    cases = {c['bucket_id']:c for c in dataset['training_cases']}
    splits = structure_split(list(cases.values()), seed)
    curves = json.loads(Path(curve_path).read_text())
    samples, receipts, duplicate_draws = {}, [], 0
    for curve in curves:
        checkpoint = Path(curve['updated_checkpoint'])
        stage = checkpoint.parent.parent/'collections'/f"train_{curve['update']}"
        binding = json.loads((stage/'collection_binding.json').read_text())
        if binding['benchmark'] or binding['diagnostic_subset'] or binding['physical_outcomes_reused']:
            raise ValueError('diagnostic or reused outcomes are not auxiliary training data')
        summary_path = stage/'panel/summary.json'
        rows = json.loads(summary_path.read_text())[Path(curve['checkpoint']).parent.name]['rows']
        if len(rows) != 448:
            raise ValueError('incomplete training panel')
        receipts.append(dict(path=str(summary_path),sha256=hash_file(summary_path)))
        seen = set()
        for row in rows:
            name = row['case']
            if name not in cases:
                raise ValueError('non-training condition in planning labels')
            directory = Path(row['directory'])
            identity = (name,str(directory))
            if identity in seen:
                duplicate_draws += 1
                continue
            seen.add(identity)
            path = directory/'prepared_initial_encoding.pt'
            prepared = torch.load(path, weights_only=True, map_location='cpu')
            decision = json.loads((directory/'decision.json').read_text())
            if not decision['actor_input_initial_only'] or decision['request']['transition_id'] is not None:
                raise ValueError('post-choice input in planning labels')
            action = initial_action(prepared, row['selected_group'])
            rejected = (directory/'planning_rejection.json').exists()
            if rejected != bool(row.get('planning_rejected')):
                raise ValueError('planning result disagrees with collection')
            if not rejected:
                job = json.loads((directory/'job.json').read_text())
                if job['prepared_initial_encoding_sha256'] != hash_file(path):
                    raise ValueError('modified pre-action encoding')
                if job['plan_sha256'] != hash_file(directory/'plan.json'):
                    raise ValueError('modified accepted plan')
            label = float(not rejected)  # Physical failure does not mean IK failure.
            if name not in samples:
                samples[name] = dict(encoding=prepared['encoding'], requests=prepared['requests'],
                    case=cases[name], split=splits[cases[name]['structural_hash']],
                    labels={}, sources=[])
            sample = samples[name]
            if sample['requests'] != prepared['requests']:
                raise ValueError('action catalog changed within one condition')
            torch.testing.assert_close(sample['encoding'], prepared['encoding'], rtol=0, atol=2e-5)
            if action in sample['labels'] and sample['labels'][action] != label:
                raise ValueError('contradictory planner outcomes for identical condition/action')
            sample['labels'][action] = label
            sample['sources'].append(dict(directory=str(directory),encoding_sha256=hash_file(path),
                                         action=action,label=label))
    if set(samples) != set(cases):
        raise ValueError('planning labels do not cover every training condition')
    ordered = [samples[k] for k in sorted(samples)]
    data = collate_transitions(ordered)
    labels = torch.full(data['mask'].shape, -1.)
    for i,s in enumerate(ordered):
        for action,label in s['labels'].items(): labels[i,action] = label
    if ((labels >= 0) & ~data['mask']).any():
        raise ValueError('label on masked action')
    payload = dict(data=data, labels=labels, samples=[{k:v for k,v in s.items()
                    if k != 'encoding'} for s in ordered], structure_splits=splits)
    torch.save(payload, output/'dataset.pt')
    statistics = {}
    for name in ('train','dev','holdout'):
        ids = [i for i,s in enumerate(ordered) if s['split']==name]
        y = labels[ids]
        statistics[name] = dict(conditions=len(ids),structures=sum(v==name for v in splits.values()),
            donors=len({ordered[i]['case']['original_episode_id'] for i in ids}),
            positive_labels=int((y==1).sum()),negative_labels=int((y==0).sum()),
            conditions_with_known_success=int((y==1).any(1).sum()),
            observed_candidates_per_condition=float((y>=0).sum(1).float().mean()))
    manifest = dict(version='initial_planning_labels_v1',dataset_path=str((output/'dataset.pt').resolve()),
        dataset_sha256=hash_file(output/'dataset.pt'),source_dataset=str(Path(dataset_path).resolve()),
        source_dataset_sha256=hash_file(dataset_path),source_curve_sha256=hash_file(curve_path),
        input_contract='archived_pre_action_encoding_only',label_contract='bounded_planner_accepted_not_task_success',
        deduplicated_draws=duplicate_draws,split_seed=seed,statistics=statistics,receipts=receipts)
    write(output/'manifest.json',manifest)
    return manifest


@torch.no_grad()
def score(model, data, ids, device):
    return torch.cat([evaluate_batch(model,data,b,device)[0].cpu()
                      for b in ids.split(64)])


def classification_metrics(logits, labels):
    known = labels >= 0
    chosen = logits.masked_fill(~known,-torch.inf).argmax(1)
    selected = labels.gather(1,chosen[:,None]).squeeze(1)
    full_choice = logits.argmax(1)
    full_labels = labels.gather(1,full_choice[:,None]).squeeze(1)
    prediction = logits[known] >= 0
    y = labels[known].bool()
    return dict(bce=float(observed_loss(logits,labels)),
        positive_recall=float(prediction[y].float().mean()) if y.any() else None,
        negative_recall=float((~prediction[~y]).float().mean()) if (~y).any() else None,
        observed_only_top1_successes=int((selected==1).sum()),conditions=len(labels),
        observed_only_top1_rate=float((selected==1).float().mean()),
        full_catalog_known_successes=int((full_labels==1).sum()),
        full_catalog_known_failures=int((full_labels==0).sum()),
        full_catalog_unknown=int((full_labels<0).sum()))


def temporal_preservation_data(samples, max_donors=32, events_per_donor=8):
    """Real train-only runtime observations; never use outcomes as policy inputs."""
    groups={}
    for sample in samples:
        if sample['split']=='train':
            groups.setdefault(sample['case']['structural_hash'],[]).append(sample)
    # Round-robin shapes prevents lexicographic case IDs from selecting only small robots.
    ordered=[group[j] for j in range(max(map(len,groups.values()),default=0))
             for _,group in sorted(groups.items()) if j<len(group)]
    selected, receipts, donors = [], [], set()
    for sample in ordered:
        donor=sample['case']['original_episode_id']
        if sample['split']!='train' or donor in donors: continue
        source=next((s for s in sample['sources'] if s['label']==1
                     and (Path(s['directory'])/'training_rollout.pt').exists()),None)
        if source is None: continue
        path=Path(source['directory'])/'training_rollout.pt'
        rollout=torch.load(path,weights_only=True,map_location='cpu')
        if (not rollout['complete'] or rollout.get('teacher_phase_supervision')
                or rollout.get('teacher_contact_supervision')):
            raise ValueError('preservation observations must be real complete actor rollouts')
        events=[e for e in rollout['episodes'][0] if not bool(e['encoding']['initial'])]
        if not events: continue
        indices=torch.linspace(0,len(events)-1,min(events_per_donor,len(events))).long().unique()
        selected.extend(dict(events[i],encoding={**events[i]['encoding'],
            **({'object_features':sample['object_features']} if 'object_features' in sample else {})})
            for i in indices.tolist());donors.add(donor)
        receipts.append(dict(path=str(path),sha256=hash_file(path),donor=donor,
                             case=sample['case']['bucket_id'],indices=indices.tolist()))
        if len(donors)>=max_donors:break
    if not selected:raise ValueError('no real temporal observations for shared-encoder preservation')
    return collate_transitions(selected),receipts


def add_object_inputs(manifest_path, output):
    """Rebuild only declared estimates from the same initial task/condition.

    Existing observations, labels and splits are unchanged. Full runtime input
    equality is checked again when the resulting models enter the planner.
    """
    from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
    from amsrr.training.request_imitation import load_episode
    from amsrr.training.request_object_conditions import expand_box_episode
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(Path(manifest_path).read_text())
    if hash_file(manifest['dataset_path'])!=manifest['dataset_sha256']:raise ValueError('dataset changed')
    payload=torch.load(manifest['dataset_path'],weights_only=True,map_location='cpu')
    groups={}
    for i,s in enumerate(payload['samples']):groups.setdefault(s['case']['record'],[]).append(i)
    physical=build_physical_model_from_config(Path(__file__).resolve().parents[2]/'configs/robot/robot_model.yaml')
    features=torch.empty(len(payload['samples']),4);receipts=[];started=time.monotonic()
    for record,indices in groups.items():
        case=payload['samples'][indices[0]]['case']
        if hash_file(record)!=case['record_sha256']:raise ValueError('record changed')
        episode=load_episode(dict(path=record,sha256=case['record_sha256']),physical)
        for i in indices:
            sample=payload['samples'][i];case=sample['case']
            if hash_file(case['condition'])!=case['condition_sha256']:raise ValueError('condition changed')
            condition=json.loads(Path(case['condition']).read_text())
            _,context=expand_box_episode(episode,condition,physical)
            features[i]=object_condition_features(context.task_spec)
            sample['object_features']=features[i].clone()
        receipts.append(dict(record=record,record_sha256=case['record_sha256'],conditions=len(indices)))
        write(output/'progress.json',dict(records=len(receipts),total_records=len(groups),seconds=time.monotonic()-started))
    payload['data']['object_features']=features
    payload['object_feature_version']=OBJECT_FEATURE_VERSION
    torch.save(payload,output/'dataset.pt')
    result={**manifest,'version':'initial_planning_labels_object_estimates_v2',
        'dataset_path':str((output/'dataset.pt').resolve()),'dataset_sha256':hash_file(output/'dataset.pt'),
        'parent_manifest_sha256':hash_file(manifest_path),'object_feature_version':OBJECT_FEATURE_VERSION,
        'object_input_sources':receipts,'seconds':time.monotonic()-started}
    write(output/'manifest.json',result)
    return result


def choose_planning_checkpoint(rows):
    """All full-catalog choices must have actual outcomes; ties retain epoch zero."""
    if not rows or not any(r['epoch']==0 for r in rows):
        raise ValueError('selection requires the unchanged baseline')
    counts={r['conditions'] for r in rows}
    if len(counts)!=1 or any(r['unknown'] or r['evaluated']!=r['conditions'] for r in rows):
        raise ValueError('selection requires complete full-catalog outcomes on identical conditions')
    return max(rows,key=lambda r:(r['accepted'],-r['epoch']))


def train_planning_labels(manifest_path, checkpoint, output, *, epochs=20,
                          batch_size=64, learning_rate=3e-4, seed=20260930, device='cpu',
                          auxiliary_weight=.2, contact_kl_limit=.02, temporal_kl_limit=.01):
    """Offline ranking plus auxiliary prediction and policy preservation, not PPO.

    Save fixed snapshots; an external actual-planner comparison selects the epoch.
    Classification loss never selects a policy checkpoint.
    """
    if epochs<1 or batch_size<1 or learning_rate<=0 or auxiliary_weight<=0:
        raise ValueError('invalid auxiliary training settings')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(Path(manifest_path).read_text())
    if hash_file(manifest['dataset_path']) != manifest['dataset_sha256']:
        raise ValueError('planning label dataset identity changed')
    payload=torch.load(manifest['dataset_path'],weights_only=True,map_location='cpu')
    data,labels,samples=payload['data'],payload['labels'],payload['samples']
    if payload.get('object_feature_version')!=OBJECT_FEATURE_VERSION or 'object_features' not in data:
        raise ValueError('rebuild planning data with declared object estimates before training')
    ids=torch.tensor([i for i,s in enumerate(samples) if s['split']=='train'])
    temporal,receipts=temporal_preservation_data(samples)
    temporal_ids=torch.arange(len(temporal['initial']))
    write(output/'preservation_inputs.json',receipts)
    torch.manual_seed(seed)
    model=RequestActorCritic.load(checkpoint).to(device).eval()
    model.enable_object_conditions()
    initial=deepcopy(model.state_dict())
    train_prefixes=('ranker.',)
    for name,p in model.named_parameters(): p.requires_grad_(name.startswith(train_prefixes))
    auxiliary=torch.nn.Sequential(torch.nn.Linear(model.ranker.contact_head[0].in_features+4,64),
                                  torch.nn.SiLU(),torch.nn.Linear(64,1)).to(device)
    captured=[]
    hook=model.ranker.contact_head.register_forward_pre_hook(lambda _module,args:captured.append(args[0]))
    def forward(dataset,ix):
        captured.clear()
        logits,_=evaluate_batch(model,dataset,ix,device)
        group=captured.pop()
        conditions=dataset['object_features'][ix].to(device)[:,None].expand(-1,group.shape[1],-1)
        features=torch.cat((group,conditions),-1)
        return logits,features
    with torch.no_grad():
        reference=score(model,data,ids,device).to(device)
        temporal_reference=score(model,temporal,temporal_ids,device).to(device)
        # Train a useful auxiliary readout before allowing it to move the encoder.
        features=torch.cat([forward(data,b)[1].detach() for b in ids.split(batch_size)])
    head_optimizer=torch.optim.Adam(auxiliary.parameters(),lr=learning_rate)
    warmup=[]
    for _ in range(100):
        head_optimizer.zero_grad(set_to_none=True)
        aux_loss=observed_loss(auxiliary(features).squeeze(-1),labels[ids].to(device))
        aux_loss.backward();head_optimizer.step();warmup.append(float(aux_loss.detach()))
    del features
    parameters=[p for p in model.parameters() if p.requires_grad]+list(auxiliary.parameters())
    optimizer=torch.optim.Adam(parameters,lr=learning_rate)
    checkpoints=[]
    def save(epoch):
        directory=output/f'epoch_{epoch}';directory.mkdir()
        saved=model.checkpoint()
        saved['planning_supervision']=dict(method='planning_ranking_separate_auxiliary_and_policy_preservation_v2',
            epoch=epoch,base_checkpoint_sha256=hash_file(checkpoint),
            auxiliary_training_only=True,ppo_update=False)
        torch.save(saved,directory/'checkpoint.pt')
        torch.save(auxiliary.state_dict(),directory/'auxiliary.pt')
        checkpoints.append(dict(epoch=epoch,checkpoint=str((directory/'checkpoint.pt').resolve()),
                                sha256=hash_file(directory/'checkpoint.pt')))
    save(0)
    history=[];started=time.monotonic();stopped=None;backtracks=[];epoch=1
    snapshots={1,max(1,epochs//4),epochs}
    while epoch<=epochs:
        previous=deepcopy(model.state_dict());previous_aux=deepcopy(auxiliary.state_dict())
        previous_optimizer=deepcopy(optimizer.state_dict())
        previous_rng=torch.get_rng_state()
        previous_logits=score(model,data,ids,device).to(device)
        order=torch.randperm(len(ids))
        for local in order.split(batch_size):
            batch=ids[local]
            optimizer.zero_grad(set_to_none=True)
            logits,features=forward(data,batch)
            aux_loss=observed_loss(auxiliary(features).squeeze(-1),labels[batch].to(device))
            ranking_loss=planning_ranking_loss(logits,labels[batch].to(device))
            policy_loss=categorical_kl_from_logits(reference[local],logits).mean()
            # Uniform temporal subset, independent of planning labels or task outcomes.
            ti=temporal_ids[torch.randperm(len(temporal_ids))[:batch_size]]
            temporal_logits,_=forward(temporal,ti)
            temporal_loss=categorical_kl_from_logits(temporal_reference[ti],temporal_logits).mean()
            loss=ranking_loss+policy_loss+temporal_loss+auxiliary_weight*aux_loss
            if not torch.isfinite(loss):raise ValueError('nonfinite auxiliary loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            logits=score(model,data,ids,device).to(device)
            kl=categorical_kl_from_logits(reference,logits)
            step_kl=categorical_kl_from_logits(previous_logits,logits)
            temporal_kl=categorical_kl_from_logits(temporal_reference,score(model,temporal,temporal_ids,device).to(device))
            aux_logits=torch.cat([auxiliary(forward(data,b)[1]).squeeze(-1).cpu() for b in ids.split(batch_size)])
        row=dict(epoch=epoch,seconds=time.monotonic()-started,contact_kl=float(kl.mean()),
                 contact_step_kl=float(step_kl.mean()),
                 max_contact_kl=float(kl.max()),temporal_kl=float(temporal_kl.mean()),
                 auxiliary_bce=float(observed_loss(aux_logits,labels[ids])),
                 ranking_loss=float(planning_ranking_loss(logits,labels[ids].to(device))),
                 changed_choices=int((logits.argmax(1)!=reference.argmax(1)).sum()),
                 actor_metrics=classification_metrics(logits.cpu(),labels[ids]))
        # Bound each optimization step, not lifetime contact learning. A fixed
        # total-KL hard cap can pin all later epochs at the boundary. The soft
        # reference KL still preserves unseen actions; temporal semantics retain
        # a cumulative cap because this stage is not meant to relearn transitions.
        if row['contact_step_kl']>contact_kl_limit or row['temporal_kl']>temporal_kl_limit:
            model.load_state_dict(previous);auxiliary.load_state_dict(previous_aux)
            optimizer.load_state_dict(previous_optimizer)
            torch.set_rng_state(previous_rng)
            rejected=dict(rejected_epoch=row,restored_model_auxiliary_optimizer_and_rng=True,
                          learning_rate=optimizer.param_groups[0]['lr'])
            if sum(r['rejected_epoch']['epoch']==epoch for r in backtracks)>=6:
                stopped=rejected;break
            for group in optimizer.param_groups:group['lr']*=.5
            backtracks.append(rejected)
            continue
        history.append(row);write(output/'history.json',history)
        if epoch in snapshots:save(epoch)
        if epoch==1 or epoch%5==0:print(json.dumps(row),flush=True)
        epoch+=1
    if history and history[-1]['epoch'] not in {c['epoch'] for c in checkpoints}:save(history[-1]['epoch'])
    hook.remove()
    changes={name:float((v-initial[name]).abs().max()) for name,v in model.state_dict().items() if v.is_floating_point()}
    for name,value in changes.items():
        if not name.startswith(train_prefixes) and value!=0:raise ValueError('non-target parameters changed')
    for c in checkpoints:
        restored=RequestActorCritic.load(c['checkpoint']).to(device).eval()
        if c['epoch']==(history[-1]['epoch'] if history else 0):
            torch.testing.assert_close(score(model,data,ids,device),score(restored,data,ids,device),atol=0,rtol=0)
    report=dict(version='planning_auxiliary_probe_v2',epochs_completed=len(history),
        seconds=time.monotonic()-started,checkpoints=checkpoints,stopped=stopped,backtracks=backtracks,
        auxiliary_weight=auxiliary_weight,contact_kl_limit=contact_kl_limit,temporal_kl_limit=temporal_kl_limit,
        warmup_bce_first=warmup[0],warmup_bce_last=warmup[-1],
        parameter_max_changes=changes,reload_exact=True,task_success_verified=False,
        selection='pending actual full-catalog dev planner outcomes; baseline included, ties earliest epoch',
        ranking_weight=1.,policy_preservation_weight=1.,
        ppo_update=False)
    write(output/'result.json',report)
    return report


def evaluate_planning_labels(manifest_path, fit_result_path, output, *, workers=8,
                             split='dev', selected_epoch=None):
    """Fresh, full-catalog selection through the unchanged bounded planner.

    Reuse only the existing content-checked planning cache, never task results.
    No Isaac rollout is required to measure this particular planning outcome.
    """
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from amsrr.training.request_execution_worker import RequestExecutionWorker
    from scripts.run_request_policy import sources
    if not 1 <= workers <= 12: raise ValueError('planning workers must be 1..12')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(Path(manifest_path).read_text())
    if hash_file(manifest['dataset_path']) != manifest['dataset_sha256']:
        raise ValueError('planning dataset changed')
    payload=torch.load(manifest['dataset_path'],weights_only=True,map_location='cpu')
    if split not in ('dev','holdout') or (split=='holdout' and selected_epoch is None):
        raise ValueError('final comparison requires a dev-selected epoch')
    holdout={s['case']['bucket_id']:(i,s) for i,s in enumerate(payload['samples']) if s['split']==split}
    fit=json.loads(Path(fit_result_path).read_text())
    models=[c for c in fit['checkpoints'] if split=='dev' or c['epoch'] in {0,selected_epoch}]
    if not models or not any(c['epoch']==0 for c in models):raise ValueError('missing baseline')
    checkpoints={str(c['epoch']):c['checkpoint'] for c in models}
    predictions=[];ids=torch.tensor([i for i,s in holdout.values()])
    for c in models:
        if hash_file(c['checkpoint'])!=c['sha256']:raise ValueError('checkpoint identity changed')
        model=RequestActorCritic.load(c['checkpoint']).eval()
        logits=score(model,payload['data'],ids,'cpu')
        for row,(_,sample) in zip(logits,holdout.values()):
            action=int(row.argmax())
            predictions.append(dict(model=str(c['epoch']),case=sample['case'],action=action,
                request=sample['requests'][action],logged_label=float(payload['labels'][holdout[sample['case']['bucket_id']][0],action])))
    write(output/'predictions.json',predictions)
    repo=Path(__file__).resolve().parents[2]
    identities={str(Path(p).resolve()):hash_file(p) for p in
        [manifest_path,fit_result_path,*checkpoints.values(),__file__]}
    identities.update(sources())
    write(output/'binding.json',dict(checkpoints=checkpoints,protected_hashes=identities,
        workers=workers,planner_timeout_s=360,task_execution=False,split=split,
        candidate_support='all current valid initial groups; no logged-action restriction',
        outcome='fresh planner acceptance; not physical feasibility proof'))
    local=threading.local();owned=[];lock=threading.Lock();cancel=threading.Event()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def run(prediction):
        if cancel.is_set():raise RuntimeError('planning comparison cancelled')
        case=prediction['case'];name=case['bucket_id'];model=prediction['model']
        index,sample=holdout[name]
        if case!=sample['case']:raise ValueError('held-out condition changed')
        for key in ('record','condition'):
            if hash_file(case[key])!=case[key+'_sha256']:raise ValueError('held-out input modified')
        with lock:
            if cancel.is_set():raise RuntimeError('planning comparison cancelled')
            if not hasattr(local,'worker'):
                local.worker=RequestExecutionWorker(root=repo,log=output/f'worker_{len(owned)}.log',env=env)
                owned.append(local.worker)
        path=output/model/name;path.parent.mkdir(exist_ok=True)
        argv=['--record',case['record'],'--object-condition',case['condition'],
              '--checkpoint',checkpoints[model],'--output',str(path),'--max-grasp-contacts','2',
              '--seed','17','--greedy','--timeout-s','360']
        receipt=local.worker.prepare(argv,log=path.with_suffix('.log'),timeout=390,allowed=(0,2))
        decision=json.loads((path/'decision.json').read_text())
        if decision['request']!=prediction['request']:
            raise ValueError('fresh inference differs from archived-input prediction')
        prepared=torch.load(path/'prepared_initial_encoding.pt',weights_only=True,map_location='cpu')
        if prepared['requests']!=sample['requests']:raise ValueError('fresh action catalog differs')
        # Compare each unpadded input against the bound archived observation.
        original=torch.load(Path(sample['sources'][0]['directory'])/'prepared_initial_encoding.pt',weights_only=True,map_location='cpu')
        current=dict(prepared['encoding'])
        if 'object_features' in sample:
            torch.testing.assert_close(current.pop('object_features'),sample['object_features'],rtol=0,atol=0)
        torch.testing.assert_close(current,original['encoding'],rtol=0,atol=2e-5)
        rejected=(path/'planning_rejection.json').exists()
        if not rejected and not (path/'job.json').exists():raise ValueError('missing plan result')
        if receipt['exit_code']!=(2 if rejected else 0):raise ValueError('planner exit/result mismatch')
        return dict(model=model,case=name,structural_hash=case['structural_hash'],
            module_count=case['module_count'],request=decision['request'],accepted=not rejected,
            previously_observed_label=prediction['logged_label'],seconds=receipt['seconds'],
            reason=json.loads((path/'planning_rejection.json').read_text())['reason'] if rejected else None,
            directory=str(path))
    # Multiple epochs often choose exactly the same action. Invoke the planner once
    # per identical condition/request, after checking every checkpoint's full-support
    # argmax above. This never reuses a physical rollout or masks an unknown action.
    unique={}
    for p in predictions:
        key=(p['case']['bucket_id'],json.dumps(p['request'],sort_keys=True))
        unique.setdefault(key,[]).append(p)
    results=[];started=time.monotonic();pool=ThreadPoolExecutor(max_workers=workers)
    try:
        futures={pool.submit(run,group[0]):group for group in unique.values()}
        for future in as_completed(futures):
            actual=future.result()
            for prediction in futures[future]:
                results.append(dict(actual,model=prediction['model'],
                    shared_identical_request=prediction['model']!=actual['model']))
            write(output/'progress.json',dict(completed=len(results),total=len(predictions),
                unique_plans=len(unique),seconds=time.monotonic()-started))
            write(output/'rows.json',results)
    except BaseException:
        cancel.set()
        for worker in owned:worker.close(force=True)
        pool.shutdown(wait=True,cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
        for worker in owned:worker.close()
    for path,digest in identities.items():
        if hash_file(path)!=digest:raise ValueError('source or model changed during comparison')
    paired={name:{r['model']:r for r in results if r['case']==name} for name in holdout}
    summary={model:dict(epoch=int(model),accepted=sum(r['accepted'] for r in results if r['model']==model),
                       conditions=len(holdout),evaluated=sum(r['model']==model for r in results),unknown=0)
             for model in checkpoints}
    best=choose_planning_checkpoint(list(summary.values())) if split=='dev' else summary[str(selected_epoch)]
    chosen=str(best['epoch'])
    report=dict(seconds=time.monotonic()-started,models=summary,selected_epoch=best['epoch'],split=split,
        unique_plans=len(unique),
        gained=sum(not r['0']['accepted'] and r[chosen]['accepted'] for r in paired.values()),
        lost=sum(r['0']['accepted'] and not r[chosen]['accepted'] for r in paired.values()),
        improved=best['accepted']>summary['0']['accepted'],
        label_disagreements=[r for r in results if r['previously_observed_label']>=0
                            and r['accepted']!=bool(r['previously_observed_label'])],
        task_success_verified=False,source_and_checkpoint_unchanged=True)
    write(output/'result.json',report)
    return report
