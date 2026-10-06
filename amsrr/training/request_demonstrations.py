"""Import successful physical request demonstrations into the shared BC format.

Only archived pre-decision encodings are inputs. Contact choices and later
requests are labels. Episode splits come from the hash-bound donor record;
neighboring observations from one condition cannot cross dataset splits.
"""
from pathlib import Path
import json

import torch

from amsrr.training.request_ppo import collate_transitions, dataset_inputs, request_runtime_contracts
from amsrr.training.request_imitation import write_json
from amsrr.utils.hashing import hash_file, stable_hash


def prepare_demonstration_dataset(manifest_path, output):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('version') != 'request_demonstrations_v1':
        raise ValueError('unsupported demonstration manifest')
    events, rows, sources, receipts = [], [], [], []
    seen_jobs, condition_splits = set(), {}
    for binding in manifest['demonstrations']:
        job_path = (manifest_path.parent / binding['job_path']).resolve()
        if job_path in seen_jobs:
            raise ValueError('duplicate demonstration job')
        seen_jobs.add(job_path)
        if hash_file(job_path) != binding['job_sha256']:
            raise ValueError('demonstration job changed')
        job = json.loads(job_path.read_text())
        root = job_path.parent
        split = binding['split']
        if split not in ('train', 'validation', 'held_out') or split != job['source_dataset_split']:
            raise ValueError('demonstration split differs from source episode')
        record = Path(job['source_record_path'])
        if hash_file(record) != job['source_record_sha256']:
            raise ValueError('demonstration source episode changed')
        source_record = json.loads(record.read_text())
        if source_record['dataset_split'] != split:
            raise ValueError('demonstration source split changed')
        for name in ('plan','scene'):
            if hash_file(root/f'{name}.json') != job[f'{name}_sha256']:
                raise ValueError('demonstration plan/scene changed')
        scene = json.loads((root/'scene.json').read_text())
        condition = stable_hash({k:scene[k] for k in ('initial_task','morphology','initial_observation')})
        if condition in condition_splits and condition_splits[condition] != split:
            raise ValueError('same initial condition crosses demonstration splits')
        condition_splits[condition] = split
        result = json.loads((root/'result.json').read_text())
        if (not result['passed'] or not result['physical_acceptance_eligible']
                or result.get('teacher_phase_supervision') or result.get('benchmark')
                or result.get('diagnostic_phase') is not None):
            raise ValueError('demonstration requires complete physical task success')
        path = root/'request_rollout.pt'
        rollout = torch.load(path,map_location='cpu',weights_only=True)
        if (result.get('source_hashes') != job['source_hashes']
                or rollout['checkpoint_sha256'] != job['checkpoint_sha256']
                or result['environment_seeds'] != rollout['environment_seeds']
                or result['environment_checkpoint_sha256'] != rollout['environment_checkpoint_sha256']
                or any(h != job['checkpoint_sha256'] for h in rollout['environment_checkpoint_sha256'])):
            raise ValueError('demonstration execution identity differs from job')
        if (not rollout['complete'] or rollout.get('diagnostic_only') or rollout.get('evaluation_only')
                or rollout.get('teacher_phase_supervision')
                or rollout.get('runtime_contracts') != request_runtime_contracts()
                or len(rollout['episodes']) != len(result['episodes'])):
            raise ValueError('invalid demonstration execution contract')
        if hash_file(root/'prepared_initial_encoding.pt') != job['prepared_initial_encoding_sha256']:
            raise ValueError('demonstration initial encoding changed')
        prepared = torch.load(root/'prepared_initial_encoding.pt',map_location='cpu',weights_only=True)
        from amsrr.simulation.request_event_execution import validate_initial_encoding
        for environment, trajectory in enumerate(rollout['episodes']):
            outcome = result['episodes'][environment]
            if (not outcome['task_success'] or not outcome['no_fallback_success'] or outcome['safety_failure']
                    or not outcome['isaac_backed'] or not outcome['full_mesh_evaluation']):
                raise ValueError('unsafe or unsuccessful demonstration environment')
            if not trajectory or not trajectory[-1]['done']:
                raise ValueError('incomplete demonstration trajectory')
            validate_initial_encoding(prepared['encoding'],trajectory[0]['encoding'])
            if prepared['requests'] != trajectory[0]['requests']:
                raise ValueError('demonstration initial catalog differs')
            for index, event in enumerate(trajectory):
                initial = bool(event['encoding']['initial'])
                action = int(event['action'])
                if initial != (index == 0) or not event['encoding']['mask'][action]:
                    raise ValueError('noncausal or masked demonstration action')
                request = event['requests'][action]
                if request != event['request']:
                    raise ValueError('demonstration label/catalog differ')
                category = ('initial_group' if initial else 'forced' if event['encoding']['mask'].sum() == 1
                            else 'transition' if request['transition_id'] is not None else 'continuation')
                rows.append(dict(split=split, initial=initial, raw_index=index, label=action,
                    category=category, requests=event['requests'],
                    committed_group_id=None if initial else job['group_id'],
                    episode_id=source_record['episode_id'], environment=environment,
                    condition_hash=condition, job_path=str(job_path)))
                events.append(event)
        binding_source=dict(path=str(record),sha256=job['source_record_sha256'])
        existing=next((s for s in sources if s['episode_id']==source_record['episode_id']),None)
        if existing is not None and (existing['binding'] != binding_source or existing['split'] != split):
            raise ValueError('inconsistent demonstration donor identity')
        if existing is None:
            sources.append(dict(episode_id=source_record['episode_id'],binding=binding_source,
                split=split,original_split=split))
        receipts.append(dict(job_path=str(job_path), job_sha256=hash_file(job_path),
            result_sha256=hash_file(root/'result.json'), rollout_sha256=hash_file(path),
            source_record_sha256=job['source_record_sha256'], split=split,
            contact_group_id=job['group_id'], condition_hash=condition))
    if not events:
        raise ValueError('no successful demonstrations')
    for split in ('train','validation'):
        if not any(r['split']==split and r['initial'] for r in rows):
            raise ValueError('demonstrations need independent train and validation episodes')
    if not any(r['split']=='train' and not r['initial'] for r in rows):
        raise ValueError('demonstrations need causal execution decisions')
    data = collate_transitions(events)
    data['candidate_features'] = data.pop('candidates')
    data.pop('initial')
    data.update(rows=rows,sources=sources,demonstrations=receipts,version='request_demonstration_imitation_v1')
    dataset_inputs(data)  # Same masks and branch semantics used by BC and PPO.
    counts={}
    for index,row in enumerate(rows):
        if row['initial']:
            n=int(data['membership'][index,row['label']].sum())
            counts[str(n)]=counts.get(str(n),0)+1
    output.mkdir(parents=True,exist_ok=False)
    torch.save(data,output/'dataset.pt')
    report=dict(version=data['version'],dataset_path=str(output/'dataset.pt'),
        dataset_sha256=hash_file(output/'dataset.pt'),manifest_sha256=hash_file(manifest_path),
        examples=len(rows),episodes=sum(r['initial'] for r in rows),
        initial_contact_count_distribution=counts,initial_inputs_before_contact_choice=True,
        runtime_contracts=request_runtime_contracts(),sources=sources,demonstrations=receipts)
    write_json(output/'manifest.json',report)
    return report
