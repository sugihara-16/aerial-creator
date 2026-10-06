#!/usr/bin/env python3
"""Convert complete training-only request archives into off-policy Q replay."""
import argparse
import json
from pathlib import Path
import torch
from amsrr.training.request_q_learning import episode_transitions
from amsrr.utils.hashing import hash_file


def prepare(dataset, collection, output):
    dataset, collection, output = map(Path, (dataset, collection, output))
    definition = json.loads(dataset.read_text())
    training = {c['episode_id']: c for c in definition['training_cases']}
    if any(c['split'] != 'train' for c in training.values()):
        raise ValueError('replay dataset contains non-training conditions')
    sources, transitions = [], []
    runtime = None
    for path in sorted((collection/'panel').glob('*/group_*/training_rollout.pt')):
        case = path.parent.parent.name
        if case not in training:
            raise ValueError('source case not in declared training pool: '+case)
        archive = torch.load(path, map_location='cpu', weights_only=True)
        if (not archive.get('complete') or archive.get('evaluation_only')
                or archive.get('diagnostic_only') or archive.get('teacher_phase_supervision')
                or archive.get('teacher_contact_supervision')):
            raise ValueError('ineligible replay source: '+str(path))
        if runtime is not None and archive['runtime_contracts'] != runtime:
            raise ValueError('replay mixes physical execution contracts')
        runtime = archive['runtime_contracts']
        record_path = Path(training[case]['record'])
        if hash_file(record_path) != training[case]['record_sha256']:
            raise ValueError('training record changed')
        record = json.loads(record_path.read_text())
        manifest_path = Path(record['case_manifest']['path'])
        if hash_file(manifest_path) != record['case_manifest']['sha256']:
            raise ValueError('training case manifest changed')
        manifest = json.loads(manifest_path.read_text())
        task_path = Path(manifest['task_spec']['path'])
        if hash_file(task_path) != manifest['task_spec']['sha256']:
            raise ValueError('authored training task changed')
        task = json.loads(task_path.read_text())
        deadlines = [g['time_limit_s'] for g in task['goals']]
        scene_path = path.parent/'scene.json'
        if scene_path.exists():
            scene = json.loads(scene_path.read_text())
            if deadlines != [g['time_limit_s'] for g in scene['initial_task']['goals']]:
                raise ValueError('condition changed the authored task deadlines')
        for i, episode in enumerate(archive['episodes']):
            rows = episode_transitions(episode, deadlines=deadlines)
            for row in rows:
                row.update(case=case, episode=str(path)+':'+str(i), success=episode[-1].get('task_success', False))
            transitions.extend(rows)
        sources.append(dict(path=str(path), sha256=hash_file(path),
                            task_sha256=hash_file(task_path), episodes=len(archive['episodes'])))
    if not sources:
        raise ValueError('no complete training sources')
    receipt = dict(version='request_physical_q_replay_v1', split='train', dataset=str(dataset),
        dataset_sha256=hash_file(dataset), collection=str(collection), sources=sources,
        episodes=sum(s['episodes'] for s in sources), runtime_contracts=runtime)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('replay output already exists')
    torch.save(dict(receipt, transitions=transitions), output)
    output.with_suffix('.json').write_text(json.dumps(dict(receipt, transitions=len(transitions)), indent=2)+'\n')
    print(json.dumps(dict(output=str(output), episodes=receipt['episodes'], transitions=len(transitions))), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--collection', required=True)
    p.add_argument('--output', required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    prepare(**vars(args))
