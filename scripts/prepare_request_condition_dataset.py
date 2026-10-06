#!/usr/bin/env python3
"""Prepare a reproducible R2 condition pool and full, bounded PPO panels.

Only original training records supply new initial scenes. No trajectory labels,
rewards or simulation outcomes are generated or used to select conditions.
"""
from pathlib import Path
from collections import Counter, defaultdict
from copy import deepcopy
import argparse
import hashlib
import itertools
import json
import random

ROOT = Path(__file__).resolve().parents[1]
AXES = ('volume_factor', 'x_over_y', 'z_over_y', 'mass_factor', 'com_x', 'com_y', 'com_z')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8*1024**2):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def checked(binding):
    path = Path(binding['path'])
    if not path.is_absolute():
        path = ROOT/path
    if sha(path) != binding['sha256']:
        raise ValueError('input hash mismatch: '+str(path))
    return path


def bounds(distribution):
    return [distribution[k] for k in AXES[:4]] + [distribution['true_com_fraction']]*3


def samples(donor_count, distribution, seed):
    """Eight variants/donor: four interior, two faces, two intersecting faces.

    A Latin hypercube covers the seven continuous axes. Face assignment cycles
    over all axes and both endpoints independently of any validation outcomes.
    """
    rng = random.Random(seed)
    n = donor_count*8
    cube = [[0.]*7 for _ in range(n)]
    for axis in range(7):
        bins = list(range(n)); rng.shuffle(bins)
        for i, bin_index in enumerate(bins):
            cube[i][axis] = (bin_index+rng.uniform(.01, .99))/n
    pairs = list(itertools.combinations(range(7), 2)); rng.shuffle(pairs)
    result = []
    for donor in range(donor_count):
        for variant in range(8):
            point = list(cube[donor*8+variant]); kind = 'interior'
            if variant in (4, 5):
                face = (donor*2+variant-4) % 14
                point[face//2] = float(face % 2); kind = 'single_boundary'
            elif variant in (6, 7):
                axes = list(pairs[(donor*2+variant-6) % len(pairs)])
                if variant == 7:
                    axes.append(rng.choice([a for a in range(7) if a not in axes]))
                for axis in axes:
                    point[axis] = float(rng.randrange(2))
                kind = 'multiple_boundaries'
            values = [lo+(hi-lo)*u for u, (lo, hi) in zip(point, bounds(distribution))]
            result.append((donor, variant, kind, dict(zip(AXES[:4], values[:4])), values[4:]))
    return result


def panels(cases, per_module, seed):
    """Partition once per cycle, balancing modules and distinct donor scenes."""
    rng = random.Random(seed)
    by_module = defaultdict(lambda: defaultdict(list))
    for case in cases:
        by_module[case['module_count']][case['record']].append(case)
    totals = {sum(len(v) for v in groups.values()) for groups in by_module.values()}
    if len(totals) != 1 or next(iter(totals)) % per_module:
        raise ValueError('condition pool must be equally divisible across module counts')
    count = next(iter(totals))//per_module
    result = [[] for _ in range(count)]
    for module, groups in sorted(by_module.items()):
        for rows in groups.values():
            rng.shuffle(rows)
        for panel in result:
            used = set()
            for _ in range(per_module):
                available = [key for key, rows in groups.items() if rows and key not in used]
                if not available:
                    raise ValueError('cannot keep distinct donors within a module panel')
                rng.shuffle(available)
                key = max(available, key=lambda k: len(groups[k]))
                panel.append(groups[key].pop()); used.add(key)
    return result


def prepare(parent_path, record_dir, output, checkpoint, seed):
    parent_path, output, checkpoint = parent_path.resolve(), output.resolve(), checkpoint.resolve()
    if output.exists():
        raise FileExistsError('use a new dataset directory')
    parent = read(parent_path)
    if parent['maximum_grasp_contacts'] != 2 or parent['training_draws_per_case'] != 16:
        raise ValueError('expected current two-contact,16-draw protocol')
    protected = {}
    held_records, held_poses = set(), set()
    for case in parent['evaluation_cases']+parent['untouched_test_cases']:
        protected[case['record']] = case['record_sha256']
        protected[case['condition']] = case['condition_sha256']
        held_records.add(str(Path(case['record']).resolve()))
        held_poses.add(read(case['record'])['pose_condition_hash'])
    donors = defaultdict(list)
    for path in sorted(record_dir.glob('*.json')):
        record = read(path)
        if record['dataset_split'] != 'train':
            held_poses.add(record['pose_condition_hash'])
            continue
        if record['status'] != 'accepted' or not record['training_eligible']:
            raise ValueError('unusable original training record: '+str(path))
        if str(path.resolve()) in held_records:
            raise ValueError('training record overlaps protected evaluation')
        manifest = read(checked(record['case_manifest']))
        if manifest['split'] != 'train':
            raise ValueError('record/case split mismatch')
        checked(manifest['task_spec']); checked(manifest['morphology_graph'])
        donors[record['module_count']].append((path.resolve(), record, manifest))
    if set(donors) != {c['module_count'] for c in parent['training_cases']}:
        raise ValueError('training module coverage differs')
    if any(record['pose_condition_hash'] in held_poses for rows in donors.values() for _, record, _ in rows):
        raise ValueError('training initial scene overlaps a held-out scene')
    for path, digest in protected.items():
        if sha(path) != digest:
            raise ValueError('protected evaluation input changed')
    cases = deepcopy(parent['training_cases'])
    known = {str(path) for rows in donors.values() for path, _, _ in rows}
    for case in cases:
        if str(Path(case['record']).resolve()) not in known or sha(case['condition']) != case['condition_sha256'] or sha(case['record']) != case['record_sha256']:
            raise ValueError('retained training case identity mismatch')
    coverage = {c['episode_id']: 'retained' for c in cases}
    # Refuse invalid initialization before creating any output.
    checkpoint_hash, state_hash = sha(checkpoint), sha(checkpoint.parent/'training_state.pt')
    output.mkdir(parents=True)
    for module, rows in sorted(donors.items()):
        for index, variant, kind, values, com in samples(len(rows), parent['distribution'], seed+module):
            path, record, manifest = rows[index]
            identifier = f"r2_train_m{module}_{record['episode_id']}_v{variant:02d}"
            condition = dict(bucket_id=identifier, split='train', **values, com_fraction=com,
                             condition_seed=seed+module, geometry_reuse_permitted=False)
            condition_path = output/'conditions'/f'{identifier}.json'; write(condition_path, condition)
            cases.append(dict(episode_id=identifier, bucket_id=identifier, split='train',
                record=str(path), record_sha256=sha(path), module_count=module,
                structural_hash=manifest['structural_hash'], pose_condition_hash=record['pose_condition_hash'],
                condition=str(condition_path), condition_sha256=sha(condition_path),
                original_episode_id=record['episode_id'], greedy=False, policy_seed=17))
            coverage[identifier] = kind
    if len({c['episode_id'] for c in cases}) != len(cases):
        raise ValueError('duplicate case identifier')
    signatures = [(c['record'], tuple(read(c['condition'])[k] for k in AXES[:4]), tuple(read(c['condition'])['com_fraction'])) for c in cases]
    if len(set(signatures)) != len(cases):
        raise ValueError('duplicate initial scene/object condition')
    batch_cases = panels(cases, 4, seed)
    # Retain learning/task settings, discard old execution history and old model selection evidence.
    keep = ('configuration_hashes','config_sha256','distribution','ppo','training_draws_per_case',
            'training_seed_base','evaluation_seed_base','evaluation_draws_per_case',
            'prepare_wallclock_timeout_s','execute_wallclock_timeout_s','evaluation_cases',
            'untouched_test_cases','selection_rule','validation_interval','stale_validations',
            'minimum_additional_updates','reward_tolerance')
    common = {key: deepcopy(parent[key]) for key in keep}
    initial_update = int(checkpoint.parent.name.removeprefix('update_'))
    common.update(source_protocol=str(parent_path), source_protocol_sha256=sha(parent_path),
        maximum_grasp_contacts=2, max_grasp_contacts=2, maximum_training_environments=16,
        maximum_evaluation_environments=16, evaluation_only=False, additional_learning=True,
        initial_optimizer_updates=initial_update, initialization=dict(checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_hash, optimizer_state=str(checkpoint.parent/'training_state.pt'),
            optimizer_state_sha256=state_hash),
        scope='R2 expanded training pool; unchanged fixed validation/test; exactly two contacts')
    panel_files = []
    for index, selected in enumerate(batch_cases):
        path = output/'panels'/f'panel_{index:03d}.json'
        write(path, dict(common, training_cases=selected))
        panel_files.append(dict(index=index, path=str(path), sha256=sha(path),
                                cases=[c['episode_id'] for c in selected], episodes=len(selected)*16))
    dataset = dict(parent_protocol=str(parent_path), parent_protocol_sha256=sha(parent_path),
        generator=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__), seed=seed),
        distribution=parent['distribution'], training_cases=cases, coverage_kind=coverage,
        evaluation_cases=parent['evaluation_cases'], untouched_test_cases=parent['untouched_test_cases'],
        initialization=common['initialization'], panels=panel_files,
        cycle_rule='Use every panel once per cycle; continue optimizer/update seeds across panels. No outcome filtering.',
        observed_training_reward_comparison='Varying panels estimate the same pool distribution; individual update differences include condition sampling variation.',
        protected_evaluation_hashes=protected)
    write(output/'dataset.json', dataset)
    coverage_report = dict(condition_count=len(cases), new_conditions=len(cases)-len(parent['training_cases']),
        original_train_scenes=len(known), structural_patterns=len({manifest['structural_hash'] for rows in donors.values() for _,_,manifest in rows}),
        module_counts=dict(Counter(c['module_count'] for c in cases)), categories=dict(Counter(coverage.values())),
        panel_count=len(batch_cases), conditions_per_panel=len(batch_cases[0]), episodes_per_panel=len(batch_cases[0])*16,
        validation_conditions=len(parent['evaluation_cases']), test_conditions=len(parent['untouched_test_cases']),
        retained_original_cases_unchanged=cases[:len(parent['training_cases'])]==parent['training_cases'],
        protected_evaluation_unchanged=True, training_scene_overlap_with_held_out=0,
        generated_demonstrations=False, ppo_executed=False)
    write(output/'coverage.json', coverage_report)
    return dataset, coverage_report


def validate_inputs(output):
    """Exercise actual causal scene expansion, two-contact catalog and actor."""
    import torch
    from amsrr.training.request_imitation import load_episode, build_physical_model_from_config
    from amsrr.training.request_object_conditions import expand_box_episode, expand_episode_grasp_contacts
    from amsrr.policies.request_actor_critic import RequestActorCritic
    torch.set_num_threads(1)
    data = read(output/'dataset.json')
    physical = build_physical_model_from_config(ROOT/'configs/robot/robot_model.yaml')
    actor = RequestActorCritic.load(data['initialization']['checkpoint']).eval()
    state = torch.load(data['initialization']['optimizer_state'], map_location='cpu', weights_only=True)
    assert state['checkpoint_sha256'] == data['initialization']['checkpoint_sha256']
    groups = defaultdict(list)
    for case in data['training_cases']:
        groups[case['record']].append(case)
    results, snapshots = [], set()
    for path, cases in sorted(groups.items()):
        episode = load_episode(dict(path=path, sha256=cases[0]['record_sha256']), physical)
        for case in cases:
            condition = read(case['condition'])
            assert sha(case['condition']) == case['condition_sha256']
            expanded, _ = expand_box_episode(episode, condition, physical)
            _, context = expand_episode_grasp_contacts(expanded, physical, max_grasp_contacts=2)
            with torch.no_grad():
                d = actor.decide(context, deterministic=True)
            valid = torch.isfinite(d['logits'])
            sizes = {len(entry.candidate_ids) for entry, active in zip(context.catalog.entries, valid) if active}
            assert valid.any() and sizes == {2}
            assert d['snapshot_hash'] not in snapshots
            snapshots.add(d['snapshot_hash'])
            obj = expanded['task'].scene.objects[0]
            results.append(dict(case=case['episode_id'], snapshot=d['snapshot_hash'],
                active_choices=int(valid.sum()), mass_kg=obj.mass_kg, density_kg_m3=obj.density_kg_m3,
                center_of_mass_object=obj.center_of_mass_object))
        print(json.dumps(dict(checked_conditions=len(results), total=len(data['training_cases']))), flush=True)
    assert len(results) == len(data['training_cases'])
    write(output/'input_audit.json', dict(passed=True, conditions=len(results), initial_snapshots_unique=True,
        contact_cardinality=2, labels_from_teacher=False, task_success_verified=False,
        checkpoint_sha256=sha(data['initialization']['checkpoint']), rows=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-protocol', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument('--validate-inputs', action='store_true')
    args = parser.parse_args()
    _, report = prepare(args.parent_protocol, args.records, args.output, args.checkpoint, args.seed)
    print(json.dumps(report), flush=True)
    if args.validate_inputs:
        validate_inputs(args.output.resolve())


if __name__ == '__main__':
    main()
