#!/usr/bin/env python3
"""Author fixed upstream anchors and paired R2 training object conditions.

No validation outcomes, policy logits, plans or simulation results participate
in selection. Existing training morphologies remain disjoint from held-out
morphologies. This is a dataset authoring step, not a policy-time anchor search.
"""
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import argparse
import random

from scripts.prepare_request_condition_dataset import read, write, sha, checked, samples


def task_morphology(case):
    from amsrr.schemas.morphology import MorphologyGraph
    record = read(case['record'])
    manifest = read(checked(record['case_manifest']))
    nominal_path = checked(manifest['nominal_artifact'])
    nominal = read(nominal_path)
    path = Path(nominal['task_conditioned_morphology_path'])
    if not path.is_absolute():
        path = nominal_path.parent/path
    return manifest, MorphologyGraph.from_dict(read(path))


def bound_pitch_pair_designs(physical, excluded, asset_manifest_path):
    """Read independently authored upstream designs with physical asset binding.

    Authoring/feasibility belongs to the existing morphology-pool pipeline;
    this training-data generator never searches designs using task outcomes.
    """
    from amsrr.simulation.order9_morphology_assets import (
        load_order9_morphology_asset_manifest,
        validate_order9_morphology_asset_manifest_bytes)
    from amsrr.schemas.order3 import Order3MorphologyPoolManifest
    assets = load_order9_morphology_asset_manifest(asset_manifest_path)
    repository = Path(__file__).resolve().parents[1]
    validate_order9_morphology_asset_manifest_bytes(assets, repository_root=repository)
    if assets.physical_model_hash != physical.stable_hash():
        raise ValueError('upstream design physical model differs')
    pool = Order3MorphologyPoolManifest.from_json(
        (repository/assets.source_pool_path).read_text())
    binding = dict(path=str(asset_manifest_path.resolve()), sha256=sha(asset_manifest_path))
    result = []
    for entry in sorted(pool.entries, key=lambda e:e.structural_hash):
        graph = entry.morphology_graph
        if entry.split.value != 'train' or entry.structural_hash in excluded:
            raise ValueError('additional upstream morphology is not independent training data')
        if entry.module_count != 6:
            continue
        anchors = [a for a in graph.robot_anchors if a.anchor_type == 'grasp']
        if len(anchors) != 2 or any(a.capability.get('dock_port_local_id') !=
                                  'pitch_connect_point_1' for a in anchors):
            continue
        asset = assets.entry_for(graph)
        if asset.source_morphology_hash != graph.stable_hash():
            raise ValueError('upstream design and physical assets differ')
        result.append(dict(morphology=graph, structural_hash=entry.structural_hash,
            ports=[a.capability['surface_port_id'] for a in anchors],
            proposal_seed=entry.accepted_proposal_seed,
            flight_feasibility=entry.feasibility_result.to_dict(), asset_manifest=binding))
    if len(result) != 2:
        raise ValueError('expected two independent, physically bound six-module pitch-pair designs')
    return result


def select_designs(pool, physical, seed, asset_manifest_path):
    """Four donor designs/module plus two missing-port designs each for6/7.

    Six-module pitch pairs use independently generated flight-feasible designs.
    Seven-module pairs minimize displacement from a donor's designated pair
    among free yaw2 ports on different modules. Neither rule guarantees task
    feasibility or supplies an action preference to PPO.
    """
    import numpy as np
    from amsrr.robot_model.gripper_surfaces import resolve_unoccupied_gripper_surfaces
    rng = random.Random(seed)
    by_module = defaultdict(dict)
    for case in pool['training_cases']:
        if case['split'] != 'train' or read(case['record'])['dataset_split'] != 'train':
            raise ValueError('non-training donor')
        by_module[case['module_count']].setdefault(case['record'], case)
    designs = []
    graphs = {}
    for module, rows in sorted(by_module.items()):
        grouped = defaultdict(list)
        for case in rows.values():
            manifest, graph = task_morphology(case)
            graphs[case['record']] = (manifest, graph)
            grouped[manifest['structural_hash']].append(case)
        keys = sorted(grouped); rng.shuffle(keys)
        for rows_for_shape in grouped.values():
            rows_for_shape.sort(key=lambda c: c['record']); rng.shuffle(rows_for_shape)
        if len(keys) > 4:
            raise ValueError('four donor slots cannot cover all training structures')
        selected = [grouped[keys[i % len(keys)]].pop() for i in range(4)]
        designs.extend(dict(case=c, ports=None, kind='original') for c in selected)
        if module == 6:
            excluded = {task_morphology(c)[0]['structural_hash'] for c in
                        pool['training_cases']+pool['evaluation_cases']+pool['untouched_test_cases']}
            proposals = bound_pitch_pair_designs(physical, excluded, asset_manifest_path)
            designs.extend(dict(case=selected[i], kind='new_morphology_pitch_pair', **value)
                           for i, value in enumerate(proposals))
            continue
        if module not in (6, 7):
            continue
        target = 'pitch_connect_point_1' if module == 6 else 'yaw_connect_point_2'
        alternatives = []
        for key in keys:
            case = next(c for c in rows.values() if graphs[c['record']][0]['structural_hash'] == key)
            _, graph = graphs[case['record']]
            surfaces = resolve_unoccupied_gripper_surfaces(graph, physical)
            originals = [a.capability['surface_port_id'] for a in graph.robot_anchors if a.anchor_type == 'grasp']
            positions = {s.port_global_id: np.array(s.grasp_contact_frame_design[:3]) for s in surfaces}
            pairs = [pair for pair in combinations(surfaces, 2)
                     if pair[0].module_id != pair[1].module_id
                     and all(s.port_local_id == target for s in pair)]
            def displacement(pair):
                a, b = (positions[s.port_global_id] for s in pair)
                x, y = (positions[p] for p in originals)
                return min(float(np.linalg.norm(a-x)+np.linalg.norm(b-y)),
                           float(np.linalg.norm(a-y)+np.linalg.norm(b-x)))
            if pairs:
                pair = min(pairs, key=lambda p: (displacement(p), tuple(s.port_global_id for s in p)))
                alternatives.append(dict(case=case, ports=[s.port_global_id for s in pair],
                    kind='upstream_'+target, donor_pair_displacement_m=displacement(pair)))
        if len(alternatives) < 2:
            raise ValueError('missing independent training structures for '+target)
        # Fixed shuffled structure order, independent of feasibility/outcomes.
        designs.extend(alternatives[:2])
    if len(designs) != 32:
        raise ValueError('expected seven module counts and32 upstream designs')
    return designs, graphs


def prepare(protocol_path, pool_path, output, seed, asset_manifest_path):
    from amsrr.training.request_imitation import build_physical_model_from_config
    protocol, pool = read(protocol_path), read(pool_path)
    if output.exists():
        raise FileExistsError('use a fresh dataset directory')
    if protocol['grasp_anchor_source'] != 'defined' or protocol['training_draws_per_case'] != 8:
        raise ValueError('expected defined anchors and8 draws')
    if protocol['evaluation_cases'] != pool['evaluation_cases'] or protocol['untouched_test_cases'] != pool['untouched_test_cases']:
        raise ValueError('held-out split mismatch')
    held_records, held_shapes = set(), set()
    protected = {}
    for case in protocol['evaluation_cases']+protocol['untouched_test_cases']:
        for key in ('record', 'condition'):
            if sha(case[key]) != case[key+'_sha256']:
                raise ValueError('held-out input changed')
            protected[case[key]] = case[key+'_sha256']
        held_records.add(case['record'])
        held_shapes.add(task_morphology(case)[0]['structural_hash'])
    physical = build_physical_model_from_config('configs/robot/robot_model.yaml')
    designs, graphs = select_designs(pool, physical, seed, asset_manifest_path)
    for design in designs:
        case = design['case']
        if case['record'] in held_records or graphs[case['record']][0]['structural_hash'] in held_shapes:
            raise ValueError('training design overlaps held-out morphology/record')
        if sha(case['record']) != case['record_sha256']:
            raise ValueError('donor identity changed')
    # Two independently sampled geometries/COMs per upstream design, each at
    # both mass endpoints. Paired objects differ only in mass, not in the robot.
    points = {(i, v): (values, com) for i, v, _, values, com in
              samples(len(designs), protocol['distribution'], seed+1) if v in (0, 1)}
    panels = [[] for _ in range(4)]
    design_records = []
    for i, design in enumerate(designs):
        donor = design['case']; manifest, graph = graphs[donor['record']]
        ports = design['ports']
        design_id = f'anchor_train_m{donor["module_count"]}_d{i:02d}'
        design_hash = design.get('structural_hash', manifest['structural_hash'])
        morphology_binding = None
        if 'morphology' in design:
            graph_path = output/'morphologies'/f'{design_id}.json'
            write(graph_path, design['morphology'].to_dict())
            morphology_binding = dict(path=str(graph_path.resolve()), sha256=sha(graph_path),
                                      asset_manifest=design['asset_manifest'])
        ids = []
        for variant in range(4):
            values, com = points[i, variant//2]
            values = dict(values, mass_factor=protocol['distribution']['mass_factor'][variant % 2])
            identifier = f'{design_id}_v{variant:02d}'
            condition = dict(bucket_id=identifier, split='train', **values,
                com_fraction=com, condition_seed=seed, geometry_reuse_permitted=False)
            if ports is not None:
                condition['designated_grasp_port_ids'] = ports
            if morphology_binding is not None:
                condition['designated_morphology'] = morphology_binding
            path = output/'conditions'/f'{identifier}.json'; write(path, condition)
            case = dict(donor, episode_id=identifier, bucket_id=identifier,
                structural_hash=design_hash, condition=str(path.resolve()),
                condition_sha256=sha(path), greedy=False, policy_seed=17)
            panels[(variant+i) % 4].append(case); ids.append(identifier)
        original_ports = [a.capability['surface_port_id'] for a in graph.robot_anchors if a.anchor_type == 'grasp']
        design_records.append(dict(design_id=design_id, record=donor['record'],
            structural_hash=design_hash, module_count=donor['module_count'],
            kind=design['kind'], ports=ports or original_ports, cases=ids,
            morphology=morphology_binding, proposal_seed=design.get('proposal_seed'),
            flight_feasibility=design.get('flight_feasibility')))
    entries = []
    for i, cases in enumerate(panels):
        path = output/'panels'/f'panel_{i:03d}.json'
        p = dict(deepcopy(protocol), training_cases=cases,
            scope='Fixed upstream anchor coverage; paired R2 object conditions; unchanged PPO and held-out42.')
        write(path, p)
        entries.append(dict(index=i, path=str(path.resolve()), sha256=sha(path),
            cases=[c['episode_id'] for c in cases], episodes=len(cases)*8))
    data = dict(training_cases=[c for panel in panels for c in panel], panels=entries,
        evaluation_cases=protocol['evaluation_cases'], untouched_test_cases=protocol['untouched_test_cases'],
        designs=design_records, distribution=protocol['distribution'],
        initialization=protocol['initialization'], protected_evaluation_hashes=protected,
        source_protocol=str(protocol_path.resolve()), source_protocol_sha256=sha(protocol_path),
        source_pool=str(pool_path.resolve()), source_pool_sha256=sha(pool_path), seed=seed,
        asset_manifest=dict(path=str(asset_manifest_path.resolve()), sha256=sha(asset_manifest_path)),
        generator_sha256=sha(__file__), outcomes_used_for_condition_selection=False,
        cycle_rule='Four32-case panels; one object variant of every upstream design per update;8 fresh draws/case. Repeat the same cycle.',
        train_validation_shape_overlap=0)
    write(output/'dataset.json', data)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--pool', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20261005)
    parser.add_argument('--asset-manifest', type=Path, required=True)
    args = parser.parse_args()
    data = prepare(args.protocol, args.pool, args.output.resolve(), args.seed, args.asset_manifest)
    print(f'Prepared {len(data["training_cases"])} conditions, {len(data["designs"])} upstream designs,4 panels.')
