"""Independent CPU initial observations/draws for a frozen collection policy.

Workers receive explicit source/configuration/data identities and RNG seeds.
Only initial decisions are parallelized; no future plan or physical outcome is
an input, and the parent retains protocol order when collecting the results.
"""
import json
from pathlib import Path

import torch

from amsrr.policies.request_actor_critic import RequestActorCritic, require_training_object_inputs
from amsrr.training.request_imitation import load_episode
from amsrr.training.request_imitation import build_physical_model_from_config
from amsrr.training.request_object_conditions import (
    expand_box_episode, expand_episode_grasp_contacts,
)
from amsrr.utils.hashing import hash_file


def draw_initial_case(job):
    torch.set_num_threads(1)
    identities = dict(job['source_hashes'])
    identities.update(job['configuration_hashes'])
    case = job['case']
    identities[case['record']] = case['record_sha256']
    if case.get('condition'):
        identities[case['condition']] = case['condition_sha256']
    for name, path in job['checkpoints'].items():
        identities[path] = job['checkpoint_identities'][name]

    def verify():
        for path, digest in identities.items():
            if hash_file(path) != digest:
                raise ValueError('initial draw input changed: ' + str(path))

    verify()
    if job['require_training_split'] and (
        case['split'] != 'train'
        or json.loads(Path(case['record']).read_text())['dataset_split'] != 'train'
    ):
        raise ValueError('initial training draws require the training split')
    physical = build_physical_model_from_config(job['physical_configuration'])
    episode = load_episode(dict(path=case['record'], sha256=case['record_sha256']), physical)
    if case.get('condition'):
        episode = expand_box_episode(episode,
            json.loads(Path(case['condition']).read_text()), physical)[0]
    _, context = expand_episode_grasp_contacts(episode, physical,
        max_grasp_contacts=job['max_grasp_contacts'],
        grasp_anchor_source=job.get('grasp_anchor_source', 'all_free'))
    draws = []
    for name, path in job['checkpoints'].items():
        actor = RequestActorCritic.load(path).eval()
        if job['require_training_split']:
            require_training_object_inputs(actor,context.task_spec)
        seeds = job['seeds'][name]
        decisions = actor.decide_many(context, deterministic=job['greedy'],
            generators=[torch.Generator().manual_seed(seed) for seed in seeds])
        for seed, decision in zip(seeds, decisions):
            draws.append(dict(model=name, seed=seed, request=decision['request'].to_dict(),
                action=decision['action'], log_prob=decision['log_prob'],
                modal=decision['action'] == int(decision['logits'].argmax()),
                snapshot=decision['snapshot_hash'],
                **({'ranking_order': decision['ranking_order']} if 'ranking_order' in decision else {})))
    verify()
    return draws
