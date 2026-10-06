"""Source-controlled request-policy collector, extracted from the R2 runner.

The sampled draws, value/log-probability replay, rejection data, reward returns,
physical-motion audit and PPO split rules are retained. A larger batch changes
only how many independent replicas share a simulation step.
"""
from copy import deepcopy
from pathlib import Path
import json

import torch

from amsrr.training.request_imitation import build_physical_model_from_config
from amsrr.policies.request_actor_critic import RequestActorCritic, require_training_object_inputs
from amsrr.training.request_ppo import (collate_transitions, evaluate_batch,
    discounted_event_returns, request_runtime_contracts)
from amsrr.policies.request_ranking import action_statistics
from amsrr.utils.hashing import stable_hash
from amsrr.training.request_object_conditions import grasp_contact_expansion_limit


def aggregate(rows):
    result = dict(rows=rows, count=len(rows), successes=sum(r['passed'] for r in rows),
        safety_failures=sum(r['safety_failure'] for r in rows),
        mean_reward=sum(r['reward'] for r in rows)/len(rows))
    if all('planning_attempts' in r for r in rows):
        result['mean_planning_attempts'] = sum(r['planning_attempts'] for r in rows)/len(rows)
    if any('anchor_distance' in r for r in rows):
        if not all('anchor_distance' in r for r in rows):
            raise ValueError('mixed anchor-distance collection records')
        result.update(mean_anchor_distance=sum(r['anchor_distance'] for r in rows)/len(rows),
                      center_selections=sum(r['anchor_distance']==0 for r in rows))
    return result


def replay_log_probability_errors(actor, events, *, tolerance=2e-5):
    """Check stored probabilities; resolve batch rounding with canonical inference.

    Batched CPU matrix products can round differently from the one-observation
    policy call. An apparent mismatch is rechecked individually, with the same
    tolerance. Genuine input/weight/action mismatches still fail the caller.
    """
    from amsrr.policies.request_q_policy import RequestQPolicy, CONTEXTUAL_RANKING_VERSION
    contextual = isinstance(actor, RequestQPolicy) and actor.q_config.get('candidate_context', False)
    def probabilities(scores, rows, *, batched=True):
        ordinary = []
        for event in rows:
            prefix = event.get('ranking_prefix')
            if contextual and prefix is not None:
                if (event.get('ranking_policy_version') != CONTEXTUAL_RANKING_VERSION
                        or not prefix or not bool(event['encoding']['initial']) or prefix[-1] != event['action']):
                    raise ValueError('conditional ranking contract mismatch')
                ordinary.append({k: v for k, v in event.items() if k != 'ranking_prefix'})
            else:
                ordinary.append(event)
        values = action_statistics(scores, ordinary)[0]
        if contextual:
            for i, event in enumerate(rows):
                if 'ranking_prefix' in event:
                    values[i] = actor.ranking_log_probability(event['encoding'], event['ranking_prefix'], batched=batched)
        return values
    data = collate_transitions(events)
    with torch.no_grad():
        scores, _ = evaluate_batch(actor, data, torch.arange(len(events)), 'cpu')
        replayed = probabilities(scores, events)
    recorded = torch.tensor([event['log_prob'] for event in events])
    errors = (replayed - recorded).abs()
    for index in (errors >= tolerance).nonzero().flatten().tolist():
        event = events[index]
        with torch.no_grad():
            scores, _ = evaluate_batch(actor, collate_transitions([event]), torch.arange(1), 'cpu')
            replayed = probabilities(scores, [event], batched=False)
        errors[index] = abs(float(replayed[0]) - float(recorded[index]))
    return errors


def ppo_update_command(*, python, checkpoint, archives, output, manifest, protocol):
    """Feed a complete on-policy panel to the existing PPO optimizer.

    Collection diagnostics must never become learning data. Hyperparameters
    remain owned by the training protocol, including the all-rejections case.
    """
    if not archives:
        raise ValueError('empty PPO collection')
    episodes = events = 0
    for path in archives:
        data = torch.load(path, map_location='cpu', weights_only=True)
        if (not data.get('complete') or data.get('diagnostic_only')
                or data.get('evaluation_only') or data.get('teacher_phase_supervision')
                or data.get('teacher_contact_supervision')):
            raise ValueError('PPO requires complete training collection without demonstrations or diagnostics')
        episodes += len(data['episodes'])
        events += sum(len(episode) for episode in data['episodes'])
    expected = len(protocol['training_cases']) * protocol['training_draws_per_case']
    if episodes != expected:
        raise ValueError(f'incomplete PPO panel: {episodes} episodes, expected {expected}')
    settings = dict(protocol['ppo'])
    contact_baseline = settings.pop('contact_baseline', 'condition_loo_v1')
    if contact_baseline not in {'condition_loo_v1', 'value_v1'}:
        raise ValueError('unknown PPO contact baseline')
    if events == episodes:
        settings = {k: v for k, v in settings.items() if k not in {
            'contact_loss_weight', 'contact_target_kl', 'temporal_target_kl', 'temporal_epochs'}}
    command = [python, 'scripts/train_request_imitation.py', 'ppo',
        '--checkpoint', str(checkpoint), '--rollouts', *map(str, archives),
        '--output', str(output), '--training-profile', 'timed_value_v1']
    if contact_baseline == 'condition_loo_v1':
        command += ['--contact-group-manifest', str(manifest)]
    # A protocol chooses one baseline for the whole run. Declare its boundary
    # once when needed, while retaining Adam history; later checkpoints already
    # carry the requested baseline and need no transition flag.
    state_path = Path(checkpoint).parent / 'training_state.pt'
    if state_path.exists():
        saved = torch.load(state_path, map_location='cpu', weights_only=True)
        previous = saved.get('contact_baseline_kind', 'value_v1')
        if previous != contact_baseline:
            command += ['--contact-baseline-transition-from', previous]
    for name, value in settings.items():
        command.extend(['--' + name.replace('_', '-'), str(value)])
    return command


def execute_panel(runtime, label, cases, checkpoints, *, update_index=None, greedy=False):
    """Collect all independent draws with the existing on-policy and motion audits.

    Runtime supplies resource-limited command execution, condition loading and
    the project's physical-motion audit. Simulator execution can use the
    persistent worker without changing this collector's learning semantics.
    """
    h = runtime
    ROOT, P = runtime.root, runtime.protocol
    contact_expansion_limit = grasp_contact_expansion_limit(P)
    load_episode = runtime.load_episode
    geometry_index = runtime.geometry_index
    benchmark = bool(getattr(runtime, "benchmark", False))
    evaluation=update_index is None
    folder=ROOT/label;folder.mkdir(parents=True,exist_ok=True)
    actors={p.parent.name:RequestActorCritic.load(p).eval() for p in checkpoints}
    paths={p.parent.name:p for p in checkpoints}
    identities={name:h.sha(p) for name,p in paths.items()}
    physical=build_physical_model_from_config(runtime.repo/'configs/robot/robot_model.yaml')
    geometry=geometry_index();rows={name:[] for name in actors};archives=[]
    draw_count=1 if greedy else P['evaluation_draws_per_case' if evaluation else 'training_draws_per_case']
    stage_physics = bool(getattr(runtime, 'stage_physics', False))
    def run_group_iter(case, base, group_index, group, unplanned):
        out=base/f'group_{group_index}';receipt=out/'collection.json'
        if receipt.exists():
            got=json.loads(receipt.read_text())
            saved = {(r['model'], r['seed']) for r in got}
            if 'ranking_order' in group[0]:
                unplanned.extend(d for d in group if (d['model'], d['seed']) not in saved)
                group[:] = [d for d in group if (d['model'], d['seed']) in saved]
            assert [(r['model'],r['seed']) for r in got]==[(d['model'],d['seed']) for d in group]
            assert all(r['checkpoint_sha256']==identities[r['model']] for r in got)
            return got, (str(out/'training_rollout.pt') if not evaluation else None)
        first=group[0]
        argv=[runtime.python,'-u','scripts/run_request_policy.py','prepare','--record',case['record'],
              '--checkpoint',str(paths[first['model']]),'--output',str(out),'--seed',str(first['seed']),'--timeout-s',str(P.get('prepare_wallclock_timeout_s',150))]
        ranked = 'ranking_order' in first
        if ranked:
            argv += ['--plan-cache', str(ROOT.parent / 'checked_plan_cache')]
        if contact_expansion_limit is not None:
            argv += ['--max-grasp-contacts', str(contact_expansion_limit)]
            if P.get('grasp_anchor_source') == 'neighborhood':
                argv += ['--grasp-anchor-source', 'neighborhood']
        if case.get('condition'):
            assert h.sha(case['condition']) == case['condition_sha256']
            argv += ['--object-condition', case['condition']]
        if greedy:argv.append('--greedy')
        cached=geometry.get((first['snapshot'],stable_hash(first['request'])))
        if cached is not None and not ranked:argv+=['--geometry',str(cached)]
        timeout = P.get('prepare_wallclock_timeout_s',150)
        h.command(argv,base/f'group_{group_index}.prepare.log',
                  timeout * (len(first['ranking_order']) if ranked else 1) + 30,(0,2))
        prepared_decision = json.loads((out/'decision.json').read_text())
        if ranked:
            prefix = prepared_decision['ranking_prefix']
            if first['ranking_order'][:len(prefix)] != prefix:
                raise ValueError('planner changed the policy ranking')
            matching = [d for d in group if d['ranking_order'][:len(prefix)] == prefix]
            unplanned.extend(d for d in group if d['ranking_order'][:len(prefix)] != prefix)
            group[:] = matching
            for draw in group:
                draw.update(request=prepared_decision['request'], action=prepared_decision['action'],
                            log_prob=prepared_decision['log_prob'],
                            ranking_prefix=prepared_decision['ranking_prefix'])
        assert prepared_decision['request']==first['request']
        rejected=(out/'planning_rejection.json').exists()
        if rejected:
            original=torch.load(out/'request_rollout.pt',weights_only=True)
            assert len(original['episodes'])==len(original['episodes'][0])==1
            episodes=[]
            for draw in group:
                e=deepcopy(original['episodes'][0]);e[0]['action']=draw['action'];e[0]['log_prob']=draw['log_prob']
                # Shared raw input, but each model owns its value prediction.
                data=collate_transitions(e)
                with torch.no_grad():_,v=evaluate_batch(actors[draw['model']],data,torch.arange(1),'cpu')
                e[0]['value']=float(v[0]);episodes.append(e)
            archive={**original,'episodes':episodes,'environment_seeds':[d['seed'] for d in group],
                     'evaluation_only':evaluation,'diagnostic_only':evaluation,
                     'environment_checkpoint_sha256':[identities[d['model']] for d in group]}
            result=None
        else:
            argv=[runtime.python,'-u','scripts/run_request_policy.py','execute','--job',str(out/'job.json'),
                  '--num-envs',str(len(group)),'--env-spacing','0','--environment-seeds',*[str(d['seed']) for d in group],
                  '--rollout-steps',str(P.get('diagnostic_rollout_steps',20000)),'--timeout-s',str(P.get('execute_wallclock_timeout_s',360))]
            if benchmark:argv.append('--benchmark')
            if evaluation:argv+=['--evaluation-checkpoints',*[str(paths[d['model']]) for d in group]]
            if stage_physics:
                yield (argv,out/'isaac.log',P.get('execute_wallclock_timeout_s',360)+30,(0,2))
            else:
                h.command(argv,out/'isaac.log',P.get('execute_wallclock_timeout_s',360)+30,(0,2))
            archive=torch.load(out/'request_rollout.pt',weights_only=True)
            result=json.loads((out/'result.json').read_text())
            assert result['physical_acceptance_eligible'] == (not benchmark) and not result['teacher_phase_supervision']
            assert result['environment_origins_world']==[[0.,0.,0.]]*len(group)
            geometry[(first['snapshot'],stable_hash(first['request']))]=out/'geometry.json'
        assert archive['complete'] and archive['evaluation_only']==evaluation
        assert archive['sampling']==('greedy' if greedy else 'categorical')
        assert not archive['teacher_phase_supervision'] and archive['runtime_contracts']==request_runtime_contracts()
        assert len(archive['episodes'])==len(group)
        audit_batch = getattr(runtime, 'audit_batch', None)
        motion_reports = audit_batch(out) if not rejected and audit_batch is not None else None
        if motion_reports is not None:
            assert len(motion_reports) == len(group)
        # Replay every event, batching the forward pass across replicas. The
        # original per-event tolerance and per-episode checks remain unchanged.
        replay_errors = [None] * len(group)
        for name, actor in actors.items():
            indices = [i for i, draw in enumerate(group) if draw['model'] == name]
            events = [event for i in indices for event in archive['episodes'][i]]
            if not events:
                continue
            errors = replay_log_probability_errors(actor, events)
            offset = 0
            for i in indices:
                length = len(archive['episodes'][i])
                replay_errors[i] = float(errors[offset:offset + length].max())
                offset += length
        got=[]
        for env,(events,draw) in enumerate(zip(archive['episodes'],group)):
            assert events[-1]['done'] and sum(bool(e['done']) for e in events)==1
            assert events[0]['request']==draw['request']
            assert archive['environment_checkpoint_sha256'][env]==identities[draw['model']]
            error=replay_errors[env];assert error<2e-5
            discounted_event_returns(events,.99)
            row=dict(case=case['episode_id'],model=draw['model'],seed=draw['seed'],environment=env,
                     checkpoint_sha256=identities[draw['model']],directory=str(out),modal=draw['modal'],
                     selected_group=draw['request']['contact_group_id'],planning_rejected=rejected,
                     reward=sum(e['reward'] for e in events),events=len(events),replay_error=error)
            if 'planning_attempts' in events[0]:
                row.update(planning_attempts=events[0]['planning_attempts'], planning_cost=events[0]['planning_cost'])
            if 'anchor_distances' in events[0]['encoding']:
                row['anchor_distance'] = float(events[0]['encoding']['anchor_distances'][events[0]['action']])
            if rejected:row.update(passed=False,safety_failure=False,failure_reason=json.loads((out/'planning_rejection.json').read_text())['reason'])
            else:
                native=result['episodes'][env];assert native['fallback_decision_count']==0
                motion=motion_reports[env] if motion_reports is not None else runtime.audit(out,environment=env)
                row.update(passed=native['task_success'] and not native['safety_failure'] and native['no_fallback_success'] and motion['physical_goal_and_motion_passed'],
                           safety_failure=native['safety_failure'],failure_reason=native['failure_reason'],motion=motion)
            got.append(row)
        if benchmark:
            archive['diagnostic_only'] = True
        if evaluation and rejected:
            torch.save(archive,out/'request_rollout.pt')
        if not evaluation:
            assert len(actors)==1 and archive['checkpoint_sha256']==next(iter(identities.values()))
            torch.save(archive,out/'training_rollout.pt')
        h.write(receipt,got)
        return got, (str(out/'training_rollout.pt') if not evaluation else None)

    def run_group(case, base, group_index, group):
        unplanned = []
        iterator = run_group_iter(case, base, group_index, group, unplanned)
        try:
            command = next(iterator)
        except StopIteration as result:
            prepared = dict(result=result.value)
        except BaseException:
            iterator.close()
            raise
        else:
            prepared = dict(iterator=iterator, command=command)
        # Prefixes that end at the same accepted assignment share physics;
        # untried ranking suffixes must not create redundant simulator jobs.
        try:
            remainder = (run_group(case, base, str(group_index) + '_next', unplanned)
                         if unplanned else [])
        except BaseException:
            if 'iterator' in prepared:
                prepared['iterator'].close()
            raise
        return [prepared, *remainder]

    def collect_group(result):
        got, archive = result
        for row in got:
            rows[row['model']].append(row)
        if archive is not None:
            archives.append(archive)
        report={n:aggregate(x) for n,x in rows.items() if x}
        h.write(folder/'summary.json',report)
        h.emit(dict(stage=label,case=got[0]['case'],counts={n:v['count'] for n,v in report.items()},
                    successes={n:v['successes'] for n,v in report.items()},means={n:v['mean_reward'] for n,v in report.items()}))

    pending = []
    executor = getattr(runtime, 'group_executor', None)
    initial_pool = getattr(runtime, 'initial_draw_pool', None)
    initial_futures = []
    def case_seeds(case_index, case):
        index = (getattr(runtime, 'evaluation_case_indices', {}).get(case['episode_id'], case_index)
                 if evaluation else next(i for i,c in enumerate(P['training_cases'])
                                         if c['episode_id'] == case['episode_id']))
        return [(17 if greedy else (P['evaluation_seed_base'] + index*100 + j if evaluation
                 else P['training_seed_base'] + update_index*10000 + index*100 + j))
                for j in range(draw_count)]
    if initial_pool is not None:
        from amsrr.training.request_initial_draws import draw_initial_case
        for index, case in enumerate(cases):
            initial_futures.append(initial_pool.submit(draw_initial_case, dict(
                case=case, checkpoints={name:str(path) for name,path in paths.items()},
                checkpoint_identities=identities, seeds={name:case_seeds(index, case) for name in actors},
                greedy=greedy, max_grasp_contacts=contact_expansion_limit,
                grasp_anchor_source=P.get('grasp_anchor_source', 'all_free'),
                require_training_split=not evaluation,
                physical_configuration=str(runtime.repo/'configs/robot/robot_model.yaml'),
                source_hashes=runtime.initial_draw_sources,
                configuration_hashes={**P['configuration_hashes'],
                    str(runtime.repo/'configs/training/order9_learning_curriculum.yaml'):P['config_sha256']})))
    for case_index,case in enumerate(cases):
        if not evaluation:
            assert case['split']=='train'
            assert json.loads(Path(case['record']).read_text())['dataset_split']=='train'
        h.verify();assert h.sha(case['record'])==case['record_sha256']
        assert all(h.sha(paths[n])==identities[n] for n in actors)
        base=folder/case['episode_id'];base.mkdir(exist_ok=True)
        groups={};draws=[]
        if initial_pool is not None:
            draws = initial_futures[case_index].result()
            expected = [(name, seed) for name in actors for seed in case_seeds(case_index, case)]
            if [(row['model'], row['seed']) for row in draws] != expected:
                raise ValueError('parallel initial draws changed protocol order/seeds')
        else:
            episode=load_episode(case,physical)
            from amsrr.training.request_object_conditions import expand_episode_grasp_contacts
            episode,context=expand_episode_grasp_contacts(episode,physical,max_grasp_contacts=contact_expansion_limit,
                grasp_anchor_source=P.get('grasp_anchor_source', 'all_free'))
            for name,actor in actors.items():
                if not evaluation:
                    require_training_object_inputs(actor,context.task_spec)
                seeds=case_seeds(case_index, case)
                decisions=actor.decide_many(context,deterministic=greedy,
                    generators=[torch.Generator().manual_seed(seed) for seed in seeds])
                for seed,x in zip(seeds,decisions):
                    draws.append(dict(model=name,seed=seed,request=x['request'].to_dict(),action=x['action'],
                         log_prob=x['log_prob'],modal=x['action']==int(x['logits'].argmax()),snapshot=x['snapshot_hash'],
                         **({'ranking_order':x['ranking_order']} if 'ranking_order' in x else {})))
        for row in draws:
            groups.setdefault(stable_hash([row['request'], row['model']] if 'ranking_order' in row else row['request']),[]).append(row)
        h.write(base/'draws.json',dict(draws=draws,checkpoint_sha256=identities))
        batch_limit=int(P.get("maximum_evaluation_environments",16) if evaluation
                        else P.get("maximum_training_environments",16))
        assert batch_limit >= 1
        chunks=[group[i:i+batch_limit] for group in groups.values() for i in range(0,len(group),batch_limit)]
        for group_index,group in enumerate(chunks):
            if executor is None:
                prepared = run_group(case,base,group_index,group)
                if stage_physics:
                    pending.append(prepared)
                else:
                    for item in prepared:
                        collect_group(item['result'])
            else:
                pending.append(executor.submit(run_group,case,base,group_index,group))
    # Stable protocol/draw ordering regardless of completion order.
    if stage_physics:
        prepared = []
        auditing = []
        try:
            for value in pending:
                prepared.extend(value.result() if executor is not None else value)
            # CPU planning workers are finished before physical execution is
            # admitted. Independent per-environment RNG/archives are retained.
            closer = getattr(runtime, 'close_workers', None)
            if closer is not None:
                closer()
            commands = [item['command'] for item in prepared if 'command' in item]
            if commands:
                runtime.run_physics_batch(commands, folder)
            if closer is not None:
                closer()
            def finish_group(item):
                if 'result' in item:
                    return item['result']
                try:
                    next(item['iterator'])
                except StopIteration as result:
                    return result.value
                raise RuntimeError('collection unexpectedly requested a second execution')
            if executor is None:
                for item in prepared:
                    collect_group(finish_group(item))
            else:
                for item in prepared:
                    auditing.append(executor.submit(finish_group, item))
                for future in auditing:
                    collect_group(future.result())
        finally:
            # A generator must not be closed while an audit thread is resuming it.
            if auditing:
                from concurrent.futures import wait
                for future in auditing:
                    future.cancel()
                wait(auditing)
            for item in prepared:
                if 'iterator' in item:
                    item['iterator'].close()
    else:
        for future in pending:
            for item in future.result():
                collect_group(item['result'])
    assert all(len(r)==len(cases)*draw_count for r in rows.values())
    report={n:aggregate(r) for n,r in rows.items()}
    h.write(folder/'summary.json',report)
    if not evaluation:
        by_case = {case['episode_id']: case for case in cases}
        entries = []
        for path in archives:
            archive = Path(path)
            case = by_case[archive.parent.parent.name]
            saved = json.loads((archive.parent.parent/'draws.json').read_text())['draws']
            snapshots = {draw['snapshot'] for draw in saved}
            assert len(snapshots) == 1
            snapshot = next(iter(snapshots))
            identity = dict(record_sha256=case['record_sha256'], snapshot_hash=snapshot)
            entries.append(dict(path=path, sha256=h.sha(path), record_path=case['record'],
                **identity, condition_hash=stable_hash(identity)))
        h.write(folder/'contact_groups.json', dict(version='same_initial_condition_rollout_groups_v1',
            group_size=P['training_draws_per_case'], rollouts=entries))
    h.write(folder/'completed.json',dict(checkpoints=identities,training=not evaluation,
        archives=archives,cases=[c['episode_id'] for c in cases],draws_per_case=draw_count))
    return report,archives
