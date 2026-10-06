#!/usr/bin/env python3
"""Resume a fixed request-PPO protocol using guarded full-panel collections.

This outer schedule never imports an artifact runner or changes the collector.
Commands, checkpoint identities, full metrics and verified raw-log archives are
retained. Interrupted stages require diagnosis; they are not retried implicitly.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def better(new, old, tolerance):
    if new['successes'] != old['successes']:
        return new['successes'] > old['successes']
    delta = new['mean_reward'] - old['mean_reward']
    if abs(delta) > tolerance:
        return delta > 0
    if new['safety_failures'] != old['safety_failures']:
        return new['safety_failures'] < old['safety_failures']
    return new['update'] < old['update']


def select_best(rows, tolerance):
    ordered = sorted(rows, key=lambda r: r['update'])
    best = ordered[0]
    for row in ordered[1:]:
        if better(row, best, tolerance):
            best = row
    return best


def plateau(last, best, contract):
    start = contract['resume_update']
    anchor = contract.get('validation_anchor_update', start)
    stale = sum(k > max(start, best['update'])
                for k in range(start + 1, last + 1)
                if (k-anchor) % contract['validation_interval'] == 0)
    return (last - start >= contract['minimum_new_updates']
            and stale >= contract['stale_validations'])


def load_training_panels(dataset_path, baseline):
    """Validate a predeclared pool without changing evaluation or PPO settings."""
    dataset = read(dataset_path)
    panels, seen = [], []
    variable = {'training_cases', 'initialization', 'initial_optimizer_updates',
                'additional_learning', 'evaluation_only', 'scope', 'source_protocol',
                'source_protocol_sha256', 'minimum_additional_updates'}
    reference = None
    for entry in dataset['panels']:
        path = Path(entry['path'])
        if sha(path) != entry['sha256']:
            raise ValueError('training panel changed: ' + str(path))
        protocol = read(path)
        fixed = {k: v for k, v in protocol.items() if k not in variable}
        if reference is not None and fixed != reference:
            raise ValueError('training panels change non-case settings')
        reference = fixed
        for key in ('evaluation_cases', 'untouched_test_cases', 'ppo', 'distribution',
                    'training_draws_per_case', 'training_seed_base', 'configuration_hashes'):
            if protocol[key] != baseline[key]:
                raise ValueError('training pool changes fixed setting: ' + key)
        cases = protocol['training_cases']
        if ([c['episode_id'] for c in cases] != entry['cases']
                or len(cases)*protocol['training_draws_per_case'] != entry['episodes']):
            raise ValueError('training panel case/count mismatch')
        if any(c['split'] != 'train' for c in cases):
            raise ValueError('non-training case in training panel')
        seen.extend(cases)
        panels.append((path, entry['sha256'], protocol))
    expected = dataset['training_cases']
    if (not panels or len({c['episode_id'] for c in seen}) != len(seen)
            or sorted(seen, key=lambda c: c['episode_id']) != sorted(expected, key=lambda c: c['episode_id'])):
        raise ValueError('training panels must cover the pool exactly once')
    return panels


def stage_command(template, root, checkpoint, index, training):
    argv = list(template)
    boundary = argv.index('scripts/run_request_collection.py')
    guard, collector = argv[:boundary], argv[boundary:]
    label = ('train_' if training else 'validation_') + str(index)
    guard[guard.index('--output') + 1] = str(root/'guards'/label)
    collector[collector.index('--output') + 1] = str(root/'collections'/label)
    collector[collector.index('--checkpoint') + 1] = str(checkpoint)
    if '--update-index' in collector or '--ppo-output' in collector:
        raise ValueError('command template must be a full validation command')
    if training:
        collector += ['--update-index', str(index), '--ppo-output', str(root/f'update_{index+1}')]
    return guard + collector


def validate_summary(values, cases, draws, checkpoint_hash, training_index, seed_base):
    rows = values['rows']
    expected = {(c['episode_id'], 17 if training_index is None else
                 seed_base + training_index*10000 + i*100 + j)
                for i, c in enumerate(cases) for j in range(draws)}
    if (len(rows) != len(cases)*draws or len(expected) != len(rows)
            or {(r['case'], r['seed']) for r in rows} != expected):
        raise ValueError('incomplete, duplicate or wrong-seed panel')
    if any(r['checkpoint_sha256'] != checkpoint_hash or not math.isfinite(r['reward']) for r in rows):
        raise ValueError('invalid panel reward or checkpoint identity')
    calculated = dict(count=len(rows), successes=sum(r['passed'] for r in rows),
        safety_failures=sum(r['safety_failure'] for r in rows),
        mean_reward=sum(r['reward'] for r in rows)/len(rows))
    if any(not math.isclose(values[k], v, rel_tol=1e-12, abs_tol=1e-12) for k, v in calculated.items()):
        raise ValueError('summary does not match complete panel')
    return calculated


def validate_optimizer_recovery(stage, failed_guard):
    """Accept an explicitly diagnosed optimizer-only recovery, never fake a guard success."""
    evidence = read(stage/'optimizer_recovery.json')
    if sha(failed_guard) != evidence['failed_guard_sha256']:
        raise ValueError('recovery failed-guard identity mismatch')
    for path, digest in evidence['input_hashes'].items():
        if sha(path) != digest:
            raise ValueError('recovery input changed: ' + path)
    original = read(stage/'optimizer_command.json')['argv']
    required = original[original.index('--rollouts')+1:original.index('--output')]
    checkpoint = Path(original[original.index('--checkpoint')+1])
    required += [str(checkpoint), str(checkpoint.parent/'training_state.pt'),
                 original[original.index('--contact-group-manifest')+1],
                 str(stage/'optimizer_command.json'), str(stage/'collection_binding.json'),
                 str(stage/'panel/summary.json'), str(stage/'panel/completed.json')]
    if not set(required) <= set(evidence['input_hashes']):
        raise ValueError('recovery omits bound collection inputs')
    recovered = evidence['optimizer_argv']
    expected = list(original)
    expected[expected.index('--device')+1] = evidence['device']
    if recovered != expected:
        raise ValueError('recovery changes more than optimizer device')
    guard = read(evidence['guard_result'])
    if (sha(evidence['guard_result']) != evidence['guard_result_sha256']
            or guard['exit_code'] != 0 or guard['error'] is not None
            or guard['command'][-len(recovered):] != recovered):
        raise ValueError('recovery optimizer guard invalid')
    result = read(Path(recovered[recovered.index('--output')+1])/'result.json')
    binding = read(stage/'collection_binding.json')
    if (not result['reload_exact']
            or result['optimization'].get('completed_actor_steps', result['optimization']['completed_epochs']) <= 0
            or result['optimization']['optimizer_updates_resumed'] != binding['collection_policy_update']):
        raise ValueError('recovery optimizer result incomplete')
    return guard['seconds'], evidence['device']


def archive_raw(stage):
    """Compress only completed, owned simulation logs; verify before unlinking."""
    manifest = stage/'raw_log_archives.json'
    data = read(manifest) if manifest.exists() else dict(files=[], restore='zstd -d --keep ARCHIVE -o ORIGINAL')
    known = {r['path']: r for r in data['files']}

    def compress(path):
        digest = sha(path)
        archive = Path(str(path) + '.zst')
        temporary = Path(str(archive) + '.tmp')
        if not archive.exists():
            subprocess.run(['zstd', '-1', '-T1', '--quiet', '--keep', str(path), '-o', str(temporary)], check=True)
            os.replace(temporary, archive)
        proc = subprocess.Popen(['zstd', '-d', '--stdout', '--quiet', str(archive)], stdout=subprocess.PIPE)
        h = hashlib.sha256()
        with proc.stdout:
            while block := proc.stdout.read(8 * 1024**2):
                h.update(block)
        if proc.wait() != 0 or h.hexdigest() != digest:
            raise ValueError('raw archive verification failed: ' + str(path))
        return dict(path=str(path), archive=str(archive), sha256=digest,
                    archive_sha256=sha(archive), bytes=path.stat().st_size, archive_bytes=archive.stat().st_size)

    with ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(compress, sorted((stage/'panel').glob('*/group_*/rollout.pt'))):
            if row['path'] in known:
                if row != known[row['path']]:
                    raise ValueError('archive identity changed')
            else:
                data['files'].append(row)
                write(manifest, data)
            Path(row['path']).unlink()


class Schedule:
    def __init__(self, root):
        self.root = root.resolve()
        self.contract = read(root/'execution_contract.json')
        self.protocol = read(root/'protocol.json')
        self.settings = read(root/'schedule_inputs.json')
        self.template = read(self.settings['command_template'])
        self.panels = []
        self.panel_offset = 0
        self.prior_training_episodes = 0
        if self.settings.get('training_dataset'):
            dataset = Path(self.settings['training_dataset'])
            if sha(dataset) != self.settings['training_dataset_sha256']:
                raise ValueError('training dataset changed')
            self.panels = load_training_panels(dataset, self.protocol)
            previous = self.settings.get('previous_training_curve')
            if previous:
                if sha(previous) != self.settings['previous_training_curve_sha256']:
                    raise ValueError('previous training curve changed')
                rows = read(previous)
                start = self.contract['resume_update']-len(rows)
                if (not rows or rows[-1]['updated_checkpoint'] != self.contract['resume_checkpoint']
                        or [r['update'] for r in rows] != list(range(start, self.contract['resume_update']))
                        or [r['training_panel'] for r in rows] != [i % len(self.panels) for i in range(len(rows))]
                        or any(r['source_protocol_sha256'] != self.panels[r['training_panel']][1] for r in rows)
                        or any(r['count'] != len(self.panels[r['training_panel']][2]['training_cases'])*self.protocol['training_draws_per_case'] for r in rows)):
                    raise ValueError('previous training does not bind the continued pool and checkpoint')
                self.panel_offset = len(rows)
                self.prior_training_episodes = sum(r['count'] for r in rows)
            if self.contract['minimum_new_updates']+self.panel_offset < len(self.panels):
                raise ValueError('minimum updates must cover the training pool')
        self.checkpoints = {int(k): Path(v) for k, v in self.settings['initial_checkpoints'].items()}
        self.validations = []
        self.curve = []
        self.child = None
        binding = dict(driver_sha256=sha(__file__), inputs_sha256=sha(root/'schedule_inputs.json'),
            contract_sha256=sha(root/'execution_contract.json'), protocol_sha256=sha(root/'protocol.json'),
            template_sha256=sha(self.settings['command_template']))
        bound = root/'schedule_binding.json'
        if bound.exists() and read(bound) != binding:
            raise ValueError('schedule inputs changed')
        if binding['protocol_sha256'] != self.contract['protocol_sha256']:
            raise ValueError('protocol identity changed')
        if sha(self.contract['resume_checkpoint']) != self.contract['resume_sha256']:
            raise ValueError('resume checkpoint changed')
        for name, digest in read(self.settings['source_manifest']).items():
            if sha(Path(self.settings['source_root'])/name) != digest:
                raise ValueError('frozen source changed: ' + name)
        self.runtime_sources = None
        write(bound, binding)
        self.binding = binding

    def verify_inputs(self):
        paths = dict(driver_sha256=Path(__file__), inputs_sha256=self.root/'schedule_inputs.json',
            contract_sha256=self.root/'execution_contract.json', protocol_sha256=self.root/'protocol.json',
            template_sha256=Path(self.settings['command_template']))
        for key, path in paths.items():
            if sha(path) != self.binding[key]:
                raise ValueError('schedule input changed during run: ' + str(path))
        for path, digest in self.settings['protected_hashes'].items():
            if sha(path) != digest:
                raise ValueError('protected source/checkpoint changed: ' + path)
        if self.panels:
            if sha(self.settings['training_dataset']) != self.settings['training_dataset_sha256']:
                raise ValueError('training dataset changed during run')
            if any(sha(path) != digest for path, digest, _ in self.panels):
                raise ValueError('training panel changed during run')
            if self.settings.get('previous_training_curve') and sha(self.settings['previous_training_curve']) != self.settings['previous_training_curve_sha256']:
                raise ValueError('previous training curve changed during run')

    def stage(self, index, training=False):
        self.verify_inputs()
        label = ('train_' if training else 'validation_') + str(index)
        stage = self.root/'collections'/label
        guard = self.root/'guards'/label
        checkpoint = self.checkpoints[index]
        argv = stage_command(self.template, self.root, checkpoint, index, training)
        protocol, protocol_hash = self.protocol, self.contract['protocol_sha256']
        panel_index = None
        if training and self.panels:
            panel_index = (self.panel_offset + index - self.contract['resume_update']) % len(self.panels)
            path, protocol_hash, protocol = self.panels[panel_index]
            argv[argv.index('--protocol') + 1] = str(path)
        storage = self.settings.get('training_file_storage_from_update')
        if training and storage is not None and index >= storage:
            argv[argv.index('--cpu-log-storage') + 1] = 'file'
        cpu_from = self.settings.get('training_ppo_cpu_from_update')
        if training and cpu_from is not None and index >= cpu_from:
            argv[argv.index('--ppo-device') + 1] = 'cpu'
        command_path = self.root/'commands'/f'{label}.json'
        if command_path.exists() and read(command_path) != argv:
            raise ValueError('stage command changed: ' + label)
        write(command_path, argv)
        if not (guard/'guard_result.json').exists():
            if stage.exists() or (guard/'process.log').exists():
                raise RuntimeError('incomplete stage requires diagnosis: ' + label)
            if shutil.disk_usage(self.root).free < 60*1024**3:
                raise RuntimeError('insufficient disk headroom for next complete stage')
            write(self.root/'progress.json', dict(stage=label, utc=datetime.now(timezone.utc).isoformat()))
            with (self.root/'commands'/f'{label}.log').open('x') as log:
                self.child = subprocess.Popen(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                code = self.child.wait()
                self.child = None
            if code:
                raise RuntimeError(f'{label} exited {code}; no automatic retry')
        receipt = read(guard/'guard_result.json')
        recovery_seconds, recovered_device = 0, None
        if receipt['exit_code'] != 0 or receipt['error'] is not None:
            if not training or not (stage/'optimizer_recovery.json').exists():
                raise ValueError('guard completion invalid: ' + label)
            recovery_seconds, recovered_device = validate_optimizer_recovery(stage, guard/'guard_result.json')
        if receipt['command'] != argv[argv.index('--')+1:]:
            raise ValueError('guard completion invalid: ' + label)
        binding = read(stage/'collection_binding.json')
        digest = sha(checkpoint)
        cases = protocol['training_cases' if training else 'evaluation_cases']
        draws = protocol['training_draws_per_case'] if training else 1
        expected = dict(checkpoint_sha256=digest, source_protocol_sha256=protocol_hash,
            collection_policy_update=index if training else None, benchmark=False,
            diagnostic_subset=False, diagnostic_steps=None, physical_outcomes_reused=False,
            optimizer_requested=training, optimizer_invoked=training,
            case_ids=[c['bucket_id'] for c in cases], device='cpu', cpu_broadphase='GPU')
        if any(binding[k] != v for k, v in expected.items()):
            raise ValueError('collection binding mismatch: ' + label)
        if self.runtime_sources is not None and self.runtime_sources != binding['runtime_sources']:
            raise ValueError('runtime changed between stages')
        self.runtime_sources = binding['runtime_sources']
        report = read(stage/'panel/summary.json')
        if set(report) != {checkpoint.parent.name}:
            raise ValueError('unexpected evaluated models')
        values = validate_summary(report[checkpoint.parent.name], cases, draws, digest,
                                  index if training else None, protocol['training_seed_base'])
        completed = read(stage/'panel/completed.json')
        if completed['training'] != training or completed['draws_per_case'] != draws or completed['cases'] != [c['episode_id'] for c in cases]:
            raise ValueError('panel completion mismatch')
        row = dict(update=index, checkpoint=str(checkpoint), checkpoint_sha256=digest,
                   seconds=receipt['seconds']+recovery_seconds, **values)
        if recovered_device is not None:
            row.update(optimizer_recovered=True, recovered_optimizer_device=recovered_device)
        if panel_index is not None:
            row.update(training_panel=panel_index, source_protocol_sha256=protocol_hash,
                       training_cycle=(self.panel_offset+index-self.contract['resume_update'])//len(self.panels))
        if training:
            optimized = read(stage/'optimizer_completed.json')
            destination = self.root/f'update_{index+1}/checkpoint.pt'
            if optimized['update_index'] != index+1 or optimized['checkpoint_sha256'] != sha(destination):
                raise ValueError('optimizer completion mismatch')
            row.update(optimizer_update=index+1, updated_checkpoint=str(destination))
            self.checkpoints[index+1] = destination
            self.curve.append(row)
            write(self.root/'learning_curve.json', self.curve)
        else:
            self.validations.append(row)
            write(self.root/'validation_curve.json', self.validations)
            write(self.root/'best.json', select_best(self.validations, self.contract['reward_tolerance']))
        print(json.dumps(dict(stage=label, **row)), flush=True)
        archive_raw(stage)
        return row

    def run(self):
        # The resume model is evaluated first so an independently completed baseline can be adopted.
        for k in sorted(self.contract['baseline_updates'], reverse=True):
            self.stage(k)
        start = self.contract['resume_update']
        k = start
        while True:
            self.stage(k, training=True)
            k += 1
            if (k-self.contract.get('validation_anchor_update', start)) % self.contract['validation_interval']:
                continue
            self.stage(k)
            best = select_best(self.validations, self.contract['reward_tolerance'])
            if not plateau(k, best, self.contract):
                continue
            while True:
                done = {r['update'] for r in self.validations}
                radius = self.contract['neighbor_radius']
                todo = sorted(i for i in self.checkpoints if abs(i-best['update']) <= radius and i not in done)
                if not todo:
                    break
                for i in todo:
                    self.stage(i)
                best = select_best(self.validations, self.contract['reward_tolerance'])
            if plateau(k, best, self.contract):
                write(self.root/'completed.json', dict(converged_under_predeclared_rule=True,
                    best=best, last_update=k, additional_updates=k-start,
                    training_episodes=sum(r['count'] for r in self.curve),
                    prior_training_episodes=getattr(self, 'prior_training_episodes', 0),
                    evaluated_updates=sorted(r['update'] for r in self.validations),
                    finished_utc=datetime.now(timezone.utc).isoformat()))
                return

    def interrupt(self, signum, frame):
        # Unwind Popen.wait before waiting again: its waitpid lock is not reentrant.
        raise InterruptedError('outer schedule interrupted')

    def stop_child(self):
        if self.child is not None:
            self.child.send_signal(signal.SIGTERM)
            self.child.wait(timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with (args.output/'schedule.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = Schedule(args.output)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, runner.interrupt)
        try:
            runner.run()
        except BaseException as exc:
            runner.stop_child()
            write(args.output/'interruption.json', dict(error=repr(exc), utc=datetime.now(timezone.utc).isoformat()))
            raise


if __name__ == '__main__':
    main()
