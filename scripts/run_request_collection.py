"""Collect request-policy PPO data using bounded, persistent Isaac workers.

An optimizer is invoked only with --ppo-output, for one complete training panel.
No automatic training restart is performed.
The existing protocol supplies conditions, sampling seeds and episode counts.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import importlib.util
import json
import os
import signal
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from amsrr.training.request_collection import execute_panel, ppo_update_command
from amsrr.training.request_execution_worker import RequestExecutionWorker
from amsrr.training.request_imitation import load_episode
from amsrr.training.request_object_conditions import expand_box_episode
from amsrr.utils.hashing import hash_file, stable_hash
from scripts.run_request_policy import sources


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


class CollectionInterrupted(RuntimeError):
    pass


def release_worker_affinity(worker, allowed):
    """Allow an owned worker to use idle cores after the work queue drains."""
    with worker._lifecycle_lock:
        if worker._closed:
            return
        worker.cpu_affinity = None
        process = worker.process
        if process is None or process.poll() is not None:
            return
        # Change the main thread first so subsequently created threads inherit
        # the wider set. Existing physics/tasking threads also need updating.
        try:
            os.sched_setaffinity(process.pid, allowed)
            for task in (Path('/proc')/str(process.pid)/'task').iterdir():
                try:
                    os.sched_setaffinity(int(task.name), allowed)
                except ProcessLookupError:
                    pass
        except (ProcessLookupError, FileNotFoundError):
            if process.poll() is None:
                raise


def interrupt_collection(*_):
    # selectors treats InterruptedError as a retryable syscall interruption.
    # A distinct exception must escape Connection.poll and close the worker.
    raise CollectionInterrupted('collection interrupted')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--update-index', type=int, help='Collection policy update; omission selects greedy validation')
    p.add_argument('--case', action='append', help='Diagnostic subset only; the binding records this restriction')
    p.add_argument('--geometry-root', type=Path, action='append', default=[])
    p.add_argument('--audit', type=Path, help='Optional archived audit for historical comparison; default uses the equivalent batch audit')
    p.add_argument('--benchmark', action='store_true', help='Exclude all collected data from PPO/acceptance')
    p.add_argument('--max-environments', type=int, default=16)
    p.add_argument('--workers', type=int, default=1, help='Persistent simulation processes (1..24), under an aggregate host guard')
    p.add_argument('--prepare-workers', type=int, help='Complete planning before physics, using 1..16 temporary workers')
    p.add_argument('--initial-workers', type=int, default=0,
                   help='Separate CPU processes for initial observation/draw work (0..4)')
    p.add_argument('--recycle-after-jobs', type=int, default=0, help='Recycle a physics process after this many jobs (0 keeps it for the panel)')
    p.add_argument('--worker-threads', type=int, default=2)
    p.add_argument('--physics-threads', type=int, choices=(0, 2), default=2)
    p.add_argument('--cpu-broadphase', choices=('MBP', 'GPU'), default='MBP')
    p.add_argument('--cpu-log-storage', choices=('ram', 'file'), default='ram')
    p.add_argument('--pin-workers', nargs='?', const='physical', choices=['physical', 'performance', 'adaptive'],
                   help='Pin every worker to a physical core, or reserve SMT performance cores for the largest jobs')
    p.add_argument('--device', choices=['cpu','cuda:0'], default='cuda:0')
    p.add_argument('--diagnostic-steps', type=int, help='Short preflight only; requires --benchmark and is excluded from learning')
    p.add_argument('--execute-timeout-s', type=int, help='Wall budget only; task deadlines are unchanged')
    p.add_argument('--ppo-output', type=Path, help='Explicitly perform one PPO update after a complete training collection')
    p.add_argument('--ppo-timeout-s', type=int, default=120)
    p.add_argument('--ppo-device', choices=['cpu', 'cuda:0'], default='cpu')
    args = p.parse_args()
    if args.ppo_output is not None:
        if args.update_index is None or args.case or args.benchmark or args.diagnostic_steps is not None:
            p.error('--ppo-output requires the complete training panel without diagnostics')
        if args.ppo_output.exists() or not 1 <= args.ppo_timeout_s <= 600:
            p.error('PPO output must be fresh, with a bounded optimizer timeout')
    if not 1 <= args.max_environments <= 16:
        p.error('validated environment batch range is 1..16')
    if not 1 <= args.workers <= 24:
        p.error('bounded worker count is 1..24')
    if not 0 <= args.recycle_after_jobs <= 32:
        p.error('worker recycle interval must be 0..32')
    if not 1 <= args.worker_threads <= 4:
        p.error('CPU worker thread count must be 1..4')
    if args.prepare_workers is not None and not 1 <= args.prepare_workers <= 16:
        p.error('planning worker count must be 1..16')
    if not 0 <= args.initial_workers <= 4 or (args.initial_workers and args.prepare_workers is None):
        p.error('initial worker count must be0..4, with staged preparation')
    if args.workers > 1 and args.device != 'cpu':
        p.error('multiple processes currently require CPU physics')
    if args.cpu_broadphase == 'GPU' and args.device != 'cpu':
        p.error('--cpu-broadphase GPU requires CPU dynamics')
    if args.cpu_log_storage == 'file' and args.device != 'cpu':
        p.error('--cpu-log-storage file requires CPU dynamics')
    if args.workers > 1 or (args.prepare_workers or 1) > 1 or args.initial_workers:
        from scripts.run_guarded_request import require_bounded_worker_environment
        require_bounded_worker_environment()
    affinity_groups = None
    if args.pin_workers:
        if args.device != 'cpu' or args.prepare_workers is None:
            p.error('--pin-workers requires CPU physics and staged --prepare-workers')
        from amsrr.training.request_execution_worker import physical_cpu_groups
        affinity_groups = physical_cpu_groups()
        if args.workers > len(affinity_groups):
            p.error('worker count exceeds available physical CPU cores')
        affinity_groups = affinity_groups[:args.workers]
        if args.pin_workers == 'performance':
            # Keep expensive jobs on the SMT-capable performance cores. The
            # remaining workers may use spare cores and migrate to a freed
            # performance core instead of leaving it idle at the panel tail.
            affinity_groups = [cpus if len(cpus) > 1 else None for cpus in affinity_groups]
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())
    protocol.update(maximum_training_environments=args.max_environments,
                    maximum_evaluation_environments=args.max_environments)
    if args.execute_timeout_s is not None:
        if not 1 <= args.execute_timeout_s <= 3600:
            p.error('invalid execution wall budget')
        protocol['execute_wallclock_timeout_s'] = args.execute_timeout_s
    if args.diagnostic_steps is not None:
        if not args.benchmark or not 1 <= args.diagnostic_steps <= 1000:
            p.error('diagnostic steps require --benchmark and a bound of 1..1000')
        protocol['diagnostic_rollout_steps'] = args.diagnostic_steps
    cases = protocol['evaluation_cases' if args.update_index is None else 'training_cases']
    if args.case:
        selected = set(args.case)
        cases = [c for c in cases if c['bucket_id'] in selected]
        if {c['bucket_id'] for c in cases} != selected:
            p.error('unknown or wrong-split case')
    current = sources()
    for path in ['amsrr/training/request_collection.py', 'amsrr/training/request_initial_draws.py',
                 'scripts/run_guarded_request.py', __file__]:
        current[str(Path(path).resolve())] = hash_file(path)
    audit_path = args.audit or REPO/'amsrr/training/request_motion_audit.py'
    current[str(audit_path.resolve())] = hash_file(audit_path)
    binding = dict(source_protocol=str(args.protocol.resolve()),
        source_protocol_sha256=hash_file(args.protocol), checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hash_file(args.checkpoint), runtime_sources=current,
        grasp_anchor_source=protocol.get('grasp_anchor_source', 'all_free'),
        diagnostic_steps=args.diagnostic_steps, execute_timeout_s=protocol.get("execute_wallclock_timeout_s",360), maximum_environments=args.max_environments, workers=args.workers, worker_threads=args.worker_threads, device=args.device, collection_policy_update=args.update_index,
        case_ids=[c['bucket_id'] for c in cases], diagnostic_subset=bool(args.case), benchmark=bool(args.benchmark or args.case),
        physical_outcomes_reused=False, optimizer_requested=args.ppo_output is not None,
        optimizer_output=str(args.ppo_output.resolve()) if args.ppo_output is not None else None,
        optimizer_invoked=False, optimizer_device=args.ppo_device,
        prepare_workers=args.prepare_workers,
        initial_workers=args.initial_workers,
        worker_cpu_affinity=affinity_groups,
        worker_affinity_strategy=args.pin_workers,
        physics_threads=args.physics_threads, cpu_broadphase=args.cpu_broadphase,
        cpu_log_storage=args.cpu_log_storage,
        recycle_after_jobs=args.recycle_after_jobs)
    binding_path = output/'collection_binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError('collection binding changed; use a fresh output directory')
    write(binding_path, binding)

    def verify():
        for path, digest in current.items():
            if hash_file(path) != digest:
                raise ValueError('runtime changed: ' + path)
        if hash_file(args.protocol) != binding['source_protocol_sha256']:
            raise ValueError('source protocol changed')
        for path, digest in protocol['configuration_hashes'].items():
            if hash_file(path) != digest:
                raise ValueError('configuration changed: ' + path)
        if hash_file(REPO/'configs/training/order9_learning_curriculum.yaml') != protocol['config_sha256']:
            raise ValueError('curriculum changed')
        import shutil
        available = next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
        if available < 8*1024**3 or shutil.disk_usage(output).free < 40*1024**3:
            raise RuntimeError('host resource floor reached')

    if args.audit:
        spec = importlib.util.spec_from_file_location('physical_motion_audit', args.audit)
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        audit_one, audit_batch = audit.audit, None
    else:
        from amsrr.training.request_motion_audit import audit_all
        audit_one, audit_batch = None, audit_all
    geometry = {}
    for root in args.geometry_root:
        for path in sorted(root.glob('**/geometry.json')):
            if not (path.parent/'collection.json').exists():
                continue
            data = json.loads(path.read_text())
            provenance = data['provenance']
            if provenance.get('contact_goal_initialization_retry', {}).get('version') == 'bounded_selected_contact_goal_v1':
                continue
            geometry[(provenance['observation_snapshot'], stable_hash(provenance['request']))] = path

    def load_condition(case, physical):
        verify()
        if hash_file(case['condition']) != case['condition_sha256']:
            raise ValueError('object condition changed')
        original = load_episode(dict(path=case['record'], sha256=case['record_sha256']), physical)
        return expand_box_episode(original, json.loads(Path(case['condition']).read_text()), physical)[0]

    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        AMSRR_CPU_ROLLOUT_STORAGE=args.cpu_log_storage,
        AMSRR_ISAAC_PHYSX_THREADS=str(args.physics_threads),
        PXR_WORK_THREAD_LIMIT=str(args.worker_threads), AMSRR_ISAAC_THREADS=str(args.worker_threads))
    import threading
    local = threading.local()
    worker_list = []
    execution_slots = []
    workers_lock = threading.Lock()
    collection_cancelled = threading.Event()
    affinity_released = threading.Event()
    allowed_cpus = set(os.sched_getaffinity(0))
    physics_started = 0
    physics_total = 0
    initial_pool = None
    if args.initial_workers:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        initial_pool = ProcessPoolExecutor(max_workers=args.initial_workers,
            mp_context=multiprocessing.get_context('spawn'))
    def get_worker(mode):
        if collection_cancelled.is_set():
            raise RuntimeError('collection cancelled')
        if not hasattr(local, 'worker') or local.worker._closed:
            with workers_lock:
                if collection_cancelled.is_set():
                    raise RuntimeError('collection cancelled')
                if mode == 'execute' and affinity_groups is not None and not hasattr(local, 'affinity'):
                    local.affinity = affinity_groups[len(execution_slots)]
                    execution_slots.append(local.affinity)
                local.worker = RequestExecutionWorker(root=REPO,
                    log=output/f'isaac_worker_{len(worker_list)}_{time.time_ns()}.log', env=env,
                    cpu_affinity=None if affinity_released.is_set() else getattr(local, 'affinity', None))
                worker_list.append(local.worker)
                local.executed_jobs = 0
        return local.worker
    def close_workers(*, force=False):
        if force:
            collection_cancelled.set()
        with workers_lock:
            closing = list(worker_list)
        for worker in closing:
            worker.close(force=force)
    def command(argv, log, timeout, allowed=(0,)):
        nonlocal physics_started
        worker = get_worker('execute' if 'execute' in argv else 'prepare')
        if 'execute' in argv:
            releasing = []
            if args.pin_workers == 'adaptive':
                with workers_lock:
                    physics_started += 1
                    if physics_started == physics_total:
                        affinity_released.set()
                        releasing = list(worker_list)
                for owned_worker in releasing:
                    release_worker_affinity(owned_worker, allowed_cpus)
                if releasing:
                    write(output/'affinity_release.json', dict(
                        reason='all physics jobs have started',
                        started_jobs=physics_started, total_jobs=physics_total,
                        allowed_cpus=sorted(allowed_cpus), wall_time=time.time()))
            arguments = argv[argv.index('execute')+1:]+['--device',args.device,
                '--cpu-broadphase', args.cpu_broadphase]
            record = worker.execute(arguments, log=log, timeout=timeout, allowed=allowed)
            local.executed_jobs += 1
            if args.recycle_after_jobs and local.executed_jobs >= args.recycle_after_jobs:
                worker.close()
        elif 'prepare' in argv:
            record = worker.prepare(argv[argv.index('prepare')+1:], log=log, timeout=timeout, allowed=allowed)
        else:
            raise ValueError('collection only supports prepare/execute')
        with workers_lock:
            with (output/'commands.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
        return record
    def run_physics_batch(commands, folder):
        nonlocal physics_total, initial_pool
        if initial_pool is not None:
            initial_pool.shutdown(wait=True, cancel_futures=True)
            initial_pool = None
        verify()
        physics_total = len(commands)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def estimated_work(item):
            argv = item[0]
            job = Path(argv[argv.index('--job')+1])
            plan = json.loads((job.parent/'plan.json').read_text())
            environments = int(argv[argv.index('--num-envs')+1])
            duration = sum(float(trajectory['horizon_s'])
                           for trajectory in plan['phase_trajectories'].values())
            return (environments + 4) * duration
        # Run larger jobs first to avoid a long final worker tail. Only
        # scheduling changes: every seed, step and archive remains intact.
        commands = sorted(commands, key=estimated_work, reverse=True)
        write(folder/'physics_schedule.json', [dict(
            job=argv[argv.index('--job')+1], estimated_work=estimated_work(item))
            for item in commands for argv in [item[0]]])
        pool = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix='request-physics')
        try:
            futures = [pool.submit(command, argv, log, timeout, allowed)
                       for argv, log, timeout, allowed in commands]
            for index, future in enumerate(as_completed(futures)):
                receipt = future.result()
                print(json.dumps(dict(stage='physics', completed_jobs=index+1,
                    total_jobs=len(futures), seconds=receipt['seconds'])), flush=True)
        except BaseException:
            close_workers(force=True)
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    runtime = SimpleNamespace(root=output, repo=REPO, python=sys.executable,
        protocol=protocol, benchmark=bool(args.benchmark or args.case), load_episode=load_condition, geometry_index=lambda:dict(geometry),
        sha=hash_file, write=write, command=command, verify=verify, audit=audit_one, audit_batch=audit_batch,
        stage_physics=bool(args.prepare_workers is not None),
        run_physics_batch=run_physics_batch, close_workers=close_workers, emit=lambda value: print(json.dumps(value), flush=True))
    runtime.initial_draw_pool = initial_pool
    runtime.initial_draw_sources = current
    from concurrent.futures import ThreadPoolExecutor
    preparing_workers = args.prepare_workers or args.workers
    executor = ThreadPoolExecutor(max_workers=preparing_workers, thread_name_prefix='request-prepare') if preparing_workers > 1 else None
    runtime.group_executor = executor
    started = time.monotonic()
    try:
        report, archives = execute_panel(runtime, 'panel', cases, [args.checkpoint.resolve()],
            update_index=args.update_index, greedy=args.update_index is None)
    except BaseException:
        close_workers(force=True)
        raise
    else:
        close_workers()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if initial_pool is not None:
            initial_pool.shutdown(wait=True, cancel_futures=True)
    write(output/'timing.json', dict(seconds=time.monotonic()-started,
        independent_trials=sum(r['count'] for r in report.values()), archives=archives))
    if args.ppo_output is not None:
        import subprocess
        verify()
        command = ppo_update_command(python=sys.executable, checkpoint=args.checkpoint.resolve(),
            archives=archives, output=args.ppo_output.resolve(), manifest=output/'panel/contact_groups.json',
            protocol=protocol)
        command += ['--device', args.ppo_device]
        optimization_started = time.monotonic()
        write(output/'optimizer_command.json', dict(argv=command, timeout_s=args.ppo_timeout_s))
        write(binding_path, {**binding, 'optimizer_invoked': True})
        with (output/'optimizer.log').open('x') as log:
            subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=args.ppo_timeout_s, check=True)
        result = json.loads((args.ppo_output/'result.json').read_text())
        if (result['optimization']['optimizer_updates_resumed'] != args.update_index
                or result['optimization'].get('completed_actor_steps', result['optimization']['completed_epochs']) <= 0
                or result['episodes'] != len(cases)*protocol['training_draws_per_case']):
            raise ValueError('PPO update receipt does not match the collected panel')
        write(output/'optimizer_completed.json', dict(seconds=time.monotonic()-optimization_started,
            total_seconds=time.monotonic()-started, checkpoint=str(args.ppo_output/'checkpoint.pt'),
            checkpoint_sha256=hash_file(args.ppo_output/'checkpoint.pt'), update_index=args.update_index+1))
    return 0


if __name__ == '__main__':
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupt_collection)
    raise SystemExit(main())
