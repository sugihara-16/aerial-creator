"""One sequential Isaac process for an on-policy collector's execution jobs.

PPO remains in the caller. Planning and execution share this worker process.
Each execution request still creates a fresh scene,
controller, RNG and rollout; only application startup is shared. The private
inherited pipe carries receipts separately from Isaac's stdout.
"""
from __future__ import annotations

import json
from multiprocessing import Pipe
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def physics_log_errors(path, start, end):
    """A successful process exit must not hide dropped PhysX contacts/pairs."""
    errors = []
    with Path(path).open('rb') as stream:
        stream.seek(start)
        for raw in iter(stream.readline, b''):
            if stream.tell() - len(raw) >= end:
                break
            line = raw.decode('utf-8', errors='replace').rstrip()
            lower = line.lower()
            if ('[error]' in lower and '[omni.physx' in lower
                    or 'physx' in lower and 'overflow' in lower
                    or 'gpu' in lower and 'capacity' in lower
                    and any(word in lower for word in ('insufficient', 'exceeded', 'increase'))):
                errors.append(line[:1000])
                if len(errors) == 4:
                    break
    return errors


def physical_cpu_groups(*, topology_root=Path('/sys/devices/system/cpu'), allowed=None):
    """Disjoint physical cores, retaining SMT siblings in the same worker.

    SMT-capable cores precede single-thread cores. On the current hybrid host
    these are the performance cores, which receive the largest scheduled jobs.
    Affinity does not alter the aggregate resource guard.
    """
    allowed = set(os.sched_getaffinity(0) if allowed is None else allowed)
    groups = {}
    for cpu in sorted(allowed):
        topology = topology_root / f'cpu{cpu}' / 'topology'
        key = (int((topology/'physical_package_id').read_text()),
               int((topology/'core_id').read_text()))
        groups.setdefault(key, []).append(cpu)
    return sorted(groups.values(), key=lambda cpus: (-len(cpus), cpus[0]))


class RequestExecutionWorker:
    def __init__(self, *, root, log, python=sys.executable, env=None, cpu_affinity=None):
        self.root = Path(root).resolve()
        self.log = Path(log).resolve()
        self.process = None
        self.connection = None
        self._lock = threading.Lock()
        # Forced shutdown must be able to interrupt an executing RPC, but it
        # must not race with Popen before the child identity is registered.
        self._lifecycle_lock = threading.Lock()
        self.python = python
        self.env = env
        self.cpu_affinity = None if cpu_affinity is None else tuple(cpu_affinity)
        if self.cpu_affinity is not None and (
                not self.cpu_affinity or len(set(self.cpu_affinity)) != len(self.cpu_affinity)
                or not set(self.cpu_affinity) <= set(os.sched_getaffinity(0))):
            raise ValueError('worker affinity must contain distinct available CPUs')
        self._stream = None
        self._closed = False

    def _start(self):
        with self._lifecycle_lock:
            self._start_locked()

    def _start_locked(self):
        if self._closed:
            raise RuntimeError('Isaac worker already closed')
        if self.process is not None:
            if self.process.poll() is not None:
                raise RuntimeError('Isaac worker exited; refusing an implicit restart')
            return
        parent, child = Pipe(duplex=True)
        self._stream = self.log.open('xb')
        try:
            command = [
                self.python, '-u', 'scripts/run_request_policy.py', 'execute-worker',
                '--connection-fd', str(child.fileno())]
            if self.cpu_affinity is not None:
                command = ['taskset', '--cpu-list', ','.join(map(str, self.cpu_affinity)), *command]
            self.process = subprocess.Popen(command, cwd=self.root, env=self.env,
                pass_fds=(child.fileno(),), stdout=self._stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            parent.close()
            self._stream.close()
            raise
        finally:
            child.close()
        self.connection = parent

    def execute(self, argv, *, log, timeout, allowed=(0, 2)):
        return self._run('execute', argv, log=log, timeout=timeout, allowed=allowed)

    def prepare(self, argv, *, log, timeout, allowed=(0, 2)):
        return self._run('prepare', argv, log=log, timeout=timeout, allowed=allowed)

    def _run(self, mode, argv, *, log, timeout, allowed):
        """Execute normal arguments and require the corresponding receipt."""
        # Reserve the receipt before sending any work. A stale destination
        # must fail without running a simulation whose result cannot be saved.
        with self._lock, Path(log).open('x') as stream:
            self._start()
            started = time.monotonic()
            offset = self.log.stat().st_size
            self.connection.send(dict(mode=mode, argv=list(argv), timeout_s=timeout))
            try:
                if not self.connection.poll(timeout):
                    raise TimeoutError('persistent Isaac execution deadline')
                reply = self.connection.recv()
            except BaseException:
                self.close(force=True)
                raise
            record = dict(mode=mode, argv=list(argv), worker_pid=self.process.pid,
                worker_log=str(self.log), log_offset_start=offset,
                log_offset_end=self.log.stat().st_size,
                seconds=time.monotonic()-started, **reply)
            if mode == 'execute':
                record['physics_errors'] = physics_log_errors(
                    self.log, offset, record['log_offset_end'])
            json.dump(record, stream, indent=2)
            if reply.get('error') or reply['exit_code'] not in allowed or record.get('physics_errors'):
                self.close(force=True)
                raise RuntimeError(f'Isaac worker failed: {record}')
            return record

    def close(self, *, force=False):
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            process = self.process
        if process is None:
            return
        if not force and process.poll() is None:
            try:
                self.connection.send(None)
                process.wait(timeout=15)
            except (OSError, EOFError, subprocess.TimeoutExpired):
                force = True
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        self.connection.close()
        self._stream.close()
        # Keep the exited process identity: accidental reuse fails closed.

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close(force=error[0] is not None)


def serve(connection, runner):
    """Run on the child's main thread, as required by SimulationApp."""
    import gc
    from amsrr.utils.hashing import hash_file
    session = runner.RequestIsaacSession()
    exit_code = 0
    initial_sources = runner.sources()
    try:
        while True:
            try:
                request = connection.recv()
            except EOFError:
                break
            if request is None:
                break
            try:
                if runner.sources() != initial_sources:
                    raise ValueError('runtime sources changed while Isaac was running')
                mode = request.get('mode', 'execute')
                if mode not in ('prepare', 'execute'):
                    raise ValueError('unsupported worker request')
                args = runner.parser().parse_args([mode, *request['argv']])
                # Preparation can try several candidates. args.timeout_s is
                # the individual planner budget; the collector supplies the
                # bounded whole-ranking RPC budget. Physics retains its own
                # execution deadline.
                wall_limit = int(request['timeout_s'])
                if mode == 'execute':
                    wall_limit = min(args.timeout_s, wall_limit)
                signal.alarm(max(1, wall_limit))
                if mode == 'execute':
                    code = runner.execute(args, session=session)
                    signal.alarm(0)
                    result = Path(args.job).parent / 'result.json'
                else:
                    code = runner.prepare(args)
                    signal.alarm(0)
                    result = Path(args.output) / ('job.json' if code == 0 else 'planning_rejection.json')
                # The job has returned. Receipt delivery and arena cleanup are
                # bounded by the parent's RPC timeout, not the job's alarm.
                if not result.exists():
                    raise RuntimeError('Isaac returned without a result receipt')
                gc.collect()
                # Return completed CPU rollout arenas before admitting another
                # scene. glibc otherwise retains the peak of the raw archive.
                import ctypes
                libc = ctypes.CDLL(None)
                if hasattr(libc, 'malloc_trim'):
                    libc.malloc_trim(0)
                import resource
                import torch
                memory = dict(max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                if torch.cuda.is_initialized():
                    memory.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                                  cuda_reserved_bytes=torch.cuda.memory_reserved())
                connection.send(dict(exit_code=code, result=str(result),
                    result_sha256=hash_file(result), memory=memory))
            except BaseException as error:
                exit_code = 1
                connection.send(dict(exit_code=1, error=repr(error)))
                break
            finally:
                signal.alarm(0)
    finally:
        connection.close()
        session.close(exit_code=exit_code)
    return exit_code
