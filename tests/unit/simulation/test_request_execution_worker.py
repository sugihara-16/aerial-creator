from pathlib import Path
from types import SimpleNamespace

from amsrr.training.request_execution_worker import serve


def test_physical_cpu_affinity_keeps_siblings_together_and_respects_allowed_set(tmp_path):
    from amsrr.training.request_execution_worker import physical_cpu_groups
    for cpu, core in [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 3)]:
        directory = tmp_path/f'cpu{cpu}'/'topology'
        directory.mkdir(parents=True)
        (directory/'physical_package_id').write_text('0')
        (directory/'core_id').write_text(str(core))
    assert physical_cpu_groups(topology_root=tmp_path, allowed={0, 1, 2, 3, 4, 5}) == [[0, 1], [2, 3], [4], [5]]
    assert physical_cpu_groups(topology_root=tmp_path, allowed={1, 2, 3, 5}) == [[2, 3], [1], [5]]


def test_existing_receipt_prevents_sending_or_starting_work(tmp_path, monkeypatch):
    import pytest
    from amsrr.training.request_execution_worker import RequestExecutionWorker
    receipt = tmp_path / 'receipt.json'
    receipt.write_text('previous evidence')
    worker = RequestExecutionWorker(root=tmp_path, log=tmp_path / 'worker.log')
    monkeypatch.setattr(worker, '_start', lambda: pytest.fail('must not start work'))
    with pytest.raises(FileExistsError):
        worker.execute([], log=receipt, timeout=1)
    assert receipt.read_text() == 'previous evidence'


class Connection:
    def __init__(self, requests):
        self.requests = iter(requests)
        self.replies = []
        self.closed = False
    def recv(self):
        return next(self.requests)
    def send(self, result):
        self.replies.append(result)
    def close(self):
        self.closed = True


def test_completed_jobs_cancel_deadline_before_cleanup_and_receipt_delivery(tmp_path, monkeypatch):
    import gc
    import amsrr.training.request_execution_worker as worker
    armed = []
    monkeypatch.setattr(worker.signal, 'alarm', lambda seconds: armed.append(seconds))
    def collect():
        if armed[-1]:
            raise TimeoutError('request job deadline')
    monkeypatch.setattr(gc, 'collect', collect)
    closed = []
    args = SimpleNamespace(timeout_s=360, output=str(tmp_path), job=str(tmp_path/'job.json'))
    runner = SimpleNamespace(
        RequestIsaacSession=lambda: SimpleNamespace(close=lambda **kw: closed.append(kw)),
        sources=lambda: {}, parser=lambda: SimpleNamespace(parse_args=lambda _: args))
    def complete(name, code):
        assert armed[-1] == 360
        (tmp_path/name).write_text('{}')
        return code
    runner.prepare = lambda _: complete('planning_rejection.json', 2)
    runner.execute = lambda _, **kw: complete('result.json', 0)
    connection = Connection([dict(mode=mode, argv=[], timeout_s=360)
                             for mode in ('prepare', 'execute')] + [None])
    assert serve(connection, runner) == 0
    assert [row['exit_code'] for row in connection.replies] == [2, 0]
    assert closed == [dict(exit_code=0)]


def test_worker_separates_jobs_retains_one_session_and_checks_receipts(tmp_path, monkeypatch):
    import scripts.run_request_policy as runner
    closed, sessions = [], []
    monkeypatch.setattr(runner, 'RequestIsaacSession', lambda: SimpleNamespace(close=lambda **kw: closed.append(kw)))
    monkeypatch.setattr(runner, 'sources', lambda: {'code': 'unchanged'})
    requests = []
    for i in range(2):
        directory = tmp_path / str(i)
        directory.mkdir()
        requests.append(dict(argv=['--job', str(directory/'job.json')], timeout_s=30))
    def execute(args, *, session):
        sessions.append(session)
        (Path(args.job).parent/'result.json').write_text('{}')
        return 0 if len(sessions) == 1 else 2
    monkeypatch.setattr(runner, 'execute', execute)
    connection = Connection([*requests, None])
    assert serve(connection, runner) == 0
    assert connection.closed and sessions[0] is sessions[1]
    assert [r['exit_code'] for r in connection.replies] == [0, 2]
    assert closed == [{'exit_code': 0}]


def test_worker_fails_on_missing_result_instead_of_reporting_success(tmp_path, monkeypatch):
    import scripts.run_request_policy as runner
    monkeypatch.setattr(runner, 'RequestIsaacSession', lambda: SimpleNamespace(close=lambda **kw: None))
    monkeypatch.setattr(runner, 'sources', lambda: {})
    monkeypatch.setattr(runner, 'execute', lambda *args, **kw: 0)
    connection = Connection([dict(argv=['--job', str(tmp_path/'job.json')], timeout_s=30)])
    assert serve(connection, runner) == 1
    assert 'without a result receipt' in connection.replies[0]['error']


def test_force_close_during_child_creation_waits_for_child_identity(tmp_path, monkeypatch):
    import threading
    import pytest
    import amsrr.training.request_execution_worker as module
    started, release, closing = threading.Event(), threading.Event(), threading.Event()
    class Child:
        pid = 871234
        exit_code = None
        def poll(self): return self.exit_code
        def wait(self, timeout): self.exit_code = -15
    child = Child()
    def create(*args, **kwargs):
        started.set()
        assert release.wait(2.)
        return child
    killed = []
    monkeypatch.setattr(module.subprocess, "Popen", create)
    monkeypatch.setattr(module.os, "killpg", lambda *args: killed.append(args))
    worker = module.RequestExecutionWorker(root=tmp_path, log=tmp_path/'worker.log')
    errors = []
    def start():
        try: worker._start()
        except BaseException as e: errors.append(e)
    def close():
        closing.set()
        worker.close(force=True)
    a = threading.Thread(target=start); a.start(); assert started.wait(2.)
    b = threading.Thread(target=close); b.start(); assert closing.wait(2.)
    # Shutdown cannot return while Popen is still creating the owned child.
    b.join(.02); assert b.is_alive()
    release.set(); a.join(2.); b.join(2.)
    assert not a.is_alive() and not b.is_alive() and not errors
    assert killed == [(child.pid, module.signal.SIGTERM)]
    assert child.poll() is not None and worker._closed
    with pytest.raises(RuntimeError, match="closed"):
        worker._start()


def test_physics_error_audit_reads_only_current_job_and_rejects_capacity_loss(tmp_path):
    from amsrr.training.request_execution_worker import physics_log_errors
    old = b'[Error] [omni.physx.plugin] previous job\n'
    normal = b'[Warning] [omni.physx.plugin] TGS velocity iteration configuration\n'
    failure = b'[Error] [omni.physx.plugin] GPU buffer overflow, increase capacity\n'
    log = tmp_path / 'worker.log'
    log.write_bytes(old + normal + failure)
    assert not physics_log_errors(log, len(old), len(old + normal))
    errors = physics_log_errors(log, len(old), log.stat().st_size)
    assert len(errors) == 1 and 'overflow' in errors[0]


def test_ranked_prepare_uses_whole_ranking_transport_deadline(tmp_path, monkeypatch):
    import amsrr.training.request_execution_worker as worker
    armed=[]
    monkeypatch.setattr(worker.signal,'alarm',lambda seconds:armed.append(seconds))
    args=SimpleNamespace(timeout_s=360,output=str(tmp_path),job=str(tmp_path/'job.json'))
    runner=SimpleNamespace(RequestIsaacSession=lambda:SimpleNamespace(close=lambda **kw:None),
        sources=lambda:{},parser=lambda:SimpleNamespace(parse_args=lambda _:args))
    def prepare(_):
        assert armed[-1]==2910  # Eight candidates plus receipt allowance.
        assert args.timeout_s==360  # Individual solver budget is unchanged.
        (tmp_path/'planning_rejection.json').write_text('{}')
        return 2
    def execute(_,**kw):
        assert armed[-1]==360
        (tmp_path/'result.json').write_text('{}')
        return 0
    runner.prepare=prepare;runner.execute=execute
    connection=Connection([dict(mode=mode,argv=[],timeout_s=2910)
                           for mode in ('prepare','execute')]+[None])
    assert serve(connection,runner)==0
    assert [x['exit_code'] for x in connection.replies]==[2,0]


def test_prepare_can_cross_one_candidate_alarm_without_interrupting_ranking(tmp_path):
    import signal
    import time
    args=SimpleNamespace(timeout_s=1,output=str(tmp_path),job=str(tmp_path/'job.json'))
    runner=SimpleNamespace(RequestIsaacSession=lambda:SimpleNamespace(close=lambda **kw:None),
        sources=lambda:{},parser=lambda:SimpleNamespace(parse_args=lambda _:args))
    def prepare(_):
        time.sleep(1.1)  # Represents consecutive candidates, longer than one budget.
        (tmp_path/'planning_rejection.json').write_text('{}')
        return 2
    runner.prepare=prepare
    def deadline(*_):raise TimeoutError('wrong whole-ranking alarm')
    previous=signal.signal(signal.SIGALRM,deadline)
    try:
        connection=Connection([dict(mode='prepare',argv=[],timeout_s=5),None])
        assert serve(connection,runner)==0
        assert connection.replies[0]['exit_code']==2
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
