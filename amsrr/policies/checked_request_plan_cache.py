"""Content-bound reuse of completed deterministic planning, never rollouts.

Actor inference and log probabilities are always recomputed by the caller.
Changing observations, requests, physics, configuration or implementation
invalidates an entry. A deadline is not a geometric rejection and is not cached.
"""
import json
import fcntl
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile

from amsrr.utils.hashing import stable_hash


class CheckedRequestPlanCache:
    def __init__(self, root, identity):
        self.root = Path(root)
        self.identity = identity
        self.key = stable_hash(identity)
        self.path = self.root / (self.key + '.json')

    @contextmanager
    def locked(self):
        """Only one worker solves a given deterministic candidate at a time."""
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / (self.key + '.lock')).open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def load(self):
        if not self.path.exists():
            return None
        entry = json.loads(self.path.read_text())
        if entry['identity'] != self.identity or stable_hash(entry['payload']) != entry['payload_sha256']:
            raise ValueError('checked plan cache identity/content changed')
        return entry['payload']

    def store(self, payload):
        if payload['status'] not in ('accepted', 'rejected'):
            raise ValueError('only completed planner results can be cached')
        if payload['status'] == 'rejected' and any(
                word in payload['reason'].lower() for word in ('timeout', 'timed out', 'deadline')):
            raise ValueError('planner deadlines cannot be cached as geometric rejection')
        self.root.mkdir(parents=True, exist_ok=True)
        entry = dict(identity=self.identity, payload=payload, payload_sha256=stable_hash(payload))
        fd, name = tempfile.mkstemp(dir=self.root, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as handle:
                json.dump(entry, handle, separators=(',', ':'))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.path)
        finally:
            Path(name).unlink(missing_ok=True)
