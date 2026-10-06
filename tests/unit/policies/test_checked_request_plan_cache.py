import json

import pytest

from amsrr.policies.checked_request_plan_cache import CheckedRequestPlanCache


def test_checked_plan_cache_binds_inputs_and_checks_content(tmp_path):
    identity = dict(snapshot='a', request={'group': 'b'}, source='c', mass=.3)
    cache = CheckedRequestPlanCache(tmp_path, identity)
    assert cache.load() is None
    payload = dict(status='accepted', plan={'trajectory': [1, 2, 3]})
    cache.store(payload)
    assert CheckedRequestPlanCache(tmp_path, identity).load() == payload
    for key, value in [('snapshot', 'd'), ('request', {'group': 'e'}), ('source', 'f'), ('mass', .4)]:
        assert CheckedRequestPlanCache(tmp_path, {**identity, key: value}).load() is None
    entry = json.loads(cache.path.read_text())
    entry['payload']['plan']['trajectory'][0] = 8
    cache.path.write_text(json.dumps(entry))
    with pytest.raises(ValueError, match='content changed'):
        cache.load()


def test_deadline_rejection_is_not_cached(tmp_path):
    cache = CheckedRequestPlanCache(tmp_path, dict(snapshot='a'))
    for reason in ('planning timeout', 'request deadline', 'planner timed out'):
        with pytest.raises(ValueError, match='deadlines'):
            cache.store(dict(status='rejected', reason=reason))
    assert cache.load() is None
    payload = dict(status='rejected', reason='joint limit along contact trajectory')
    cache.store(payload)
    assert cache.load() == payload
