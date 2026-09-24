"""Caching must preserve the selected planner and its timing contract."""
from types import SimpleNamespace
from copy import deepcopy
import pytest
from scripts import run_request_policy as runner


def test_reviewed_cache_cannot_replace_authored_tail_with_generic_tail(monkeypatch):
    request = SimpleNamespace(to_dict=lambda: {'contact_group_id': 'chosen'})
    context = SimpleNamespace(catalog=SimpleNamespace(snapshot_hash='observation'))
    reviewed = SimpleNamespace(phase_trajectories={'lift': {'horizon_s': 3.}})
    cache = {'provenance': {'observation_snapshot': 'observation',
             'request': request.to_dict(), 'teacher_geometry_reused': True},
             'phases': {'lift': {'horizon_s': 30.}}}
    def reviewed_route(actual_context, actual_request, actual_bundle):
        assert actual_context is context and actual_request is request
        assert actual_bundle is reviewed
        return deepcopy(reviewed)
    monkeypatch.setattr(runner, 'reuse_reviewed_request_geometry', reviewed_route)
    monkeypatch.setattr(runner, 'complete_request_phases',
                        lambda *_: pytest.fail('generic tail replaces reviewed timing'))
    restored = runner.restore_cached_request_geometry(cache, context, request, reviewed)
    assert restored.phase_trajectories == reviewed.phase_trajectories
    assert cache['phases']['lift']['horizon_s'] == 30.


@pytest.mark.parametrize('field', ['observation_snapshot', 'request'])
def test_geometry_cache_rejects_other_decision_before_restoring(field):
    request = SimpleNamespace(to_dict=lambda: {'contact_group_id': 'chosen'})
    context = SimpleNamespace(catalog=SimpleNamespace(snapshot_hash='observation'))
    provenance = {'observation_snapshot': 'observation', 'request': request.to_dict()}
    provenance[field] = 'different'
    with pytest.raises(ValueError, match='different request/initial observation'):
        runner.restore_cached_request_geometry({'provenance': provenance}, context, request, None)
