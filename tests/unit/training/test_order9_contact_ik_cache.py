from types import SimpleNamespace
from amsrr.training.order9_articulated_teacher import _cached_contact_ik


def test_cache_invalidates_seed_and_geometry_and_isolates_returned_values():
    class Solver:
        config = {'iterations': 60}
        calls = 0
        def solve(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(q=[kwargs['seed']], target=kwargs['target'])
    solver = Solver(); cache = {}
    first = _cached_contact_ik(solver, cache, seed=1., target=[0., 1.])
    first.q[0] = 999.
    second = _cached_contact_ik(solver, cache, seed=1., target=[0., 1.])
    assert solver.calls == 1 and second.q == [1.]
    _cached_contact_ik(solver, cache, seed=2., target=[0., 1.])
    _cached_contact_ik(solver, cache, seed=1., target=[0., 2.])
    solver.config = {'iterations': 120}
    _cached_contact_ik(solver, cache, seed=1., target=[0., 1.])
    assert solver.calls == 4
    for seed in range(30):
        _cached_contact_ik(solver, cache, seed=seed, target=[0., 1.])
    assert len(cache) == 16
