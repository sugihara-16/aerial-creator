import pytest
from scripts.run_guarded_request import trip_reason

@pytest.mark.parametrize('field,value', [('cpu_c',85),('gpu_c',75),('gpu_memory_mib',20000),
    ('ram_available_gib',7.9),('disk_free_gib',39.9),('kernel_taint',128),('kernel_taint',16),
    ('no_turbo',0),('gpu_power_limit_w',500)])
def test_guard_retains_existing_machine_protection_thresholds(field,value):
    healthy=dict(cpu_c=45,gpu_c=50,gpu_memory_mib=8000,ram_available_gib=16,disk_free_gib=80,
                 kernel_taint=12289,no_turbo=1,gpu_power_limit_w=250)
    assert trip_reason(healthy) is None
    assert trip_reason({**healthy,field:value}) is not None
