from __future__ import annotations

from amsrr.training.order9_runtime_load import Order9RuntimeLoadMonitor


def test_runtime_load_monitor_records_gpu_and_process_peaks() -> None:
    sample_index = 0

    def gpu_probe(index: int):
        nonlocal sample_index
        sample_index += 1
        return {
            "gpu_index": index,
            "gpu_uuid": "GPU-unit",
            "gpu_memory_total_mib": 24564.0,
            "gpu_memory_used_mib": 1000.0 + sample_index,
            "gpu_utilization_percent": 20.0 * sample_index,
            "gpu_memory_utilization_percent": 10.0,
            "gpu_power_draw_w": 80.0,
            "gpu_temperature_c": 55.0,
        }

    monitor = Order9RuntimeLoadMonitor(
        sample_interval_s=60.0,
        device="cuda:0",
        gpu_probe=gpu_probe,
        rss_probe=lambda: 512.0,
    )
    monitor.start()
    report = monitor.stop()

    assert report["gpu_monitor_available"] is True
    assert report["gpu_sample_count"] == 2
    assert report["gpu_memory_used_mib_peak"] == 1002.0
    assert report["gpu_utilization_percent_mean"] == 30.0
    assert report["process_rss_mib_peak"] == 512.0
    assert len(report["samples"]) == 2


def test_runtime_load_monitor_is_explicitly_unavailable_on_cpu() -> None:
    monitor = Order9RuntimeLoadMonitor(
        sample_interval_s=60.0,
        device="cpu",
        gpu_probe=lambda _index: {},
        rss_probe=lambda: 128.0,
    )
    monitor.start()
    report = monitor.stop()

    assert report["gpu_index"] is None
    assert report["gpu_monitor_available"] is False
    assert report["process_rss_mib_peak"] == 128.0
