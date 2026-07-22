from __future__ import annotations

"""Low-overhead runtime telemetry for Order 9 collection and PPO updates."""

import math
import os
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any


ORDER9_RUNTIME_LOAD_VERSION = "order9_runtime_load_v2_system"

_GPU_FIELDS = (
    "gpu_index",
    "gpu_uuid",
    "gpu_memory_total_mib",
    "gpu_memory_used_mib",
    "gpu_utilization_percent",
    "gpu_memory_utilization_percent",
    "gpu_power_draw_w",
    "gpu_temperature_c",
)
_NVIDIA_QUERY_FIELDS = (
    "index",
    "uuid",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "temperature.gpu",
)
_SYSTEM_FIELDS = (
    "system_load_1m",
    "system_load_5m",
    "system_load_15m",
    "system_load_per_cpu_1m",
    "system_memory_used_mib",
    "system_memory_available_mib",
)


class Order9RuntimeLoadMonitor:
    """Sample global GPU load and process RSS without depending on NVML Python."""

    def __init__(
        self,
        *,
        sample_interval_s: float = 1.0,
        device: str = "cuda:0",
        gpu_probe: Callable[[int], Mapping[str, Any]] | None = None,
        rss_probe: Callable[[], float] | None = None,
        system_probe: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        if not math.isfinite(sample_interval_s) or sample_interval_s <= 0.0:
            raise ValueError("Order9 runtime-load sample interval must be positive")
        self.sample_interval_s = float(sample_interval_s)
        self.device = str(device)
        self.gpu_index = _gpu_index(self.device)
        self._gpu_probe = gpu_probe or _nvidia_smi_probe
        self._rss_probe = rss_probe or _process_rss_mib
        self._system_probe = system_probe or _system_load
        self._started_at: float | None = None
        self._samples: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, *, torch_module: Any | None = None) -> None:
        if self._started_at is not None:
            raise RuntimeError("Order9 runtime-load monitor is already running")
        self._started_at = time.perf_counter()
        self._reset_torch_peaks(torch_module)
        self._sample()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="order9-runtime-load",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, torch_module: Any | None = None) -> dict[str, Any]:
        if self._started_at is None:
            raise RuntimeError("Order9 runtime-load monitor was not started")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, 2.0 * self.sample_interval_s))
        self._sample()
        elapsed = time.perf_counter() - self._started_at
        with self._lock:
            samples = [dict(sample) for sample in self._samples]
            errors = list(self._errors)
        numeric = {
            field: [
                float(sample[field])
                for sample in samples
                if isinstance(sample.get(field), (int, float))
                and not isinstance(sample.get(field), bool)
                and math.isfinite(float(sample[field]))
            ]
            for field in (*_GPU_FIELDS[2:], "process_rss_mib", *_SYSTEM_FIELDS)
        }
        summary: dict[str, Any] = {
            "telemetry_version": ORDER9_RUNTIME_LOAD_VERSION,
            "device": self.device,
            "gpu_index": self.gpu_index,
            "sample_interval_s": self.sample_interval_s,
            "wall_elapsed_s": elapsed,
            "sample_count": len(samples),
            "gpu_sample_count": len(numeric["gpu_memory_used_mib"]),
            "gpu_monitor_available": bool(numeric["gpu_memory_used_mib"]),
            "probe_error_count": len(errors),
            "probe_errors": errors[-5:],
            "samples": samples,
        }
        for field, values in numeric.items():
            if not values:
                continue
            summary[f"{field}_mean"] = sum(values) / len(values)
            summary[f"{field}_peak"] = max(values)
        summary.update(self._torch_peaks(torch_module))
        return summary

    def latest_sample(self) -> dict[str, Any]:
        """Return a copy of the newest sample for live observers."""

        if self._started_at is None:
            raise RuntimeError("Order9 runtime-load monitor was not started")
        with self._lock:
            return dict(self._samples[-1]) if self._samples else {}

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval_s):
            self._sample()

    def _sample(self) -> None:
        if self._started_at is None:
            return
        sample: dict[str, Any] = {
            "elapsed_s": time.perf_counter() - self._started_at,
        }
        try:
            sample["process_rss_mib"] = float(self._rss_probe())
        except Exception as error:  # telemetry must not stop a training run
            self._record_error("rss", error)
        try:
            sample.update(dict(self._system_probe()))
        except Exception as error:  # telemetry must not stop a training run
            self._record_error("system", error)
        if self.gpu_index is not None:
            try:
                sample.update(dict(self._gpu_probe(self.gpu_index)))
            except Exception as error:  # telemetry must not stop a training run
                self._record_error("nvidia-smi", error)
        with self._lock:
            self._samples.append(sample)

    def _record_error(self, source: str, error: Exception) -> None:
        message = f"{source}:{type(error).__name__}:{error}"
        with self._lock:
            if not self._errors or self._errors[-1] != message:
                self._errors.append(message)

    def _reset_torch_peaks(self, torch_module: Any | None) -> None:
        if torch_module is None or self.gpu_index is None:
            return
        try:
            if torch_module.cuda.is_available():
                torch_module.cuda.reset_peak_memory_stats(self.gpu_index)
        except Exception as error:
            self._record_error("torch_reset_peak", error)

    def _torch_peaks(self, torch_module: Any | None) -> dict[str, float]:
        if torch_module is None or self.gpu_index is None:
            return {}
        try:
            if not torch_module.cuda.is_available():
                return {}
            divisor = 1024.0 * 1024.0
            return {
                "torch_cuda_memory_allocated_peak_mib": (
                    torch_module.cuda.max_memory_allocated(self.gpu_index) / divisor
                ),
                "torch_cuda_memory_reserved_peak_mib": (
                    torch_module.cuda.max_memory_reserved(self.gpu_index) / divisor
                ),
            }
        except Exception as error:
            self._record_error("torch_read_peak", error)
            return {}


def _gpu_index(device: str) -> int | None:
    if not device.startswith("cuda"):
        return None
    _, separator, raw_index = device.partition(":")
    if not separator:
        return 0
    try:
        index = int(raw_index)
    except ValueError:
        return None
    return index if index >= 0 else None


def _nvidia_smi_probe(gpu_index: int) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=" + ",".join(_NVIDIA_QUERY_FIELDS),
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu_index),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("nvidia-smi returned an unexpected GPU row count")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != len(_GPU_FIELDS):
        raise RuntimeError("nvidia-smi returned an unexpected GPU field count")
    output: dict[str, Any] = {}
    for name, value in zip(_GPU_FIELDS, values):
        if name == "gpu_uuid":
            output[name] = value
        elif name == "gpu_index":
            output[name] = int(value)
        else:
            output[name] = _finite_float(value, name)
    return output


def _finite_float(value: str, name: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise RuntimeError(f"nvidia-smi field {name} is unavailable: {value}") from error
    if not math.isfinite(result):
        raise RuntimeError(f"nvidia-smi field {name} is non-finite")
    return result


def _process_rss_mib() -> float:
    status = f"/proc/{os.getpid()}/status"
    with open(status, encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                kib = float(line.split()[1])
                return kib / 1024.0
    raise RuntimeError("VmRSS is absent from proc status")


def _system_load() -> dict[str, float]:
    load_1m, load_5m, load_15m = os.getloadavg()
    values: dict[str, float] = {}
    with open("/proc/meminfo", encoding="utf-8") as stream:
        for line in stream:
            key, separator, raw = line.partition(":")
            if not separator or key not in {"MemTotal", "MemAvailable"}:
                continue
            values[key] = float(raw.split()[0]) / 1024.0
    if set(values) != {"MemTotal", "MemAvailable"}:
        raise RuntimeError("system memory totals are absent from proc meminfo")
    cpu_count = max(1, int(os.cpu_count() or 1))
    return {
        "system_load_1m": float(load_1m),
        "system_load_5m": float(load_5m),
        "system_load_15m": float(load_15m),
        "system_load_per_cpu_1m": float(load_1m) / cpu_count,
        "system_memory_used_mib": values["MemTotal"] - values["MemAvailable"],
        "system_memory_available_mib": values["MemAvailable"],
    }


__all__ = ["ORDER9_RUNTIME_LOAD_VERSION", "Order9RuntimeLoadMonitor"]
