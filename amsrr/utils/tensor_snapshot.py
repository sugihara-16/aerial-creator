"""Owned CPU snapshots with one device transfer per tensor dtype/device.

For small batched runtime observations, scalar device reads are much more
expensive than their arithmetic. Keep decision logic on the host without a
separate device synchronization for every field or environment.
"""
from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace

import torch


def cpu_snapshot(value):
    groups = {}
    slots = []

    def visit(item):
        if isinstance(item, torch.Tensor):
            if item.device.type == "cpu":
                # Already on the host: packing adds a second copy and several
                # temporary views. Keep the same owned, contiguous snapshot.
                snapshot = item.detach().clone(memory_format=torch.contiguous_format)
                return lambda: snapshot
            index = len(slots)
            slots.append(None)
            groups.setdefault((item.device, item.dtype), []).append((index, item))
            return lambda: slots[index]
        if is_dataclass(item) and not isinstance(item, type):
            children = {f.name: visit(getattr(item, f.name)) for f in fields(item)}
            return lambda: replace(item, **{k: v() for k, v in children.items()})
        if isinstance(item, SimpleNamespace):
            children = {k: visit(v) for k, v in vars(item).items()}
            return lambda: SimpleNamespace(**{k: v() for k, v in children.items()})
        if isinstance(item, dict):
            children = {k: visit(v) for k, v in item.items()}
            return lambda: {k: v() for k, v in children.items()}
        if isinstance(item, (list, tuple)):
            children = [visit(v) for v in item]
            return lambda: type(item)(v() for v in children)
        return lambda: item

    rebuild = visit(value)
    for tensors in groups.values():
        packed = torch.cat([t.detach().reshape(-1) for _, t in tensors]).to(
            device="cpu", copy=True)
        offset = 0
        for index, tensor in tensors:
            count = tensor.numel()
            slots[index] = packed[offset:offset + count].reshape(tensor.shape)
            offset += count
    return rebuild()
