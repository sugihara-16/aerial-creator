"""Replay pure tensor calculations with dataclass inputs and owned outputs.

Used by the rollout reward, phase gate and force estimator. Validation stays
outside capture. One graph per instance bounds cache memory; this object is
owned by a single rollout thread and must not be called concurrently.
"""
from dataclasses import fields, is_dataclass

import torch


def _flatten(value):
    tensors = []

    def visit(node):
        if isinstance(node, torch.Tensor):
            tensors.append(node)
            return ("tensor", len(tensors) - 1)
        if is_dataclass(node) and not isinstance(node, type):
            return ("dataclass", type(node), tuple((f.name, visit(getattr(node, f.name))) for f in fields(node)))
        if isinstance(node, dict):
            return ("dict", tuple((key, visit(item)) for key, item in node.items()))
        if isinstance(node, tuple):
            return ("tuple", tuple(visit(item) for item in node))
        if node is None:
            return ("none",)
        raise TypeError(f"unsupported CUDA graph leaf: {type(node).__name__}")

    spec = visit(value)
    return tensors, spec


def _unflatten(tensors, spec):
    kind = spec[0]
    if kind == "tensor":
        return tensors[spec[1]]
    if kind == "none":
        return None
    if kind == "tuple":
        return tuple(_unflatten(tensors, child) for child in spec[1])
    if kind == "dict":
        return {key: _unflatten(tensors, child) for key, child in spec[1]}
    return spec[1](**{key: _unflatten(tensors, child) for key, child in spec[2]})


class TensorDataclassGraph:
    def __init__(self):
        self._cache = None

    def call(self, function, args=(), kwargs=None, *, configuration):
        kwargs = {} if kwargs is None else kwargs
        inputs, structure = _flatten((args, kwargs))
        if (not inputs or inputs[0].device.type != "cuda"
                or any(t.device != inputs[0].device or t.requires_grad for t in inputs)):
            return function(*args, **kwargs)
        key = (structure, configuration, tuple((tuple(t.shape), t.dtype, t.device) for t in inputs))
        cached = self._cache
        if cached is None or cached[0] != key:
            with torch.no_grad():
                static = [t.detach().clone() for t in inputs]
                static_args, static_kwargs = _unflatten(static, structure)
                stream = torch.cuda.Stream(device=inputs[0].device)
                stream.wait_stream(torch.cuda.current_stream(inputs[0].device))
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        function(*static_args, **static_kwargs)
                torch.cuda.current_stream(inputs[0].device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    result = function(*static_args, **static_kwargs)
                outputs, output_structure = _flatten(result)
            cached = (key, static, graph, outputs, output_structure)
            self._cache = cached
        _, static, graph, outputs, output_structure = cached
        with torch.no_grad():
            for dest, value in zip(static, inputs):
                dest.copy_(value)
            graph.replay()
            # A later replay must not overwrite a retained reward or next_state.
            return _unflatten([t.clone() for t in outputs], output_structure)
