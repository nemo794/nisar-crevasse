"""Pickle-free checkpoint I/O: safetensors for every tensor, a JSON sidecar for
everything else. Two files share one stem: <path>.safetensors + <path>.json.

Drop-in replacement for `torch.save(ckpt_dict, path)` / `torch.load(path,
weights_only=False)` -- `load_checkpoint` reconstructs the exact original nested
dict/list shape and types (tensors stay tensors, numpy arrays stay numpy arrays with
their original dtype, plain scalars stay plain scalars), so no consumer code needs to
change beyond the load/save call itself. `weights_only=False` was needed everywhere in
this project because checkpoints bundle non-tensor Python objects (args dicts, channel
lists, mu/sd floats, optimizer state) alongside tensors -- that's exactly what makes
`torch.load` run arbitrary pickle deserialization. safetensors can't execute code on
load at all, since its file format is just a flat tensor namespace plus a JSON header.
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def _join(prefix, key):
    return str(key) if prefix == "" else f"{prefix}.{key}"


def _flatten(obj, prefix, tensors):
    if isinstance(obj, np.generic):          # numpy scalar (np.float64 etc) -> plain Python
        obj = obj.item()
    if torch.is_tensor(obj):
        tensors[prefix] = obj.detach().cpu().contiguous()
        return {"__type__": "tensor", "key": prefix}
    if isinstance(obj, np.ndarray):
        tensors[prefix] = torch.from_numpy(np.ascontiguousarray(obj))
        return {"__type__": "ndarray", "key": prefix, "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {"__type__": "dict", "items": {
            str(k): _flatten(v, _join(prefix, k), tensors) for k, v in obj.items()}}
    if isinstance(obj, (list, tuple)):
        return {"__type__": "list", "items": [
            _flatten(v, _join(prefix, i), tensors) for i, v in enumerate(obj)]}
    return {"__type__": "scalar", "value": obj}   # int/float/str/bool/None


def _unflatten(node, tensors):
    t = node["__type__"]
    if t == "tensor":
        return tensors[node["key"]]
    if t == "ndarray":
        return tensors[node["key"]].numpy().astype(node["dtype"])
    if t == "dict":
        # JSON object keys are always strings; recover int keys (e.g. optimizer
        # state's per-parameter-id keys, which torch's load_state_dict requires as
        # ints) heuristically -- every dict key in these checkpoints is either a
        # non-numeric name (model/arg keys) or a plain non-negative integer id, never
        # a string that merely looks numeric, so this round-trips correctly here.
        return {(int(k) if k.isdigit() else k): _unflatten(v, tensors)
                for k, v in node["items"].items()}
    if t == "list":
        return [_unflatten(v, tensors) for v in node["items"]]
    return node["value"]


def save_checkpoint(path, ckpt_dict):
    """`path` is the checkpoint's stem, no extension (e.g. "models/RUN/unet_best").
    Writes `<path>.safetensors` (every tensor/ndarray found anywhere in `ckpt_dict`,
    however deeply nested) and `<path>.json` (the rest -- a skeleton mirroring
    `ckpt_dict`'s exact shape, with tensor/ndarray leaves replaced by a small pointer
    back into the safetensors file)."""
    path = Path(path)
    tensors = {}
    skeleton = _flatten(ckpt_dict, "", tensors)
    save_file(tensors, str(path) + ".safetensors")
    path.with_suffix(".json").write_text(json.dumps(skeleton, indent=2))


def load_checkpoint(path, device="cpu"):
    """Returns the reconstructed dict, same nested shape/types as the original
    `ckpt_dict` passed to `save_checkpoint`. `device` may be a `torch.device`, an int,
    or a string -- safetensors' own `load_file` only accepts str/int, not a
    `torch.device` (what every consumer in this project already has on hand), so that
    one case is converted; a plain str/int is passed through untouched (stringifying an
    int device index, e.g. 0 -> "0", is NOT equivalent for safetensors)."""
    path = Path(path)
    if isinstance(device, torch.device):
        device = str(device)
    tensors = load_file(str(path) + ".safetensors", device=device)
    skeleton = json.loads(path.with_suffix(".json").read_text())
    return _unflatten(skeleton, tensors)
