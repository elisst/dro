"""
read and write saved fits for the fast experiment
"""

import hashlib
import json

import h5py
import numpy as np


def json_value(value):
    """convert NumPy values and infinity to portable JSON values"""
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def config_fingerprint(config):
    """map from config to hash fingerprint for folder naming"""
    settings = {
        key: value
        for key, value in vars(config).items()
        if key not in ("n", "p", "k", "fit_min_n")
    }
    encoded = json.dumps(json_value(settings), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:12]


def run_metadata(config, n, p, repetition):
    """record the current settings for one dataset and fit"""
    settings = {**vars(config), "n": n, "p": p, "k": repetition}
    settings.pop("fit_min_n")
    return settings


def save_run(path, row, metadata):
    """save one fit and its settings"""

    # finish the write before making the result available for reuse
    temporary = path.with_suffix(".tmp.h5")
    with h5py.File(temporary, "w") as file:
        for key, value in row.items():
            file.create_dataset(key, data=value)
        file.attrs["metadata"] = json.dumps(json_value(metadata))
        file.attrs["completed"] = True
    temporary.replace(path)


def load_run(path, metadata):
    """read one fit and check that it belongs to the requested setup"""
    with h5py.File(path, "r") as file:
        if not file.attrs.get("completed", False):
            raise ValueError(f"incomplete fit: {path}")
        if json.loads(file.attrs["metadata"]) != json_value(metadata):
            raise ValueError(f"configuration mismatch: {path}")
        row = {key: file[key][()] for key in file}
    return {
        key: value.decode() if isinstance(value, bytes) else value
        for key, value in row.items()
    }
