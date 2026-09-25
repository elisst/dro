"""
tests for saved eta and CVX comparisons

check that paired fits use the same datasets and speedups use total times
"""

import argparse

import h5py
import pytest

from DRO.experiments.fast import rate
from DRO.experiments.fast.eta_cvx_diag import load_pair, comparison_summary


@pytest.fixture
def paired(tmp_path, monkeypatch):
    config = rate.default_config()
    config.n, config.p, config.k = [16], [2.0], 2
    config.d, config.s = 3, 2
    eta_config = argparse.Namespace(**vars(config), solver="eta")
    folders = {"cvx": tmp_path / "fast", "eta": tmp_path / "fast_eta"}
    monkeypatch.setattr(
        rate, "default_directory",
        lambda cfg, solver=None: folders[solver or getattr(cfg, "solver", "cvx")],
    )
    # simulate a baseline made under an older seed rule
    with monkeypatch.context() as historical:
        historical.setattr(rate, "dataset_seed", lambda n, p, rep: 900 + rep)
        rate.run(config)
    rate.run(eta_config)
    return folders, eta_config


def test_reuses_historical_seeds(paired):
    folders, _ = paired
    _, cvx, eta = load_pair(folders["cvx"], folders["eta"])
    assert [row["seed"] for row in cvx[2.0][0]] == [901, 902]
    assert [row["seed"] for row in eta[2.0][0]] == [901, 902]


def test_speedup_is_ratio_of_total_times(paired):
    folders, _ = paired
    config, cvx, eta = load_pair(folders["cvx"], folders["eta"])
    for row, seconds in zip(cvx[2.0][0], [1.0, 9.0]):
        row["fit_seconds"] = seconds
    for row, seconds in zip(eta[2.0][0], [1.0, 3.0]):
        row["fit_seconds"] = seconds
    assert comparison_summary(config, cvx, eta)["2"]["speedup"] == 2.5


def test_rejects_mismatched_saved_seeds(paired):
    folders, eta_config = paired
    path = next(folders["eta"].glob("*.h5"))
    with h5py.File(path, "r+") as file:
        file["seed"][()] = 999999
    with pytest.raises(ValueError, match="seed mismatch"):
        load_pair(folders["cvx"], folders["eta"])
    with pytest.raises(ValueError, match="seed mismatch"):
        rate.run(eta_config, read_only=True)
