"""Integration tests for the public Python API.

Run via: source .venv/bin/activate && pytest tests/test_public_api.py
(maturin develop --release must have been run first to install the rust ext).
"""
from __future__ import annotations

import name_cluster as nc
import polars as pl
import pyarrow as pa
import pytest


def test_normalize_basic():
    assert nc.normalize("Acme Corporation Inc") == "acme"
    assert nc.normalize("00 IBM") == "ibm"
    assert nc.normalize("") == ""


def test_cluster_names_dedups_acme():
    ids = nc.cluster_names(["Acme Corp", "ACME Inc", "Brightspoke"])
    assert ids[0] == ids[1]
    assert ids[0] != ids[2]


def test_cluster_names_handles_nulls():
    ids = nc.cluster_names(["Acme Corp", None, "", "Acme Inc"])
    assert ids[0] is not None
    assert ids[1] is None
    assert ids[2] is None
    assert ids[3] == ids[0]


def test_cluster_polars():
    df = pl.DataFrame({
        "name": ["Acme Corp", "ACME Inc", "Apple Inc", None],
    })
    result = nc.cluster(df, name_col="name")
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["name", "cluster_id", "canonical_name"]
    assert result["cluster_id"][0] == result["cluster_id"][1]
    assert result["cluster_id"][3] is None
    assert result["canonical_name"][0] == "acme"


def test_cluster_pyarrow():
    tbl = pa.table({"name": ["Acme Corp", "ACME Inc", "Apple Inc"]})
    result = nc.cluster(tbl, name_col="name")
    assert isinstance(result, pa.Table)
    cids = result["cluster_id"].to_pylist()
    assert cids[0] == cids[1]
    assert cids[0] != cids[2]


def test_cluster_threshold_changes_partition():
    ds = nc.generate_examples(n_entities=20, difficulty="medium", seed=0)
    strict = nc.cluster(ds, name_col="variant_name", threshold=0.99)
    loose = nc.cluster(ds, name_col="variant_name", threshold=0.5)
    n_strict = len(set(strict["cluster_id"].to_pylist()))
    n_loose = len(set(loose["cluster_id"].to_pylist()))
    assert n_strict >= n_loose, f"stricter threshold should leave >= clusters: strict={n_strict}, loose={n_loose}"


def test_cluster_deterministic():
    df = pl.DataFrame({"name": ["Acme Corp", "Beta Inc", "ACME Corp"]})
    a = nc.cluster(df, name_col="name", seed=42)
    b = nc.cluster(df, name_col="name", seed=42)
    assert a["cluster_id"].to_list() == b["cluster_id"].to_list()
    assert a["canonical_name"].to_list() == b["canonical_name"].to_list()


def test_lsh_calibrate_returns_sensible_config():
    cfg = nc.lsh_calibrate(target_jaccard=0.7, target_recall=0.95)
    assert cfg["bands"] >= 1
    assert cfg["rows"] >= 1
    assert cfg["bands"] * cfg["rows"] == cfg["num_perm"]
    assert cfg["p_at_target"] >= 0.95
    # FP rate should be lower than target rate (sharp transition)
    assert cfg["p_at_fp"] < cfg["p_at_target"]


def test_lsh_calibrate_validates_inputs():
    with pytest.raises(ValueError):
        nc.lsh_calibrate(target_jaccard=0.0)
    with pytest.raises(ValueError):
        nc.lsh_calibrate(target_jaccard=1.0)
    with pytest.raises(ValueError):
        nc.lsh_calibrate(target_jaccard=0.5, target_recall=1.5)


def test_generate_then_cluster_then_score_round_trip():
    """Easy round trip with non-wrapping pool size should achieve high ARI."""
    # Cap n_entities <= toy-canonical pool size so the generator doesn't wrap
    # (wrap appends " 2" disambig which can falsely cluster with originals).
    ds = nc.generate_examples(n_entities=40, difficulty="easy", seed=42)
    result = nc.cluster(ds, name_col="variant_name")
    metrics = nc.score_clusters(
        result["cluster_id"].to_pylist(),
        ds["true_entity_id"].to_pylist(),
    )
    assert metrics["recall"] >= 0.95, f"easy round-trip recall too low: {metrics}"
    assert metrics["adjusted_rand"] > 0.85, f"easy round-trip ARI too low: {metrics}"


def test_invalid_threshold_raises():
    df = pl.DataFrame({"name": ["X"]})
    with pytest.raises(ValueError):
        nc.cluster(df, name_col="name", threshold=1.5)
    with pytest.raises(ValueError):
        nc.cluster(df, name_col="name", threshold=-0.1)
