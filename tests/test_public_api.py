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


def test_normalize_and_strip_unifies_ampersand_and_word():
    """`&` -> 'and' (step 3) then 'and' stripped via List 2; both spelling
    variants and the no-connector form all canonicalize identically."""
    assert nc.normalize("Smith & Jones") == nc.normalize("Smith and Jones")
    assert nc.normalize("Smith and Jones") == nc.normalize("Smith Jones")
    assert nc.normalize("Procter & Gamble") == "procter gamble"
    assert nc.normalize("S&P 500") == "s p 500"


def test_normalize_abbrev_variants():
    """Manuf/Manufac/MFTG all reach the same canonical short form as Mfg."""
    base = nc.normalize("Acme Mfg Inc")
    for v in ["Manufacturing", "Manuf", "Manufac", "MFR", "MFTG"]:
        assert nc.normalize(f"Acme {v} Inc") == base, f"variant {v!r} drifted"
    # Service singular + abbrev variants -> svc
    base_svc = nc.normalize("Acme Services Inc")
    for v in ["Service", "Serv", "Ser", "Srv", "SRVC"]:
        assert nc.normalize(f"Acme {v} Inc") == base_svc, f"service variant {v!r} drifted"
    # New canonicals: management / information / department
    assert nc.normalize("Acme Management") == "acme mgmt"
    assert nc.normalize("Acme MGT") == "acme mgmt"
    assert nc.normalize("Acme Information") == "acme info"
    assert nc.normalize("Acme Department") == "acme dept"


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


def test_candidates_returns_scored_pairs():
    df = pl.DataFrame({
        "name": [
            "Foothill Industries",
            "Foothill Inds Limited",
            "Sherwin-Williams Co",
            "Sherwin Williams Company",
            "Apple Inc",
        ],
    })
    cand = nc.candidates(df, name_col="name", min_score=0.0)
    assert isinstance(cand, pl.DataFrame)
    assert set(cand.columns) == {"name_a", "name_b", "normalized_a", "normalized_b", "score"}
    assert cand.height >= 1
    foothill = cand.filter(
        pl.col("normalized_a").str.contains("foothill")
        & pl.col("normalized_b").str.contains("foothill")
    )
    assert foothill.height == 1
    assert foothill["score"][0] >= 0.7

    high = nc.candidates(df, name_col="name", min_score=0.99)
    assert high.height < cand.height


def test_aliases_force_merge_acronym_with_expansion():
    df = pl.DataFrame({
        "name": [
            "IBM Corp",
            "I.B.M. Inc",
            "International Business Machines",
            "International Business Machines Corporation",
            "Apple Inc",
        ],
    })
    r0 = nc.cluster(df, name_col="name")
    cids0 = r0["cluster_id"].to_list()
    assert cids0[0] != cids0[2], "without aliases, IBM and the expansion should NOT cluster"

    r1 = nc.cluster(
        df, name_col="name",
        aliases={"International Business Machines": ["IBM", "I.B.M."]},
    )
    cids1 = r1["cluster_id"].to_list()
    assert cids1[0] == cids1[1] == cids1[2] == cids1[3], (
        "all 4 IBM/expansion variants should share a cluster: " + repr(cids1)
    )
    assert cids1[4] != cids1[0], "Apple should remain a separate cluster"
    assert r1["canonical_name"][0] == "intl business machines"


def test_aliases_empty_dict_noop():
    """Passing aliases={} should be identical to passing aliases=None."""
    df = pl.DataFrame({"name": ["Acme Corp", "ACME Inc", "Apple Inc"]})
    a = nc.cluster(df, name_col="name", aliases=None)
    b = nc.cluster(df, name_col="name", aliases={})
    assert a["cluster_id"].to_list() == b["cluster_id"].to_list()


def test_parallel_matches_serial_byte_identical():
    """Cluster a non-trivial corpus across three thread counts; outputs must
    match byte-for-byte. Determinism is the contract; rayon's collect preserves
    order. Larger n + extra config increases the chance of catching contention
    races that wouldn't surface at low thread count."""
    ds = nc.generate_examples(n_entities=800, difficulty="hard", seed=7)
    baseline = nc.cluster(ds, name_col="variant_name", n_threads=1)
    base_ids = baseline["cluster_id"].to_pylist()
    base_canon = baseline["canonical_name"].to_pylist()
    for nt in (4, 8, 16):
        r = nc.cluster(ds, name_col="variant_name", n_threads=nt)
        assert r["cluster_id"].to_pylist() == base_ids, f"cluster_id drift at n_threads={nt}"
        assert r["canonical_name"].to_pylist() == base_canon, f"canonical drift at n_threads={nt}"


def test_acronym_map_finds_corpus_pairs():
    df = pl.DataFrame({
        "name": [
            "IBM Corp",
            "I.B.M. Inc",
            "International Business Machines",
            "AA Inc",
            "American Airlines",
            "NASA",
            "National Aeronautics and Space Administration",
            "Apple Inc",
            "Apple Computer Co.",
        ],
    })
    ac = nc.acronym_map(df, name_col="name")
    assert isinstance(ac, pl.DataFrame)
    assert set(ac.columns) == {"acronym", "expansion_count", "expansions", "acronym_examples"}

    acronyms = set(ac["acronym"].to_list())
    assert "ibm" in acronyms, f"ibm not found: {acronyms}"
    assert "aa" in acronyms, f"aa not found: {acronyms}"
    assert "nasa" in acronyms, f"nasa not found (stopword skip should help)"

    ibm_row = ac.filter(pl.col("acronym") == "ibm").row(0, named=True)
    assert ibm_row["expansion_count"] == 1
    assert "International Business Machines" in ibm_row["expansions"]
    assert any("IBM" in n for n in ibm_row["acronym_examples"])


def test_acronym_map_high_confidence_feeds_aliases():
    df = pl.DataFrame({
        "name": [
            "IBM Corp", "International Business Machines", "International Business Machines Inc",
            "Apple Inc", "Apple Computer Co.",
        ],
    })
    ac = nc.acronym_map(df, name_col="name")
    high = ac.filter(pl.col("expansion_count") == 1)
    aliases = {
        row["expansions"][0]: row["acronym_examples"]
        for row in high.iter_rows(named=True)
    }
    result = nc.cluster(df, name_col="name", aliases=aliases)
    cids = result["cluster_id"].to_list()
    assert cids[0] == cids[1] == cids[2], f"IBM/expansion should merge via derived aliases; got {cids}"


def test_aliases_duplicate_alias_resolution_is_deterministic():
    """When the same alias appears under two canonicals, the lex-greater
    canonical wins (build_alias_map sorts ascending and last-write wins)."""
    df = pl.DataFrame({"name": ["XX Inc"]})
    a = nc.cluster(
        df, name_col="name",
        aliases={"Alpha": ["XX"], "Bravo": ["XX"]},
    )
    b = nc.cluster(
        df, name_col="name",
        aliases={"Bravo": ["XX"], "Alpha": ["XX"]},
    )
    assert a["canonical_name"][0] == b["canonical_name"][0] == "bravo", (
        "lex-greater canonical (bravo > alpha) should win regardless of dict order; "
        f"got a={a['canonical_name'][0]!r} b={b['canonical_name'][0]!r}"
    )


def test_explain_returns_cluster_diagnostics():
    df = pl.DataFrame({
        "name": [
            "Foothill Industries",
            "Foothill Inds Limited",
            "Apple Inc",
        ],
    })
    result = nc.cluster(df, name_col="name", threshold=0.7)
    fcid = result.filter(pl.col("name") == "Foothill Industries")["cluster_id"][0]
    info = nc.explain(result, fcid)
    assert info["canonical"].startswith("foothill")
    assert info["size"] == 2
    assert info["hub_radius"] >= 1
    assert len(info["edges"]) == 1
    edge = info["edges"][0]
    assert edge[2] >= 0.7

    apple_cid = result.filter(pl.col("name") == "Apple Inc")["cluster_id"][0]
    info_singleton = nc.explain(result, apple_cid)
    assert info_singleton["size"] == 1
    assert info_singleton["edges"] == []
    assert info_singleton["hub_radius"] == 0

    with pytest.raises(ValueError):
        nc.explain(result, 99999)
