import numpy as np
import pytest

from transport import const
from transport.data import check_headline_split, check_no_leak, make_shift
from transport.eval import ablate, evaluate, target_success_pct


def test_leak_injection_fails():
    split = make_shift(30, 30, seed=1)
    leaked = np.concatenate([split["source_ids"], split["target_ids"]])
    with pytest.raises(ValueError, match="leak"):
        check_no_leak(leaked, split["target_ids"])


def test_mixing_source_labels_into_target_score_fails():
    split = make_shift(24, 24, seed=2)
    pred_t = split["yt"].copy()
    mixed_pred = np.concatenate([pred_t, split["ys"]])
    mixed_y = np.concatenate([split["yt"], split["ys"]])
    mixed_ids = np.concatenate([split["target_ids"], split["source_ids"]])
    with pytest.raises(ValueError, match="leak"):
        target_success_pct(mixed_pred, mixed_y, mixed_ids, split["source_ids"])


def test_source_and_target_ids_are_disjoint():
    split = make_shift(40, 40, seed=0)
    check_no_leak(split["source_ids"], split["target_ids"])
    assert len(split["source_ids"]) == 40
    assert len(split["target_ids"]) == 40


def test_iid_shuffle_is_not_the_headline_split():
    split = make_shift(36, 36, seed=3)
    mixed_dom = np.concatenate([split["source_domain"], split["target_domain"]])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(mixed_dom))
    cut = len(mixed_dom) // 2
    with pytest.raises(ValueError, match="leak"):
        check_headline_split(mixed_dom[perm[:cut]], mixed_dom[perm[cut:]])


def test_caps_match_design():
    assert const.N_SOURCE <= const.MAX_SAMPLES
    assert const.N_TARGET <= const.MAX_SAMPLES
    assert const.MAX_SAMPLES <= 5000


def test_ot_beats_no_ot_on_target_and_clears_floor():
    m = evaluate(n_source=120, n_target=120, seed=0)
    assert m["success_pct"] == m["ot_pct"]
    assert m["n_source"] <= const.MAX_SAMPLES
    assert m["success_pct"] >= const.FLOOR_PCT
    assert m["success_pct"] > m["no_ot_pct"]
    assert m["no_ot_pct"] < 70
    assert m["split"] == "source_train_target_test"


def test_ablation_same_split_has_ot_and_no_ot():
    rows = ablate(n_source=60, n_target=60, seed=0)
    assert len(rows) >= 4
    costs = {r["cost"] for r in rows}
    epss = {r["eps"] for r in rows}
    assert "sqeuclidean" in costs
    assert const.EPS in epss
    for r in rows:
        assert "ot_pct" in r and "no_ot_pct" in r
        assert r["n"] <= const.MAX_SAMPLES
