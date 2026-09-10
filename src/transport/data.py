"""Source/target Gaussian shift. Headline split never mixes domains."""

from __future__ import annotations

import numpy as np

from . import const


def check_no_leak(source_ids, target_ids) -> None:
    overlap = set(np.asarray(source_ids).tolist()) & set(np.asarray(target_ids).tolist())
    if overlap:
        raise ValueError(f"leak: {len(overlap)} ids in both source and target")


def check_headline_split(train_domain, test_domain) -> None:
    if set(np.asarray(train_domain).tolist()) != {"source"}:
        raise ValueError("leak: headline train is not source-only")
    if set(np.asarray(test_domain).tolist()) != {"target"}:
        raise ValueError("leak: headline test is not target-only")


def make_shift(
    n_source: int = const.N_SOURCE,
    n_target: int = const.N_TARGET,
    seed: int = const.SEED,
) -> dict:
    if n_source > const.MAX_SAMPLES or n_target > const.MAX_SAMPLES:
        raise ValueError(f"cap: samples over {const.MAX_SAMPLES}")
    rng = np.random.default_rng(seed)
    means = np.array([[0.0, 0.0], [3.4, 0.2], [1.6, 2.9]])
    cov = np.array([[0.18, 0.02], [0.02, 0.18]])
    shift = np.array(const.SHIFT, dtype=float)

    def blobs(n, offset, id0):
        y = np.repeat(np.arange(const.N_CLASSES), n // const.N_CLASSES)
        extra = n - len(y)
        if extra:
            y = np.concatenate([y, np.arange(extra)])
        x = np.stack([rng.multivariate_normal(means[c] + offset, cov) for c in y])
        ids = np.arange(id0, id0 + n)
        return x, y, ids

    xs, ys, sid = blobs(n_source, np.zeros(2), 0)
    xt, yt, tid = blobs(n_target, shift, n_source)
    check_no_leak(sid, tid)
    src_dom = np.array(["source"] * n_source)
    tgt_dom = np.array(["target"] * n_target)
    check_headline_split(src_dom, tgt_dom)
    return {
        "xs": xs,
        "ys": ys,
        "xt": xt,
        "yt": yt,
        "source_ids": sid,
        "target_ids": tid,
        "source_domain": src_dom,
        "target_domain": tgt_dom,
        "n_classes": const.N_CLASSES,
    }
