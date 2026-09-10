"""Sinkhorn OTDA. Score only on target. Never invent %."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import const
from .data import check_headline_split, check_no_leak, make_shift


def _cost(xs: np.ndarray, xt: np.ndarray, metric: str = "sqeuclidean") -> np.ndarray:
    if metric == "sqeuclidean":
        x2 = np.sum(xs * xs, axis=1, keepdims=True)
        t2 = np.sum(xt * xt, axis=1, keepdims=True).T
        c = np.maximum(x2 + t2 - 2.0 * xs @ xt.T, 0.0)
    elif metric == "euclidean":
        c = np.sqrt(_cost(xs, xt, "sqeuclidean"))
    else:
        xs_n = xs / np.clip(np.linalg.norm(xs, axis=1, keepdims=True), 1e-12, None)
        xt_n = xt / np.clip(np.linalg.norm(xt, axis=1, keepdims=True), 1e-12, None)
        c = np.maximum(1.0 - xs_n @ xt_n.T, 0.0)
    scale = float(np.mean(c)) or 1.0
    return c / scale


def sinkhorn(xs: np.ndarray, xt: np.ndarray, eps: float, metric: str = "sqeuclidean") -> np.ndarray:
    c = _cost(xs, xt, metric)
    a = np.full(len(xs), 1.0 / len(xs))
    b = np.full(len(xt), 1.0 / len(xt))
    try:
        import ot

        return np.asarray(ot.sinkhorn(a, b, c, reg=eps, numItermax=400), dtype=float)
    except Exception:
        # ponytail: numpy Sinkhorn if POT missing; same plan, same ε
        k = np.exp(-c / max(eps, 1e-8))
        u = np.ones_like(a)
        for _ in range(400):
            v = b / np.clip(k.T @ u, 1e-12, None)
            u = a / np.clip(k @ v, 1e-12, None)
        return u[:, None] * k * v[None, :]


def barycentric_map(gamma: np.ndarray, xt: np.ndarray) -> np.ndarray:
    mass = np.clip(gamma.sum(axis=1, keepdims=True), 1e-12, None)
    return (gamma @ xt) / mass


def _fit(x, y) -> LogisticRegression:
    clf = LogisticRegression(max_iter=400, solver="lbfgs")
    clf.fit(x, y)
    return clf


def _pct(pred, y) -> float:
    return round(100.0 * float(np.mean(pred == y)), 2)


def _cloud(xs, ys, xt, yt, n: int = 48) -> dict:
    if xs.shape[1] < 2:
        return {"source": [], "target": []}
    rng = np.random.default_rng(0)
    i_s = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    i_t = rng.choice(len(xt), size=min(n, len(xt)), replace=False)
    pack = lambda x, y, idx: [
        {"x": float(x[i, 0]), "y": float(x[i, 1]), "c": int(y[i])} for i in idx
    ]
    return {"source": pack(xs, ys, i_s), "target": pack(xt, yt, i_t)}


def target_success_pct(pred, y_target, target_ids, source_ids) -> float:
    check_no_leak(source_ids, target_ids)
    if len(pred) != len(y_target) or len(y_target) != len(target_ids):
        raise ValueError("leak: score length is not target-only")
    return _pct(pred, y_target)


def evaluate(
    n_source: int = const.N_SOURCE,
    n_target: int = const.N_TARGET,
    seed: int = const.SEED,
    eps: float = const.EPS,
    metric: str = "sqeuclidean",
    pca_dim: int | None = None,
    feature_dim: int | None = None,
) -> dict:
    split = make_shift(n_source, n_target, seed)
    xs, ys, xt, yt = split["xs"], split["ys"], split["xt"], split["yt"]
    sid, tid = split["source_ids"], split["target_ids"]
    check_no_leak(sid, tid)
    check_headline_split(split["source_domain"], split["target_domain"])

    if feature_dim is not None and feature_dim > xs.shape[1]:
        rng = np.random.default_rng(seed + 17)
        pad_s = rng.normal(0, 0.05, size=(len(xs), feature_dim - xs.shape[1]))
        pad_t = rng.normal(0, 0.05, size=(len(xt), feature_dim - xt.shape[1]))
        xs, xt = np.hstack([xs, pad_s]), np.hstack([xt, pad_t])

    if pca_dim is not None and pca_dim < xs.shape[1]:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=pca_dim, random_state=seed).fit(xs)
        xs, xt = pca.transform(xs), pca.transform(xt)

    clf_base = _fit(xs, ys)
    no_ot = target_success_pct(clf_base.predict(xt), yt, tid, sid)

    gamma = sinkhorn(xs, xt, eps, metric)
    xs_map = barycentric_map(gamma, xt)
    clf_ot = _fit(xs_map, ys)
    ot_pct = target_success_pct(clf_ot.predict(xt), yt, tid, sid)

    counts = np.bincount(ys, minlength=split["n_classes"])
    majority = _pct(np.full(len(yt), int(np.argmax(counts))), yt)
    chance = round(100.0 / split["n_classes"], 2)
    return {
        "success_pct": ot_pct,
        "ot_pct": ot_pct,
        "no_ot_pct": no_ot,
        "majority_pct": majority,
        "chance_pct": chance,
        "n_source": int(n_source),
        "n_target": int(n_target),
        "n_classes": int(split["n_classes"]),
        "eps": float(eps),
        "cost": metric,
        "feature_dim": int(xs.shape[1]),
        "seed": int(seed),
        "illustrative": False,
        "blocker": None,
        "split": "source_train_target_test",
        "cloud": _cloud(xs, ys, xt, yt),
    }


def ablate(
    n_source: int = const.N_SOURCE,
    n_target: int = const.N_TARGET,
    seed: int = const.SEED,
) -> list[dict]:
    rows = []
    knobs = [
        {"eps": 0.02, "metric": "sqeuclidean", "pca_dim": None, "feature_dim": None, "n": n_source},
        {"eps": const.EPS, "metric": "sqeuclidean", "pca_dim": None, "feature_dim": None, "n": n_source},
        {"eps": 0.4, "metric": "sqeuclidean", "pca_dim": None, "feature_dim": None, "n": n_source},
        {"eps": const.EPS, "metric": "euclidean", "pca_dim": None, "feature_dim": None, "n": n_source},
        {"eps": const.EPS, "metric": "cosine", "pca_dim": None, "feature_dim": None, "n": n_source},
        {"eps": const.EPS, "metric": "sqeuclidean", "pca_dim": 2, "feature_dim": 8, "n": n_source},
        {"eps": const.EPS, "metric": "sqeuclidean", "pca_dim": None, "feature_dim": None, "n": min(100, n_source)},
    ]
    for k in knobs:
        m = evaluate(
            n_source=k["n"],
            n_target=k["n"],
            seed=seed,
            eps=k["eps"],
            metric=k["metric"],
            pca_dim=k["pca_dim"],
            feature_dim=k["feature_dim"],
        )
        rows.append(
            {
                "eps": k["eps"],
                "cost": k["metric"],
                "feature_dim": m["feature_dim"],
                "n": k["n"],
                "ot_pct": m["ot_pct"],
                "no_ot_pct": m["no_ot_pct"],
                "success_pct": m["success_pct"],
            }
        )
    return rows


def print_report(m: dict) -> None:
    tag = "ILLUSTRATIVE" if m.get("illustrative") else "measured"
    print(
        f"OT success_pct={m['success_pct']:.1f}  no-OT success_pct={m['no_ot_pct']:.1f}  "
        f"majority_pct={m['majority_pct']:.1f}  chance_pct={m['chance_pct']:.1f}  [{tag}]"
    )


def write_metrics(m: dict, path: str = const.METRICS_PATH) -> None:
    Path(path).write_text(json.dumps(m, indent=2), encoding="utf-8")


def write_data_ts(m: dict, path: str = const.METRICS_GEN) -> None:
    p = Path(path)
    if not p.parent.exists():
        return
    ablations = m.get("ablations") or []
    cloud = m.get("cloud") or {"source": [], "target": []}
    body = (
        "// Generated by scripts/run.py. Numbers match metrics.json. Do not hand-edit.\n\n"
        f"export const ILLUSTRATIVE = {str(bool(m.get('illustrative'))).lower()}\n\n"
        "export const PROTOCOL = {\n"
        f"  nSource: {int(m['n_source'])},\n"
        f"  nTarget: {int(m['n_target'])},\n"
        f"  nClasses: {int(m['n_classes'])},\n"
        f"  eps: {float(m['eps'])},\n"
        f"  cost: {m['cost']!r},\n"
        f"  featureDim: {int(m['feature_dim'])},\n"
        f"  split: {m['split']!r},\n"
        f"  floorPct: {const.FLOOR_PCT},\n"
        "}\n\n"
        "export const METRICS = {\n"
        f"  successPct: {float(m['success_pct'])},\n"
        f"  otPct: {float(m['ot_pct'])},\n"
        f"  noOtPct: {float(m['no_ot_pct'])},\n"
        f"  majorityPct: {float(m['majority_pct'])},\n"
        f"  chancePct: {float(m['chance_pct'])},\n"
        "}\n\n"
        f"export const ABLATION: {{ eps: number; cost: string; feature_dim: number; n: number; ot_pct: number; no_ot_pct: number; success_pct: number }}[] = {json.dumps(ablations, indent=2)}\n\n"
        "export type CloudPt = { x: number; y: number; c: number }\n"
        f"export const CLOUD: {{ source: CloudPt[]; target: CloudPt[] }} = {json.dumps(cloud)}\n"
    )
    p.write_text(body, encoding="utf-8")
    print(f"wrote {path}")
