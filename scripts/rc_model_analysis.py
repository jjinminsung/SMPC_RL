#!/usr/bin/env python3
"""RC(위험 가치망) 모델 검증: 에피소드 단위 K-fold 교차검증으로 '학습에 안 쓴 에피소드'에 대한 예측만 모아 평가한다.

    python3 scripts/rc_model_analysis.py --data 'results/rc/iter0/data_s*.npz' --out results/rc/analysis_iter0

비교 대상(모두 같은 fold):
  const      학습 fold 의 나쁜 결과 비율(정보 없음)
  peak       2 s 트리의 지평선 안 최대 위험(점수 그대로, 학습 없음) — 트리가 이미 아는 것
  speed      -자차 속도(점수 그대로) — '서 있으면 위험' 만으로 얼마나 맞히나
  logreg     모든 특징의 선형 로지스틱 회귀(L2) — 비선형이 필요한가
  RC         MLP 앙상블(rl/risk_critic.RCNet, 에피소드 부트스트랩) — 실제로 쓰는 모델
지표: AUC, 평균 정밀도(AP), 로그 손실, 브라이어, 보정 오차(ECE). 95 % 신뢰구간 = 에피소드 부트스트랩.
정지(v < 1 m/s)·주행(v ≥ 1 m/s) 안에서 따로 잰 AUC: 판단은 같은 상황 안에서 '지금 바꿀지'를 가르므로 이것이 중요하다.
특징 묶음별 순열 중요도(AUC 감소), 학습 곡선(학습 에피소드 25/50/100 %), 문턱 θ 별 막는 비율·잡는 비율.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from smpc_sim.rl.features2 import FEATURE_NAMES  # noqa: E402
from smpc_sim.rl.train_rc import fit_net, predict  # noqa: E402

NAMES = list(FEATURE_NAMES) + ["logpeak", "connector"]
GROUPS = {"ego_speed": [0], "ego_lateral": [1], "lane_id": [2, 3, 4], "lane_end": [5, 6, 7],
          "cur_lane_veh": [8, 9, 10, 11], "tgt_lane_veh": [12, 13, 14, 15], "nxt_lane_veh": [16, 17, 18, 19],
          "peak": [20], "connector": [21]}


def auc(y, p):
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def avg_precision(y, p):
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-np.asarray(p), kind="mergesort")
    ys = y[o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float(prec[ys].mean())


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, bins=(0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.01)):
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def metrics(y, p):
    return {"auc": auc(y, p), "ap": avg_precision(y, p), "logloss": logloss(y, p) if np.all((p >= 0) & (p <= 1)) else None,
            "brier": float(np.mean((p - y) ** 2)) if np.all((p >= 0) & (p <= 1)) else None,
            "ece": ece(y, p) if np.all((p >= 0) & (p <= 1)) else None}


def fit_logreg(X, y, l2=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32))
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.tensor(float(np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6))), requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")
    lossf = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = lossf(Xt @ w + b, yt) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    W, B = w.detach().numpy(), float(b.detach())

    def f(Z):
        return 1.0 / (1.0 + np.exp(-(((Z - mu) / sd) @ W + B)))
    f.coef = W / sd
    return f


def fit_rc(X, y, E, seed, M=5, hidden=64):
    """train_rc 와 같은 방식: 학습 에피소드 15 % 로 조기 종료, 나머지로 에피소드 부트스트랩 M 개."""
    rng = np.random.default_rng(seed)
    eps = np.unique(E)
    es = set(rng.choice(eps, max(1, int(0.15 * len(eps))), replace=False).tolist())
    m_es = np.array([e in es for e in E])
    tr_eps = np.unique(E[~m_es])
    nets = []
    for m in range(M):
        bs = set(rng.choice(tr_eps, len(tr_eps), replace=True).tolist())
        sel = (~m_es) & np.array([e in bs for e in E])
        nets.append(fit_net(X[sel], y[sel], X[m_es], y[m_es], seed * 100 + m, hidden=hidden))
    return lambda Z: predict(nets, Z.astype(np.float32))


def boot_ci(y, preds, E, n=300, seed=0):
    rng = np.random.default_rng(seed)
    eps = np.unique(E)
    idx_of = {e: np.nonzero(E == e)[0] for e in eps}
    out = {k: {"auc": [], "ap": []} for k in preds}
    diff = {"RC-logreg": [], "RC-peak": []}
    for _ in range(n):
        ii = np.concatenate([idx_of[e] for e in rng.choice(eps, len(eps), replace=True)])
        a = {}
        for k, p in preds.items():
            a[k] = auc(y[ii], p[ii])
            out[k]["auc"].append(a[k])
            out[k]["ap"].append(avg_precision(y[ii], p[ii]))
        diff["RC-logreg"].append(a["RC"] - a["logreg"])
        diff["RC-peak"].append(a["RC"] - a["peak"])
    ci = {k: {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in d.items()} for k, d in out.items()}
    ci["diff_auc"] = {k: [float(np.percentile(v, 2.5)), float(np.mean(v)), float(np.percentile(v, 97.5))] for k, v in diff.items()}
    return ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="glob")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--no_curve", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    fs = sorted(glob.glob(a.data))
    D = [np.load(f) for f in fs]
    X = np.concatenate([d["X"] for d in D]).astype(np.float32)
    y = np.concatenate([d["y"] for d in D]).astype(np.int64)
    E = np.concatenate([d["ep"] for d in D])
    why = np.concatenate([d["why"] for d in D])
    v = X[:, 0] * 25.0
    stand = v < 1.0
    print(f"files {len(fs)}  samples {len(y)}  bad {y.sum()} ({100 * y.mean():.1f} %)  episodes {len(np.unique(E))}")
    rng = np.random.default_rng(0)
    eps = np.unique(E)
    fold_of = dict(zip(rng.permutation(eps), np.arange(len(eps)) % a.folds))
    F = np.array([fold_of[e] for e in E])
    oof = {k: np.zeros(len(y)) for k in ("const", "peak", "speed", "logreg", "RC")}
    perm_drop = {g: [] for g in GROUPS}
    curve = {0.25: np.zeros(len(y)), 0.5: np.zeros(len(y))}
    coefs = []
    for k in range(a.folds):
        te, tr = F == k, F != k
        oof["const"][te] = y[tr].mean()
        oof["peak"][te] = X[te, 20]
        oof["speed"][te] = -v[te]
        lr = fit_logreg(X[tr], y[tr])
        coefs.append(lr.coef)
        oof["logreg"][te] = lr(X[te])
        rc = fit_rc(X[tr], y[tr], E[tr], seed=k + 1, M=a.M, hidden=a.hidden)
        oof["RC"][te] = rc(X[te])
        base = auc(y[te], oof["RC"][te])
        for g, cols in GROUPS.items():         # 순열 중요도: 이 묶음을 행끼리 섞으면 AUC 가 얼마나 떨어지나
            Xp = X[te].copy()
            Xp[:, cols] = Xp[rng.permutation(te.sum())][:, cols]
            perm_drop[g].append(base - auc(y[te], rc(Xp)))
        if not a.no_curve:
            tr_eps = np.unique(E[tr])
            for frac in curve:
                sub = set(rng.choice(tr_eps, int(frac * len(tr_eps)), replace=False).tolist())
                m = tr & np.array([e in sub for e in E])
                curve[frac][te] = fit_rc(X[m], y[m], E[m], seed=50 + k, M=a.M, hidden=a.hidden)(X[te])
        print(f"fold {k}: AUC RC {base:.3f}  logreg {auc(y[te], oof['logreg'][te]):.3f}  peak {auc(y[te], X[te, 20]):.3f}")

    R = {"n": int(len(y)), "bad": int(y.sum()), "episodes": int(len(eps)), "why": {w: int((why == w).sum()) for w in set(why.tolist())},
         "overall": {k: metrics(y, p) for k, p in oof.items()},
         "standstill": {"n": int(stand.sum()), "bad_rate": float(y[stand].mean()),
                        **{k: auc(y[stand], p[stand]) for k, p in oof.items() if k != "const"}},
         "moving": {"n": int((~stand).sum()), "bad_rate": float(y[~stand].mean()),
                    **{k: auc(y[~stand], p[~stand]) for k, p in oof.items() if k != "const"}},
         "perm_importance_auc_drop": {g: [float(np.mean(d)), float(np.std(d))] for g, d in perm_drop.items()},
         "logreg_coef_mean": dict(zip(NAMES, np.mean(coefs, axis=0).round(4).tolist()))}
    R["ci"] = boot_ci(y, {k: oof[k] for k in ("peak", "speed", "logreg", "RC")}, E)
    if not a.no_curve:
        R["learning_curve_auc"] = {"0.25": auc(y, curve[0.25]), "0.5": auc(y, curve[0.5]), "1.0": R["overall"]["RC"]["auc"]}
    ops = []
    for th in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        veto = oof["RC"] > th
        ops.append({"theta": th, "veto_%": 100 * float(veto.mean()), "bad_caught_%": 100 * float(veto[y == 1].mean()),
                    "good_blocked_%": 100 * float(veto[y == 0].mean()),
                    "moving_veto_%": 100 * float(veto[~stand].mean()), "moving_bad_caught_%": 100 * float(veto[~stand & (y == 1)].mean()),
                    "stand_veto_%": 100 * float(veto[stand].mean()), "stand_bad_caught_%": 100 * float(veto[stand & (y == 1)].mean())})
    R["operating_points"] = ops
    cal = []
    for lo, hi in zip((0, 0.02, 0.05, 0.1, 0.2, 0.4), (0.02, 0.05, 0.1, 0.2, 0.4, 1.01)):
        m = (oof["RC"] >= lo) & (oof["RC"] < hi)
        if m.any():
            cal.append([lo, hi, int(m.sum()), float(oof["RC"][m].mean()), float(y[m].mean())])
    R["calibration"] = cal
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(out / "report.json", "w"), indent=1, ensure_ascii=False)
    np.savez(out / "oof.npz", y=y, E=E, **{k: p for k, p in oof.items()})
    print(json.dumps({k: R[k] for k in ("overall", "standstill", "moving")}, indent=1, ensure_ascii=False))
    print("CI", json.dumps(R["ci"]["diff_auc"]), "curve", R.get("learning_curve_auc"))
    print("perm", {g: round(d[0], 3) for g, d in R["perm_importance_auc_drop"].items()})
    for o in ops:
        print(o)


if __name__ == "__main__":
    main()
