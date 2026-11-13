"""RC(위험 가치망) 학습: collect_rc 가 모은 (특징, 나쁜 결과 0/1) 로 Q_c(s, 지금 바꾸기) = P(나쁜 결과) 를 배운다.

    cd sim
    python3 -m smpc_sim.rl.train_rc --data ../results/rc/iter0/data_s*.npz --out ../results/rc/RC_iter0.pt

몬테카를로 정책 평가: 리턴(이 차선 변경이 나쁘게 끝났나)이 0/1 이라 가치 회귀 = 이진 분류(교차 엔트로피).
앙상블 M 개: 에피소드 단위 부트스트랩 + 초기값이 다르다. 검증 = 에피소드 15 %(학습에 안 쓴 에피소드)로
조기 종료하고, 로그 손실·AUC·브라이어 점수를 상수 예측과 '지평선 안 위험(peak)만 쓰는 로지스틱 회귀'와 비교한다.
peak 만으로도 안다면 RC 가 배운 것이 없다는 뜻이다.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .risk_critic import DIM, RCNet, save_rc


def auc(y, p):
    """순위 기반 AUC(동점은 평균 순위)."""
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(p)
    r = np.empty(len(p))
    r[order] = np.arange(1, len(p) + 1)
    ps = np.asarray(p)[order]
    i = 0
    while i < len(ps):            # 동점 평균 순위
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_net(Xtr, ytr, Xva, yva, seed, hidden=64, epochs=400, lr=1e-3, wd=1e-4, patience=30, cols=None):
    torch.manual_seed(seed)
    dim = Xtr.shape[1]
    net = RCNet(dim=dim, hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))
    xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva.astype(np.float32))
    rng = np.random.default_rng(seed)
    best, best_sd, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(xt))
        for k in range(0, len(perm), 256):
            b = perm[k:k + 256]
            opt.zero_grad()
            lossf(net(xt[b]), yt[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(lossf(net(xv), yv))
        if v < best - 1e-5:
            best, best_sd, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_sd)
    net.eval()
    return net


def predict(nets, X):
    with torch.no_grad():
        return torch.sigmoid(torch.stack([n(torch.from_numpy(X)) for n in nets])).numpy().mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(2)
    Xs, ys, Es = [], [], []
    for f in a.data:
        d = np.load(f)
        Xs.append(d["X"]); ys.append(d["y"]); Es.append(d["ep"])
    X, y, E = np.concatenate(Xs).astype(np.float32), np.concatenate(ys).astype(np.int64), np.concatenate(Es)
    assert X.shape[1] == DIM, (X.shape, DIM)
    rng = np.random.default_rng(a.seed)
    eps = np.unique(E)
    va_eps = set(rng.choice(eps, int(len(eps) * a.val_frac), replace=False).tolist())
    va = np.array([e in va_eps for e in E])
    Xtr, ytr, Xva, yva = X[~va], y[~va], X[va], y[va]
    print(f"samples {len(y)} (bad {y.sum()}, {100 * y.mean():.1f} %), train {len(ytr)}, val {len(yva)} (bad {yva.sum()})")

    # 기준: 상수(정보 없음), 지평선 안 위험 peak 를 점수 그대로(학습 없음, AUC 만 의미 있음).
    # 자세한 비교(로지스틱 회귀, 교차검증, 신뢰구간)는 scripts/rc_model_analysis.py.
    p_const = np.full(len(yva), ytr.mean())
    p_peak = Xva[:, DIM - 2]

    tr_eps = np.unique(E[~va])
    nets = []
    for m in range(a.M):
        bs = set(rng.choice(tr_eps, len(tr_eps), replace=True).tolist())     # 에피소드 부트스트랩
        sel = np.array([e in bs for e in E[~va]])
        nets.append(fit_net(Xtr[sel], ytr[sel], Xva, yva, a.seed + 1 + m, hidden=a.hidden))
    p_rc = predict(nets, Xva)
    res = {}
    for k, p in (("const", p_const), ("RC", p_rc)):
        res[k] = {"logloss": logloss(yva, p), "auc": auc(yva, p), "brier": float(np.mean((p - yva) ** 2))}
        print(f"{k:10s} logloss {res[k]['logloss']:.4f}  AUC {res[k]['auc']:.3f}  Brier {res[k]['brier']:.4f}")
    res["peak_score"] = {"auc": auc(yva, p_peak)}
    print(f"peak 점수  AUC {res['peak_score']['auc']:.3f}")
    # 보정(calibration): 예측 확률 구간별 실제 비율
    bins = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.01]
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p_rc >= lo) & (p_rc < hi)
        if m.any():
            cal.append((lo, hi, int(m.sum()), float(p_rc[m].mean()), float(yva[m].mean())))
            print(f"  p∈[{lo:.2f},{hi:.2f}) n {m.sum():5d}  예측 {p_rc[m].mean():.3f}  실제 {yva[m].mean():.3f}")
    meta = {"kappa": a.kappa, "data": a.data, "n": int(len(y)), "bad": int(y.sum()), "val": res, "calibration": cal}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_rc(a.out, nets, meta)
    json.dump(meta, open(Path(a.out).with_suffix(".json"), "w"), indent=1, ensure_ascii=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
