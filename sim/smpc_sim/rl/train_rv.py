"""RV(RL 위험 값) 학습: collect_tc --risk_log 데이터에서 y_t = 이 차선 변경을 마칠 때까지 실제로 마주친 최대 위험.

    cd sim
    python3 -m smpc_sim.rl.train_rv --data ../results/rv/iter0/data_s*.npz --out ../results/rv/RV_iter0.pt

소프트 라벨 BCE(최소점 = E[y]), 에피소드 단위 부트스트랩 앙상블, 보류 에피소드 15 % 로 조기 종료.
지표: Brier(상수 기준과 비교), y > ε(0.04) 를 가려내는 AUC, 예측 구간별 보정.
--leaf: 잎 학습판. 입력 = 지금 바꾸기를 명령한 틱의 예측 잎(collect_tc --leaf_log, 가지별 무게 p_b),
    라벨 = 그 차선 변경의 실제 결과(risk_value.leaf_targets). 플래너는 루트 0 에만 쓴다.
--target trace (--leaf 와 함께): RVT(시점별 위험 값). y_j = 명령 뒤 jΔ 의 위험 신호(j = 1..40, Δ = 0.2 s), 출력 40개 BCE.
    플래너는 max_j Σ_b p_b V_bj 를 쓴다(risk_value 머리말). 지표 = 원소별 Brier, max_j 예측의 중단 AUC·ε 넘는 비율.
--target abort: RA(중단 값). y = (t, t_c] 의 중단 횟수, 푸아송 NLL(출력 = log 기댓값), 지표 = 편차·y > 0 AUC.
"""
import argparse
import glob
import json

import numpy as np
import torch

from smpc_sim.rl.risk_value import leaf_targets, risk_targets, save_rv
from smpc_sim.rl.terminal_cost import TCNet, add_lateral_speed

EPS = 0.04


def auc(s, l):
    pos, neg = s[l], s[~l]
    if not len(pos) or not len(neg):
        return float("nan")
    o = np.argsort(np.concatenate([pos, neg]))
    rk = np.empty(len(o)); rk[o] = np.arange(1, len(o) + 1)
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load(paths):
    X, E, T, eps = [], [], [], {}
    for p in paths:
        z = np.load(p)
        X.append(z["X"]); E.append(z["ep"]); T.append(z["t"])
        for e in json.load(open(p[:-4] + ".json"))["episodes"]:
            eps[e["seed"]] = e
    X, E, T = np.concatenate(X), np.concatenate(E), np.concatenate(T)
    return add_lateral_speed(X, T, E).astype(np.float32), E, T, eps


def load_leaf(paths, target):
    """*_leaf.npz + 같은 이름 .json → (X 잎 행, y, w, E)."""
    X, y, w, E = [], [], [], []
    for p in paths:
        z = np.load(p[:-4] + "_leaf.npz")
        eps = {e["seed"]: e for e in json.load(open(p[:-4] + ".json"))["episodes"]}
        yr, vr, wr = leaf_targets(z["REC_ep"], z["REC_t"], eps, target)
        k = z["LREC"]
        keep = vr[k] & ~z["LA"]
        X.append(z["LF"][keep]); y.append(yr[k][keep]); w.append((z["LW"] * wr[k])[keep]); E.append(z["REC_ep"][k][keep])
    return (np.concatenate(X).astype(np.float32), np.concatenate(y).astype(np.float32),
            np.concatenate(w).astype(np.float32), np.concatenate(E))


def fit_member(X, y, Xv, yv, hidden, epochs, seed, target="risk", w=None, wv=None):
    torch.manual_seed(seed)
    net = TCNet(dim=X.shape[1], hidden=hidden, out=y.shape[1] if y.ndim == 2 else 1)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-6)
    xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    xv, yvt = torch.from_numpy(Xv), torch.from_numpy(yv)
    w = np.ones(len(y), np.float32) if w is None else w
    wv = np.ones(len(yv), np.float32) if wv is None else wv
    wt, wvt = torch.from_numpy(w), torch.from_numpy(wv)

    def base(o, t):
        if target == "abort":
            return torch.nn.functional.poisson_nll_loss(o, t, log_input=True, full=False, reduction="none")
        l = torch.nn.functional.binary_cross_entropy_with_logits(o, t, reduction="none")
        return l.mean(-1) if l.ndim == 2 else l

    def lossf(o, t, ww):
        return (base(o, t) * ww).sum() / ww.sum()
    best, best_state = np.inf, None
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for k in range(0, len(xt), 1024):
            b = perm[k:k + 1024]
            loss = lossf(net(xt[b]), yt[b], wt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            vl = lossf(net(xv), yvt, wvt).item()
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--target", choices=["risk", "abort", "trace"], default="risk")
    ap.add_argument("--leaf", action="store_true")
    a = ap.parse_args()
    paths = sorted(p for d in a.data for p in glob.glob(d) if not p.endswith(("_roots.npz", "_leaf.npz")))
    if a.target == "trace" and not a.leaf:
        raise SystemExit("--target trace 는 --leaf 와 함께")
    if a.leaf:
        X, y, W, E = load_leaf(paths, a.target)
        if a.target == "trace":
            _, y_ab, _, _ = load_leaf(paths, "abort")       # 같은 행 순서(유효 여부는 라벨 종류와 무관): 지표용
    else:
        X, E, T, eps = load(paths)
        y, clean = risk_targets(X, T, E, eps, target=a.target)
        X, E, y = X[clean], E[clean], y[clean].astype(np.float32)
        W = np.ones(len(y), np.float32)
    ueps = np.unique(E)
    rng = np.random.default_rng(0)
    val_eps = set(rng.choice(ueps, max(1, int(0.15 * len(ueps))), replace=False).tolist())
    vm = np.array([e in val_eps for e in E])
    Xv, yv = X[vm], y[vm]
    tr_eps = np.array(sorted(set(ueps.tolist()) - val_eps))
    torch.set_num_threads(2)
    nets = []
    for m in range(a.M):
        bs = rng.choice(tr_eps, len(tr_eps), replace=True)
        cnt = {e: c for e, c in zip(*np.unique(bs, return_counts=True))}
        idx = np.concatenate([np.repeat(np.nonzero(E == e)[0], c) for e, c in cnt.items()])
        nets.append(fit_member(X[idx], y[idx], Xv, yv, a.hidden, a.epochs, seed=m, target=a.target, w=W[idx], wv=W[vm]))
        print(f"member {m} done", flush=True)
    if a.target == "trace":
        with torch.no_grad():
            pv = torch.stack([torch.sigmoid(n(torch.from_numpy(Xv))) for n in nets]).mean(0).numpy()
        const = y[~vm].mean(0)
        V, Vy, ab = pv.max(1), yv.max(1), y_ab[vm] > 0
        wv = W[vm]
        bins = [0, 0.01, 0.02, 0.04, 0.08, 1.01]
        calib = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (V >= lo) & (V < hi)
            if m.any():
                calib.append([lo, hi, int(m.sum()), round(float(np.average(V[m], weights=wv[m])), 4),
                              round(float(np.average(ab[m], weights=wv[m])), 3), round(float(np.average(Vy[m], weights=wv[m])), 4)])
        meta = {"target": "trace", "leaf": True, "trace_dt": 0.2, "trace_T": 8.0, "attempts_w": round(float(W.sum()), 1),
                "rows": int(len(y)), "episodes": int(len(ueps)), "val_rows": int(vm.sum()),
                "y_mean_peak_of_mean": round(float(np.average(y, axis=0, weights=W).max()), 4),
                "y_mean_of_peak": round(float(np.average(y.max(1), weights=W)), 4),
                "val_brier": float(((pv - yv) ** 2).mean()), "const_brier": float(((const - yv) ** 2).mean()),
                "val_auc_abort_maxj": auc(V, ab), "val_frac_maxj_gt_eps(w)": round(float(np.average(V > EPS, weights=wv)), 4),
                "val_abort_rate(w)": round(float(np.average(ab, weights=wv)), 4),
                "calib_maxj[lo,hi,n,pred,P(abort),realized max]": calib, "hidden": a.hidden, "M": a.M,
                "data": ["/".join(p.split("/")[-2:]) for p in paths]}
        save_rv(a.out, nets, meta)
        print(json.dumps(meta))
        return
    link = torch.exp if a.target == "abort" else torch.sigmoid
    with torch.no_grad():
        pv = torch.stack([link(n(torch.from_numpy(Xv))) for n in nets]).mean(0).numpy()
    const = float(y[~vm].mean())
    if a.target == "abort":
        def dev(mu):
            mu = np.maximum(mu, 1e-9)
            return float(2 * np.mean(np.where(yv > 0, yv * np.log(np.maximum(yv, 1e-9) / mu), 0.0) - (yv - mu)))
        bins = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 100]
        calib = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (pv >= lo) & (pv < hi)
            if m.any():
                calib.append([lo, hi, int(m.sum()), round(float(pv[m].mean()), 4), round(float(yv[m].mean()), 4)])
        meta = {"target": "abort", "leaf": a.leaf, "attempts_w": round(float(W.sum()), 1), "rows": int(len(y)), "episodes": int(len(ueps)), "val_rows": int(vm.sum()),
                "y_mean": round(float(y.mean()), 4), "y_gt0": round(float((y > 0).mean()), 4),
                "val_dev": dev(pv), "const_dev": dev(np.full_like(yv, const)), "val_auc_gt0": auc(pv, yv > 0),
                "val_pred_mean": float(pv.mean()), "val_y_mean": float(yv.mean()),
                "calib[lo,hi,n,pred,y]": calib, "hidden": a.hidden, "M": a.M,
                "data": ["/".join(p.split("/")[-2:]) for p in paths]}
        save_rv(a.out, nets, meta)
        print(json.dumps(meta))
        return
    pos = yv > EPS
    bins = [0, 0.005, 0.02, 0.04, 0.08, 1.01]
    calib = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pv >= lo) & (pv < hi)
        if m.any():
            calib.append([lo, hi, int(m.sum()), round(float(pv[m].mean()), 4), round(float(yv[m].mean()), 4),
                          round(float((yv[m] > EPS).mean()), 3)])
    meta = {"leaf": a.leaf, "attempts_w": round(float(W.sum()), 1), "rows": int(len(y)), "episodes": int(len(ueps)), "val_rows": int(vm.sum()),
            "y_mean": round(float(y.mean()), 4), "y_gt_eps": round(float((y > EPS).mean()), 4),
            "val_brier": float(((pv - yv) ** 2).mean()), "const_brier": float(((const - yv) ** 2).mean()),
            "val_auc_gt_eps": auc(pv, pos), "val_pred_mean": float(pv.mean()), "val_y_mean": float(yv.mean()),
            "calib[lo,hi,n,pred,y,P(y>eps)]": calib, "hidden": a.hidden, "M": a.M,
            "data": ["/".join(p.split("/")[-2:]) for p in paths]}
    save_rv(a.out, nets, meta)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
