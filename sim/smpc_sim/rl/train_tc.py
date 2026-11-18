"""TC(종단 비용) 학습: collect_tc 가 모은 (상태 특징, 남은 J) 로 V(s) ≈ E[남은 J] 를 회귀한다(몬테카를로 정책 평가).

    cd sim
    python3 -m smpc_sim.rl.train_tc --data ../results/tc/iter0/data_s*.npz --out ../results/tc/TC_iter0.pt

검증 = 학습에 안 쓴 에피소드 15 %. 비교 기준: 상수(학습 평균), 선형 회귀(최소제곱). 지표: R², RMSE [s].
앙상블 M 개: 에피소드 단위 부트스트랩 + 초기값이 다르다. 출력 = 평균.
--roots: 수집 때 남긴 루트 기록(<data>_roots.npz)도 쓴다. 판단 틱에 실제로 따른 루트의 SMPC 루트 값
    Σ_b p_b [c_b + τ + V(예측 잎)] 이 그 틱부터의 실제 남은 J 와 같도록 V 를 맞춘다(예측 잎은 기회 제약을 통과한 가지라
    실제 상태보다 낙관적이다. 실제 상태 표본만으로는 '곧 중단될 바꾸기의 잎' 같은 상태가 데이터에 없다).
    손실 = 실제 상태 MSE 평균 + 루트 값 MSE 평균.
"""
import argparse
import glob
import json

import numpy as np
import torch

from smpc_sim.rl.terminal_cost import SCALE, TCNet, add_lateral_speed, lane_targets, root_targets, root_values, save_tc


def load(paths):
    X, G, E, T = [], [], [], []
    for p in paths:
        z = np.load(p)
        X.append(z["X"]); G.append(z["G"]); E.append(z["ep"]); T.append(z["t"])
    return np.concatenate(X), np.concatenate(G), np.concatenate(E), np.concatenate(T)


def r2(pred, y):
    return float(1.0 - ((pred - y) ** 2).mean() / max(1e-9, ((y - y.mean()) ** 2).mean()))


def take(R, idx):
    return {k: v[idx] for k, v in R.items()}


def root_loss(net, R, y):
    return (((root_values(lambda F: SCALE * net(F), R) - torch.as_tensor(y)) / SCALE) ** 2).mean()


def fit_member(X, y, Xv, yv, hidden, epochs, seed, R=None, yr=None, Rv=None, yrv=None, state_w=1.0):
    torch.manual_seed(seed)
    net = TCNet(dim=X.shape[1], hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-6)
    xt, yt = torch.from_numpy(X), torch.from_numpy(y / SCALE)
    xv, yv_t = torch.from_numpy(Xv), torch.from_numpy(yv / SCALE)
    best, best_state = np.inf, None
    steps = max(1, (len(xt) + 1023) // 1024)
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        rperm = torch.randperm(len(yr)) if R is not None else None
        rb = (len(yr) + steps - 1) // steps if R is not None else 0
        for j, k in enumerate(range(0, len(xt), 1024)):
            b = perm[k:k + 1024]
            loss = state_w * ((net(xt[b]) - yt[b]) ** 2).mean()
            if R is not None:
                ib = rperm[j * rb:(j + 1) * rb].numpy()
                if len(ib):
                    loss = loss + root_loss(net, take(R, ib), yr[ib])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            vl = state_w * ((net(xv) - yv_t) ** 2).mean().item()
            if Rv is not None:
                vl += root_loss(net, Rv, yrv).item()
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    return net


def predict(nets, X):
    with torch.no_grad():
        return SCALE * torch.stack([n(torch.from_numpy(X)) for n in nets]).mean(0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--target", default="lane", choices=["lane", "episode"], help="lane = 지금 차선 변경까지의 J")
    ap.add_argument("--roots", action="store_true", help="루트 기록으로 루트 값도 실제 남은 J 에 맞춘다(lane 라벨)")
    ap.add_argument("--state_w", type=float, default=1.0, help="--roots 일 때 실제 상태 MSE 무게(0 = 루트 값만으로 학습)")
    ap.add_argument("--notchange_keep", action="store_true", help="--roots: 바꾸지 않은 틱은 기다리기 루트 잎으로 맞춘다(TCb 비교용)")
    a = ap.parse_args()
    paths = sorted({p for d in a.data for p in glob.glob(d) if not p.endswith(("_roots.npz", "_leaf.npz"))})
    X, G, E, T = load(paths)
    if a.target == "lane":
        eps = {e["seed"]: e for p in paths for e in json.load(open(p[:-4] + ".json"))["episodes"]}
        G, clean = lane_targets(X, G, T, E, eps)
    else:
        clean = np.ones(len(G), dtype=bool)
    R = None
    if a.roots:
        zs = [dict(np.load(p[:-4] + "_roots.npz")) for p in paths]
        if a.notchange_keep:
            for z in zs:            # 바꾸지 않은 틱(준비·기다리기)의 라벨은 기다리기 루트 잎에 건다: Qk = Q(지금 안 바꿈)
                if "keep_ok" in z:
                    m = (z["act"] != 0) & z["keep_ok"]
                    for k in ("F", "C", "term", "t_arr", "tau"):
                        z[k][m] = z["keep_" + k][m]
        Bm = max(z["P"].shape[1] for z in zs)
        keys = ("F", "P", "C", "term", "t_arr", "mask", "tau", "t", "ep", "lane", "act", "explored", "Qsel")

        def pad(v):
            return v if v.ndim == 1 or v.shape[1] == Bm else np.concatenate(
                [v, np.zeros((v.shape[0], Bm - v.shape[1]) + v.shape[2:], v.dtype)], axis=1)
        R = {k: np.concatenate([pad(z[k]) for z in zs]) for k in keys}
        yr, rclean = root_targets(R["t"], R["ep"], R["lane"], R["act"], R["explored"], eps)
        n_roots_all = len(yr)
        R, yr = take(R, rclean), yr[rclean]
    X = add_lateral_speed(X, T, E)
    n_all = len(G)
    X, G, E, T = X[clean], G[clean], E[clean], T[clean]          # 탐색이 남은 표본은 탐욕 정책의 값이 아니다
    torch.set_num_threads(4)
    rng = np.random.default_rng(0)
    eps = np.unique(E)
    val_eps = set(rng.choice(eps, size=max(1, int(0.15 * len(eps))), replace=False).tolist())
    val = np.isin(E, list(val_eps))
    Xt, Gt, Et = X[~val], G[~val], E[~val]
    Xv, Gv = X[val], G[val]
    # 기준: 상수, 선형 회귀
    const = np.full(len(Gv), Gt.mean())
    A = np.c_[Xt, np.ones(len(Xt))]
    w = np.linalg.lstsq(A, Gt, rcond=None)[0]
    lin = np.c_[Xv, np.ones(len(Xv))] @ w
    tr_eps = np.unique(Et)
    if R is not None:
        rv = np.isin(R["ep"], list(val_eps))
        Rt, yrt, Rv, yrv = take(R, ~rv), yr[~rv], take(R, rv), yr[rv]
        Rt_f = {k: Rt[k] for k in ("F", "P", "C", "term", "t_arr", "mask", "tau")}
        Rv_f = {k: Rv[k] for k in ("F", "P", "C", "term", "t_arr", "mask", "tau")}
    nets = []
    for m in range(a.M):
        bs = rng.choice(tr_eps, len(tr_eps), replace=True)
        cnt = {e: c for e, c in zip(*np.unique(bs, return_counts=True))}
        idx = np.concatenate([np.repeat(np.nonzero(Et == e)[0], c) for e, c in cnt.items()])
        if R is not None:
            ridx = np.concatenate([np.repeat(np.nonzero(Rt["ep"] == e)[0], c) for e, c in cnt.items()])
            nets.append(fit_member(Xt[idx], Gt[idx], Xv, Gv, a.hidden, a.epochs, m, take(Rt_f, ridx), yrt[ridx], Rv_f, yrv,
                                   a.state_w))
        else:
            nets.append(fit_member(Xt[idx], Gt[idx], Xv, Gv, a.hidden, a.epochs, m))
        print(f"member {m} done", flush=True)
    pv = predict(nets, Xv)
    moving = Xv[:, 0] * 25.0 >= 1.0
    stats = {"rows": int(len(G)), "rows_all": int(n_all), "episodes": int(len(eps)), "val_rows": int(val.sum()), "G_mean": float(G.mean()),
             "G_std": float(G.std()),
             "val": {"TC": {"r2": r2(pv, Gv), "rmse": float(np.sqrt(((pv - Gv) ** 2).mean()))},
                     "linear": {"r2": r2(lin, Gv), "rmse": float(np.sqrt(((lin - Gv) ** 2).mean()))},
                     "const": {"r2": r2(const, Gv), "rmse": float(np.sqrt(((const - Gv) ** 2).mean()))}},
             "val_moving_r2": {"TC": r2(pv[moving], Gv[moving]), "linear": r2(lin[moving], Gv[moving])},
             "val_stand_r2": {"TC": r2(pv[~moving], Gv[~moving]), "linear": r2(lin[~moving], Gv[~moving])},
             "hidden": a.hidden, "M": a.M, "target": a.target, "state_w": a.state_w if a.roots else None, "data": ["/".join(p.split("/")[-2:]) for p in paths]}
    if R is not None:
        with torch.no_grad():
            qv = torch.stack([root_values(lambda F, n=n: SCALE * n(F), Rv) for n in nets]).mean(0).numpy()
        cn = Rv["act"] == 0
        stats["roots"] = {"rows": int(len(yr)), "rows_all": int(n_roots_all), "val_rows": int(rv.sum()),
                          "val_r2": r2(qv, yrv), "val_rmse": float(np.sqrt(((qv - yrv) ** 2).mean())),
                          "val_change_now": {"n": int(cn.sum()), "r2": r2(qv[cn], yrv[cn]), "bias": float((qv[cn] - yrv[cn]).mean()),
                                             "collect_model_bias": float((Rv["Qsel"][cn] - yrv[cn]).mean())},
                          "val_keep_bias": float((qv[Rv["act"] == 2] - yrv[Rv["act"] == 2]).mean())}
    save_tc(a.out, nets, stats)
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
