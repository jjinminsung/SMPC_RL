"""TR 학습: log10(꼬리 위험 + 1e-4) 회귀, 에피소드 단위 검증, 앙상블.

    python3 -m smpc_sim.rl.train_tail --data ../results/tail/data.npz --out ../results/tail/TR.pt"""
import argparse
import json

import numpy as np
import torch
from torch import nn

from smpc_sim.rl.tail_risk import TailNet, save_tail, to_target


def auc(score, y):
    pos, neg = score[y], score[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ensemble", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--agg", default="max")
    a = ap.parse_args()
    z = np.load(a.data)
    X, Y, A, E = z["X"], z["Y"], z["A"], z["E"]
    X, Y, E = X[A], Y[A], E[A]
    T = to_target(Y).astype(np.float32)
    rng = np.random.default_rng(0)
    eps = np.unique(E)
    val_eps = set(rng.choice(eps, size=max(1, len(eps) // 10), replace=False).tolist())
    val = np.array([e in val_eps for e in E])
    xt, tt = torch.from_numpy(X[~val]), torch.from_numpy(T[~val])
    xv, tv = torch.from_numpy(X[val]), torch.from_numpy(T[val])
    w_t = torch.from_numpy(np.where(Y[~val] > 1e-3, 3.0, 1.0).astype(np.float32))   # 위험이 있는 행을 더 본다
    nets, preds = [], []
    for m in range(a.ensemble):
        torch.manual_seed(m)
        net = TailNet(dim=X.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-6)
        best, best_state = np.inf, None
        for ep in range(a.epochs):
            perm = torch.randperm(len(xt))
            for k in range(0, len(xt), 1024):
                b = perm[k:k + 1024]
                loss = (w_t[b] * (net(xt[b]) - tt[b]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                vl = nn.functional.mse_loss(net(xv), tv).item()
            if vl < best:
                best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        nets.append(net)
        with torch.no_grad():
            preds.append(net(xv).numpy())
        print(f"member {m}: val mse {best:.4f}", flush=True)
    P = np.max(preds, axis=0) if a.agg == "max" else np.mean(preds, axis=0)
    yv = T[val]
    r2 = 1 - ((P - yv) ** 2).mean() / ((yv - yv.mean()) ** 2).mean()
    stats = {"rows": int(len(Y)), "val_rows": int(val.sum()), "r2_log": float(r2), "agg": a.agg}
    for th in (0.01, 0.04):
        lab = Y[val] > th
        pr = P > np.log10(th + 1e-4)
        stats[f"auc_gt{th}"] = auc(P, lab)
        stats[f"pos_rate_gt{th}"] = float(lab.mean())
        stats[f"recall_gt{th}"] = float((pr & lab).sum() / max(1, lab.sum()))
        stats[f"false_pos_gt{th}"] = float((pr & ~lab).sum() / max(1, (~lab).sum()))
    save_tail(a.out, nets, {"agg": a.agg, **stats})
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
