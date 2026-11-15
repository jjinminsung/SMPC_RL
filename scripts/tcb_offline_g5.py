#!/usr/bin/env python3
"""G5(2 s + 5 s 판단) 수집 데이터에서 TC 모델 평가: 2 s 가 지금 바꾸려는 결정마다 Qc·Qk 를 다시 계산해
  1) 모델의 거절(Qc > Qk)이 5 s 의 거절과 얼마나 맞나(AUC, 거절률)
  2) 무작위 비교: 막힌 결정에서 강제 바꾸기 대 거절, 통과한 결정에서 바꾸기 대 강제 기다리기의 실제 차선 J

    python3 scripts/tcb_offline_g5.py --model M1.pt [M2.pt ...] --data "results/tc_v9/iter0/data_s9*.npz" [--seeds a-b]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcb_offline import load, root_q  # noqa: E402
from smpc_sim.rl.terminal_cost import load_tc_fn, root_targets  # noqa: E402


def wants_starts(R):
    """2 s 가 지금 바꾸려는 틱이 이어지는 구간의 첫 틱, 그때 막혔나, 강제였나, 무작위 비교에 쓸 수 있나(그 차선 탐색 전)."""
    o = np.lexsort((R["t"], R["ep"]))
    idx, used, prev = [], set(), None
    for k in o:
        if R["wants"][k] and R["chg_ok"][k] and R["keep_ok"][k]:
            new = prev is None or R["ep"][prev] != R["ep"][k] or not R["wants"][prev] or R["t"][k] - R["t"][prev] > 0.1
            key = (int(R["ep"][k]), int(R["lane"][k]))
            if new:
                idx.append((k, key not in used))
            if R["forced"][k]:
                used.add(key)
        prev = k
    return np.array([i for i, _ in idx]), np.array([r for _, r in idx])


def auc(s, l):
    pos, neg = s[l], s[~l]
    return float(np.mean([(x > neg).mean() + 0.5 * (x == neg).mean() for x in pos])) if len(pos) and len(neg) else float("nan")


def cmp(yd, a, b):
    if a.sum() < 3 or b.sum() < 3:
        return None
    g = yd[a].mean() - yd[b].mean()
    se = np.sqrt(yd[a].var() / a.sum() + yd[b].var() / b.sum())
    return {"n": [int(a.sum()), int(b.sum())], "J": [round(float(yd[a].mean()), 2), round(float(yd[b].mean()), 2)],
            "diff": round(float(g), 2), "ci95": [round(float(g - 1.96 * se), 2), round(float(g + 1.96 * se), 2)]}


def cmp_adj(yd, a, b, Z):
    """무작위 비교의 회귀 보정: y ~ 1 + 처치 + Z(처치와 무관한 상태 함수, 예: 모델의 Qc·Qk). 처치 계수와 HC1 95 % 구간."""
    m = a | b
    if a.sum() < 3 or b.sum() < 3:
        return None
    X = np.c_[np.ones(m.sum()), a[m].astype(float), Z[m]]
    y = yd[m]
    XtX = np.linalg.pinv(X.T @ X)
    beta = XtX @ X.T @ y
    r = y - X @ beta
    V = XtX @ (X.T * r ** 2) @ X @ XtX * len(y) / max(1, len(y) - X.shape[1])
    g, se = beta[1], np.sqrt(V[1, 1])
    return {"diff": round(float(g), 2), "ci95": [round(float(g - 1.96 * se), 2), round(float(g + 1.96 * se), 2)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", default=None)
    a = ap.parse_args()
    eps, R = load(a.data)
    if a.seeds:
        lo, hi = map(int, a.seeds.split("-"))
        m = (R["ep"] >= lo) & (R["ep"] <= hi)
        R = {k: v[m] for k, v in R.items()}
    y, _ = root_targets(R["t"], R["ep"], R["lane"], R["act"], R["explored"], eps)
    d, rnd = wants_starts(R)
    dis, f = R["gate_dis"][d], R["forced"][d]
    yd = y[d]
    base = {"episodes": len(np.unique(R["ep"])), "starts": int(len(d)), "blocked": int(dis.sum()),
            # 막힌 결정: 강제 바꾸기 J - 거절 J (양수면 거절이 이득) / 통과한 결정: 바꾸기 J - 강제 기다리기 J (음수면 바꾸기가 이득)
            "blocked: change - decline": cmp(yd, dis & rnd & f, dis & rnd & ~f),
            "passed: change - wait": cmp(yd, ~dis & rnd & ~f, ~dis & rnd & f)}
    print("data", json.dumps(base, ensure_ascii=False))
    for mp in a.model:
        fn = load_tc_fn(mp)
        qc, qk = root_q(fn, R, "chg", d), root_q(fn, R, "keep", d)
        dec = qc > qk
        out = {"AUC(Qc-Qk, 5 s 막음)": round(auc(qc - qk, dis), 3), "decline|blocked": round(float(dec[dis].mean()), 3),
               "decline|passed": round(float(dec[~dis].mean()), 3),
               "calib Qc (바꾼 결정)": [round(float(qc[R["act"][d] == 0].mean()), 2), round(float(yd[R["act"][d] == 0].mean()), 2)],
               "calib Qk (안 바꾼 결정)": [round(float(qk[R["act"][d] != 0].mean()), 2), round(float(yd[R["act"][d] != 0].mean()), 2)],
               # 모델이 거절할 결정 안에서의 무작위 비교(바꾸기 J - 안 바꾸기 J, 양수면 거절이 맞음)
               "model-decline: change - notchange": cmp(yd, dec & rnd & (R["act"][d] == 0), dec & rnd & (R["act"][d] != 0)),
               "model-accept: change - notchange": cmp(yd, ~dec & rnd & (R["act"][d] == 0), ~dec & rnd & (R["act"][d] != 0)),
               # 같은 비교를 Qc·Qk 로 회귀 보정(분산 축소, 무작위라 편향 없음)
               "adj blocked: change - decline": cmp_adj(yd, dis & rnd & f, dis & rnd & ~f, np.c_[qc, qk]),
               "adj passed: change - wait": cmp_adj(yd, ~dis & rnd & ~f, ~dis & rnd & f, np.c_[qc, qk]),
               "adj model-decline: change - notchange": cmp_adj(yd, dec & rnd & (R["act"][d] == 0), dec & rnd & (R["act"][d] != 0), np.c_[qc, qk])}
        print(Path(mp).name, json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
