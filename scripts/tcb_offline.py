#!/usr/bin/env python3
"""TCb 오프라인 평가: 결정 탐색 데이터(collect_tc --record --explore_decision)에서 모델의 Qc·Qk 를 다시 계산하고,
'Qc > Qk(거절)' 인 결정에서 실제로 기다린 결과가 바꾼 결과보다 나은지 무작위 비교로 잰다.

    python3 scripts/tcb_offline.py --model results/tc_v7/TC_iter0.pt --data "results/tc_v7/iter0/data_s7*.npz" [--eps 70000-72999 의 일부]

결정 = 탐욕 정책이 지금 바꾸기를 고르기 시작한 틱. 결정 탐색이 살아 있던 결정(그 차선의 첫 강제 기다리기까지)만
무작위 비교에 쓴다(각 결정이 확률 p 로 기다리기). 라벨 = 그 틱부터 이 차선을 마칠 때까지의 J(root_targets).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from smpc_sim.rl.terminal_cost import load_tc_fn, root_targets, root_values  # noqa: E402


def load(pattern):
    paths = sorted(p for p in glob.glob(pattern) if not p.endswith("_roots.npz"))
    eps, Rs = {}, []
    for p in paths:
        for e in json.load(open(p[:-4] + ".json"))["episodes"]:
            eps[e["seed"]] = e
        Rs.append(dict(np.load(p[:-4] + "_roots.npz")))
    Bm = max(r["P"].shape[1] for r in Rs)

    def pad(v):
        return v if v.ndim == 1 or v.shape[1] == Bm else np.concatenate(
            [v, np.zeros((v.shape[0], Bm - v.shape[1]) + v.shape[2:], v.dtype)], 1)
    return eps, {k: np.concatenate([pad(r[k]) for r in Rs]) for k in Rs[0]}


def decisions(R):
    """결정 틱 인덱스, 강제 기다리기 여부, 무작위 비교에 쓸 수 있는지."""
    o = np.lexsort((R["t"], R["ep"]))
    idx, forced, rand = [], [], []
    used = set()                                   # (ep, lane): 그 차선의 결정 탐색이 이미 쓰였나
    prev = None
    for k in o:
        start = R["forced"][k] or (R["act"][k] == 0 and not R["explored"][k] and (
            prev is None or R["ep"][prev] != R["ep"][k] or not (R["act"][prev] == 0 or R["forced"][prev])
            or R["t"][k] - R["t"][prev] > 0.1))
        if start and R["chg_ok"][k] and R["keep_ok"][k]:
            key = (int(R["ep"][k]), int(R["lane"][k]))
            idx.append(k)
            forced.append(bool(R["forced"][k]))
            rand.append(key not in used)
            if R["forced"][k]:
                used.add(key)
        prev = k
    return np.array(idx), np.array(forced), np.array(rand)


def root_q(fn, R, x, sel):
    S = {"F": R[f"{x}_F"][sel], "P": R["P"][sel], "C": R[f"{x}_C"][sel], "term": R[f"{x}_term"][sel],
         "t_arr": R[f"{x}_t_arr"][sel], "mask": R["mask"][sel], "tau": R[f"{x}_tau"][sel]}
    with torch.no_grad():
        return root_values(lambda F: torch.as_tensor(fn(F.numpy())), S).numpy()


def report(fn, eps, R, tag=""):
    y, _ = root_targets(R["t"], R["ep"], R["lane"], R["act"], R["explored"], eps)
    d, f, rnd = decisions(R)
    qc, qk = root_q(fn, R, "chg", d), root_q(fn, R, "keep", d)
    yd = y[d]
    dec = qc > qk
    out = {"decisions": int(len(d)), "forced": int(f.sum()), "randomized": int(rnd.sum()),
           "calib_change": [float(qc[~f].mean()), float(yd[~f].mean())], "calib_wait": [float(qk[f].mean()), float(yd[f].mean())],
           "decline_frac": float(dec.mean())}
    for name, m in (("decline", dec & rnd), ("accept", ~dec & rnd)):
        c, w = m & ~f, m & f
        if c.sum() and w.sum():
            diff = yd[c].mean() - yd[w].mean()
            se = np.sqrt(yd[c].var() / c.sum() + yd[w].var() / w.sum())
            out[name] = {"n_change": int(c.sum()), "n_wait": int(w.sum()), "J_change": float(yd[c].mean()),
                         "J_wait": float(yd[w].mean()), "wait_gain": float(diff), "ci95": [float(diff - 1.96 * se), float(diff + 1.96 * se)]}
    n_eps = len(np.unique(R["ep"]))
    if "decline" in out:
        # 에피소드당 기대 이득 ≈ (에피소드당 거절 결정 수) × (거절 결정에서 기다리기의 이득). 거절 뒤 정책은 2 s 그대로라 1단계 개선
        out["gain_per_ep_1step"] = float((dec & rnd).sum() / max(1, rnd.sum()) * len(d) / n_eps * out["decline"]["wait_gain"])
    print(tag, json.dumps(out, ensure_ascii=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", default=None, help="a-b: 이 범위 에피소드만(학습에 안 쓴 시드로 평가)")
    a = ap.parse_args()
    eps, R = load(a.data)
    if a.seeds:
        lo, hi = map(int, a.seeds.split("-"))
        m = (R["ep"] >= lo) & (R["ep"] <= hi)
        R = {k: v[m] for k, v in R.items()}
    for mp in a.model:
        report(load_tc_fn(mp), eps, R, tag=Path(mp).name)


if __name__ == "__main__":
    main()
