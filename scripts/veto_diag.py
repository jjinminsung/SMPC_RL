#!/usr/bin/env python3
"""2 s 플래너가 '지금 바꾸기'를 명령한 틱마다: RV 가 막았을지, 5 s 판이 막았을지, 그 차선 변경이 중단됐는지.

2 s 정책 그대로 달리면서(판단은 바꾸지 않음) 같은 틱에
  - RV: 루트 0(지금 바꾸기)의 꼬리 Σ_b p_b V_b, max(지평선 안, 꼬리) ≤ ε 인 대응 조합이 있는지
  - 5 s: 같은 상태에서 5 s 지평선판의 판단(지금 바꾸기를 고르나), 루트 0 가능 여부와 지평선 안 위험
을 기록하고, 결과(다음 중단/완료)와 맞춘다.

    OMP_NUM_THREADS=1 python3 scripts/veto_diag.py --rv results/rv/RV_iter1.pt --seeds 6000-6299 --workers 5 \
        --out results/rv/veto_diag_val.json
"""
import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "sim"))
from run_batch import planner_factory  # noqa: E402
from smpc_sim.world import World, EpisodeConfig  # noqa: E402
from smpc_sim.rl.risk_value import leaf_risk, load_rv_fn  # noqa: E402

B = "rx_mt_pf_fl_tfs_rr_ng6_nv20_nm_h10_rf10"
CFG = {"switch_delay_sec": 0.55, "acc_floor_real": True, "acc_tf_on_start": True}
RV_PATH = None


def wrap_solve(pl, rv_fn):
    f = pl._solve

    def g(r, br, root_cost, mask, eps):
        out = f(r, br, root_cost, mask, eps)
        if getattr(pl, "_roots_eval", False) and not getattr(pl, "_mon_eval", False) and mask[0]:
            info = {"feas0": bool(out[0][0]), "raw0": float(pl._raw_risk[0])}
            if rv_fn is not None:
                RV = leaf_risk(pl, r, mask, rv_fn)
                B_ = len(br)
                p = np.array([b[3] for b in br])
                opts = [[0, 1] if br[b][2] else [0] for b in range(B_)]
                combos = np.array(list(itertools.product(*opts)))
                ok = np.ones(len(combos), dtype=bool)
                rk = np.zeros((len(combos), r["risk"].shape[-1]))
                for b in range(B_):
                    ok &= r["allowed"][b, combos[:, b], 0]
                    rk += p[b] * r["risk"][b, combos[:, b], 0]
                peak = rk.max(axis=1)
                tail = sum(p[b] * RV[b, combos[:, b], 0] for b in range(B_))
                if tail.ndim == 2:                         # 시점별(RVT)
                    tail = tail.max(axis=1)
                good = ok & (peak <= eps)
                info["rv_feas0"] = bool((good & (tail <= eps)).any())
                info["tail_min0"] = float(tail[good].min()) if good.any() else float("nan")
            pl._diag = info
        return out
    pl._solve = g


def run(seed):
    rv_fn = load_rv_fn(RV_PATH)
    w = World(planner_factory(B), EpisodeConfig(seed=seed, **CFG))
    pl = w.planner
    gate = planner_factory(B.replace("_h10_", "_h25_"))(w.P, w.track)    # G5 와 같은 방식: 같은 상태로 매 틱 판단
    wrap_solve(pl, rv_fn)
    wrap_solve(gate, None)
    rec = []
    f = pl.decide

    def dec(now, t, v_e, mission, acc_target, acc_status):
        gate.nudging = getattr(pl, "nudging", False)
        gate._diag = None
        dg = gate.decide(now, t, v_e, mission, acc_target, acc_status)
        pl._diag = None
        out = f(now, t, v_e, mission, acc_target, acc_status)
        if out.change_left and not w.sup.active and w.sup.pending_at is None and pl._diag is not None:
            d = dict(pl._diag)
            d.update({"t": round(now - w.t0, 2), "g_change": bool(dg.change_left),
                      "g_feas0": (gate._diag or {}).get("feas0"), "g_raw0": (gate._diag or {}).get("raw0")})
            rec.append(d)
        return out
    pl.decide = dec
    res = w.run()
    ev = [(e[0], e[1]) for e in res.events if e[0] in ("abort", "complete")]
    for d in rec:
        nxt = [k for k, tt in ev if tt > d["t"] - 1e-6]
        d["outcome"] = nxt[0] if nxt else ("collision" if res.collision else "end")
    return {"seed": seed, "aborts": res.aborts, "success": res.success, "t_end": res.t_end, "cmds": rec}


def init(p):
    global RV_PATH
    RV_PATH = p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rv", required=True)
    ap.add_argument("--seeds", default="6000-6299")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    with Pool(a.workers, initializer=init, initargs=(str(R / a.rv) if not a.rv.startswith("/") else a.rv,)) as pool:
        eps = pool.map(run, range(lo, hi + 1), chunksize=1)
    json.dump({"episodes": eps}, open(a.out, "w"))
    C = [c for e in eps for c in e["cmds"] if c["outcome"] in ("abort", "complete")]
    ab = np.array([c["outcome"] == "abort" for c in C])
    rvv = np.array([not c["rv_feas0"] for c in C])
    g5v = np.array([not c["g_change"] for c in C])
    print(json.dumps({"cmds": len(C), "P(abort)": round(float(ab.mean()), 3),
                      "RV veto": [int(rvv.sum()), round(float(ab[rvv].mean()), 3) if rvv.any() else None,
                                  round(float(ab[~rvv].mean()), 3)],
                      "5s veto": [int(g5v.sum()), round(float(ab[g5v].mean()), 3) if g5v.any() else None,
                                  round(float(ab[~g5v].mean()), 3)],
                      "both": int((rvv & g5v).sum()), "aborts caught RV/5s": [int((ab & rvv).sum()), int((ab & g5v).sum()),
                                                                              int(ab.sum())]}))


if __name__ == "__main__":
    main()
