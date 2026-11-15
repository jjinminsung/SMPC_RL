#!/usr/bin/env python3
"""짝 비교(공통 난수): 같은 시드를 다시 돌리면 결정 시작 틱까지 똑같이 흘러가므로, 그 틱부터 D 초 기다리기를 강제한
실행과 원래 실행(바로 바꾸기)의 에피소드 J 차이가 같은 교통 흐름에서의 '기다리기 - 바꾸기' 가 된다(무작위 두 팔 비교보다 분산이 작다).

    python3 scripts/crn_wait.py --planner rx_mt_pf_fl_tfs_rr_ng6_nv20_nm_h10_rf10 --seed0 96000 --n 200 --D 1 2 3 \
        --out results/crn/crn_s96000.json

출력: 에피소드마다 원래 실행의 J·결정 시작 시각, 결정마다 D 별 J(같은 시드 재실행).
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "sim"))
from run_batch import planner_factory  # noqa: E402
from smpc_sim.world import World, EpisodeConfig  # noqa: E402
from smpc_sim.rl.terminal_cost import FAIL_TIME  # noqa: E402

CFG = {"switch_delay_sec": 0.55, "acc_floor_real": True, "acc_tf_on_start": True}


def run(args):
    planner, seed, crn = args
    w = World(planner_factory(planner), EpisodeConfig(seed=seed, crn_wait=crn, **CFG))
    res = w.run()
    J = (res.t_merge if res.success else FAIL_TIME) + 60.0 * res.collision + 2.0 * res.aborts
    return {"seed": seed, "crn": crn, "J": J, "success": res.success, "collision": res.collision, "aborts": res.aborts,
            "t0": w.t0, "starts": list(w.planner.decision_starts),
            "completes": list(res.lane_change_times), "abort_times": [ev[1] for ev in res.events if ev[0] == "abort"]}


def workers(default):
    f = R / "results" / "rc" / "workers.txt"
    return int(f.read_text().strip()) if f.exists() else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", required=True)
    ap.add_argument("--seed0", type=int, default=96000)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--D", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    ap.add_argument("--max_starts", type=int, default=8, help="에피소드당 짝 비교할 결정 시작 수(앞에서부터)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    W = a.workers or workers(4)
    seeds = list(range(a.seed0, a.seed0 + a.n))
    with Pool(W) as pool:
        base = pool.map(run, [(a.planner, s, None) for s in seeds], chunksize=1)
        jobs = [(a.planner, b["seed"], (t, D)) for b in base for t in b["starts"][:a.max_starts] for D in a.D]
        alt = pool.map(run, jobs, chunksize=1)
    out = {"planner": a.planner, "D": a.D, "base": base, "alt": alt}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out + ".tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, a.out)
    print(f"episodes {len(base)}  decisions {sum(min(len(b['starts']), a.max_starts) for b in base)}  replays {len(alt)}")


if __name__ == "__main__":
    main()
