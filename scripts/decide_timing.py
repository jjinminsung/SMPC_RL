#!/usr/bin/env python3
"""플래너별 틱당 계산 시간: 같은 시드를 한 프로세스로 돌리며 decide(판단)·monitor(실행 중 모니터)·nudge 모니터 시간을 잰다.
가능한 최대 판단 주파수 = 1 / (틱당 시간). Python 시뮬 기준이라 절대값보다 플래너 사이 비율을 본다.

    OMP_NUM_THREADS=1 python3 scripts/decide_timing.py --seeds 6000-6019 --out results/timing/t.json \
        rx_..._h10_rf10 rx_..._h10_rf10_RV@results/rv/RV_iter0.pt rx_..._h25_rf10 rx_..._h40_rf10
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "sim"))
from run_batch import planner_factory  # noqa: E402
from smpc_sim.world import World, EpisodeConfig  # noqa: E402

CFG = {"switch_delay_sec": 0.55, "acc_floor_real": True, "acc_tf_on_start": True}


def timed(w, name, store):
    """플래너 메서드 시간을 world 시각(틱)별로 남긴다. 같은 틱의 판단·모니터 시간을 더해 틱당 시간을 만든다."""
    f = getattr(w.planner, name)

    def g(*a, **k):
        t0 = time.perf_counter()
        out = f(*a, **k)
        store.append((round(w.t, 3), time.perf_counter() - t0))
        return out
    setattr(w.planner, name, g)


def run(planner, seed):
    w = World(planner_factory(planner), EpisodeConfig(seed=seed, **CFG))
    rec = {"decide": [], "monitor": [], "nudge_monitor_tick": []}
    for m in rec:
        if hasattr(w.planner, m):
            timed(w, m, rec[m])
    res = w.run()
    tick = {}
    for m, xs in rec.items():
        for i, dt in xs:
            tick[i] = tick.get(i, 0.0) + dt
    return {"decide": [d for _, d in rec["decide"]], "monitor": [d for _, d in rec["monitor"]],
            "tick": list(tick.values()), "aborts": res.aborts, "t_merge": res.t_merge}


def stats(x):
    x = np.asarray(x) * 1e3
    if not len(x):
        return None
    return {"n": int(len(x)), "mean": round(float(x.mean()), 2), "p50": round(float(np.percentile(x, 50)), 2),
            "p95": round(float(np.percentile(x, 95)), 2), "p99": round(float(np.percentile(x, 99)), 2),
            "max": round(float(x.max()), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("planners", nargs="+")
    ap.add_argument("--seeds", default="6000-6019")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    out = {}
    for pl in a.planners:
        pl = pl.replace("@results/", f"@{R}/results/")
        D, M, T = [], [], []
        for s in range(lo, hi + 1):
            r = run(pl, s)
            D += r["decide"]; M += r["monitor"]; T += r["tick"]
        st = {"decide_ms": stats(D), "monitor_ms": stats(M), "tick_ms": stats(T)}
        st["max_hz_p99"] = round(1000.0 / st["tick_ms"]["p99"], 1)
        st["max_hz_worst"] = round(1000.0 / st["tick_ms"]["max"], 1)
        key = pl.split("/")[-1] if "@" in pl else pl
        out[key] = st
        print(key, json.dumps(st), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
