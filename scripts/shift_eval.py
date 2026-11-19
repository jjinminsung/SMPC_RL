#!/usr/bin/env python3
"""학습 때 없던 조건(분포 이동)에서 플래너 평가. collect_tc.run_seed 를 그대로 쓰고 EpisodeConfig 만 덧씌운다.

    cd sim && python3 ../scripts/shift_eval.py <cond> <planner> <seed0> <n> <workers> <out.json>
    예: 조건 lidar dense hgrid × 플래너(2 s + RVg + VTp, 2 s, 5 s, 8 s), 시드 3000–3299 → results/tc/eval/shift/<cond>_<vtp|2s|5s|8s>.json
        요약·짝 비교: python3 scripts/shift_pair.py

cond: lidar  = 라이다 수준 인지(속도 잡음 σ1 m/s, 60 m 밖 뒤차 미검출, 뒤차 깜빡임 50 %)
      dense  = 교통 1.3 배 촘촘
      hgrid  = MORAI phase R 의 H 격자 재현: NPC 스폰 속도 30/50/70 × 목표 속도 40/60/80/100 km/h (시드마다 돌아가며)
      base   = 학습·평가와 같은 조건
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from smpc_sim.rl import collect_tc  # noqa: E402

BASE = dict(collect_tc.CFG)
HGRID = [(s, d) for s in (30, 50, 70) for d in (40, 60, 80, 100)]


def extra(cond, i):
    if cond == "lidar":
        return {"perception": "lidar_like"}
    if cond == "dense":
        return {"spawn_scale": 1.3}
    if cond == "hgrid":
        s, d = HGRID[i % len(HGRID)]
        return {"npc_override": {"spawn_kph": s, "desired_kph": d}}
    return {}


def work(a):
    cond, name, seed, i = a
    collect_tc.CFG = {**BASE, **extra(cond, i)}
    ep = collect_tc.run_seed((name, seed, 0.0, 0.0, None, False, 0.0, "wait", False, False))[0]
    ep["cond"] = extra(cond, i)
    return ep


def main():
    cond, name, seed0, n, workers, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    t0 = time.time()
    with Pool(workers) as p:
        eps = p.map(work, [(cond, name, seed0 + i, i) for i in range(n)], chunksize=1)
    json.dump({"cond": cond, "planner": name, "episodes": eps, "wall_s": time.time() - t0}, open(out, "w"))
    print(f"{cond} {name.split('@')[0]} n={n} {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
