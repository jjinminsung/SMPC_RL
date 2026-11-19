"""분포 이동 결과 요약 + 짝 비교(같은 시드, 부트스트랩 95 %). 기준 조건(base)은 기존 시험 결과의 3000–3299."""
import glob
import json
import os

from pathlib import Path

import numpy as np

R = str(Path(__file__).resolve().parents[1] / "results/tc/eval") + "/"
O = str(Path.home() / "github/smpc_lane_change_rl/research/results/rc/rcf") + "/"   # 2 s·5 s 시험 기준(rcf)
SH = R + "shift/"
SEEDS = range(3000, 3300)
PL = ["vtp", "2s", "5s", "8s"]
NAME = {"vtp": "2s+RVg+VTp", "2s": "2s", "5s": "5s", "8s": "8s"}


def load(fs):
    E = {}
    for f in fs:
        for e in json.load(open(f))["episodes"]:
            if e["seed"] in SEEDS:
                E[e["seed"]] = e
    return E


def J(e):
    return (e["t_merge"] if e["success"] else 120) + 60 * e["collision"] + 2 * e["aborts"]


def base():
    return {"vtp": load([R + "vttest/p1_s3000.json"]), "2s": load(sorted(glob.glob(O + "test3000_2s_s*.json"))),
            "5s": load(sorted(glob.glob(O + "test3000_5s_s*.json"))), "8s": load([R + "test8/p0_s3000.json"])}


def main():
    rng = np.random.default_rng(0)
    conds = {"base": base()}
    for c in ("lidar", "dense", "hgrid"):
        d = {k: load([SH + f"{c}_{k}.json"]) for k in PL if os.path.exists(SH + f"{c}_{k}.json")}
        if d:
            conds[c] = d
    for c, d in conds.items():
        print(f"== {c}")
        for k in PL:
            if k not in d:
                continue
            A = d[k]
            v = list(A.values())
            print(f"  {NAME[k]:11s} n {len(v)} J {np.mean([J(e) for e in v]):6.2f} fail {sum(not e['success'] for e in v):2d} "
                  f"coll {sum(e['collision'] for e in v)} ab {np.mean([e['aborts'] for e in v]):.3f} "
                  f"minsep<0.5 {sum(e['min_sep'] < 0.5 for e in v)}")
        if "vtp" in d:
            for k in ("2s", "5s", "8s"):
                if k not in d:
                    continue
                s = sorted(set(d["vtp"]) & set(d[k]))
                x = np.array([J(d["vtp"][i]) - J(d[k][i]) for i in s])
                bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
                print(f"    VTp - {k:3s}: {x.mean():+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}] (n {len(s)})")
        if c == "hgrid" and "vtp" in d:
            print("    격자별 J (스폰/목표 km/h):")
            cells = {}
            for k in PL:
                for e in d.get(k, {}).values():
                    key = (e["cond"]["npc_override"]["spawn_kph"], e["cond"]["npc_override"]["desired_kph"])
                    cells.setdefault(key, {}).setdefault(k, []).append(J(e))
            for key in sorted(cells):
                print("     ", key, "  ".join(f"{NAME[k]} {np.mean(cells[key][k]):6.2f}" for k in PL if k in cells[key]))


if __name__ == "__main__":
    main()
