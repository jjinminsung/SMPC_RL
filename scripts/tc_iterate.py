#!/usr/bin/env python3
"""TC(학습한 종단 비용) 근사 정책 반복. 다시 실행하면 이미 있는 결과는 건너뛰고 이어서 한다.

    python3 scripts/tc_iterate.py --max_iter 4 [--out results/tc] [--mode TC|TCb] [--n 3000] [--explore_lane 0.5] [--roots TC.pt]

반복 k:
  1. 수집: π_k 로 학습 시드 N 회, 탐색 ε 0.05·기다리기 0.1 (k = 0: 2 s, k ≥ 1: 2 s + TC_{k-1})
  2. 학습: iter0..k 데이터 전체로 TC_k (train_tc)
  3. 주행 검증: 검증 시드 6000–6299 에서 2 s + TC_k 를 2 s·5 s 와 같은 시드로 비교
멈춤: 두 반복 연속 검증 J 개선 없음 또는 --max_iter. 끝에 검증 J 최고 TC 로 시험 시드 3000–3999.
기준 에피소드: results/ref/{val_2s,val_5s,test_2s,test_5s}.json (없으면 여기서 돌린다).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
SIM = R / "sim"
TC = R / "results" / "tc"
REF = R / "results" / "ref"
BASE = "rx_mt_pf_fl_tfs_rr_ng6_nv20_nm_h10_rf10"
VAL0, VALN = 6000, 300
TEST0, TESTN = 3000, 1000
EXPLORE = ["--explore", "0.05", "--explore_wait", "0.1"]
sys.path.insert(0, str(R / "scripts"))
from rc_iterate import cool, episodes_of, summarize, workers  # noqa: E402


def log(*a):
    msg = time.strftime("%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    TC.mkdir(parents=True, exist_ok=True)
    with open(TC / "iterate.log", "a") as f:
        f.write(msg + "\n")


def run(cmd, cwd=SIM):
    log("$", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        log("FAILED", r.stderr[-2000:])
        raise SystemExit(1)
    return r.stdout


def collect(planner, seed0, n, prefix, extra, chunk=500):
    for s in range(seed0, seed0 + n, chunk):
        out = Path(f"{prefix}_s{s}.npz")
        if out.exists():
            continue
        cool()
        run(["python3", "-m", "smpc_sim.rl.collect_tc", "--planner", planner, "--seed0", str(s),
             "--n", str(min(chunk, seed0 + n - s)), "--workers", str(workers()), "--out", str(out)] + extra)
    return episodes_of(sorted(Path(prefix).parent.glob(Path(prefix).name + "_s*.json")))


def ref(name, planner, seed0, n):
    p = REF / f"{name}.json"
    if not p.exists():
        eps = collect(planner, seed0, n, str(REF / name / "run"), [])
        REF.mkdir(parents=True, exist_ok=True)
        json.dump({"episodes": list(eps.values())}, open(p, "w"))
    return {e["seed"]: e for e in json.load(open(p))["episodes"] if seed0 <= e["seed"] < seed0 + n}


def main():
    global TC
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_iter", type=int, default=4)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default=str(TC), help="결과 폴더. iter0/data_s*.npz·json 이 이미 있으면 그 수집은 건너뛴다")
    ap.add_argument("--mode", default="TC", choices=["TC", "TCb"])
    ap.add_argument("--explore_lane", type=float, default=None, help="탐색 시작: 차선마다 탐색할 확률(collect_tc 참고)")
    ap.add_argument("--no_test", action="store_true", help="반복만 하고 시험 시드는 돌리지 않는다")
    ap.add_argument("--roots", default=None, metavar="TC.pt",
                    help="루트 기록 수집 + 루트 값 맞춤 학습(train_tc --roots). 값 = k = 0 수집에서 잎 특징을 만들 TC 가중치"
                         "(_TCr@: 판단은 2 s 그대로)")
    ap.add_argument("--explore_decision", type=float, default=None,
                    help="결정 탐색(collect_tc 참고). 주면 틱마다의 기다리기 탐색(0.1)은 끈다")
    ap.add_argument("--seed0", type=int, default=50000, help="k 회차 수집 시드 = seed0 + 10000·k")
    ap.add_argument("--explore_rand", type=float, default=0.05, help="--explore_decision 과 함께 쓰는 무작위 후보 탐색 확률")
    a = ap.parse_args()
    explore = EXPLORE + (["--explore_lane", str(a.explore_lane)] if a.explore_lane is not None else [])
    if a.explore_decision is not None:
        explore = ["--explore", str(a.explore_rand), "--explore_wait", "0", "--explore_decision", str(a.explore_decision)] + explore[4:]
    if a.roots:
        explore = explore + ["--record"]
    TC = Path(a.out).resolve()
    tag = f"_{a.mode}@"
    st_path = TC / "state.json"
    st = json.load(open(st_path)) if st_path.exists() else {"iters": []}
    v2, v5 = ref("val_2s", BASE, VAL0, VALN), ref("val_5s", BASE.replace("_h10_", "_h25_"), VAL0, VALN)
    log("val 2 s:", json.dumps(summarize(v2)), "5 s:", json.dumps(summarize(v5)))
    k = len(st["iters"])
    while k < a.max_iter:
        it = {"k": k, "N": a.n}
        pol = BASE if k == 0 else f"{BASE}{tag}{st['iters'][-1]['model']}"
        if k == 0 and a.roots:
            pol = f"{BASE}_TCr@{Path(a.roots).resolve()}"
        it["collect"] = summarize(collect(pol, a.seed0 + 10000 * k, a.n, str(TC / f"iter{k}" / "data"), explore))
        log(f"iter{k} collect ({pol.split('@')[0]}):", json.dumps(it["collect"]))
        model = TC / f"TC_iter{k}.pt"
        if not model.exists():
            cool()
            data = sorted(str(p) for j in range(k + 1) for p in (TC / f"iter{j}").glob("data_s*.npz")
                          if not p.name.endswith("_roots.npz"))
            run(["python3", "-m", "smpc_sim.rl.train_tc", "--data", *data, "--out", str(model)] + (["--roots"] if a.roots else []))
        import torch
        it["model"] = str(model)
        it["model_check"] = torch.load(model, map_location="cpu", weights_only=True)["meta"]
        log(f"iter{k} model:", json.dumps({x: it["model_check"][x] for x in ("rows", "val", "val_moving_r2", "val_stand_r2", "roots")
                                           if x in it["model_check"]}))
        ev = collect(f"{BASE}{tag}{model}", VAL0, VALN, str(TC / "val" / f"iter{k}"), [], chunk=300)
        it["val"] = summarize(ev, v2)
        it["val"]["dJ_vs_5s"] = summarize(ev, v5)["dJ_vs_ref"]
        it["val_J"] = it["val"]["J"]
        log(f"iter{k} val:", json.dumps(it["val"]))
        st["iters"].append(it)
        json.dump(st, open(st_path, "w"), indent=1, ensure_ascii=False)
        Js = [x["val_J"] for x in st["iters"]]
        if len(Js) >= 3 and min(Js[-2:]) >= min(Js[:-2]) - 0.05:
            log("두 반복 연속 검증 J 개선 없음 → 멈춤")
            break
        k += 1
    if a.no_test:
        return
    bi = min(st["iters"], key=lambda x: x["val_J"])
    t2, t5 = ref("test_2s", BASE, TEST0, TESTN), ref("test_5s", BASE.replace("_h10_", "_h25_"), TEST0, TESTN)
    te = collect(f"{BASE}{tag}{bi['model']}", TEST0, TESTN, str(TC / "test" / f"best_iter{bi['k']}"), [])
    out = {"model": bi["model"], "TC": summarize(te), "2s": summarize(t2), "5s": summarize(t5),
           "TC_vs_2s": summarize(te, t2)["dJ_vs_ref"], "TC_vs_5s": summarize(te, t5)["dJ_vs_ref"]}
    st["test"] = out
    json.dump(st, open(st_path, "w"), indent=1, ensure_ascii=False)
    log("TEST:", json.dumps(out))


if __name__ == "__main__":
    main()
