#!/usr/bin/env python3
"""RC 근사 정책 반복 자동 실행(docs/rc_design.md). 다시 실행하면 이미 있는 결과는 건너뛰고 이어서 한다.

    python3 scripts/rc_iterate.py --max_iter 4

반복 k:
  1. 수집: π_k 로 학습 시드 N_k 회(k = 0: RC 없는 2 s, k ≥ 1: 2 s + RC_{k-1}(θ_{k-1}) + RC 탐색)
  2. 학습: iter0..k 데이터 전체로 RC_k (train_rc)
  3. 모델 검증: 교차검증 분석(rc_model_analysis) + 탐색 없는 검증 시드 데이터에서 AUC
  4. 주행 검증: 검증 시드 6000–6299 에서 θ 격자, RC 없는 2 s 와 같은 시드로 비교 → θ_k = J 최소
  5. 늘릴지 판단(규칙은 rc_design.md 4 절) → 다음 반복 설정
멈춤: 두 반복 연속 검증 J 개선 없음 또는 --max_iter. 끝에 최고 (k, θ) 로 시험 시드 2000–2999 1000 회(2·5·8 s 와 짝 비교).
CPU: 매 실행 전 70 °C 아래까지 기다리고(sensors), 워커 수는 results/rc/workers.txt(없으면 5).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]           # 저장소 루트
SIM = R / "sim"
RC = R / "results" / "rc"
BASE = "rx_mt_pf_fl_tfs_rr_ng6_nv20_nm_h10_rf10"
VAL0, VALN = 6000, 300
TEST0, TESTN = 2000, 1000
THETAS = (0.2, 0.3, 0.5, 0.7, 0.85)
EXPLORE = ["--explore", "0.05", "--explore_wait", "0.1"]
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(R / "scripts"))


def log(*a):
    msg = time.strftime("%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(RC / "iterate.log", "a") as f:
        f.write(msg + "\n")


def temp():
    try:
        out = subprocess.run(["sensors"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return 0
    for line in out.splitlines():
        if line.startswith("Package id 0"):
            return int(float(line.split()[3].strip("+°C")))
    return 0


def cool():
    while temp() >= 70:
        time.sleep(10)


def workers():
    try:
        return int((RC / "workers.txt").read_text().strip())
    except Exception:
        return 5


def run(cmd, cwd=SIM):
    log("$", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        log("FAILED", r.stderr[-2000:])
        raise SystemExit(1)
    return r.stdout


def collect(planner, seed0, n, prefix, extra):
    """500 회 덩어리로 수집(덩어리마다 온도 대기·워커 수 다시 읽기). 이미 있는 덩어리는 건너뛴다."""
    for s in range(seed0, seed0 + n, 500):
        out = Path(f"{prefix}_s{s}.npz")
        if out.exists():
            continue
        cool()
        run(["python3", "-m", "smpc_sim.rl.collect_rc", "--planner", planner, "--seed0", str(s), "--n", str(min(500, seed0 + n - s)),
             "--workers", str(workers()), "--out", str(out)] + extra)


def episodes_of(json_paths):
    eps = []
    for p in json_paths:
        eps += json.load(open(p))["episodes"]
    return {e["seed"]: e for e in eps}


def Jv(e):
    return (e["t_merge"] if e["success"] else 120.0) + 60.0 * e["collision"] + 2.0 * e["aborts"]


def summarize(eps, ref=None):
    s = sorted(eps)
    J = np.array([Jv(eps[k]) for k in s])
    d = {"n": len(s), "J": float(J.mean()), "success_%": 100 * float(np.mean([eps[k]["success"] for k in s])),
         "collisions": int(sum(eps[k]["collision"] for k in s)), "aborts_per_ep": float(np.mean([eps[k]["aborts"] for k in s])),
         "timeouts": int(sum((not eps[k]["success"]) and eps[k]["t_end"] >= 119.0 for k in s)),
         "minsep<0.5": int(sum(eps[k]["min_sep"] < 0.5 for k in s)),
         "lane_end_stops": float(np.mean([eps[k].get("lane_end_stops", 0) for k in s])),
         "rc_vetoes_per_ep": float(np.mean([eps[k].get("rc_vetoes", 0) for k in s]))}
    if ref is not None:
        diff = J - np.array([Jv(ref[k]) for k in s])
        rng = np.random.default_rng(0)
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        d["dJ_vs_ref"] = [float(np.percentile(bs, 2.5)), float(diff.mean()), float(np.percentile(bs, 97.5))]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_iter", type=int, default=4)
    ap.add_argument("--n0", type=int, default=3000)
    a = ap.parse_args()
    st_path = RC / "state.json"
    st = json.load(open(st_path)) if st_path.exists() else {"iters": [], "N": a.n0, "hidden": 64, "M": 5}

    # RC 없는 2 s 검증 기준(탐색 없음): 주행 비교 기준 + 수집 정책 그대로의 보류 데이터(분포 이동 확인)
    collect(BASE, VAL0, VALN, str(RC / "val" / "base"), [])
    ref = episodes_of(sorted((RC / "val").glob("base_s*.json")))
    log("val 2 s:", json.dumps(summarize(ref)))

    k = len(st["iters"])
    while k < a.max_iter:
        it = {"k": k, "N": st["N"], "hidden": st["hidden"], "M": st["M"]}
        # 1. 수집
        if k == 0:
            collect(BASE, 30000, st["N"], str(RC / "iter0" / "data"), EXPLORE)
        else:
            prev = st["iters"][-1]
            seed0 = 30000 + 10000 * k
            pol = f"{BASE}_RC{int(round(prev['theta'] * 100))}@{prev['model']}"
            collect(pol, seed0, st["N"], str(RC / f"iter{k}" / "data"), EXPLORE + ["--rc_explore", "0.3"])
        it["collect"] = summarize(episodes_of(sorted((RC / f"iter{k}").glob("data_s*.json"))))
        log(f"iter{k} collect:", json.dumps(it["collect"]))
        # 2. 학습(지금까지 데이터 전체)
        data = sorted(str(p) for j in range(k + 1) for p in (RC / f"iter{j}").glob("data_s*.npz"))
        model = RC / f"RC_iter{k}.pt"
        if not model.exists():
            cool()
            run(["python3", "-m", "smpc_sim.rl.train_rc", "--data", *data, "--out", str(model), "--hidden", str(st["hidden"]),
                 "--M", str(st["M"])])
        it["model"] = str(model)
        # 3. 모델 검증
        ana = RC / f"analysis_iter{k}"
        if not (ana / "report.json").exists():
            cool()
            run(["python3", "scripts/rc_model_analysis.py", "--data", str(RC / f"iter[0-{k}]" / "data_s*.npz"),
                 "--out", str(ana), "--hidden", str(st["hidden"]), "--M", str(st["M"])], cwd=R)
        rep = json.load(open(ana / "report.json"))
        from smpc_sim.rl.risk_critic import load_rc_fn
        from rc_model_analysis import auc
        fn = load_rc_fn(model)
        Vx = np.concatenate([np.load(p)["X"] for p in sorted((RC / "val").glob("base_s*.npz"))])
        Vy = np.concatenate([np.load(p)["y"] for p in sorted((RC / "val").glob("base_s*.npz"))])
        it["model_check"] = {"cv_auc": rep["overall"]["RC"]["auc"], "cv_auc_logreg": rep["overall"]["logreg"]["auc"],
                             "diff_ci": rep["ci"]["diff_auc"]["RC-logreg"], "moving_auc": rep["moving"]["RC"],
                             "stand_auc": rep["standstill"]["RC"], "ece": rep["overall"]["RC"]["ece"],
                             "curve": rep.get("learning_curve_auc"), "holdout_onpolicy_auc": auc(Vy, fn(Vx)),
                             "holdout_n": int(len(Vy)), "holdout_bad": int(Vy.sum())}
        mc = it["model_check"]
        log(f"iter{k} model:", json.dumps(mc))
        # 4. 주행 검증(θ 격자)
        res = {}
        for th in THETAS:
            pol = f"{BASE}_RC{int(round(th * 100))}@{model}"
            pre = RC / "val" / f"iter{k}_th{int(round(th * 100))}"
            collect(pol, VAL0, VALN, str(pre), [])
            res[th] = summarize(episodes_of(sorted(pre.parent.glob(pre.name + "_s*.json"))), ref)
            log(f"iter{k} θ={th}:", json.dumps(res[th]))
        best = min(THETAS, key=lambda t: (res[t]["J"], res[t]["aborts_per_ep"]))
        it["val"] = {str(t): r for t, r in res.items()}
        it["theta"], it["val_J"] = best, res[best]["J"]
        log(f"iter{k} best θ={best} J={res[best]['J']:.2f} (2 s {summarize(ref)['J']:.2f})")
        # 5. 늘릴지 판단(rc_design.md 4 절)
        model_ok = (mc["diff_ci"][0] > 0 and mc["moving_auc"] >= 0.85 and mc["stand_auc"] >= 0.80 and mc["ece"] <= 0.02)
        curve_flat = mc["curve"] is not None and mc["curve"]["1.0"] - mc["curve"]["0.5"] < 0.005
        drive_ok = res[best]["dJ_vs_ref"][2] < 0 and res[best]["collisions"] <= summarize(ref)["collisions"]
        it["decision"] = {"model_ok": model_ok, "curve_flat": curve_flat, "drive_ok": drive_ok}
        if not model_ok or not curve_flat or not drive_ok:
            st["N"] = st["N"] + 3000
        prev_best = min([x["val_J"] for x in st["iters"]], default=np.inf)
        if len(st["iters"]) >= 1 and it["val_J"] >= prev_best - 0.05 and model_ok and curve_flat:
            st["hidden"], st["M"] = min(256, st["hidden"] * 2), min(8, st["M"] + 3)    # 데이터로 안 늘면 모델을 키운다
        log(f"iter{k} decision:", json.dumps(it["decision"]), "next N", st["N"], "hidden", st["hidden"], "M", st["M"])
        st["iters"].append(it)
        json.dump(st, open(st_path, "w"), indent=1, ensure_ascii=False)
        Js = [x["val_J"] for x in st["iters"]]
        if len(Js) >= 3 and min(Js[-2:]) >= min(Js[:-2]) - 0.05:
            log("두 반복 연속 검증 J 개선 없음 → 멈춤")
            break
        k += 1

    # 최종 시험: 검증 J 최고 (k, θ)
    bi = min(st["iters"], key=lambda x: x["val_J"])
    pol = f"{BASE}_RC{int(round(bi['theta'] * 100))}@{bi['model']}"
    pre = RC / "test" / f"best_iter{bi['k']}_th{int(round(bi['theta'] * 100))}"
    collect(pol, TEST0, TESTN, str(pre), [])
    te = episodes_of(sorted(pre.parent.glob(pre.name + "_s*.json")))
    out = {"policy": pol}
    for name, h in (("2s", 10), ("5s", 25), ("8s", 40)):
        pre_h = RC / "test" / f"fixed_h{h}"
        collect(BASE.replace("_h10_", f"_h{h}_"), TEST0, TESTN, str(pre_h), [])
        ref_t = episodes_of(sorted(pre_h.parent.glob(pre_h.name + "_s*.json")))
        out[name] = summarize(ref_t)
        out[f"RC_vs_{name}"] = summarize(te, ref_t)["dJ_vs_ref"]
    out["RC"] = summarize(te)
    st["test"] = out
    json.dump(st, open(st_path, "w"), indent=1, ensure_ascii=False)
    log("TEST:", json.dumps(out))


if __name__ == "__main__":
    main()
