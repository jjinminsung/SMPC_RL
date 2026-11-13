"""RC(위험 가치망) 학습 데이터 모으기: 정책 π 로 에피소드를 돌리며 '지금 바꾸기'를 시작한 순간마다 (특징, 결과)를 남긴다.

    cd sim
    python3 -m smpc_sim.rl.collect_rc --planner rx_mt_pf_fl_tfs_rr_ng6_nv20_nm_h10_rf10 --seed0 30000 --n 500 \
        --explore 0.05 --explore_wait 0.1 --out ../results/rc/iter0/data_s30000.npz

결과(라벨) = 그 차선 변경 하나의 끝(몬테카를로 리턴, 할인 없음):
    bad = 1  중단(복귀)으로 끝남, 또는 차선 변경 중 충돌, 또는 차선 변경 중 TTC < 1 s (0.5 s 구간 최소값)
    bad = 0  완료
    바꾸기 요청 뒤 1.5 s 안에 감독기가 승인하지 않은 요청은 라벨 없음(버림).
탐색: --explore (판단마다 실행 가능한 후보 중 무작위), --explore_wait (바꿀 수 있어도 0.5–3 s 기다림),
      --rc_explore (RC 정책일 때 차선마다 이 확률로 RC 로 막지 않음).
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

from run_batch import planner_factory  # noqa: E402
from smpc_sim.world import World, EpisodeConfig  # noqa: E402

CFG = {"switch_delay_sec": 0.55, "acc_floor_real": True, "acc_tf_on_start": True}
NEAR_TTC_S = 1.0


def label_commits(res, t0, log):
    """[(t_commit_rel, 특징, rc_q, bad, why)]"""
    ev = res.events
    out = []
    end_t = res.t_end
    log = [(t_abs - t0, x, q) for t_abs, x, q in log]
    for acc in (e for e in ev if e[0] == "accept"):
        # 판단은 승인 대기(switch delay) 동안에도 매 틱 '바꾸기'를 낸다. 승인 하나에는 그 직전 1.5 s 안의 첫 요청(결정한 순간) 하나만 쓴다.
        ta = acc[1]
        first = next(((tc, x, q) for tc, x, q in log if ta - 1.5 - 1e-6 <= tc <= ta + 1e-6), None)
        if first is None:
            continue
        tc, x, q = first
        fin = next((e for e in ev if e[0] in ("complete", "abort") and e[1] >= ta - 1e-9), None)
        t_fin = fin[1] if fin is not None else end_t
        why = ""
        if fin is not None and fin[0] == "abort":
            why = "abort"
        elif fin is None and res.collision:
            why = "collision"
        elif fin is None and not res.success:
            why = "offroad" if end_t < 119.0 else "timeout"
        if not why:
            for b0, ttc, _gap in res.risk_buckets:
                if b0 + 0.5 > ta and b0 <= t_fin and ttc < NEAR_TTC_S:
                    why = "near"
                    break
        out.append((round(tc, 2), x, q, int(why in ("abort", "collision", "near", "offroad")), why))
    return out


def run_seed(args):
    name, seed, ex, exw, rcx = args
    cfg = EpisodeConfig(seed=seed, record_risk=True, explore_eps=ex, explore_wait_eps=exw, **CFG)
    w = World(planner_factory(name), cfg)
    pl = w.planner
    pl.rc_log = []
    pl.rc_explore = rcx
    res = w.run()
    lab = label_commits(res, w.t0, pl.rc_log)
    return {"seed": seed, "success": res.success, "collision": res.collision, "t_end": res.t_end, "t_merge": res.t_merge,
            "aborts": res.aborts, "min_sep": res.min_sep, "min_ttc": res.min_ttc, "lane_end_stops": res.lane_end_stops,
            "stop_time": res.stop_time, "nudges": res.nudges, "rc_vetoes": getattr(pl, "rc_vetoes", 0), "commits": lab}


def J(e):
    return (e["t_merge"] if e["success"] else 120.0) + 60.0 * e["collision"] + 2.0 * e["aborts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", required=True)
    ap.add_argument("--seed0", type=int, default=30000)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--explore", type=float, default=0.0)
    ap.add_argument("--explore_wait", type=float, default=0.0)
    ap.add_argument("--rc_explore", type=float, default=0.0)
    ap.add_argument("--out", required=True, help=".npz (특징·라벨) + 같은 이름 .json (에피소드 요약)")
    a = ap.parse_args()
    t_start = time.time()
    seeds = list(range(a.seed0, a.seed0 + a.n))
    with Pool(a.workers) as pool:
        eps = pool.map(run_seed, [(a.planner, s, a.explore, a.explore_wait, a.rc_explore) for s in seeds], chunksize=1)
    X, y, E, why, q = [], [], [], [], []
    for e in eps:
        for tc, x, qq, bad, wy in e["commits"]:
            X.append(x); y.append(bad); E.append(e["seed"]); why.append(wy); q.append(np.nan if qq is None else qq)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=np.array(X, np.float32), y=np.array(y, np.int8), ep=np.array(E), why=np.array(why), q=np.array(q))
    summ = {"planner": a.planner, "cfg": CFG, "explore": a.explore, "explore_wait": a.explore_wait, "rc_explore": a.rc_explore,
            "n": len(eps), "J": float(np.mean([J(e) for e in eps])), "success": float(np.mean([e["success"] for e in eps])),
            "collisions": int(sum(e["collision"] for e in eps)), "aborts_per_ep": float(np.mean([e["aborts"] for e in eps])),
            "commits": len(y), "bad": int(sum(y)), "why": {k: why.count(k) for k in set(why)},
            "rc_vetoes_per_ep": float(np.mean([e["rc_vetoes"] for e in eps])), "wall_s": round(time.time() - t_start, 1)}
    json.dump({"summary": summ, "episodes": [{k: v for k, v in e.items() if k != "commits"} for e in eps]},
              open(out.with_suffix(".json"), "w"))
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
