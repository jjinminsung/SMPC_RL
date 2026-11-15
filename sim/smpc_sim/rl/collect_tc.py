"""TC(종단 비용) 학습 데이터: 정책 π 로 에피소드를 돌리며 0.5 s 마다 (실제 상태 특징, 남은 J)를 남긴다.

    cd sim
    python3 -m smpc_sim.rl.collect_tc --planner rx_mt_pf_fl_tfs_rr_ng6_nv20_nm_h10_rf10 --seed0 50000 --n 500 \\
        --explore 0.05 --explore_wait 0.1 --out ../results/tc/iter0/data_s50000.npz

남은 J = (T_end - t) + 2·(t 뒤 중단 수) + 60·충돌, T_end = 합류 시각(실패면 120 s). rl/terminal_cost.py 참고.
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
from smpc_sim.rl.terminal_cost import FAIL_TIME, j_to_go  # noqa: E402

CFG = {"switch_delay_sec": 0.55, "acc_floor_real": True, "acc_tf_on_start": True}


def J(e):
    return (e["t_merge"] if e["success"] else FAIL_TIME) + 60.0 * e["collision"] + 2.0 * e["aborts"]


ACT = {"change_now": 0, "hold": 1, "yield": 1, "accel": 1, "keep": 2, "nudge": 3}
REC_DT = 0.25         # 루트 기록 간격 [s] (지금 바꾸기·탐색 틱은 모두 남긴다)


def roots_of(recs, t0):
    """planner.tc_records → 가지 수가 다른 기록 목록(시각은 에피소드 기준)."""
    out, last = [], -1e9
    for r in recs:
        t = r["t"] - t0
        if r["action"] != "change_now" and not r["explored"] and not r.get("forced") and not r.get("wants") \
                and t - last < REC_DT - 1e-6:
            continue
        last = t
        out.append((round(t, 2), r["lane"], ACT.get(r["action"], 3), r["explored"], r["tau"], r["Qc"], r["Qk"], r["Qsel"],
                    np.asarray(r["p"], np.float32), np.asarray(r["c"], np.float32), np.asarray(r["F"], np.float32),
                    np.asarray(r["term"], bool), np.asarray(r["t_arr"], np.float32), r.get("chg"), r.get("keep"), bool(r.get("forced")),
                    bool(r.get("wants")), bool(r.get("gate_dis"))))
    return out


def pack_roots(per_ep):
    """[(seed, 기록 목록)] → 가지 축을 최대 가지 수로 채운 배열들(mask = 실제 가지)."""
    rows = [(s, r) for s, rs in per_ep for r in rs if len(r) >= 13]
    n = len(rows)
    Bm = max((len(r[8]) for _, r in rows), default=1)
    D = next((r[10].shape[1] for _, r in rows), 1)
    z = {k: np.zeros(n, np.float32) for k in ("t", "tau", "Qc", "Qk", "Qsel")}
    z.update(ep=np.zeros(n, np.int64), lane=np.zeros(n, np.int8), act=np.zeros(n, np.int8), explored=np.zeros(n, bool),
             P=np.zeros((n, Bm), np.float32), C=np.zeros((n, Bm), np.float32), F=np.zeros((n, Bm, D), np.float32),
             term=np.zeros((n, Bm), bool), t_arr=np.zeros((n, Bm), np.float32), mask=np.zeros((n, Bm), bool),
             forced=np.zeros(n, bool), wants=np.zeros(n, bool), gate_dis=np.zeros(n, bool))
    for x in ("chg", "keep"):         # 같은 틱의 지금 바꾸기·기다리기 루트 잎(TCb 오프라인 평가용, ok = 기회 제약을 지키는 대응이 있음)
        z.update({f"{x}_ok": np.zeros(n, bool), f"{x}_tau": np.zeros(n, np.float32), f"{x}_C": np.zeros((n, Bm), np.float32),
                  f"{x}_F": np.zeros((n, Bm, D), np.float32), f"{x}_term": np.zeros((n, Bm), bool),
                  f"{x}_t_arr": np.zeros((n, Bm), np.float32)})
    for j, (s, r) in enumerate(rows):
        b = len(r[8])
        for x, lf in (("chg", r[13]), ("keep", r[14])):
            if lf is not None:
                z[f"{x}_ok"][j], z[f"{x}_tau"][j] = True, lf["tau"]
                z[f"{x}_C"][j, :b], z[f"{x}_F"][j, :b] = lf["c"], lf["F"]
                z[f"{x}_term"][j, :b], z[f"{x}_t_arr"][j, :b] = lf["term"], lf["t_arr"]
        z["ep"][j], z["t"][j], z["lane"][j], z["act"][j], z["explored"][j] = s, r[0], r[1], r[2], r[3]
        z["tau"][j], z["Qc"][j], z["Qk"][j], z["Qsel"][j] = r[4], r[5], r[6], r[7]
        z["P"][j, :b], z["C"][j, :b], z["F"][j, :b], z["term"][j, :b], z["t_arr"][j, :b] = r[8], r[9], r[10], r[11], r[12]
        z["mask"][j, :b] = True
        z["forced"][j] = len(r) > 15 and r[15]
        z["wants"][j] = len(r) > 16 and r[16]
        z["gate_dis"][j] = len(r) > 17 and r[17]
    return z


def run_seed(args):
    name, seed, ex, exw, exl, rec, exd, exm, rlog, lf = args
    cfg = EpisodeConfig(seed=seed, record_states=True, explore_eps=ex, explore_wait_eps=exw, explore_lane_prob=exl,
                        tc_record=rec, explore_decision=exd, explore_decision_mode=exm, risk_log=rlog, leaf_log=lf, **CFG)
    w = World(planner_factory(name), cfg)
    res = w.run()
    t_end_eff = res.t_merge if res.success else FAIL_TIME
    aborts = [ev[1] for ev in res.events if ev[0] == "abort"]
    ts = np.array([s[0] for s in res.samples])
    G = j_to_go(ts, t_end_eff, aborts, res.collision) if len(ts) else np.zeros(0)
    ep = {"seed": seed, "success": res.success, "collision": res.collision, "t_end": res.t_end, "t_merge": res.t_merge,
          "aborts": res.aborts, "min_sep": res.min_sep, "min_ttc": res.min_ttc, "lane_end_stops": res.lane_end_stops,
          "stop_time": res.stop_time, "nudges": res.nudges, "tc_keeps": getattr(w.planner, "tc_keeps", 0),
          "tc_declines": getattr(w.planner, "tc_declines", 0), "completes": list(res.lane_change_times), "abort_times": aborts,
          "accept_times": [round(ev[1], 2) for ev in res.events if ev[0] == "accept"]}
    if exl is not None:
        ep["explore_events"] = [(round(a - w.t0, 2), round(b - w.t0, 2)) for a, b in w.planner.explore_events]
        ep["decision_events"] = [(round(a - w.t0, 2), round(b - w.t0, 2)) for a, b in w.planner.explore_decisions]
    if rlog:
        ep["risk_log"] = [(round(a - w.t0, 2), float(r), k) for a, r, k in w.planner.risk_log]
    X = np.array([s[1] for s in res.samples], np.float32).reshape(len(ts), -1)
    roots = roots_of(w.planner.tc_records, w.t0) if rec else []
    leaf = None
    if lf:      # 명령 틱마다 가지별 잎(그 가지가 고른 대응의 잎), 무게 p_b
        rows, wts, arrs, rid, rt = [], [], [], [], []
        for k, (now, F, arr, p, q) in enumerate(w.planner.leaf_log):
            rt.append(round(now - w.t0, 2))
            for b in range(len(p)):
                rows.append(F[2 * b + q[b]]); wts.append(p[b]); arrs.append(bool(arr[2 * b + q[b]])); rid.append(k)
        D = X.shape[1] + 1 if len(X) else 21
        leaf = (np.array(rows, np.float32).reshape(-1, D), np.array(wts, np.float32), np.array(arrs, bool),
                np.array(rid, np.int32), np.array(rt, np.float32))
    return ep, X, G.astype(np.float32), ts.astype(np.float32), roots, leaf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", required=True)
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--explore", type=float, default=0.0)
    ap.add_argument("--explore_wait", type=float, default=0.0)
    ap.add_argument("--explore_lane", type=float, default=None, help="탐색 시작: 차선마다 탐색할 확률(탐색은 차선당 한 번)")
    ap.add_argument("--explore_decision", type=float, default=0.0,
                    help="탐색할 차선에서 탐욕 정책이 지금 바꾸기를 고른 틱에 이 확률로 0.5–3 s 기다린다(--explore_lane 필요)")
    ap.add_argument("--explore_decision_mode", default="wait", choices=["wait", "yield", "change", "both"],
                    help="yield = 1–2.5 s 감속 양보, change = (G5 플래너) 5 s 가 막은 바꾸기를 그대로 실행, "
                         "both = 막힌 결정은 바꾸기·통과한 결정은 기다리기로(차선당 한 번)")
    ap.add_argument("--record", action="store_true", help="판단 틱의 루트 잎 기록(플래너에 TC 가중치 필요, _TCr@ 이면 판단은 그대로)"
                    " → <out>_roots.npz")
    ap.add_argument("--leaf_log", action="store_true", help="지금 바꾸기를 명령한 틱의 루트 0 잎(가지별) 기록(RV·RA 잎 학습용)")
    ap.add_argument("--risk_log", action="store_true", help="실제 주행 중 플래너가 계산한 지평선 안 위험 기록(RL 위험 값 학습용)")
    ap.add_argument("--out", required=True, help=".npz (특징·남은 J) + 같은 이름 .json (에피소드 요약)")
    a = ap.parse_args()
    t0 = time.time()
    seeds = list(range(a.seed0, a.seed0 + a.n))
    with Pool(a.workers) as pool:
        res = pool.map(run_seed, [(a.planner, s, a.explore, a.explore_wait, a.explore_lane, a.record, a.explore_decision, a.explore_decision_mode, a.risk_log, a.leaf_log) for s in seeds], chunksize=1)
    eps = [r[0] for r in res]
    X = np.concatenate([r[1] for r in res])
    G = np.concatenate([r[2] for r in res])
    T = np.concatenate([r[3] for r in res])
    E = np.concatenate([np.full(len(r[2]), r[0]["seed"]) for r in res])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, G=G, t=T, ep=E)
    if a.record:
        np.savez(out.with_name(out.stem + "_roots.npz"), **pack_roots([(r[0]["seed"], r[4]) for r in res]))
    if a.leaf_log:
        LF, LW, LA, LREC, REC_ep, REC_t, off = [], [], [], [], [], [], 0
        for r in res:
            F, wts, arrs, rid, rt = r[5]
            LF.append(F); LW.append(wts); LA.append(arrs); LREC.append(rid + off)
            REC_ep.append(np.full(len(rt), r[0]["seed"])); REC_t.append(rt); off += len(rt)
        np.savez(out.with_name(out.stem + "_leaf.npz"), LF=np.concatenate(LF), LW=np.concatenate(LW), LA=np.concatenate(LA),
                 LREC=np.concatenate(LREC), REC_ep=np.concatenate(REC_ep), REC_t=np.concatenate(REC_t))
    summ = {"planner": a.planner, "cfg": CFG, "explore": a.explore, "explore_wait": a.explore_wait, "explore_lane": a.explore_lane,
            "explore_decision": a.explore_decision, "explore_decision_mode": a.explore_decision_mode, "n": len(eps),
            "J": float(np.mean([J(e) for e in eps])), "success": float(np.mean([e["success"] for e in eps])),
            "collisions": int(sum(e["collision"] for e in eps)), "aborts_per_ep": float(np.mean([e["aborts"] for e in eps])),
            "samples": int(len(G)), "tc_keeps_per_ep": float(np.mean([e["tc_keeps"] for e in eps])),
            "tc_declines_per_ep": float(np.mean([e["tc_declines"] for e in eps])), "roots": int(sum(len(r[4]) for r in res)),
            "wall_s": round(time.time() - t0, 1)}
    json.dump({"summary": summ, "episodes": eps}, open(out.with_suffix(".json"), "w"))
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
