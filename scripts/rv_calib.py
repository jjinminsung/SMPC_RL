"""RV(지평선 뒤 위험 값) 보정 검증 데이터: 배치 플래너(2 s + RVg + VTp)로 시험 시드를 돌리며
지금 바꾸기를 명령한 틱마다
  - 루트 0 잎(가지별, collect_tc --leaf_log 와 같은 형식) → 나중에 RVT 예측을 다시 계산
  - 같은 틱, 같은 관측으로 8 s 플래너(그림자, 판단에 영향 없음)의 지금 바꾸기 루트 걸음별 위험 Σ_b p_b r_b(k)
  - 배치 플래너 자신의 지평선 안(2 s) 걸음별 위험
을 남긴다. 실제 결과(위험 기록, 승인·완료·중단 시각, 충돌)는 에피소드 요약에.

    cd sim && python3 ../scripts/rv_calib.py collect 3000 1000 6 ../results/rv/calib/calib_s3000
    cd sim && python3 ../scripts/rv_calib.py analyze ../results/rv/calib/calib_s3000
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
import numpy as np  # noqa: E402

B = "rx_mt_pf_fl_tfs_rr_ng6_nv20_nm_h10_rf10"
MAIN = f"{B}_RVg_VTp@{ROOT}/results/rv/RVT_1.pt+{ROOT}/results/tc/VT_lane0.pt"
SHADOW = B.replace("_h10_", "_h40_")
RV_PATH = ROOT / "results/rv/RVT_1.pt"
EPS = 0.04


def _curve(risk, allowed, p, combo, ret_code):
    """루트 0 의 고른 대응(가지별 계속/복귀)의 걸음별 위험 Σ_b p_b r_b(k)."""
    q = (np.asarray(combo[1:]) == ret_code).astype(int)
    return sum(p[b] * risk[b, q[b]] for b in range(len(p)))


def _hook_solve(pl, cap):
    orig = pl._solve

    def solve(r, br, root_cost, mask, eps):
        out = orig(r, br, root_cost, mask, eps)
        if getattr(pl, "_roots_eval", False) and not getattr(pl, "_mon_eval", False) and "risk" not in cap:
            cap.update(risk=r["risk"][:, :, 0, :].copy(), allowed=r["allowed"][:, :, 0].copy(),
                       p=np.array([b[3] for b in br]), feas0=bool(out[0][0]), combo0=np.array(out[2][0]),
                       mask0=bool(mask[0]), eps=float(eps), peak0=float(out[3][0]))
        return out
    pl._solve = solve


def run_seed(seed):
    from smpc_sim.rl import collect_tc
    from run_batch import planner_factory
    from smpc_sim.world import World, EpisodeConfig
    from smpc_sim.tree_reactive import TREE_RETURN

    cfg = EpisodeConfig(seed=seed, risk_log=True, leaf_log=True, **collect_tc.CFG)
    w = World(planner_factory(MAIN), cfg)
    sh = planner_factory(SHADOW)(w.P, w.track)
    for k, v in dict(explore_eps=0.0, explore_wait_eps=0.0, explore_rng=np.random.default_rng(0), explore_lane_prob=None,
                     explore_decision=0.0, explore_decision_mode="wait", explore_events=[], explore_decisions=[],
                     tc_record=False, tc_records=[], crn_wait=cfg.crn_wait, decision_starts=[], risk_log=None,
                     leaf_log=None, _sup_busy=False, nudging=False, _leaf0=None).items():
        setattr(sh, k, v)
    pl = w.planner
    mcap, scap = {}, {}
    _hook_solve(pl, mcap)
    _hook_solve(sh, scap)
    recs = []
    orig_dec = pl.decide

    def decide(now, t, v_e, mission, acc_target, acc_status):
        mcap.clear()
        d = orig_dec(now, t, v_e, mission, acc_target, acc_status)
        if d.change_left and not pl._sup_busy and pl._leaf0 is not None:
            scap.clear()
            sh.nudging = getattr(pl, "nudging", False)
            sh._sup_busy = False
            t0 = time.perf_counter()
            d8 = sh.decide(now, t, v_e, mission, acc_target, acc_status)
            dt8 = time.perf_counter() - t0
            m2 = _curve(mcap["risk"], mcap["allowed"], mcap["p"], mcap["combo0"], TREE_RETURN) if "risk" in mcap else None
            s8 = (_curve(scap["risk"], scap["allowed"], scap["p"], scap["combo0"], TREE_RETURN)
                  if scap.get("mask0") else None)
            recs.append(dict(t=round(now - w.t0, 2), m2=None if m2 is None else np.asarray(m2, np.float32),
                             s8=None if s8 is None else np.asarray(s8, np.float32),
                             s8_feas0=bool(scap.get("feas0", False)), s8_valid0=bool(scap.get("mask0", False)),
                             s8_change=bool(d8.change_left), s8_ms=round(1000 * dt8, 1)))
        else:
            sh.hist.update(t, now)
            sh.scene.update(t, now, sh.P)
        return d
    pl.decide = decide
    res = w.run()
    aborts = [ev[1] for ev in res.events if ev[0] == "abort"]
    ep = {"seed": seed, "success": res.success, "collision": res.collision, "t_end": res.t_end, "t_merge": res.t_merge,
          "aborts": res.aborts, "min_sep": res.min_sep, "completes": list(res.lane_change_times), "abort_times": aborts,
          "accept_times": [round(ev[1], 2) for ev in res.events if ev[0] == "accept"],
          "risk_log": [(round(a - w.t0, 2), float(r), k) for a, r, k in pl.risk_log]}
    leaf = [(round(now - w.t0, 2), F, arr, p, q) for now, F, arr, p, q in pl.leaf_log]
    return ep, leaf, recs


def collect(seed0, n, workers, out):
    t0 = time.time()
    with Pool(workers) as pool:
        res = pool.map(run_seed, range(seed0, seed0 + n), chunksize=1)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 틱 단위 배열(leaf_log 틱과 그림자 기록은 같은 틱에서 같은 순서로 쌓인다)
    T, EP, P, Q, ARR, F, M2, S8, S8F, S8V, S8C, S8MS = [], [], [], [], [], [], [], [], [], [], [], []
    Bm = max((len(l[3]) for _, lf, _ in res for l in lf), default=1)
    for ep, lf, recs in res:
        assert len(lf) == len(recs), (ep["seed"], len(lf), len(recs))
        for (tt, Fm, arr, p, q), rc in zip(lf, recs):
            assert abs(tt - rc["t"]) < 1e-6
            b = len(p)
            rows = np.stack([Fm[2 * i + q[i]] for i in range(b)])
            F.append(np.pad(rows, ((0, Bm - b), (0, 0))))
            ARR.append(np.pad([bool(arr[2 * i + q[i]]) for i in range(b)], (0, Bm - b)))
            P.append(np.pad(np.asarray(p, float), (0, Bm - b)))
            T.append(tt); EP.append(ep["seed"])
            M2.append(rc["m2"] if rc["m2"] is not None else np.full(10, np.nan, np.float32))
            S8.append(rc["s8"] if rc["s8"] is not None else np.full(40, np.nan, np.float32))
            S8F.append(rc["s8_feas0"]); S8V.append(rc["s8_valid0"]); S8C.append(rc["s8_change"]); S8MS.append(rc["s8_ms"])
    np.savez(out.with_suffix(".npz"), t=np.array(T, np.float32), ep=np.array(EP), P=np.array(P, np.float32),
             ARR=np.array(ARR, bool), F=np.array(F, np.float32), M2=np.array(M2, np.float32), S8=np.array(S8, np.float32),
             S8F=np.array(S8F), S8V=np.array(S8V), S8C=np.array(S8C), S8MS=np.array(S8MS, np.float32))
    eps = [r[0] for r in res]
    J = [(e["t_merge"] if e["success"] else 120) + 60 * e["collision"] + 2 * e["aborts"] for e in eps]
    summ = {"planner": MAIN, "shadow": SHADOW, "n": n, "seed0": seed0, "J": float(np.mean(J)), "collisions": sum(e["collision"] for e in eps),
            "ticks": len(T), "wall_s": round(time.time() - t0, 1)}
    json.dump({"summary": summ, "episodes": eps}, open(out.with_suffix(".json"), "w"))
    print(json.dumps(summ))


def wmean(x, w):
    return float(np.sum(x * w) / np.sum(w)) if np.sum(w) > 0 else float("nan")


def auc(s, l):
    pos, neg = s[l], s[~l]
    if not len(pos) or not len(neg):
        return float("nan")
    o = np.argsort(np.concatenate([pos, neg]), kind="stable")
    rk = np.empty(len(o)); rk[o] = np.arange(1, len(o) + 1)
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def analyze(base):
    from smpc_sim.rl.risk_value import leaf_targets, load_rv_fn
    z = np.load(base + ".npz")
    D = json.load(open(base + ".json"))
    eps = {e["seed"]: e for e in D["episodes"]}
    fn = load_rv_fn(RV_PATH)
    n, Bm, dim = z["F"].shape
    V = fn(z["F"].reshape(-1, dim)).reshape(n, Bm, -1)
    V = np.where(z["ARR"][:, :, None], 0.0, V)
    RV = np.einsum("nb,nbj->nj", z["P"], V)                  # 틱별 Σ_b p_b V_bj (j = 0.2..8 s)
    y, valid, w = leaf_targets(z["ep"], z["t"], eps, "trace")
    ya, _, _ = leaf_targets(z["ep"], z["t"], eps, "abort")
    m2, s8 = z["M2"], z["S8"]
    m2pk = np.nanmax(m2, axis=1)
    rvpk = RV.max(1)
    rvtail = RV[:, 10:].max(1)                               # 지평선(2 s) 뒤만
    s8ok = z["S8V"] & ~np.isnan(s8).any(1)
    s8pk = np.where(s8ok, np.nanmax(np.nan_to_num(s8, nan=0), axis=1), np.nan)
    s8tail = np.where(s8ok, np.nan_to_num(s8, nan=0)[:, 10:].max(1), np.nan)
    v = valid
    W = w[v]
    out = {"ticks": int(n), "valid_ticks": int(v.sum()), "attempts(w)": round(float(W.sum()), 1),
           "check_max(m2,rv)<=eps(frac)": round(float(np.mean(np.maximum(m2pk, rvpk) <= EPS + 1e-9)), 4)}
    yv, rv, ab = y[v], RV[v], ya[v] > 0
    # 1) 시점별 평균: 예측 vs 실현(가중 평균 궤적)
    out["mean_curve_pred"] = [round(wmean(rv[:, j], W), 4) for j in range(40)]
    out["mean_curve_real"] = [round(wmean(yv[:, j], W), 4) for j in range(40)]
    s8v = s8ok[v]
    out["mean_curve_s8"] = [round(wmean(np.nan_to_num(s8[v][:, j]), W * s8v), 4) for j in range(40)]
    out["brier_rv"] = float(np.average(((rv - yv) ** 2).mean(1), weights=W))
    const = np.average(yv, axis=0, weights=W)
    out["brier_const"] = float(np.average(((const[None] - yv) ** 2).mean(1), weights=W))
    out["brier_s8"] = float(np.average(((np.nan_to_num(s8[v]) - yv) ** 2).mean(1)[s8v], weights=W[s8v]))
    # 2) 구간별 보정: 예측 max_j Σ p V 로 묶고, 묶음 안 실현 궤적의 평균의 최댓값(= 같은 순서) 과 비교
    bins = [0, 0.005, 0.01, 0.02, 0.03, 0.04, 1.01]

    def calib(score):
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (score >= lo) & (score < hi)
            if m.sum() < 5:
                continue
            ww = W[m]
            pred_curve = np.average(rv[m] if score is rvpk_v else np.nan_to_num(s8[v][m]), axis=0, weights=ww)
            real_curve = np.average(yv[m], axis=0, weights=ww)
            rows.append({"bin": [lo, hi], "n_ticks": int(m.sum()), "attempts(w)": round(float(ww.sum()), 1),
                         "pred_max_of_mean": round(float(pred_curve.max()), 4),
                         "real_max_of_mean": round(float(real_curve.max()), 4),
                         "real_mean_of_max": round(wmean(yv[m].max(1), ww), 4),
                         "P(real_max>eps)": round(wmean((yv[m].max(1) > EPS).astype(float), ww), 3),
                         "P(real_max>2eps)": round(wmean((yv[m].max(1) > 2 * EPS).astype(float), ww), 3),
                         "P(abort)": round(wmean(ab[m].astype(float), ww), 3),
                         "collision_attempts": int((yv[m].max(1) >= 1.0).sum())})
        return rows
    rvpk_v = rvpk[v]
    out["calib_rv"] = calib(rvpk_v)
    out["auc_abort_rv"] = auc(rvpk_v, ab)
    out["auc_abort_m2"] = auc(m2pk[v], ab)
    out["auc_abort_rvtail"] = auc(rvtail[v], ab)
    s8pk_v = np.where(s8v, s8pk[v], -1)
    out["auc_abort_s8"] = auc(s8pk_v[s8v], ab[s8v])
    out["calib_s8"] = calib(np.where(s8v, s8pk[v], -1.0))
    # 3) RV 꼬리 vs 8 s 모델 꼬리(같은 틱, 지평선 2 s 뒤)
    a, b = rvtail[v][s8v], s8tail[v][s8v]
    ww = W[s8v]
    rk = lambda x: np.argsort(np.argsort(x))
    out["tail_vs_s8"] = {"n": int(s8v.sum()), "rv_tail_mean": round(wmean(a, ww), 4), "s8_tail_mean": round(wmean(b, ww), 4),
                         "spearman": round(float(np.corrcoef(rk(a), rk(b))[0, 1]), 3),
                         "s8_tail>eps(frac,w)": round(wmean((b > EPS).astype(float), ww), 3),
                         "s8_feas0(frac,w)": round(wmean(z["S8F"][v][s8v].astype(float), ww), 3),
                         "s8_would_change(frac,w)": round(wmean(z["S8C"][v][s8v].astype(float), ww), 3)}
    # 8 s 가 거절(지금 바꾸기 불가)한 시도 vs 받은 시도의 실제 결과
    rej = ~z["S8F"][v] & s8v
    acc = z["S8F"][v] & s8v
    for name, m in (("s8_accepts", acc), ("s8_rejects", rej)):
        ww = W[m]
        out[name] = {"attempts(w)": round(float(ww.sum()), 1), "P(abort)": round(wmean(ab[m].astype(float), ww), 3),
                     "real_mean_of_max": round(wmean(yv[m].max(1), ww), 4),
                     "P(real_max>2eps)": round(wmean((yv[m].max(1) > 2 * EPS).astype(float), ww), 3),
                     "collision_attempts": int((yv[m].max(1) >= 1.0).sum())}
    out["episodes"] = {"n": len(eps), "collisions": int(sum(e["collision"] for e in eps.values())),
                       "aborts_per_ep": round(float(np.mean([e["aborts"] for e in eps.values()])), 3),
                       "J": round(float(np.mean([(e["t_merge"] if e["success"] else 120) + 60 * e["collision"] + 2 * e["aborts"]
                                                 for e in eps.values()])), 3)}
    out["shadow_ms_mean"] = round(float(z["S8MS"].mean()), 1)
    json.dump(out, open(base + "_analysis.json", "w"), indent=1)
    np.savez(base + "_curves.npz", RV=RV, y=y, valid=valid, w=w, ya=ya, M2=m2, S8=s8, S8OK=s8ok)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("mean_curve")}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    else:
        analyze(sys.argv[2])
