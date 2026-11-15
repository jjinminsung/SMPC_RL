"""TR 학습 데이터: 긴 지평선 플래너가 실제 판단 때 굴린 롤아웃에서 (k0 시점 입력, k0 뒤 최대 위험) 을 모은다.

    python3 -m smpc_sim.rl.collect_tail --planner rx_mt_pf_fl_h25_rf10 --seed0 30000 --n 300 --out ../results/tail/data.npz

판단(루트 평가) 10 번에 1 번(0.5 s)마다, 시작 시각 ≤ 1 s 인 유효 루트(2 s 플래너가 보는 루트)의 모든 (가지, 대응)."""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM_DIR))
CFG = {"switch_delay_sec": 0.55, "acc_floor_real": True}


def one(args):
    planner, seed, k0, every, max_start, every_mon = args
    import smpc_sim.tree_reactive as TRM
    from run_batch import planner_factory
    from smpc_sim.rollouts import execution_start
    from smpc_sim.rl.tail_risk import leaf_inputs
    from smpc_sim.world import World, EpisodeConfig
    tfs = "tfs" in planner.split("@")[0].split("_")          # 예측 ACC 규칙(tfs)과 시뮬 ACC 를 같이 바꾼다
    w = World(planner_factory(planner), EpisodeConfig(seed=seed, acc_tf_on_start=tfs, **CFG))
    p = w.planner
    X, Y, A, M, n = [], [], [], [], [0, 0]
    orig = p._solve
    orig_jr = TRM.joint_rollout

    def jr(ctx, specs, t, vehicles, P, A_, RP, track, branches, accel_of, mid_T_rem=None):
        r = orig_jr(ctx, specs, t, vehicles, P, A_, RP, track, branches, accel_of, mid_T_rem=mid_T_rem)
        if mid_T_rem is not None and every_mon > 0:            # 실행 중 모니터 롤아웃(계속·복귀)
            n[1] += 1
            if n[1] % every_mon == 0 and r["risk"].shape[-1] > k0 + 1:
                p._leaf_t = t
                F, idx = leaf_inputs(p, r, branches, [0.0], [0.0], [0], k0)
                tail = r["risk"][:, :, :, k0 + 1:].max(axis=-1)
                X.append(F)
                Y.append(np.array([tail[b, q, i] for b, q, i in idx]))
                A.append(np.array([bool(r["allowed"][b, q, i]) for b, q, i in idx]))
                M.append(np.ones(len(idx), dtype=bool))
        return r
    TRM.joint_rollout = jr

    def solve(r, br, root_cost, mask, eps):
        if getattr(p, "_roots_eval", False):
            n[0] += 1
            if n[0] % every == 0:
                starts = [execution_start(sp, p.P) for sp in p._specs]
                roots = [i for i in range(len(starts)) if mask[i] and starts[i] <= max_start + 1e-6]
                if roots:
                    F, idx = leaf_inputs(p, r, br, starts, [sp.prep_accel for sp in p._specs], roots, k0)
                    tail = r["risk"][:, :, :, k0 + 1:].max(axis=-1)
                    X.append(F)
                    Y.append(np.array([tail[b, q, i] for b, q, i in idx]))
                    A.append(np.array([bool(r["allowed"][b, q, i]) for b, q, i in idx]))
                    M.append(np.zeros(len(idx), dtype=bool))
        return orig(r, br, root_cost, mask, eps)
    p._solve = solve
    w.run()
    if not X:
        return None
    return np.concatenate(X), np.concatenate(Y), np.concatenate(A), np.concatenate(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="rx_mt_pf_fl_h25_rf10")
    ap.add_argument("--seed0", type=int, default=30000)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k0", type=int, default=9)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--max_start", type=float, default=1.0)
    ap.add_argument("--every_mon", type=int, default=5, help="모니터 롤아웃 몇 번에 1 번(0 = 모니터 안 모음)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with mp.Pool(a.workers) as pool:
        res = pool.map(one, [(a.planner, s, a.k0, a.every, a.max_start, a.every_mon) for s in range(a.seed0, a.seed0 + a.n)])
    X, Y, A, E, MO = [], [], [], [], []
    for e, r in enumerate(res):
        if r is None:
            continue
        X.append(r[0]); Y.append(r[1]); A.append(r[2]); MO.append(r[3]); E.append(np.full(len(r[1]), e))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, X=np.concatenate(X).astype(np.float32), Y=np.concatenate(Y).astype(np.float32),
                        A=np.concatenate(A), E=np.concatenate(E), MON=np.concatenate(MO))
    Y = np.concatenate(Y)
    print(f"rows {len(Y)}  tail>0.04 {100 * (Y > 0.04).mean():.2f} %  tail>0.01 {100 * (Y > 0.01).mean():.2f} %")


if __name__ == "__main__":
    main()
