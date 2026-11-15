"""학습한 종단 비용(TC): 지평선 끝 예측 상태에서 남은 J 의 기댓값 V(x) [s].

SMPC 루트 값(tree_reactive._solve, 'TC' 플래그):
    J_root(a) = min_c Σ_b p_b [ c_resp(c_b) + τ_a + V(x_τa^{b,c_b}(a)) ]   s.t. max_k Σ_b p_b r_b(k) ≤ ε (기존 기회 제약)
    c_resp(복귀) = 2 s(J 의 중단 가중). 마지막 차선 합류가 잎 전에 끝나면 그 합류 시각.
    잎 시각 τ_a = 닫힌 고리가 그 루트를 실제로 따르는 끝:
      지금 바꾸기·머리 넣기·기다리기 = 지평선 끝(실행에 들어가면 그대로 간다).
      준비(hold/yield/accel@t_lc) = 실행 시작 t_sw. 그 뒤 바꾸기는 '약속'일 뿐 다음 틱에 다시 고르므로,
      t_sw 뒤 예측 상태로 값을 매기면 지켜지지 않는 약속으로 미루기를 계속 고른다.
V 는 J(합류 시각, 실패 120 s, 충돌 +60 s, 중단 +2 s)와 같은 단위라 무게 없이 더한다. 막는 문턱(θ)은 없다.

학습 = 몬테카를로 정책 평가: 정책 π 로 돌린 에피소드에서 0.5 s 마다 실제 상태 s_t(features2)와 남은 J
    G_t = (T_end - t) + 2·(t 뒤 중단 수) + 60·충돌,  T_end = 합류 시각(실패면 120 s)
를 모아 V(s_t) ≈ E[G_t] 로 회귀한다.
라벨 'lane'(기본): 지금 차선 변경을 마칠 때까지의 J 만 쓴다(마칠 시각 - t + 2·그 사이 중단 + 그 차선에서 충돌이면 60).
    마지막 차선에서는 G_t 와 같고, 앞 차선에서는 다음 차선들의 J 를 판단과 무관한 상수로 본다. 에피소드 전체 라벨은
    다음 차선들의 교통 운이 분산 대부분이라 지금 차선의 판단 차이(중단 위험 등)가 묻힌다.
    잎이 목표 차선에 도착했으면 비용 = 도착 시각. 잎 특징은 학습 상태와 같은 틀(features2.frame_features_batch)로 만든다.
입력 = features2 + 횡속도(지난 0.5 s 횡 이동 / 0.5 s, 차선 폭 단위). 횡속도가 없으면 '머리 넣고 멈춘 상태'와
'차선 변경 한가운데'가 같은 진행도로 보여 구별되지 않는다.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features2 import DIM, LW, frame_features_batch, target_set_vehicles
from ..rollouts import ACCEL, HOLD, YIELD, execution_start
from ..track import FINAL_LANE

ABORT_COST = 2.0
COLLISION_COST = 60.0
FAIL_TIME = 120.0
SCALE = 60.0          # 학습 목표 G/SCALE
LAG = 0.5             # 횡속도 차분 간격 [s] (= 수집 표본 간격)
TC_DIM = DIM + 1


def lane_targets(X, G, T, E, episodes):
    """에피소드 남은 J 라벨 → (차선 단위 남은 J, clean). episodes: seed → 에피소드 요약(t_merge, success, collision,
    있으면 completes = 차선 변경 완료 시각들, explore_events = 탐색 구간들). 완료 시각이 없으면 앞뒤 표본 시각의 가운데.
    clean = 그 차선을 마칠 때까지 남은 탐색이 없는 표본(라벨 = 탐욕 정책의 값). 탐색 기록이 없으면 모두 True."""
    X, G, T, E = np.asarray(X), np.asarray(G, float), np.asarray(T, float), np.asarray(E)
    lane = np.where(X[:, 2:5].max(axis=1) > 0.5, X[:, 2:5].argmax(axis=1), FINAL_LANE)
    out = np.zeros(len(G), dtype=np.float32)
    clean = np.ones(len(G), dtype=bool)
    for seed in np.unique(E):
        ix = np.nonzero(E == seed)[0]
        ix = ix[np.argsort(T[ix])]
        ep = episodes[int(seed)]
        t_end = ep["t_merge"] if ep["success"] else FAIL_TIME
        coll = float(ep["collision"])
        t, ln = T[ix], lane[ix]
        n_after = np.rint((G[ix] - (t_end - t) - COLLISION_COST * coll) / ABORT_COST)
        comp = sorted(ep.get("completes", []))
        for L in np.unique(ln):
            here, later = ln == L, ln > L
            if later.any():
                t_last, j = t[here].max(), int(np.argmax(later))
                t_first = t[j]
                c = [x for x in comp if t_last - 1e-6 <= x <= t_first + 1e-6]
                t_c = c[0] if c else 0.5 * (t_last + t_first)
                n_next, cc = n_after[j], 0.0
            else:
                t_c, n_next, cc = t_end, 0.0, coll
            out[ix[here]] = (t_c - t[here]) + ABORT_COST * (n_after[here] - n_next) + COLLISION_COST * cc
            for a, b in ep.get("explore_events", []):
                if a < t_c:
                    clean[ix[here & (t <= b + 1e-6)]] = False
    return out, clean


def root_targets(t, ep, lane, act, explored, episodes):
    """루트 기록(판단 틱)의 차선 단위 남은 J 와 clean. 이 차선을 마치는 시각 t_c = t 뒤 첫 완료(없으면 끝 시각),
    라벨 = (t_c - t) + 2·(t, t_c] 중단 수 + 60·충돌(마지막 차선). clean = t 뒤 t_c 전에 시작하는 탐색이 없고,
    t 가 기다리기 탐색 구간 안이면 그 틱의 행동은 기다리기(act = keep)여야 한다. 탐색으로 고른 틱 자체는 그 행동의 값이다.
    결정 탐색(decision_events: 지금 바꾸기를 고르기 시작한 틱의 강제 기다리기)은 앞 표본을 clean 에서 빼지 않는다
    (빼면 중단된 바꾸기 뒤에만 다음 결정이 있으므로 중단된 바꾸기 표본만 골라 빠진다). 그 구간 안 틱은 기다리기 값이다."""
    t, ep = np.asarray(t, float), np.asarray(ep)
    y = np.zeros(len(t), np.float32)
    clean = np.ones(len(t), bool)
    for seed in np.unique(ep):
        ix = np.nonzero(ep == seed)[0]
        e = episodes[int(seed)]
        t_end = e["t_merge"] if e["success"] else FAIL_TIME
        comp = np.sort(np.asarray(e.get("completes", []), float))
        ab = np.sort(np.asarray(e.get("abort_times", []), float))
        ev = e.get("explore_events", [])
        dv = e.get("decision_events", [])
        for j in ix:
            k = np.searchsorted(comp, t[j] + 1e-6)
            last = k >= len(comp)
            t_c = t_end if last else comp[k]
            n_ab = np.searchsorted(ab, t_c + 1e-6) - np.searchsorted(ab, t[j] + 1e-6)
            y[j] = (t_c - t[j]) + ABORT_COST * n_ab + COLLISION_COST * float(e["collision"]) * last
            for a, b in ev:
                if t[j] + 1e-6 < a < t_c:
                    clean[j] = False
                elif b > a and a - 1e-6 <= t[j] <= b + 1e-6 and act[j] != 2:
                    clean[j] = False
            for a, b in dv:
                if a - 1e-6 <= t[j] <= b + 1e-6 and act[j] != 2:
                    clean[j] = False
    return y, clean


def root_values(nets_fn, R):
    """루트 기록 R(collect_tc pack_roots) → SMPC 루트 값 Σ_b p_b [c_b + (도착 ? t_arr : τ + V(잎))]. nets_fn: [M, D] tensor → [M]."""
    F = torch.as_tensor(R["F"])
    n, Bm, D = F.shape
    v = nets_fn(F.reshape(-1, D)).reshape(n, Bm)
    tau = torch.as_tensor(R["tau"])[:, None]
    leaf = torch.where(torch.as_tensor(R["term"]), torch.as_tensor(R["t_arr"]), tau + v)
    P = torch.as_tensor(R["P"]) * torch.as_tensor(R["mask"])
    return (P * (torch.as_tensor(R["C"]) + leaf)).sum(1)


def add_lateral_speed(X, T, E):
    """수집 표본(features2, 0.5 s 간격) → TC 입력. 횡속도 = (y_t - y_{t-0.5}) / 0.5 / LW, y = (출발 차선 + 진행도)·LW."""
    X, T, E = np.asarray(X), np.asarray(T, float), np.asarray(E)
    lane = np.where(X[:, 2:5].max(axis=1) > 0.5, X[:, 2:5].argmax(axis=1), FINAL_LANE)
    y = (lane + X[:, 1]) * LW
    o = np.lexsort((T, E))
    vy = np.zeros(len(X), dtype=np.float32)
    prev = np.r_[False, (E[o][1:] == E[o][:-1]) & (np.abs(T[o][1:] - T[o][:-1] - LAG) < 1e-3)]
    d = np.r_[0.0, np.diff(y[o])] / (LAG * LW)
    vy[o] = np.where(prev, d, 0.0)
    return np.c_[X, vy].astype(np.float32)


def j_to_go(times, t_end_eff, abort_times, collision):
    """시각 times [s] 마다의 남은 J. t_end_eff = 합류 시각(실패면 120 s)."""
    times = np.asarray(times, dtype=float)
    ab = np.sort(np.asarray(abort_times, dtype=float))
    n_after = len(ab) - np.searchsorted(ab, times - 1e-9)
    return (t_end_eff - times) + ABORT_COST * n_after + COLLISION_COST * float(collision)


def leaf_costs(planner, r, mask, fn):
    """잎 비용 τ + V(잎) [B, 2, N]. 마지막 차선에 지평선 안에서 도착한 잎은 도착 시각."""
    t, P = planner._leaf_t, planner.P
    X, Y, V = r["x"], r["y"], r["v"]
    B, _, N, K = X.shape
    dt = P.prediction_dt_sec
    cur, tgt = int(t.current_lane_id), int(t.target_lane_id)
    y_tgt = planner.track.center_y(tgt)
    tol = P.ego_reference_preview_target_completion_max_abs_d_m
    meta = {m[0]: m for m in r["tv_meta"]}
    ids = [uid for uid, _ in r["tv_s"]]
    TS = np.stack([S for _, S in r["tv_s"]], axis=-1) if ids else np.zeros((B, 2, N, K, 0))
    oth = [v for v in target_set_vehicles(t) if v.unique_id not in meta]
    o_s = np.array([v.s for v in oth])
    o_v = np.array([max(0.0, v.v_long) for v in oth])
    lanes = np.array([meta[u][1] for u in ids] + [v.lane_id for v in oth], dtype=int)
    lens = np.array([meta[u][2] for u in ids] + [v.length for v in oth], dtype=float)
    specs = getattr(planner, "_specs", None)
    lane_target = getattr(fn, "meta", {}).get("target") == "lane"
    out = np.zeros((B, 2, N))
    rows = []
    for i in range(N):
        if not mask[i]:
            continue
        ok = np.nonzero(r["mask"][i])[0]
        kk = int(ok[-1]) if len(ok) else K - 1
        if specs is not None and specs[i].action in (HOLD, YIELD, ACCEL):
            kk = min(kk, max(0, int(round(execution_start(specs[i], P) / dt)) - 1))
        tau = (kk + 1) * dt
        ts = TS[:, :, i, kk, :].reshape(2 * B, -1)
        tv = (ts - TS[:, :, i, kk - 1, :].reshape(2 * B, -1)) / dt if kk > 0 else np.zeros_like(ts)
        vs = np.concatenate([ts, np.tile(o_s + o_v * tau, (2 * B, 1))], axis=1)
        vv = np.concatenate([np.maximum(tv, 0.0), np.tile(o_v, (2 * B, 1))], axis=1)
        if tgt != cur:
            hit = np.abs(Y[:, :, i, :kk + 1] - y_tgt).reshape(2 * B, -1) <= tol
            arr = hit.any(axis=1)
            t_arr = (hit.argmax(axis=1) + 1) * dt
        else:
            arr, t_arr = np.zeros(2 * B, dtype=bool), np.zeros(2 * B)
        term = arr if lane_target else arr & (tgt >= FINAL_LANE)
        F = frame_features_batch(np.where(arr, tgt, cur), np.where(arr, np.minimum(tgt + 1, FINAL_LANE), tgt),
                                 X[:, :, i, kk].ravel(), Y[:, :, i, kk].ravel(), V[:, :, i, kk].ravel(),
                                 vs, np.tile(lanes, (2 * B, 1)), vv, np.tile(lens, (2 * B, 1)))
        f = kk - LAG / dt                                  # 스텝 k 는 시각 (k+1)·dt: 잎 0.5 s 앞을 보간
        if f >= 0:
            lo = int(np.floor(f)); w = f - lo
            y_lag = (1 - w) * Y[:, :, i, lo] + w * Y[:, :, i, min(lo + 1, kk)]
            vy = (Y[:, :, i, kk] - y_lag).ravel() / (LAG * LW)
        else:
            vy = (Y[:, :, i, kk] - Y[:, :, i, 0]).ravel() / (max(kk, 1) * dt * LW) if kk > 0 else np.zeros(2 * B)
        F = np.c_[F, vy].astype(np.float32)
        rows.append((F, i, term, t_arr, tau))
    if getattr(planner, "tc_record", False):
        planner._leaf_rows = {i: (F.reshape(B, 2, -1), term.reshape(B, 2), t_arr.reshape(B, 2), tau)
                              for F, i, term, t_arr, tau in rows}
    if rows:
        vals = fn(np.concatenate([x[0] for x in rows]))
        o = 0
        for F, i, term, t_arr, tau in rows:
            v = vals[o:o + len(F)]
            o += len(F)
            out[:, :, i] = np.where(term, t_arr, tau + v).reshape(B, 2)
    return out


class TCNet(nn.Module):
    def __init__(self, dim=TC_DIM, hidden=64, out=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def save_tc(path, nets, meta=None):
    torch.save({"ensemble": [n.state_dict() for n in nets], "meta": meta or {}}, path)


def load_tc_fn(path):
    """입력 [M, DIM] → 남은 J [M] (앙상블 평균, 초)."""
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    nets = []
    for sd in ck["ensemble"]:
        n = TCNet(dim=sd["net.0.weight"].shape[1], hidden=sd["net.0.weight"].shape[0])
        n.load_state_dict(sd)
        n.eval()
        nets.append(n)
    torch.set_num_threads(1)

    def fn(x):
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            y = torch.stack([n(xt) for n in nets]).numpy()
        return SCALE * y.mean(axis=0)
    fn.meta = ck.get("meta", {})
    return fn
