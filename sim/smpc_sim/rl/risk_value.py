"""RL 위험 값(RV): 지평선 끝 상태에서 '이 차선 변경을 마칠 때까지 플래너가 실제로 마주칠 최대 위험'.

SMPC 의 위험은 걸음별 위험 Σ_b p_b r_b(k) 의 최댓값(peak)으로 쓰이고(기회 제약 peak ≤ ε, 비용 80·peak),
최댓값은 지평선 앞뒤로 나뉜다: 전체 최댓값 = max(지평선 안, 지평선 뒤). 그래서 지평선 뒤 부분을
최댓값 벨만 식 V(s) = max(r(s), V(s')) 의 값으로 배우면 같은 단위(확률)로 그대로 넣을 수 있다.

걸음별 위험 신호 r_t(world 의 risk_log): 실제 주행 중 매 틱 플래너가 계산한 지평선 안 위험.
    지금 바꾸기를 고른 틱 = 그 루트의 위험, 실행 중 = 모니터의 '계속' 위험, 머리 넣은 동안 = 남기 위험.
라벨(몬테카를로): y_t = max{ r_t' : t < t' ≤ t_c },  t_c = 지금 차선 변경을 마친 시각(없으면 에피소드 끝).
    그 차선에서 충돌이면 1. 탐색이 그 사이에 있으면 학습에서 뺀다(라벨 = 탐욕 정책의 값).
학습 = 소프트 라벨 BCE(최소점 = E[y]), MLP 앙상블. 입력 = TC 와 같은 상태 특징(features2 + 횡속도).

중단 값(RA, 같은 파일): A(s) = 이 차선 변경을 마칠 때까지 앞으로 일어날 중단(복귀) 횟수의 기댓값.
    더하기 벨만 식(중단 한 번 = 보상 1), 라벨 = (t, t_c] 의 중단 횟수, 학습 = 푸아송 NLL(최소점 = E[횟수]).
    위험은 중단으로 끝난 시도를 낮게 기록한다(중단이 위험을 2ε 근처에서 끊는다). 그 대가는 시간과 중단이라
    SMPC 는 지평선 안에서 이를 대응 비용(복귀 30)으로 매긴다. RA 는 같은 값 30 을 지평선 뒤 중단 기댓값에 곱한다.

시점별 위험 값(RVT, 잎 학습판, target='trace'): SMPC 의 peak = max_k Σ_b p_b r_b(k) 는 기댓값을 먼저, 최댓값을 나중에 한다.
    RV 라벨(실현 궤적마다 최댓값을 먼저) 의 기댓값은 E[max_k r(k)] ≥ max_k E[r(k)] 라 같은 ε 와 비교하면 과하다
    (실제로 잘 끝난 시도도 실행 중 위험이 ε~2ε 를 오간다: 모니터 문턱이 2ε). 그래서 시점별로
    V_j(잎) = E[r(t_r + jΔ)], j = 1..TRACE_T/Δ 를 한꺼번에 배우고(출력 40개), 플래너는 max_j Σ_b p_b V_bj 를 쓴다
    = SMPC 지평선 안과 같은 순서. r(τ) = 그 틱의 위험 신호(구간 최댓값, 기록이 없으면 직전 값), 시도가 끝난 뒤 0, 충돌 뒤 1.

플래너(tree_reactive._solve):
    'RV' 플래그 = 종단 제약: peak = max(지평선 안, Σ_b p_b V_b) 를 기회 제약과 비용 80·peak 모두에 쓴다.
    'RVc' 플래그 = 종단 비용: 비용 80·peak 에만 쓰고 기회 제약은 지평선 안 그대로.
    'RA' 플래그 = 종단 비용: 루트 비용에 Σ_b p_b · 30 · A(잎_b) 를 더한다(30 = 기존 복귀 대응 비용).
    잎 = 지평선 끝 예측 상태(모든 루트). 잎이 목표 차선에 도착했으면 그 차선 변경은 끝났으므로 0.
"""
from pathlib import Path

import numpy as np
import torch

from .features2 import LW, frame_features_batch, target_set_vehicles
from .terminal_cost import LAG, TCNet
from ..track import FINAL_LANE

RISK_FLOOR = 1e-4
ACCEPT_LAG = 0.6      # 명령 묶음이 끝난 뒤 승인까지(시뮬 switch_delay_sec 0.55 + 한 틱). 라벨 짝짓기용
TRACE_DT = 0.2        # 시점별 위험(RVT) 라벨 간격 = SMPC 예측 간격
TRACE_T = 8.0         # 명령 뒤 8 s(= 8 s 지평선 판이 보는 범위). 시도(명령→완료) 중앙값 4.75 s, 10 s 넘는 시도 0.1 %


def risk_targets(X, T, E, episodes, target="risk"):
    """실제 상태 표본(0.5 s) → (라벨 y, clean). episodes[seed] 에 risk_log, completes, abort_times, explore_events.
    target='risk': y = (t, t_c] 의 최대 위험(충돌 차선이면 1). target='abort': y = (t, t_c] 의 중단 횟수."""
    X, T, E = np.asarray(X), np.asarray(T, float), np.asarray(E)
    lane = np.where(X[:, 2:5].max(axis=1) > 0.5, X[:, 2:5].argmax(axis=1), 99)
    y = np.zeros(len(T), dtype=np.float32)
    clean = np.ones(len(T), dtype=bool)
    for seed in np.unique(E):
        ix = np.nonzero(E == seed)[0]
        ix = ix[np.argsort(T[ix])]
        ep = episodes[int(seed)]
        t_end = ep["t_end"]
        rl = ep.get("risk_log", [])
        tr = np.array([a for a, _, _ in rl], float)
        rr = np.array([r for _, r, _ in rl], float)
        comp = sorted(ep.get("completes", []))
        ab = np.array(sorted(ep.get("abort_times", [])), float)
        t, ln = T[ix], lane[ix]
        for L in np.unique(ln):
            here, later = ln == L, ln > L
            if later.any():
                t_last, t_first = t[here].max(), t[int(np.argmax(later))]
                c = [x for x in comp if t_last - 1e-6 <= x <= t_first + 1e-6]
                t_c, coll = (c[0] if c else 0.5 * (t_last + t_first)), False
            else:
                t_c, coll = t_end, bool(ep["collision"])
            for j in np.nonzero(here)[0]:
                if target == "abort":
                    y[ix[j]] = float(((ab > t[j] + 1e-9) & (ab <= t_c + 1e-6)).sum())
                    continue
                m = (tr > t[j] + 1e-9) & (tr <= t_c + 1e-6)
                y[ix[j]] = 1.0 if coll else (rr[m].max() if m.any() else 0.0)
            for a, b in ep.get("explore_events", []):
                if a < t_c:
                    clean[ix[here & (t <= b + 1e-6)]] = False
    return y, clean


def _trace(tr, rr, t_r, t_c, coll, t_end):
    """시점별 위험 신호 [J]: τ_j = t_r + jΔ. 구간 (τ_j − Δ, τ_j] 최댓값, 기록이 없으면 직전 값, 구간이 t_c 뒤에서 시작하면 0(충돌이면 1)."""
    J = int(round(TRACE_T / TRACE_DT))
    out = np.zeros(J, np.float32)
    last = 0.0
    for j in range(J):
        tau = t_r + (j + 1) * TRACE_DT
        if tau - TRACE_DT >= t_c - 1e-6:                 # 이 구간은 시도가 끝난 뒤
            if coll:
                out[j:] = 1.0
            break
        m = (tr > tau - TRACE_DT + 1e-9) & (tr <= min(tau, t_c) + 1e-6)
        if m.any():
            last = float(rr[m].max())
        out[j] = last
    return out


def leaf_targets(rec_ep, rec_t, episodes, target="risk"):
    """잎 학습판 라벨: 지금 바꾸기를 명령한 틱(collect_tc --leaf_log) → (y, valid, w).

    명령 틱이 0.05 s 간격으로 이어진 묶음 = 한 번의 요청. 묶음 끝 + ACCEPT_LAG(시뮬 전환 지연 0.55 s + 한 틱) 안에
    승인이 있으면 그 차선 변경의 결과를,
    없으면(요청 철회) 제외. t_c = 이번 시도의 끝(승인 뒤 첫 완료 또는 중단, 없으면 에피소드 끝).
    y: 위험 = (t_r, t_c] 실제 최대 위험(충돌이면 1), 중단 = 이번 시도가 중단으로 끝났으면 1. w = 1 / 묶음 크기(차선 변경 하나가 한 번 세이게). 명령 뒤 (묶음 끝, t_c] 에 탐색이 끼면 제외.
    잎이 예측 상태라 '계속한 뒤 살아남은' 실제 상태만 남는 편향이 없다(결과를 알기 전에 기록)."""
    rec_ep, rec_t = np.asarray(rec_ep), np.asarray(rec_t, float)
    y = np.zeros((len(rec_t), int(round(TRACE_T / TRACE_DT))) if target == "trace" else len(rec_t), np.float32)
    valid = np.zeros(len(rec_t), bool)
    w = np.zeros(len(rec_t), np.float32)
    for seed in np.unique(rec_ep):
        ix = np.nonzero(rec_ep == seed)[0]
        ix = ix[np.argsort(rec_t[ix])]
        ep = episodes[int(seed)]
        acc = np.array(sorted(ep.get("accept_times", [])), float)
        comp = np.array(sorted(ep.get("completes", [])), float)
        ab = np.array(sorted(ep.get("abort_times", [])), float)
        rl = ep.get("risk_log", [])
        tr = np.array([a for a, _, _ in rl], float)
        rr = np.array([r for _, r, _ in rl], float)
        ex = ep.get("explore_events", [])
        t = rec_t[ix]
        cut = np.nonzero(np.diff(t) > 0.075)[0] + 1
        for ch in np.split(np.arange(len(ix)), cut):
            t0, t1 = t[ch[0]], t[ch[-1]]
            a = acc[(acc >= t0 - 1e-6) & (acc <= t1 + ACCEPT_LAG + 1e-6)]
            if not len(a):
                continue
            c = np.sort(np.r_[comp, ab])
            c = c[c > a[0] + 1e-6]                          # 이번 시도의 끝 = 승인 뒤 첫 완료 또는 중단
            t_c = c[0] if len(c) else ep["t_end"]
            if any(e0 < t_c and e1 > t1 + 1e-6 for e0, e1 in ex):   # 명령 뒤에 낀 탐색만 뺀다(명령 자체가 탐색이면 그 행동의 값)
                continue
            coll = (not len(c)) and bool(ep["collision"])
            for j in ch:
                tj = t[j]
                if target == "abort":
                    y[ix[j]] = float(((ab > tj + 1e-9) & (ab <= t_c + 1e-6)).sum())
                elif target == "trace":
                    y[ix[j]] = _trace(tr, rr, tj, t_c, coll, ep["t_end"])
                else:
                    m = (tr > tj + 1e-9) & (tr <= t_c + 1e-6)
                    y[ix[j]] = 1.0 if coll else (rr[m].max() if m.any() else 0.0)
                valid[ix[j]], w[ix[j]] = True, 1.0 / len(ch)
    return y, valid, w


def leaf_features(planner, r, mask, kk_at=None):
    """모든 루트의 지평선 끝 잎 특징. 반환 [(F [2B, DIM+1], i, 도착 [2B])]. 특징 틀은 TC(leaf_costs)와 같다.
    kk_at = 잎 대신 예측 걸음 kk_at 의 상태(점검용)."""
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
    rows = []
    for i in range(N):
        if not mask[i]:
            continue
        ok = np.nonzero(r["mask"][i])[0]
        kk = int(ok[-1]) if len(ok) else K - 1
        if kk_at is not None:
            kk = min(kk_at, kk)
        tau = (kk + 1) * dt
        ts = TS[:, :, i, kk, :].reshape(2 * B, -1)
        tv = (ts - TS[:, :, i, kk - 1, :].reshape(2 * B, -1)) / dt if kk > 0 else np.zeros_like(ts)
        vs = np.concatenate([ts, np.tile(o_s + o_v * tau, (2 * B, 1))], axis=1)
        vv = np.concatenate([np.maximum(tv, 0.0), np.tile(o_v, (2 * B, 1))], axis=1)
        if tgt != cur:
            arr = (np.abs(Y[:, :, i, :kk + 1] - y_tgt).reshape(2 * B, -1) <= tol).any(axis=1)
        else:
            arr = np.zeros(2 * B, dtype=bool)
        F = frame_features_batch(np.where(arr, tgt, cur), np.where(arr, np.minimum(tgt + 1, FINAL_LANE), tgt),
                                 X[:, :, i, kk].ravel(), Y[:, :, i, kk].ravel(), V[:, :, i, kk].ravel(),
                                 vs, np.tile(lanes, (2 * B, 1)), vv, np.tile(lens, (2 * B, 1)))
        f = kk - LAG / dt
        if f >= 0:
            lo = int(np.floor(f)); w = f - lo
            y_lag = (1 - w) * Y[:, :, i, lo] + w * Y[:, :, i, min(lo + 1, kk)]
            vy = (Y[:, :, i, kk] - y_lag).ravel() / (LAG * LW)
        else:
            vy = (Y[:, :, i, kk] - Y[:, :, i, 0]).ravel() / (max(kk, 1) * dt * LW) if kk > 0 else np.zeros(2 * B)
        rows.append((np.c_[F, vy].astype(np.float32), i, arr))
    return rows


def leaf_risk(planner, r, mask, fn):
    """잎 위험 값 [B, 2, N] (시점별 RVT 면 [B, 2, N, J]). 도착한 잎은 0."""
    B, _, N, _ = r["x"].shape
    J = getattr(fn, "out", 1)
    out = np.zeros((B, 2, N, J)) if J > 1 else np.zeros((B, 2, N))
    rows = leaf_features(planner, r, mask)
    if rows:
        vals = fn(np.concatenate([x[0] for x in rows]))
        o = 0
        for F, i, arr in rows:
            v = vals[o:o + len(F)]
            o += len(F)
            v = np.where(arr[:, None], 0.0, v) if J > 1 else np.where(arr, 0.0, v)
            out[:, :, i] = v.reshape((B, 2, J) if J > 1 else (B, 2))
    return out


def tail_peak(p, V):
    """지평선 뒤 위험 Σ_b p_b V_b (V [B] 또는 시점별 [B, J]). 시점별이면 max_j Σ_b p_b V_bj (기댓값 먼저, 최댓값 나중)."""
    s = np.tensordot(np.asarray(p, float), V, axes=(0, 0))
    return float(s.max()) if np.ndim(s) else float(s)


def save_rv(path, nets, meta=None):
    torch.save({"ensemble": [n.state_dict() for n in nets], "meta": meta or {}}, path)


def load_rv_fn(path):
    """입력 [M, DIM] → 위험 값 [M] (앙상블 확률 평균). meta target='abort' 면 중단 기댓값(exp 평균)."""
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    nets = []
    for sd in ck["ensemble"]:
        n = TCNet(dim=sd["net.0.weight"].shape[1], hidden=sd["net.0.weight"].shape[0], out=sd["net.4.weight"].shape[0])
        n.load_state_dict(sd)
        n.eval()
        nets.append(n)
    torch.set_num_threads(1)
    link = torch.exp if ck.get("meta", {}).get("target") == "abort" else torch.sigmoid

    def fn(x):
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            return torch.stack([link(n(xt)) for n in nets]).mean(0).numpy()
    fn.meta = ck.get("meta", {})
    fn.leaf = bool(fn.meta.get("leaf", False))      # 잎 학습판이면 플래너가 루트 0(지금 바꾸기)에만 쓴다
    fn.out = int(nets[0].net[4].weight.shape[0])     # 1 = 값 하나, J = 시점별(RVT)
    return fn
