"""학습한 꼬리 위험(TR): 짧은 지평선이 잘라 낸 미래 위험을 학습한 값으로 채운다(POLO·MPQ(λ) 의 끝 가치와 같은 자리).

긴 지평선(5 s) 반응형 트리는 단계마다 충돌 위험 r[b, q, i, k] 를 계산하고 Σ_b p_b r(k) 의 최댓값 ≤ ε 로 판단한다.
2 s 지평선은 k ≤ k0 (= 2 s) 까지만 본다. TR 은 k0 시점 예측 상태(잎 특징) + 가지·대응·루트 정보로
'k0 뒤 같은 가지의 최대 위험' max_{k > k0} r[b, q, i, k] 를 예측한다. 2 s 플래너는
    peak = max( max_{k ≤ k0} Σ_b p_b r_b(k),  Σ_b p_b TR_b )      (뒤 항은 꼬리 최댓값의 위 한계)
로 5 s 와 같은 규칙(peak ≤ ε)을 쓴다. 학습 데이터는 5 s 플래너가 실제 판단 때 굴린 롤아웃이다(모델 기반 가치 학습).
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features2 import DIM as FDIM, frame_features_batch, target_set_vehicles

RISK_FLOOR = 1e-4
BR_REAR = ("yield", "nonyield")
BR_FRONT = ("base", "brake", "accel")
DIM = FDIM + len(BR_REAR) + len(BR_FRONT) + 1 + 2 + 1      # 잎 특징 + 가지 + 대응(복귀) + 루트(준비 가속, 시작 시각) + k0 위험


def leaf_inputs(planner, r, br, starts, accels, roots, kk):
    """롤아웃 r 의 k = kk 시점 입력 [len(roots)·B·2, DIM] 과 (b, q, i) 목록."""
    t, track, P = planner._leaf_t, planner.track, planner.P
    X, Y, V = r["x"], r["y"], r["v"]
    B = X.shape[0]
    dt = P.prediction_dt_sec
    cur, tgt = int(t.current_lane_id), int(t.target_lane_id)
    meta = {m[0]: m for m in r["tv_meta"]}
    ids = [uid for uid, _ in r["tv_s"]]
    N, K = X.shape[2], X.shape[3]
    TS = np.stack([S for _, S in r["tv_s"]], axis=-1) if ids else np.zeros((B, 2, N, K, 0))
    oth = [v for v in target_set_vehicles(t) if v.unique_id not in meta]
    o_s = np.array([v.s for v in oth]); o_v = np.array([max(0.0, v.v_long) for v in oth])
    lanes = np.array([meta[u][1] for u in ids] + [v.lane_id for v in oth], dtype=int)
    lens = np.array([meta[u][2] for u in ids] + [v.length for v in oth], dtype=float)
    tau = (kk + 1) * dt
    brf = np.zeros((B, len(BR_REAR) + len(BR_FRONT)))
    for b, x in enumerate(br):
        brf[b, BR_REAR.index(x[0])] = 1.0
        brf[b, len(BR_REAR) + BR_FRONT.index(x[1])] = 1.0
    Fs, idx = [], []
    for i in roots:
        ts = TS[:, :, i, kk, :].reshape(2 * B, -1)
        tv = (ts - TS[:, :, i, kk - 1, :].reshape(2 * B, -1)) / dt if kk > 0 else np.zeros_like(ts)
        vs = np.concatenate([ts, np.tile(o_s + o_v * tau, (2 * B, 1))], axis=1)
        vv = np.concatenate([np.maximum(tv, 0.0), np.tile(o_v, (2 * B, 1))], axis=1)
        F = frame_features_batch(np.full(2 * B, cur), np.full(2 * B, tgt), X[:, :, i, kk].ravel(), Y[:, :, i, kk].ravel(),
                                 V[:, :, i, kk].ravel(), vs, np.tile(lanes, (2 * B, 1)), vv, np.tile(lens, (2 * B, 1)))
        extra = np.zeros((2 * B, DIM - FDIM), dtype=np.float32)
        for b in range(B):
            for q in range(2):
                row = 2 * b + q
                extra[row, :5] = brf[b]
                extra[row, 5] = q
                extra[row, 6] = accels[i] / 3.0
                extra[row, 7] = starts[i] / 2.0
                extra[row, 8] = np.log10(r["risk"][b, q, i, kk] + RISK_FLOOR) / 4.0
                idx.append((b, q, i))
        Fs.append(np.concatenate([F, extra], axis=1))
    return (np.concatenate(Fs) if Fs else np.zeros((0, DIM), np.float32)), idx


class TailNet(nn.Module):
    def __init__(self, dim=DIM, hidden=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def to_target(tail):
    return np.log10(np.asarray(tail) + RISK_FLOOR)


def from_target(y):
    return np.maximum(0.0, 10.0 ** np.asarray(y) - RISK_FLOOR)


def save_tail(path, nets, meta=None):
    torch.save({"ensemble": [n.state_dict() for n in nets], "meta": meta or {}}, path)


def load_tail_fn(path):
    """입력 [M, DIM] → 꼬리 위험 [M] (앙상블 최댓값: 보수적으로)."""
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    nets = []
    for sd in ck["ensemble"]:
        n = TailNet(dim=sd["net.0.weight"].shape[1], hidden=sd["net.0.weight"].shape[0])
        n.load_state_dict(sd)
        n.eval()
        nets.append(n)
    torch.set_num_threads(1)
    agg = ck.get("meta", {}).get("agg", "max")

    def fn(x):
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            y = torch.stack([n(xt) for n in nets]).numpy()
        return from_target(y.max(axis=0) if agg == "max" else y.mean(axis=0))
    fn.meta = ck.get("meta", {})
    return fn
