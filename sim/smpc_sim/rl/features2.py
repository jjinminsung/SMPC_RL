"""차선 변경 틀(출발 차선 cur, 목표 차선 tgt) 기준 상태 특징. TR(꼬리 위험)과 RC(위험 비평가)의 입력.

학습 상태(시뮬레이터 실제 상태)와 트리 잎(지평선 끝 예측 상태)이 같은 틀·같은 정보를 쓰도록
둘 다 frame_features 하나로 만든다.
  틀: 시뮬레이터 TargetSet 과 같다. 차선 변경 중에는 (출발, 목표), 아니면 (지금 차선, 다음 차선).
      잎에서는 목표 차선 도착(|y - y_tgt| ≤ 완료 허용폭)이면 (목표, 그다음 차선), 아니면 (출발, 목표).
  차량: cur·tgt·tgt+1 세 차선의 가장 가까운 앞차·뒤차 간격과 상대 속도. 학습 = 인지된 주변 차량 전부(실제 상태),
        잎 = 트리가 반응형으로 굴린 차량(cf·tf·tf2·뒤차) + 나머지 주변 차량은 등속 예측.
  자차: 속도, 횡 진행도 p(출발 차선 0 → 목표 차선 1), 출발 차선 one-hot, 차선 끝까지 거리·시간, 남은 차선 수.
"""
import numpy as np

from ..track import CSV_END_S, FINAL_LANE

LW = 3.5
EGO_FRONT, EGO_REAR = 3.845, 0.79
GAP_CAP, DV_CAP, DIST_CAP, TTLE_CAP = 100.0, 10.0, 400.0, 30.0
FEATURE_NAMES = (["v", "progress"] + [f"lane{i}" for i in range(3)] + ["dist_end", "ttle", "lanes_left"] +
                 [f"{k}_{ln}" for ln in ("cur", "tgt", "nxt") for k in ("gap_f", "dv_f", "gap_r", "dv_r")])
DIM = len(FEATURE_NAMES)


def frame_features(cur, tgt, ego_s, ego_y, ego_v, vehicles):
    """vehicles: [(s 중심, 차선, 속도, 길이)]. 반환 [DIM]."""
    f = np.zeros(DIM, dtype=np.float32)
    f[0] = ego_v / 25.0
    f[1] = float(np.clip((ego_y - cur * LW) / ((tgt - cur) * LW), -0.5, 1.5)) if tgt != cur else 0.0
    if cur < FINAL_LANE:
        f[2 + cur] = 1.0
    dist = max(0.0, CSV_END_S[cur] - ego_s) if cur < FINAL_LANE else DIST_CAP
    ttle = dist / ego_v if ego_v > 0.1 else TTLE_CAP
    f[5] = min(dist, DIST_CAP) / DIST_CAP
    f[6] = min(ttle, TTLE_CAP) / TTLE_CAP
    f[7] = (FINAL_LANE - cur) / 3.0
    lanes = (cur, tgt, tgt + 1 if tgt != cur else -1)
    for j, lane in enumerate(lanes):
        gf, dvf, gr, dvr = GAP_CAP, 0.0, GAP_CAP, 0.0
        if 0 <= lane <= FINAL_LANE:
            for s, ln, v, length in vehicles:
                if ln != lane:
                    continue
                if s >= ego_s:
                    g = (s - 0.5 * length) - (ego_s + EGO_FRONT)
                    if g < gf:
                        gf, dvf = g, v - ego_v
                else:
                    g = (ego_s - EGO_REAR) - (s + 0.5 * length)
                    if g < gr:
                        gr, dvr = g, v - ego_v
        f[8 + 4 * j: 12 + 4 * j] = [min(max(gf, 0.0), GAP_CAP) / GAP_CAP, float(np.clip(dvf, -DV_CAP, DV_CAP)) / DV_CAP,
                                    min(max(gr, 0.0), GAP_CAP) / GAP_CAP, float(np.clip(dvr, -DV_CAP, DV_CAP)) / DV_CAP]
    return f


def target_set_vehicles(t):
    out, seen = [], set()
    for v in [t.current_front, t.target_front, t.target_rear] + list(t.nearby_vehicles):
        if v.valid and v.unique_id not in seen:
            seen.add(v.unique_id)
            out.append(v)
    return out


def state_features2(t, ego):
    """시뮬레이터 실제 상태 → 특징(학습 샘플). 틀 = TargetSet 의 (current_lane_id, target_lane_id)."""
    veh = [(v.s, v.lane_id, max(0.0, v.v_long), v.length) for v in target_set_vehicles(t)]
    return frame_features(int(t.current_lane_id), int(t.target_lane_id), ego.s, ego.y, ego.v, veh)


def frame_features_batch(cur, tgt, ego_s, ego_y, ego_v, vs, vl, vv, vlen):
    """frame_features 의 벡터판(트리 잎용). cur, tgt, ego_*: [M]; vs, vl, vv, vlen: [M, Nv] (vl < 0 = 빈 칸). 반환 [M, DIM]."""
    cur, tgt = np.asarray(cur, int), np.asarray(tgt, int)
    M = len(ego_s)
    if np.shape(vs)[1] == 0:                       # 주변 차량 없음: 빈 칸 하나(차선 -1)
        vs, vv, vlen = np.zeros((M, 1)), np.zeros((M, 1)), np.ones((M, 1))
        vl = np.full((M, 1), -1)
    F = np.zeros((M, DIM), dtype=np.float32)
    F[:, 0] = ego_v / 25.0
    same = tgt == cur
    F[:, 1] = np.where(same, 0.0, np.clip((ego_y - cur * LW) / np.where(same, 1.0, (tgt - cur) * LW), -0.5, 1.5))
    for ln in range(FINAL_LANE):
        F[:, 2 + ln] = cur == ln
    end = np.array([CSV_END_S[k] for k in range(FINAL_LANE + 1)])
    dist = np.where(cur < FINAL_LANE, np.maximum(0.0, end[np.clip(cur, 0, FINAL_LANE)] - ego_s), DIST_CAP)
    ttle = np.where(ego_v > 0.1, dist / np.maximum(ego_v, 1e-9), TTLE_CAP)
    F[:, 5] = np.minimum(dist, DIST_CAP) / DIST_CAP
    F[:, 6] = np.minimum(ttle, TTLE_CAP) / TTLE_CAP
    F[:, 7] = (FINAL_LANE - cur) / 3.0
    es = np.asarray(ego_s)[:, None]
    front = vs >= es
    g_front = (vs - 0.5 * vlen) - (es + EGO_FRONT)
    g_rear = (es - EGO_REAR) - (vs + 0.5 * vlen)
    rows = np.arange(M)
    for j, lane in enumerate((cur, tgt, np.where(same, -1, tgt + 1))):
        inl = (vl == lane[:, None]) & (lane[:, None] >= 0) & (lane[:, None] <= FINAL_LANE)
        for col, m, g_all in ((0, inl & front, g_front), (2, inl & ~front, g_rear)):
            g = np.where(m, g_all, np.inf)
            k = np.argmin(g, axis=1)
            gk = g[rows, k]
            has = gk < GAP_CAP
            dv = np.where(has, vv[rows, k] - ego_v, 0.0)
            F[:, 8 + 4 * j + col] = np.where(has, np.maximum(gk, 0.0), GAP_CAP) / GAP_CAP
            F[:, 9 + 4 * j + col] = np.clip(dv, -DV_CAP, DV_CAP) / DV_CAP
    return F
