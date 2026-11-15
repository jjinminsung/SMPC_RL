"""1 차원 Frenet 고속도로 시뮬레이터. 한 에피소드 = 차선 0 에서 출발해 0→1→2→3 합류.

틱 순서(0.05 s): NPC → 인지 → SMPC(planner) → supervisor → ACC → 자차.
"""
from dataclasses import dataclass, field
import copy
import math

import numpy as np

from .acc import AccPlanner
from .npc import Npc, NpcModel, idm_accel
from .params import load_params
from .perception import build_targets
from .plant import Ego, EgoPlant
from .rollouts import nudge_len
from .supervisor import Supervisor
from .track import Track

# MORAI 실측 스폰(시나리오 highway_spawn50).
# 차선: (위치 s, 13.9 m/s 로 나타날 확률, 간격 범위 s, 동시 최대 대수 maximumSpawnVehicle)
SPAWN = {1: (-81.0, 0.83, (3.3, 6.3), 15),
         2: (-54.0, 0.89, (4.1, 4.5), 15),
         3: (-16.0, 0.52, (4.1, 4.5), 5)}


@dataclass
class EpisodeConfig:
    seed: int = 0
    dt: float = 0.05
    warmup_sec: float = 45.0
    time_limit_sec: float = 120.0
    mission_speed: float = 25.0
    ego_s0: tuple = (-27.0, -4.0)
    ego_v0: tuple = (10.8, 16.0)
    spawn_scale: float = 1.0          # >1 이면 교통이 더 촘촘하다
    npc_lane_change_rate_per_min: float = 0.0
    record: bool = False
    param_overrides: dict = None      # smpc_decision 파라미터 덮어쓰기 (민감도 분석)
    perception: str = "gt"            # 'gt' | 'lidar_like' (속도 잡음 + 뒤차 미검출·깜빡임)
    npc_override: dict = None         # MORAI 시나리오 재현: {"spawn_kph": 50, "desired_kph": 80} (모든 차선)
    # NPC 가 자차에 반응하는 조건: 자차가 NPC 차선을 이만큼 침범하면 앞차로 보고, 이 감속까지 쓴다
    npc_ego_intrusion_m: float = 0.3
    npc_ego_max_brake: float = 6.0
    # 첫 차선 변경 요청부터 경로 전환까지(MORAI ≈ 0.55 s, 예측의 실행 지연과 같은 값). 0 = 확인 즉시 전환
    switch_delay_sec: float = 0.0
    acc_floor_real: bool = False      # ACC 하한(merge_keep_rolling)을 실제 C++ 규칙대로(acc.py floor_real)
    acc_tf_on_start: bool = False     # ACC 가 차선 변경 실행 시작부터 목표 차선 앞차를 정상 추종(acc.py tf_on_start)
    record_risk: bool = False         # 0.5 s 구간별 최소 TTC·간격을 남긴다(risk_buckets, RC 라벨용)
    record_states: bool = False       # 0.5 s 마다 (시각, 상태 특징 features2) 를 남긴다(samples, TC 학습용)
    explore_eps: float = 0.0          # 수집용 탐색: 판단 틱마다 이 확률로 실행 가능한 후보 중 하나를 무작위로 고른다
    explore_wait_eps: float = 0.0     # 수집용 탐색: 바꿀 수 있는 판단에서 이 확률로 0.5–3 s 일부러 기다린다
    # 탐색 시작(exploring starts): 차선마다 이 확률로만 탐색하고, 탐색은 그 차선에서 한 번(위 두 확률은 틱당 발생률).
    # 탐색 구간을 남겨 그 뒤 표본만 탐욕 정책의 값으로 쓴다. None = 틱마다 탐색(기존)
    explore_lane_prob: float = None
    explore_decision: float = 0.0     # 탐색 시작과 함께: 탐색할 차선에서 탐욕 정책이 지금 바꾸기를 고른 틱에 이 확률로 0.5–3 s 기다린다
    explore_decision_mode: str = "wait"   # 'yield' = 기다리는 대신 1–2.5 s 감속 양보
    tc_record: bool = False           # TC 수집: 판단 틱마다 실제로 따른 루트의 잎 기록(planner.tc_records)
    crn_wait: tuple = None            # 짝 비교(같은 시드 재실행): (t_d, D) = 시각 t_d 의 결정 시작에서 D 초 기다리기를 강제
    leaf_log: bool = False            # 지금 바꾸기를 명령한 틱마다 루트 0 의 지평선 끝 잎 특징(가지별)을 남긴다(RV·RA 잎 학습용)
    risk_log: bool = False            # 위험 기록: 실제 주행 중 플래너가 계산한 지평선 안 위험(바꾸기 고른 틱, 실행 중 모니터, 머리 넣기)


@dataclass
class EpisodeResult:
    seed: int
    planner: str
    success: bool
    collision: bool
    final_lane: int
    t_end: float
    t_merge: float                   # 차선 3 진입 완료 시각 (실패면 nan)
    lane_end_stops: int              # 차선 끝 근처에서 정지(<0.5 m/s)한 횟수
    stop_time: float                 # 차선 0~2 에서 정지해 있던 시간
    aborts: int
    accepts: int
    min_gap: float                   # GT 최소 bumper 간격(같은 차선)
    min_ttc: float
    lane_change_times: list = field(default_factory=list)
    events: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    stop_events: list = field(default_factory=list)   # 차선 끝 정지 시각
    min_sep: float = math.inf        # 2-D 박스 분리 최소값 max(종 간격, 횡 간격) (음수 = 겹침)
    near_lat_ticks: int = 0          # 나란히(종 겹침) 횡 간격 < 0.5 m 인 틱 수
    risk_buckets: list = field(default_factory=list)  # record_risk: [(구간 시작 t, 최소 TTC, 최소 간격)] 0.5 s 구간
    samples: list = field(default_factory=list)       # record_states: [(t, 상태 특징)]
    nudges: int = 0                  # 머리 넣기 시작 횟수
    nudge_aborts: int = 0            # 머리 넣었다가 뺀 횟수(차선 변경으로 이어지지 않음)


def obb_overlap(e, n, track):
    """자차(후륜축 기준)와 NPC(중심) 박스가 겹치는지. 차선이 직선이라 축 정렬 박스로 충분하다."""
    e_front, e_back = e.s + e.front, e.s - e.rear
    n_front, n_back = n.s + 0.5 * n.length, n.s - 0.5 * n.length
    if e_back >= n_front or n_back >= e_front:
        return False
    return abs(e.y - n.y) < 0.5 * (e.width + n.width)


class World:
    def __init__(self, planner_factory, cfg=None, P=None):
        self.cfg = cfg or EpisodeConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        # 공통 난수(CRN): 차선별 스폰, NPC 별(차선, 순번), 인지 잡음이 각자 난수열을 쓴다. 한 난수열을 같이 쓰면
        # 두 플래너가 처음 갈라진 뒤 교통 전체가 달라져 짝 비교가 약해진다.
        self.rng_spawn = {lane: np.random.default_rng([self.cfg.seed, 1, lane]) for lane in SPAWN}
        self.spawn_count = {lane: 0 for lane in SPAWN}
        self.rng_percep = np.random.default_rng([self.cfg.seed, 3])
        self.P = P or load_params("smpc_decision")
        for k, v in (self.cfg.param_overrides or {}).items():
            setattr(self.P, k, v)
        self.P_sel = load_params("target_selector")
        self.P_acc = load_params("acc_planner")
        self.P_sup = load_params("lane_change_supervisor") if (self._has("lane_change_supervisor")) else load_params("supervisor")
        self.track = Track()
        self.model = NpcModel.from_measurements()
        ov = self.cfg.npc_override or {}
        if "spawn_kph" in ov:
            self.model.spawn_speed = ov["spawn_kph"] / 3.6
        if "desired_kph" in ov:
            vd = ov["desired_kph"] / 3.6
            self.model.desired_quantiles = ((0.1, vd), (0.5, vd), (0.9, vd))
        self.spawn_all_fast = "spawn_kph" in ov      # 사용자 지정 스폰 속도면 모든 차가 그 속도로 나타난다
        self.npcs = []
        self.next_uid = 100
        self.next_spawn = {lane: 0.0 for lane in SPAWN}
        self.t = 0.0
        self.planner = planner_factory(self.P, self.track)
        self.use_monitor = getattr(self.planner, "use_monitor", False)
        if self.use_monitor:
            self.P_sup.active_rear_guard_enabled = False
        self.ego = None

    @staticmethod
    def _has(name):
        from .params import CONFIG_DIR
        return (CONFIG_DIR / f"{name}.yaml").exists()

    # ------------------------------------------------------------ NPC
    def spawn(self):
        for lane, (s_sp, p_fast, period, max_alive) in SPAWN.items():
            if self.t < self.next_spawn[lane]:
                continue
            if sum(1 for n in self.npcs if n.spawn_lane == lane) >= max_alive:
                self.next_spawn[lane] = self.t + 0.5
                continue
            occupied = any(n.lane == lane and abs(n.s - s_sp) < 12.0 for n in self.npcs)
            if self.ego is not None and self.ego.lane == lane and abs(self.ego.s - s_sp) < 15.0:
                occupied = True
            if occupied:
                self.next_spawn[lane] = self.t + 0.5
                continue
            m = self.model
            r = np.random.default_rng([self.cfg.seed, 2, lane, self.spawn_count[lane]])
            self.spawn_count[lane] += 1
            v0 = m.spawn_speed if (self.spawn_all_fast or r.random() < p_fast) else 0.0
            n = Npc(uid=self.next_uid, lane=lane, s=s_sp, v=v0, born=self.t,
                    v_desired=m.sample_desired(r),
                    accel_cap=float(np.clip(r.normal(m.spawn_accel_mean, m.spawn_accel_std),
                                            m.spawn_accel_min, m.spawn_accel_max)),
                    length=m.length, width=m.width)
            n.rng = r
            n.v_target = n.v_desired
            n.y, n.v_lat, n.spawn_lane, n.merge_to = self.track.center_y(lane), 0.0, lane, None
            self.npcs.append(n)
            self.next_uid += 1
            gap = self.rng_spawn[lane].uniform(*period)
            self.next_spawn[lane] = self.t + gap / self.cfg.spawn_scale

    def occupies(self, y, width, lane):
        """폭 width 인 차량이 lane 을 0.3 m 이상 침범하는지."""
        return abs(y - self.track.center_y(lane)) < 0.5 * self.track.lane_width + 0.5 * width - 0.3

    # ---- NPC 차선 끝 강제 합류 (MORAI 실측: 차선 2 끝 s 411–417 에서 v≈10 m/s, 횡속도 2.4 m/s 로 왼쪽 합류.
    #      틈이 없으면 끝 앞에서 서서 기다린다. 틈 기준은 MORAI NPC 설정 SafetyDist 5 m, TTC 1.0 s) ----
    MERGE_ZONE_M, MERGE_LAT_RATE, MERGE_SAFE_GAP, MERGE_TTC = 80.0, 2.4, 5.0, 1.0

    def _lane_occupants(self, lane, exclude):
        """lane 을 0.3 m 이상 침범한 차량 [(s_front, s_rear, v)] (NPC + 자차)."""
        out = []
        for o in self.npcs:
            if o is not exclude and self.occupies(o.y, o.width, lane):
                out.append((o.s + 0.5 * o.length, o.s - 0.5 * o.length, o.v))
        e = self.ego
        if e is not None and self.occupies(e.y, e.width, lane):
            out.append((e.s + e.front, e.s - e.rear, e.v))
        return out

    def _merge_gap_ok(self, n, lane):
        front, rear = n.s + 0.5 * n.length, n.s - 0.5 * n.length
        for of, orr, ov in self._lane_occupants(lane, n):
            if orr >= front:                     # 앞차: 거리 + 제동으로 따라붙을 수 있는지(TTC 1 s, 감속 4 m/s²)
                g, c = orr - front, n.v - ov
                need = max(self.MERGE_SAFE_GAP, 0.5 * n.v)
                if c > 0.1:
                    need = max(need, self.MERGE_SAFE_GAP + c * c / (2.0 * 4.0), c * self.MERGE_TTC)
                if g < need:
                    return False
            elif of <= rear:                     # 뒤차
                g, c = rear - of, ov - n.v
                if g < self.MERGE_SAFE_GAP or (c > 0.1 and g / c < self.MERGE_TTC):
                    return False
            else:
                return False                     # 나란히 겹침
        return True

    def step_npcs(self, dt):
        m = self.model
        order = sorted(self.npcs, key=lambda n: -n.s)
        for n in order:
            end = self.track.road_end[n.lane] if n.lane < self.track.final_lane else math.inf
            target = getattr(n, "merge_to", None)
            # 합류 시작 판단
            if target is None and math.isfinite(end) and end - n.s < self.MERGE_ZONE_M and \
                    self._merge_gap_ok(n, n.lane + 1):
                n.merge_to = target = n.lane + 1
            # 앞차: 지금 차선 + (합류 중이면) 목표 차선을 침범한 차량
            lead_gap, lead_v = math.inf, 0.0
            lanes = [n.lane] + ([target] if target is not None else [])
            for o in self.npcs:
                if o is n or o.s <= n.s:
                    continue
                if not any(self.occupies(o.y, o.width, L) for L in lanes):
                    continue
                g = (o.s - 0.5 * o.length) - (n.s + 0.5 * n.length)
                if g < lead_gap:
                    lead_gap, lead_v = g, o.v
            e = self.ego
            lead_is_ego = False
            m_in = self.cfg.npc_ego_intrusion_m
            if e is not None and e.s + e.front > n.s and any(
                    abs(e.y - self.track.center_y(L)) < 0.5 * self.track.lane_width + 0.5 * e.width - m_in for L in lanes):
                g = (e.s - e.rear) - (n.s + 0.5 * n.length)
                if g < lead_gap and g > -0.5 * n.length:
                    lead_gap, lead_v, lead_is_ego = g, e.v, True
            # 합류 못 했으면 차선 끝이 정지 장애물
            if target is None and math.isfinite(end):
                g_end = (end - 2.0) - (n.s + 0.5 * n.length)
                if g_end < lead_gap:
                    lead_gap, lead_v, lead_is_ego = g_end, 0.0, False
            a_free = n.profile_accel(self.t, dt, m, n.rng)
            n.a = idm_accel(n.v, lead_gap, lead_v, a_free, m)
            if lead_is_ego:
                n.a = max(n.a, -self.cfg.npc_ego_max_brake)
            v_new = max(0.0, n.v + n.a * dt)
            n.s += 0.5 * (n.v + v_new) * dt
            n.v = v_new
            # 횡 이동
            if target is not None:
                y_t = self.track.center_y(target)
                step = min(self.MERGE_LAT_RATE * dt, abs(y_t - n.y)) if n.v > 0.3 else 0.0
                n.v_lat = math.copysign(step / dt, y_t - n.y) if step > 0 else 0.0
                n.y += math.copysign(step, y_t - n.y)
                if abs(n.y - y_t) < 0.5 * self.track.lane_width:
                    n.lane = target
                if abs(n.y - y_t) < 1e-3:
                    n.y, n.v_lat, n.merge_to = y_t, 0.0, None
        self.npcs = [n for n in self.npcs if n.s < self.track.road_end[n.lane] + (5.0 if n.lane < 3 else 0.0)]

    # ------------------------------------------------------------ 인지 잡음
    def lidar_like_view(self, e):
        """라이다 트래커 수준의 인지(1 단계 측정): 속도 잡음(AR(1), σ 1.0 m/s), 60 m 밖 뒤차 미검출,
        뒤차 검출 깜빡임(프레임당 50 %), 앞차는 90 m 까지 검출(프레임당 90 %)."""
        out = []
        for n in self.npcs:
            ds = n.s - e.s
            if ds < -60.0 or ds > 90.0:
                continue
            p_det = 0.5 if ds < 0 else 0.9
            if self.rng_percep.random() > p_det:
                continue
            noise = getattr(n, "_vnoise", 0.0)
            noise = 0.8 * noise + 0.6 * self.rng_percep.normal(0.0, 1.0)      # 정상 상태 σ ≈ 1.0 m/s
            n._vnoise = noise
            c = copy.copy(n)
            c.v = max(0.0, n.v + noise)
            out.append(c)
        return out

    # ------------------------------------------------------------ run
    def reset(self):
        cfg = self.cfg
        while self.t < cfg.warmup_sec:
            self.spawn()
            self.step_npcs(cfg.dt)
            self.t += cfg.dt
        s0 = self.rng.uniform(*cfg.ego_s0)
        v0 = self.rng.uniform(*cfg.ego_v0)
        self.ego = Ego(s=s0, y=self.track.center_y(0), v=v0, lane=0)
        self.plant = EgoPlant(self.ego, self.track)
        self.sup = Supervisor(self.P_sup, self.track, initial_lane=0)
        self.sup.switch_delay = cfg.switch_delay_sec
        self.acc = AccPlanner(self.P_acc, self.track)
        self.acc.floor_real = cfg.acc_floor_real
        self.acc.tf_on_start = cfg.acc_tf_on_start
        self.acc.f = v0
        self.lc_connector = False
        self.t0 = self.t
        self.planner.explore_eps = cfg.explore_eps
        self.planner.explore_wait_eps = cfg.explore_wait_eps
        self.planner.explore_rng = np.random.default_rng(cfg.seed + 99991)
        self.planner.explore_lane_prob = cfg.explore_lane_prob
        self.planner.explore_decision = cfg.explore_decision
        self.planner.explore_decision_mode = cfg.explore_decision_mode
        self.planner.explore_events = []
        self.planner.explore_decisions = []
        self.planner.tc_record = cfg.tc_record
        self.planner.tc_records = []
        self.planner.crn_wait = cfg.crn_wait
        self.planner.decision_starts = []
        self.planner.risk_log = [] if cfg.risk_log else None
        self.planner.leaf_log = [] if cfg.leaf_log else None

    def run(self):
        cfg = self.cfg
        self.reset()
        e = self.ego
        res = EpisodeResult(seed=cfg.seed, planner=getattr(self.planner, "name", "?"), success=False,
                            collision=False, final_lane=0, t_end=0.0, t_merge=math.nan, lane_end_stops=0,
                            stop_time=0.0, aborts=0, accepts=0, min_gap=math.inf, min_ttc=math.inf)
        stopped = False
        last_change_left = -1e9
        tree_return_count = 0
        nudging, nudge_block_until, nudge_miss, mon_ret = False, -1e9, 0, 0
        while self.t - self.t0 < cfg.time_limit_sec:
            now = self.t
            self.spawn()
            self.step_npcs(cfg.dt)
            visible = self.npcs if cfg.perception == "gt" else self.lidar_like_view(e)
            t = build_targets(now, e, visible, self.track, self.P_sel, forced_pair=self.sup.forced_pair)
            if cfg.record_states and (now - self.t0) + 1e-9 >= len(res.samples) * 0.5:
                from .rl.features2 import state_features2
                res.samples.append((round(now - self.t0, 2), state_features2(t, e).tolist()))
            nudge_mon = getattr(self.planner, "nudge_monitor", False)
            self.planner.nudging = nudging and nudge_mon
            mon_n = (self.planner.nudge_monitor_tick(now, t, e.v, cfg.mission_speed)
                     if nudging and nudge_mon and not self.sup.active else None)
            mon_ret = mon_ret + 1 if mon_n == "return" else 0
            self.planner._sup_busy = self.sup.active or self.sup.pending_at is not None
            self.planner._leaf0 = None
            dec = self.planner.decide(now, t, e.v, cfg.mission_speed, self.acc.f, self.acc.status)
            if cfg.leaf_log and dec.change_left and not self.planner._sup_busy and self.planner._leaf0 is not None:
                self.planner.leaf_log.append((now,) + self.planner._leaf0)
            if dec.change_left:
                last_change_left = now
                self.acc.last_change_left = now
            n_ev = len(self.sup.events)
            self.sup.on_decision(now, dec.change_left, t, e)
            self.sup.on_targets(now, t, e)
            if self.sup.active and self.use_monitor:
                # 트리 플래너: 차선 변경 중 재계획이 rear_abort 를 대체한다. supervisor 처럼 연속 2 틱이면 복귀.
                if self.planner.monitor(now, t, e.v, cfg.mission_speed) == "return":
                    tree_return_count += 1
                    if tree_return_count >= 2:
                        self.sup._abort(now, "tree_return", self.sup.source)
                        tree_return_count = 0
                else:
                    tree_return_count = 0
            else:
                tree_return_count = 0
            # 머리 넣고 기다리기(NUDGE): 플래너가 고르면 원 차선 경로 안에서 침범 깊이까지 옆으로 간다(움직일 때만).
            # 차선 변경 승인으로 이어지면 plant.set_path 가 이어받고, 플래너가 더 고르지 않으면 원 차선 가운데로 뺀다.
            want_n = (dec.request.active and dec.request.action == "nudge" and not self.sup.active
                      and now >= nudge_block_until)
            if self.sup.active or self.sup.pending_at is not None:
                nudging, nudge_miss = False, 0
            elif want_n:
                nudge_miss = 0
                if not nudging:
                    sgn = 1.0 if t.target_lane_id > t.current_lane_id else -1.0
                    y_goal = self.track.center_y(t.current_lane_id) + sgn * (0.5 * self.track.lane_width - 0.5 * e.width +
                                                                            dec.request.accel)
                    self.plant.nudge_to(y_goal, nudge_len(e.v))
                    nudging = True
                    res.nudges += 1
                    res.events.append(("nudge", round(now - self.t0, 2), t.current_lane_id, t.target_lane_id,
                                       f"{dec.request.accel:.1f}", round(e.v, 2), round(t.distance_to_lane_end, 1)))
            elif nudging and not dec.change_left and (
                    mon_ret >= 2 if nudge_mon else (nudge_miss := nudge_miss + 1) >= 4):
                # 모니터 없음: 플래너가 0.2 s(4 틱) 연속 머리 넣기를 고르지 않으면 원 차선 가운데로 뺀다.
                # 모니터('nm'): 머리 넣기가 안 뽑혀도 모니터가 연속 2 틱 '빼기'라고 할 때만 뺀다(차선 변경 모니터와 같다)
                nudge_miss, mon_ret = 0, 0
                self.plant.nudge_to(self.track.center_y(self.plant.path_lane), nudge_len(e.v))
                nudging = False
                nudge_block_until = now + 2.0
                res.nudge_aborts += 1
                res.events.append(("nudge_abort", round(now - self.t0, 2), t.current_lane_id, t.target_lane_id, "",
                                   round(e.v, 2), round(t.distance_to_lane_end, 1)))
            self.acc.nudge_active = nudging
            self.sup.tick(now, e)
            for ev in self.sup.events[n_ev:]:
                res.events.append((ev.kind, round(ev.t - self.t0, 2), ev.source, ev.target, ev.detail,
                                   round(e.v, 2), round(t.distance_to_lane_end, 1)))
                if ev.kind == "accept":
                    res.accepts += 1
                    # waypoint 는 정지(≤ 1 m/s)·원 차선 30 m 이내에서 승인되면 connector 로 간다(shouldUseEndpointConnector)
                    self.lc_connector = t.distance_to_lane_end <= 30.0 and e.v <= 1.0
                elif ev.kind == "abort":
                    res.aborts += 1
                elif ev.kind == "complete":
                    res.lane_change_times.append(round(ev.t - self.t0, 2))
            self.plant.set_path(now, self.sup.lane)
            intent = self.sup.active or now - last_change_left <= 0.5 or dec.request.active
            lc_elapsed = now - self.sup.change_start if self.sup.active else 0.0
            # 정지 connector 차선 변경: 실제 ACC 처럼 차선 끝 상한 대신 목표 차선별 connector 상한
            conn = self.sup.active and self.lc_connector
            caps = self.P_acc.endpoint_connector_speed_cap_by_target_lane_mps
            tgt = min(max(int(t.target_lane_id), 0), 3)
            v_cmd = self.acc.step(now, cfg.dt, t, e.v, e.y, cfg.mission_speed, dec.request,
                                  self.sup.active, lc_elapsed, intent,
                                  connector_cap=(caps[tgt] if tgt < len(caps) else self.P_acc.endpoint_connector_speed_cap_mps)
                                  if conn else None,
                                  connector_active=conn)
            self.plant.step(now, cfg.dt, v_cmd)
            self.t += cfg.dt

            # 지표
            tick_ttc, tick_gap = math.inf, math.inf
            for n in self.npcs:
                if obb_overlap(e, n, self.track):
                    res.collision = True
                g_long = max((n.s - 0.5 * n.length) - (e.s + e.front), (e.s - e.rear) - (n.s + 0.5 * n.length))
                g_lat = abs(e.y - n.y) - 0.5 * (e.width + n.width)
                res.min_sep = min(res.min_sep, max(g_long, g_lat))
                if g_long < 0.0 and 0.0 <= g_lat < 0.5:
                    res.near_lat_ticks += 1
                if self.occupies(n.y, n.width, e.lane) or abs(n.y - e.y) < 0.5 * (e.width + n.width):
                    if n.s > e.s:
                        g = (n.s - 0.5 * n.length) - (e.s + e.front)
                        c = e.v - n.v
                    else:
                        g = (e.s - e.rear) - (n.s + 0.5 * n.length)
                        c = n.v - e.v
                    if abs(n.y - e.y) < 0.5 * (e.width + n.width) + 0.3:
                        res.min_gap = min(res.min_gap, g)
                        tick_gap = min(tick_gap, g)
                        if c > 0.1 and g > 0:
                            res.min_ttc = min(res.min_ttc, g / c)
                            tick_ttc = min(tick_ttc, g / c)
            if cfg.record_risk:
                kb = int((now - self.t0) / 0.5 + 1e-9)
                while len(res.risk_buckets) <= kb:
                    res.risk_buckets.append([round(len(res.risk_buckets) * 0.5, 2), math.inf, math.inf])
                bk = res.risk_buckets[kb]
                bk[1], bk[2] = min(bk[1], tick_ttc), min(bk[2], tick_gap)
            lane_now = self.sup.lane if not self.sup.active else self.sup.source
            if e.v < 0.5 and lane_now < self.track.final_lane:
                res.stop_time += cfg.dt
                if not stopped and self.track.csv_end[lane_now] - e.s < 60.0:
                    res.lane_end_stops += 1
                    res.stop_events.append(round(self.t - self.t0, 2))
                stopped = True
            elif e.v > 1.0:
                stopped = False
            if cfg.record:
                res.trace.append((round(now - self.t0, 2), e.s, e.y, e.v, self.sup.lane, int(self.sup.active),
                                  dec.selected, dec.reason, v_cmd))
            if res.collision:
                break
            if not self.sup.active and self.sup.lane == self.track.final_lane:
                res.success = True
                res.t_merge = self.t - self.t0
                break
            lane_now = self.sup.lane
            if lane_now < self.track.final_lane and e.s > self.track.csv_end[lane_now] + 5.0:
                break      # 차선 끝을 지나쳤다(경로 이탈)
        res.final_lane = self.sup.lane
        res.t_end = self.t - self.t0
        return res
