"""반응형 트리 플래너.

TreePlanner 의 판단 골격(루트 후보, 비용, 준비 행동 요청)은 그대로 두고 세 가지를 바꾼다.
  1. 루트 평가: 가지 = (뒤차 yield/nonyield) × (앞차 base/brake/accel), 자차·타겟 차량을 reactive.joint_rollout 으로
     함께 굴린다. 확률 = 기본 1 − Σ예외, 예외는 최솟값 p_exc_floor. 대응 = 계속 / 복귀(예외가 드러나는 시점부터).
     루트 값 = min over 가지별 대응 조합 Σ_b p_b J_b, 단 모든 스텝에서 Σ_b p_b r_b(k) ≤ ε.
  2. 공통 조건 G6/G7/G8 을 쓰지 않는다(drop_common). G7 이 보던 목표 차선 뒤·옆 차량은 트리가 모두 본다.
  3. 모니터: 판단과 같은 예측으로 '지금부터 계속' 과 '지금 복귀' 를 비교한다. 뒤차가 반응 지연이 지났는데도
     감속하지 않으면 nonyield 확률을 올린다(두 가설 갱신). 계속 위험 > 2ε 이고 복귀가 더 안전하면 복귀.
single=True 는 비교군: 가지 없이 기본 예측 하나(뒤차 yield, 앞차 base)로만 판단한다.

RL 종단 비용(TR, rl/tail_risk.py): 짧은 지평선이 잘라 낸 지평선 뒤 위험을 학습한 꼬리 위험으로 채운다.
판단과 모니터 모두 peak = max(지평선 안 max_k Σ_b p_b r_b(k), Σ_b p_b TR_b) 로 같은 규칙(peak ≤ ε)을 쓴다.
"""
from dataclasses import dataclass
import copy
import itertools
import math

import numpy as np

from .baseline import waypoint_blend_duration
from .reactive import CONTINUE, RETURN, ReactiveParams, joint_rollout, load_acc_params, target_lane_vehicles
from .rollouts import CHANGE_NOW, NUDGE, ActionSpec, RolloutContext
from .tree import RETURN as TREE_RETURN, TreePlanner


@dataclass
class _Roots:
    """root_trajectories 가 넘기는 것: 가지마다 굴려야 하므로 궤적 대신 입력을 들고 다닌다."""
    ctx: object
    specs: list
    t: object
    valid: np.ndarray


class ReactiveTreePlanner(TreePlanner):
    def __init__(self, P, track, cfg=None, rp=None, single=False, drop_common=True):
        super().__init__(P, track, cfg)
        self.rp = rp or ReactiveParams()
        self.A = load_acc_params()
        self.single, self.drop_common = single, drop_common
        self.name += "_rx" + ("_single" if single else "") + ("" if drop_common else "_common")
        self.intr_since = None            # 실행 중 자차가 목표 차선에 들어간 시각
        self.p_rear_post = None
        self.lc_connector = None          # 마지막 지금 바꾸기 승인의 실행 방식(connector 출발인가)

    # ------------------------------------------------------------ 훅
    def root_trajectories(self, ctx, specs, t):
        vehicles = self._vehicles(t)
        r = joint_rollout(ctx, specs, t, vehicles, self.P, self.A, self.rp, self.track,
                          [("yield", "base", None)], self._accel_of)
        return _Roots(ctx, specs, t, r["valid"])

    def common_ok(self, g6, g7, g8):
        return True if self.drop_common else (g6 and g7 and g8)

    @staticmethod
    def _accel_of(v):
        return getattr(v, "accel", 0.0)

    @staticmethod
    def _vehicles(t):
        out, seen = [], set()
        for v in [t.current_front, t.target_front, t.target_rear] + list(t.nearby_vehicles):
            if v.valid and v.unique_id not in seen:
                seen.add(v.unique_id)
                out.append(v)
        return out

    def _branches(self, t, p_rear=None):
        cf, tf, rears = target_lane_vehicles(t, self._vehicles(t))
        if self.single:
            return [("yield", "base", None, 1.0)]
        fl = self.rp.p_exc_floor
        flr = fl if self.rp.p_rear_floor is None else self.rp.p_rear_floor
        flb = fl if self.rp.p_brake_floor is None else self.rp.p_brake_floor
        pr = flr if p_rear is None else p_rear
        # passed_front_as_rear: 지나칠 수 있는 목표 차선 앞차가 있으면 뒤차 가지를 만든다(C++ rxBranches 와 같다)
        has_rear = bool(rears) or (self.rp.passed_front_as_rear and tf is not None)
        rb = [("yield", 1.0 - pr, None), ("nonyield", pr, "rear")] if has_rear else [("yield", 1.0, None)]
        fb = [("base", 1.0 - flb - fl, None), ("brake", flb, "front"), ("accel", fl, None)] if (cf or tf) else [("base", 1.0, None)]
        out = []
        for (rm, p1, t1), (fm, p2, t2) in itertools.product(rb, fb):
            out.append((rm, fm, t2 or t1, p1 * p2))
        return out

    # ------------------------------------------------------------ 루트 평가
    def evaluate_tree(self, traj, specs, t, v_e, R, vehicles, root_cost, candidate_mask=None):
        """TreePlanner.evaluate_tree 와 같은 반환: (feas, cost, combo, peak, branches)."""
        P = self.P
        rev = getattr(self, "nudge_monitor", False) and getattr(self, "nudging", False) and self.p_rear_post is not None
        br = self._branches(t, self.p_rear_post if rev else None)    # 머리 넣은 채 관측한 뒤차 양보 여부를 쓴다
        rp = self._rp_revealed() if rev else self.rp
        if rev:
            br = [(b[0], b[1], "now" if b[2] == "rear" else b[2], b[3]) for b in br]
        self._leaf_t = t
        self._specs, self._roots_eval = specs, True
        roots = traj
        r = joint_rollout(roots.ctx, specs, t, self._vehicles(t), P, self.A, rp, self.track,
                          [b[:3] for b in br], self._accel_of)
        out = self._solve(r, br, root_cost, roots.valid if candidate_mask is None else roots.valid & candidate_mask,
                          P.risk_epsilon)
        self._roots_eval = False                           # 모니터(monitor_core)의 _solve 는 루트 평가가 아니다
        return out

    def _solve(self, r, br, root_cost, mask, eps):
        c = self.cfg
        B = len(br)
        N = r["risk"].shape[2]
        p = np.array([b[3] for b in br])
        opts = [[CONTINUE, RETURN] if br[b][2] else [CONTINUE] for b in range(B)]
        combos = np.array(list(itertools.product(*opts)))                    # [C, B]
        cost_r = np.array([0.0, c.cost_response[TREE_RETURN]])
        feas = np.zeros(N, dtype=bool)
        best_cost = np.full(N, np.inf)
        best_combo = np.zeros((N, B + 1), dtype=int)
        best_risk = np.ones(N)
        TT = None                                          # 학습한 꼬리 위험(TR) [B, 2, N]: 지평선 뒤 같은 가지의 최대 위험
        if getattr(self, "tail_fn", None) is not None and getattr(self, "_roots_eval", False):
            from .rl.tail_risk import leaf_inputs
            from .rollouts import execution_start
            roots = [i for i in range(N) if mask[i]]
            if roots:
                mon = getattr(self, "_mon_eval", False)            # 모니터 루트: 시작 0 s, 준비 가속 0
                Fx, idx = leaf_inputs(self, r, br, [0.0] * len(self._specs) if mon else
                                      [execution_start(sp, self.P) for sp in self._specs],
                                      [0.0] * len(self._specs) if mon else [sp.prep_accel for sp in self._specs],
                                      roots, r["risk"].shape[-1] - 1)
                TT = np.zeros((B, 2, N))
                for (b, q, i), v in zip(idx, self.tail_fn(Fx)):
                    TT[b, q, i] = v
        for i in range(N):
            if not mask[i]:
                continue
            ok = np.ones(len(combos), dtype=bool)
            rk = np.zeros((len(combos), r["risk"].shape[-1]))
            jc = np.zeros(len(combos))
            for b in range(B):
                q = combos[:, b]
                ok &= r["allowed"][b, q, i]
                rk += p[b] * r["risk"][b, q, i]
                jc += p[b] * cost_r[q]
            peak = rk.max(axis=1)
            if TT is not None:                             # 꼬리 최댓값의 위 한계 Σ_b p_b TR_b 와 지평선 안 최댓값 중 큰 값
                peak = np.maximum(peak, sum(p[b] * TT[b, combos[:, b], i] for b in range(B)))
            total = root_cost[i] + jc + c.risk_weight * peak
            good = ok & (peak <= eps)
            if good.any():
                k = int(np.argmin(np.where(good, total, np.inf)))
                feas[i], best_cost[i], best_risk[i] = True, total[k], peak[k]
            elif ok.any():
                k = int(np.argmin(np.where(ok, peak, np.inf)))
                best_risk[i] = peak[k]
            # TreePlanner 형식: [0] 은 nominal 자리, 이후 가지별 대응(복귀 = TREE_RETURN)
            best_combo[i, 1:] = np.where(combos[k] == RETURN, TREE_RETURN, 0) if (good.any() or ok.any()) else 0
        self.last_tree = {"branches": [(f"{b[0]}/{b[1]}", round(b[3], 3), b[2]) for b in br]}
        return feas, best_cost, best_combo, best_risk, br

    # ------------------------------------------------------------ 모니터
    def monitor(self, now, t, v_e, mission):
        self.mission = mission
        self.hist.update(t, now)
        self._update_rear_belief(now, t)
        return self.monitor_core(t, v_e, mission, connector=self.lc_connector)

    def _update_rear_belief(self, now, t):
        """뒤차 양보 여부 관측 갱신. 자차가 목표 차선에 기준 이상 들어간 뒤 반응 지연 + 알아채는 시간이 지났는데
        뒤차가 다가오면서(자차보다 빠름) 감속하지 않으면 nonyield, 감속하면 yield 쪽으로 기울인다."""
        rp, W = self.rp, self.P.lane_width_m
        intr = 0.5 * W + 0.5 * self.P.ego_width_m - abs(t.ego_d_target)
        if intr < rp.rear_react_intrusion_m:
            self.intr_since, self.p_rear_post = None, None
            return
        if self.intr_since is None:
            self.intr_since = now
        _, _, rears = target_lane_vehicles(t, self._vehicles(t))
        if not rears:
            return
        rear = rears[0]
        if now - self.intr_since < rp.rear_react_delay_s + rp.rear_detect_s:
            return
        a = self._accel_of(rear)
        if a <= -0.5:
            self.p_rear_post = rp.p_exc_floor * 0.2
        elif a > -0.1 and rear.v_long > getattr(self, "_v_e", 0.0) + 0.5:
            self.p_rear_post = 0.9

    def _rp_revealed(self):
        rp = copy.copy(self.rp)
        rp.rear_revealed = True
        return rp

    def nudge_monitor_tick(self, now, t, v_e, mission):
        """머리 넣은 동안 매 틱(world 가 부른다): 뒤차 양보 관측 갱신 + 남기 대 지금 빼기 위험 비교.
        차선 변경 모니터(monitor_core)와 같은 규칙: 남기 위험(가지별 나중 대응 포함) > abort_risk_factor·ε 이고
        빼기 위험이 더 작으면 'return'."""
        self.mission = mission
        self._v_e = v_e
        self.hist.update(t, now)
        self._update_rear_belief(now, t)
        P, c = self.P, self.cfg
        rev = self.p_rear_post is not None
        rp = self._rp_revealed() if rev else self.rp
        P2 = copy.copy(P)
        P2.behavior_execution_delay_sec = 0.0          # 이미 머리를 넣은 상태: 지금부터 유지·빼기
        D = waypoint_blend_duration(v_e) * P.ego_reference_preview_duration_scale
        tf = t.target_front
        ctx = RolloutContext(v0=v_e, s0=t.ego_s_current, lane=t.current_lane_id, target_lane=t.target_lane_id,
                             ego_d_target=t.ego_d_target, pre_cap=mission, mission=mission,
                             tf_speed=tf.v_long if tf.valid else math.nan, D=D, yield_ref_speed=v_e,
                             preview_mode="blend", connector_join_arc=20.0)
        spec = ActionSpec(NUDGE, 0.0, 0.0, D, False, depth=self.nudge_depth, go=getattr(self, "nudge_go", False))
        br = self._branches(t, self.p_rear_post)
        self._leaf_t = t
        r = joint_rollout(ctx, [spec], t, self._vehicles(t), P2, self.A, rp, self.track,
                          [(b[0], b[1], "now") for b in br], self._accel_of)
        p = np.array([b[3] for b in br])
        if not r["allowed"][:, RETURN, 0].all():
            return None
        back = float((p[:, None] * r["risk"][:, RETURN, 0]).sum(axis=0).max())
        br2 = [(b[0], b[1], "now" if (b[2] == "rear" and rev) else b[2], b[3]) for b in br]
        r2 = joint_rollout(ctx, [spec], t, self._vehicles(t), P2, self.A, rp, self.track,
                           [b[:3] for b in br2], self._accel_of)
        _, _, _, risk2, _ = self._solve(r2, br2, np.zeros(1), np.ones(1, dtype=bool), P.risk_epsilon)
        stay = float(risk2[0])
        self.last_nudge_mon = (round(stay, 4), round(back, 4), self.p_rear_post)
        if stay > c.abort_risk_factor * P.risk_epsilon and back < stay:
            return "return"
        return None

    def monitor_core(self, t, v_e, mission, connector=None):
        """connector = 이 차선 변경의 실행 방식(C++ rxMonitorReturn 과 같다). 실행 중에는 승인 때 고정한 값,
        승인 전 일치 검사에서는 승인 평가의 값. 속도로 다시 추정하면 정지 출발 직후 blend 로 바뀌어 위험이 튄다."""
        P, c, rp = self.P, self.cfg, self.rp
        self._v_e = v_e
        d_t = abs(t.ego_d_target)
        if d_t <= c.abort_before_abs_d:
            return None
        T_rem = max(0.5, waypoint_blend_duration(v_e) * d_t / P.lane_width_m)
        near_end = connector if connector is not None else (t.distance_to_lane_end <= 30.0 and v_e <= 1.0)
        tf = t.target_front
        ctx = RolloutContext(v0=v_e, s0=t.ego_s_current, lane=t.current_lane_id, target_lane=t.target_lane_id,
                             ego_d_target=t.ego_d_target, pre_cap=mission, mission=mission,
                             tf_speed=tf.v_long if tf.valid else math.nan, D=T_rem, yield_ref_speed=v_e,
                             preview_mode="connector" if near_end else "blend", connector_join_arc=20.0)
        br = self._branches(t, self.p_rear_post)
        self._leaf_t = t
        mon_spec = ActionSpec(CHANGE_NOW, 0.0, 0.0, T_rem, True)
        # 모니터의 루트 하나 = 지금부터 계속, 대응 1 = 지금 복귀(모든 가지에서). tb_rule 'now'.
        r = joint_rollout(ctx, [mon_spec], t, self._vehicles(t), P, self.A, rp,
                          self.track, [(b[0], b[1], "now") for b in br], self._accel_of, mid_T_rem=T_rem)
        p = np.array([b[3] for b in br])
        go = (p[:, None] * r["risk"][:, CONTINUE, 0]).sum(axis=0).max()
        if not r["allowed"][:, RETURN, 0].all():
            return None
        back = (p[:, None] * r["risk"][:, RETURN, 0]).sum(axis=0).max()
        if getattr(self, "tail_fn", None) is not None:
            # 학습한 꼬리 위험(TR): 모니터도 지평선 뒤 위험을 본다(계속·지금 복귀 각각 Σ_b p_b TR_b 와 비교)
            from .rl.tail_risk import leaf_inputs
            Fx, idx = leaf_inputs(self, r, br, [0.0], [0.0], [0], r["risk"].shape[-1] - 1)
            tv = self.tail_fn(Fx)
            TT = np.zeros((len(br), 2))
            for (b, q, _), v in zip(idx, tv):
                TT[b, q] = v
            go = max(go, float((p * TT[:, CONTINUE]).sum()))
            back = max(back, float((p * TT[:, RETURN]).sum()))
        cont = go
        if c.monitor_tree:
            # 계속 쪽도 승인처럼 가지별 나중 대응(앞차 급감속은 알아챈 뒤, 뒤차 비양보는 관측했으면 지금)을 넣어 푼다
            br2 = [(b[0], b[1], "now" if (b[2] == "rear" and self.p_rear_post is not None) else b[2], b[3]) for b in br]
            r2 = joint_rollout(ctx, [mon_spec], t, self._vehicles(t), P, self.A, rp,
                               self.track, [b[:3] for b in br2], self._accel_of, mid_T_rem=T_rem)
            if getattr(self, "tail_fn", None) is not None:      # _solve 가 TR 을 쓰도록(루트 = 이 모니터 루트 하나)
                self._specs, self._roots_eval, self._mon_eval = [mon_spec], True, True
            _, _, _, risk2, _ = self._solve(r2, br2, np.zeros(1), np.ones(1, dtype=bool), P.risk_epsilon)
            self._roots_eval, self._mon_eval = False, False
            cont = float(risk2[0])
        if cont > c.abort_risk_factor * P.risk_epsilon and back < cont:
            return "return"
        return None


def make_reactive_planner(name):
    """run_batch.py 용 반응형 플래너 이름: 'rx' + '_' 로 이은 플래그.

    예측·모니터:
      'mt'   = 모니터가 계속 쪽에도 가지별 나중 대응을 넣음(기본)   'mo' = 단순 모니터(끝까지 계속 vs 지금 복귀)
      'mK'   = 모니터 복귀 문턱 Kε                                  'fNN' = 예외 확률 최솟값 0.NN
      'pf'   = 지나친 목표 차선 앞차를 뒤차로 본다                    'fl' = ACC merge_keep_rolling 하한
      'tfs'  = 예측 속 자차 ACC 가 실행 시작부터 목표 차선 앞차를 추종(시뮬 cfg acc_tf_on_start 와 짝)
      'rr'   = 예측 속 뒤차 양보 반응을 시뮬 NPC 수준으로(지연 0, 최대 감속 6, IDM b 2.5, s0 5)
      'adNN' = 앞차 없는 차의 가속도 감쇠 시간 NN/10 s(기본 1.5 s)
    지평선:
      'hNN'  = 예측 지평선 NN 스텝(0.2 s, 기본 40 = 8 s). 이 플래너의 P 사본에만 적용(롤아웃·위험·모니터 모두)
      'rfNN' = 지평선 맞춤 완화: 실행 시작 뒤 NN/10 s 만 지평선 안이면 후보로 받는다(지평선 끝 완료 검사 생략)
    머리 넣기:
      'ngNN' = 머리 넣고 기다리기 + 이어 가기(probe): 침범 NN/10 m 까지 들어가 기다리다가 뒤차 반응 시점에
               양보 가지는 차선 변경 계속, 비양보 가지는 복귀 또는 계속 기다림. 이어 가기는 목표 차선 간격이 열렸을 때만
      'ndNN' = 머리 넣고 기다리기만(이어 가기 없음)                   'nvNN' = 머리 넣기 후보 최대 속도 [m/s] (기본 3)
      'nm'   = 머리 넣은 동안 모니터: 뒤차 양보 관측을 갱신하고, 남기 > 2ε 이고 빼기가 더 안전하면 뺀다
      'nudly'= 머리 넣기에도 실행 지연을 둔다
    민감도: 'sgNN' = 차량 위치 σ 시간 인자 상한 NN/10 s, 'slNN' = 횡 σ 상한, 'prNN' / 'pbNN' = 뒤차 비양보 / 앞차 급감속
      확률 NN/100, 'single' = 가지 없는 비교군, 'common' = G6–G8 유지.
    RL 종단 비용: '<플래너>@<TR 가중치>' = 학습한 꼬리 위험(rl/tail_risk.py)을 판단·모니터의 지평선 끝에 더한다.
    예: rx_mt_pf_fl_tfs_rr_ng6_nv20_nm_h10_rf10@models/TR.pt"""
    from .tree import TreeConfig
    tr_path = None
    if "@" in name:
        name, tr_path = name.split("@", 1)
    flags = set(name.split("_"))
    cfg = TreeConfig(probability_table="gt", keep_gates=False, decision="tree", monitor=True)
    floor = next((int(f[1:]) / 100.0 for f in flags if f.startswith("f") and f[1:].isdigit()), None)
    factor = next((float(f[1:]) for f in flags if f.startswith("m") and f[1:].isdigit()), None)
    horizon = next((int(f[1:]) for f in flags if f.startswith("h") and f[1:].isdigit()), None)
    decay = next((int(f[2:]) / 10.0 for f in flags if f.startswith("ad") and f[2:].isdigit()), None)
    relax = next((int(f[2:]) / 10.0 for f in flags if f.startswith("rf") and f[2:].isdigit()), None)
    sig_cap = next((int(f[2:]) / 10.0 for f in flags if f.startswith("sg") and f[2:].isdigit()), None)
    lat_cap = next((int(f[2:]) / 10.0 for f in flags if f.startswith("sl") and f[2:].isdigit()), None)
    rear_npc = "rr" in flags
    nudge_mon = "nm" in flags
    nudge = next((int(f[2:]) / 10.0 for f in flags if f.startswith("nd") and f[2:].isdigit()), None)
    probe = next((int(f[2:]) / 10.0 for f in flags if f.startswith("ng") and f[2:].isdigit()), None)
    nudge_vmax = next((float(f[2:]) for f in flags if f.startswith("nv") and f[2:].isdigit()), 3.0)
    p_rear = next((int(f[2:]) / 100.0 for f in flags if f.startswith("pr") and f[2:].isdigit()), None)
    p_brake = next((int(f[2:]) / 100.0 for f in flags if f.startswith("pb") and f[2:].isdigit()), None)
    if factor is not None:
        cfg.abort_risk_factor = factor
    if "mt" in flags:
        cfg.monitor_tree = True
    if "mo" in flags:
        cfg.monitor_tree = False

    def factory(P, track):
        if horizon is not None or relax is not None:
            P = copy.copy(P)
            if horizon is not None:
                P.horizon_steps = horizon
            if relax is not None:
                P.fit_relax_after_start_s = relax
        rp = ReactiveParams()
        if floor is not None:
            rp.p_exc_floor = floor
        if decay is not None:
            rp.accel_decay_s = decay
        rp.passed_front_as_rear = "pf" in flags
        rp.sigma_cap_s, rp.p_rear_floor, rp.p_brake_floor = sig_cap, p_rear, p_brake
        rp.tf_on_start = "tfs" in flags
        rp.lat_sigma_cap_s = lat_cap
        rp.nudge_exec_delay = (nudge is not None and probe is None) or "nudly" in flags
        if rear_npc:
            rp.rear_react_delay_s, rp.rear_max_brake, rp.rear_idm_b, rp.rear_idm_s0 = 0.0, 6.0, 2.5, 5.0
        rp.merge_floor = "fl" in flags
        tp = ReactiveTreePlanner(P, track, cfg=TreeConfig(**cfg.__dict__), rp=rp, single="single" in flags,
                                 drop_common="common" not in flags)
        tp.name = name
        if tr_path:
            from .rl.tail_risk import load_tail_fn
            tp.tail_fn = load_tail_fn(tr_path)
        tp.nudge_depth = probe if probe is not None else nudge
        tp.nudge_go = probe is not None
        tp.nudge_max_speed = nudge_vmax
        tp.nudge_monitor = nudge_mon
        return tp
    return factory
