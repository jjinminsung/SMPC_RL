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

ABORT_COST = 2.0      # J 의 중단 가중(rl/terminal_cost.py 와 같다)


def _root0(mask):
    """루트 0 만 남긴 마스크(잎 학습판 RV·RA 는 지금 바꾸기 루트에만 쓴다)."""
    m = np.zeros_like(mask)
    m[0] = mask[0]
    return m


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
        raw_risk = np.ones(N)                              # 꼬리 항(TR 등)을 넣기 전 지평선 안 위험(위험 기록용)
        TV = None                                          # 학습한 종단 비용(TC) [B, 2, N]: τ + V(잎)
        if getattr(self, "tc_fn", None) is not None and getattr(self, "_roots_eval", False) \
                and not getattr(self, "_mon_eval", False):
            from .rl.terminal_cost import leaf_costs
            TV = leaf_costs(self, r, mask, self.tc_fn)
            self._tc_root = np.full(N, np.inf)             # TC 루트 값(기회 제약을 지키는 대응 중 최소). 모니터의 _solve 는 건드리지 않는다
            self._tc_combo, self._tc_p = {}, p             # 루트마다 그 최소를 준 가지별 대응(수집 기록용)
        cost_j = np.array([0.0, ABORT_COST])               # TC: 대응 비용은 J 의 중단 가중(복귀 = 2 s)
        RV = None                                          # RL 위험 값(RV) [B, 2, N]: 잎에서 이 차선 변경을 마칠 때까지의 최대 위험
        if getattr(self, "rv_fn", None) is not None and getattr(self, "_roots_eval", False) \
                and (not getattr(self, "_mon_eval", False) or getattr(self, "rv_monitor", False)):
            from .rl.risk_value import leaf_risk
            RV = leaf_risk(self, r, _root0(mask) if getattr(self.rv_fn, "leaf", False) else mask, self.rv_fn)
            # 잎 학습판: 명령(지금 바꾸기) 루트 0 의 잎만 계산(나머지 루트는 0)
        AV = None                                          # RL 중단 값(RA) [B, 2, N]: 잎에서 이 차선 변경을 마칠 때까지의 중단 기댓값
        if getattr(self, "ra_fn", None) is not None and getattr(self, "_roots_eval", False) \
                and not getattr(self, "_mon_eval", False):
            from .rl.risk_value import leaf_risk
            AV = leaf_risk(self, r, _root0(mask) if getattr(self.ra_fn, "leaf", False) else mask, self.ra_fn)
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
            peak_raw = peak
            if TT is not None:                             # 꼬리 최댓값의 위 한계 Σ_b p_b TR_b 와 지평선 안 최댓값 중 큰 값
                peak = np.maximum(peak, sum(p[b] * TT[b, combos[:, b], i] for b in range(B)))
            peak_cost = peak
            if RV is not None:                             # 지평선 뒤 최대 위험(RL): 전체 최댓값 = max(지평선 안, 뒤)
                tail = sum(p[b] * RV[b, combos[:, b], i] for b in range(B))
                if tail.ndim == 2:                         # 시점별(RVT): max_j Σ_b p_b V_bj = SMPC 와 같은 순서
                    tail = tail.max(axis=1)
                both = np.maximum(peak, tail)
                if not getattr(self, "rv_constraint_only", False):  # RVg = 기회 제약에만(비용은 지평선 안 그대로)
                    peak_cost = both
                if not getattr(self, "rv_cost_only", False):   # RV = 종단 제약(기회 제약에도), RVc = 종단 비용(비용에만)
                    peak = both
            total = root_cost[i] + jc + c.risk_weight * peak_cost
            if AV is not None:                             # 지평선 뒤 중단도 지평선 안 복귀와 같은 값(30)으로 매긴다
                total = total + cost_r[1] * sum(p[b] * AV[b, combos[:, b], i] for b in range(B))
            good = ok & (peak <= eps)
            if TV is not None:                             # TC: 루트 값 = Σ_b p_b [대응 비용 + τ + V(잎)]
                tc_total = sum(p[b] * (cost_j[combos[:, b]] + TV[b, combos[:, b], i]) for b in range(B))
                if good.any():
                    kt = int(np.argmin(np.where(good, tc_total, np.inf)))
                    self._tc_root[i], self._tc_combo[i] = tc_total[kt], combos[kt]
                if getattr(self, "tc_add", False):         # VT: 휴리스틱 루트 비용 + 지평선 뒤 남은 시간(모든 루트에 같은 함수)
                    total = total + tc_total
                elif not getattr(self, "tc_binary", False):  # 전체 모드: 모든 루트를 TC 값으로 고른다(휴리스틱 루트 비용 없음)
                    total = tc_total
            if good.any():
                k = int(np.argmin(np.where(good, total, np.inf)))
                feas[i], best_cost[i], best_risk[i] = True, total[k], peak[k]
                raw_risk[i] = peak_raw[k]
            elif ok.any():
                k = int(np.argmin(np.where(ok, peak, np.inf)))
                best_risk[i] = peak[k]
                raw_risk[i] = peak_raw[k]
            # TreePlanner 형식: [0] 은 nominal 자리, 이후 가지별 대응(복귀 = TREE_RETURN)
            best_combo[i, 1:] = np.where(combos[k] == RETURN, TREE_RETURN, 0) if (good.any() or ok.any()) else 0
        self.last_tree = {"branches": [(f"{b[0]}/{b[1]}", round(b[3], 3), b[2]) for b in br]}
        self._raw_risk = raw_risk
        if getattr(self, "leaf_log", None) is not None and getattr(self, "_roots_eval", False) \
                and not getattr(self, "_mon_eval", False) and mask[0] and feas[0]:
            from .rl.risk_value import leaf_features
            m0 = np.zeros(N, dtype=bool)
            m0[0] = True
            F, _, arr = leaf_features(self, r, m0)[0]
            q = (best_combo[0, 1:] == TREE_RETURN).astype(int)
            self._leaf0 = (F, arr, p, q)                 # world 가 지금 바꾸기를 명령한 틱에 기록한다
        return feas, best_cost, best_combo, best_risk, br

    # ------------------------------------------------------------ 모니터
    def monitor(self, now, t, v_e, mission):
        self.mission = mission
        self.hist.update(t, now)
        self._update_rear_belief(now, t)
        self._mon_raw = None
        mh = getattr(self, "mon_horizon", None)          # 'MhNN': 모니터만 지평선 NN 스텝(분해 실험)
        if mh:
            h0, self.P.horizon_steps = self.P.horizon_steps, mh
        try:
            out = self.monitor_core(t, v_e, mission, connector=self.lc_connector)
        finally:
            if mh:
                self.P.horizon_steps = h0
        if getattr(self, "risk_log", None) is not None and self._mon_raw is not None:
            self.risk_log.append((now, self._mon_raw, "mon"))     # 실행 중: 계속했을 때의 지평선 안 위험
        return out

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
        mh = getattr(self, "mon_horizon", None)
        if not mh:
            return self._nudge_monitor_tick(now, t, v_e, mission)
        h0, self.P.horizon_steps = self.P.horizon_steps, mh
        try:
            return self._nudge_monitor_tick(now, t, v_e, mission)
        finally:
            self.P.horizon_steps = h0

    def _rv_back(self, r, p):
        """RVm: 지금 복귀(빼기) 쪽의 지평선 뒤 위험 Σ_b p_b V_b(복귀 잎). RV 를 모니터에 안 쓰면 0."""
        if getattr(self, "rv_fn", None) is None or not getattr(self, "rv_monitor", False):
            return 0.0
        from .rl.risk_value import leaf_risk, tail_peak
        return tail_peak(p, leaf_risk(self, r, np.ones(1, dtype=bool), self.rv_fn)[:, RETURN, 0])

    def _nudge_monitor_tick(self, now, t, v_e, mission):
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
        back = max(back, self._rv_back(r, p))
        br2 = [(b[0], b[1], "now" if (b[2] == "rear" and rev) else b[2], b[3]) for b in br]
        r2 = joint_rollout(ctx, [spec], t, self._vehicles(t), P2, self.A, rp, self.track,
                           [b[:3] for b in br2], self._accel_of)
        if getattr(self, "rv_monitor", False):           # _solve 가 RV 를 쓰도록(루트 = 남기 하나)
            self._specs, self._roots_eval, self._mon_eval = [spec], True, True
        _, _, _, risk2, _ = self._solve(r2, br2, np.zeros(1), np.ones(1, dtype=bool), P.risk_epsilon)
        self._roots_eval, self._mon_eval = False, False
        stay = float(risk2[0])
        if getattr(self, "risk_log", None) is not None:
            self.risk_log.append((now, float(self._raw_risk[0]), "nudge"))
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
        back = max(back, self._rv_back(r, p))
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
        if getattr(self, "rv_monitor", False) and getattr(self, "rv_fn", None) is not None and not c.monitor_tree:
            from .rl.risk_value import leaf_risk, tail_peak
            go = max(go, tail_peak(p, leaf_risk(self, r, np.ones(1, dtype=bool), self.rv_fn)[:, CONTINUE, 0]))
        cont = go
        self._mon_raw = float((p[:, None] * r["risk"][:, CONTINUE, 0]).sum(axis=0).max())
        if c.monitor_tree:
            # 계속 쪽도 승인처럼 가지별 나중 대응(앞차 급감속은 알아챈 뒤, 뒤차 비양보는 관측했으면 지금)을 넣어 푼다
            br2 = [(b[0], b[1], "now" if (b[2] == "rear" and self.p_rear_post is not None) else b[2], b[3]) for b in br]
            r2 = joint_rollout(ctx, [mon_spec], t, self._vehicles(t), P, self.A, rp,
                               self.track, [b[:3] for b in br2], self._accel_of, mid_T_rem=T_rem)
            if getattr(self, "tail_fn", None) is not None or getattr(self, "rv_monitor", False):
                self._specs, self._roots_eval, self._mon_eval = [mon_spec], True, True   # _solve 가 TR·RV 를 쓰도록
            _, _, _, risk2, _ = self._solve(r2, br2, np.zeros(1), np.ones(1, dtype=bool), P.risk_epsilon)
            self._roots_eval, self._mon_eval = False, False
            cont = float(risk2[0])
            self._mon_raw = float(self._raw_risk[0])
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
    예: rx_mt_pf_fl_tfs_rr_ng6_nv20_nm_h10_rf10@models/TR.pt
    RL 모니터: '<플래너>_RCNN@<RC 가중치>' = RC(rl/risk_critic.py) > NN/100 이면 '지금 바꾸기'를 막는다.
      'rcf' = RC 가 막은 틱에는 차선 끝 전에 설 수 있는 준비 후보만 남긴다.
    예: rx_mt_pf_fl_tfs_rr_ng6_nv20_nm_h10_rf10_rcf_RC50@models/RC.pt
    학습한 종단 비용: '<플래너>_TC@<TC 가중치>' = 루트 값 = Σ_b p_b [대응 비용 + τ + V(잎)](rl/terminal_cost.py),
      기다리기 루트 추가, 문턱 없음.
      '_TCb@' = 지금 바꾸기와 기다리기만 TC 값으로 비교한다. 바꾸기가 비싸면 지금 바꾸지 않고, 준비 후보는 기존 SMPC 비용으로 고른다.
      '_TCr@' = TC 값을 계산만 하고 판단은 기존 플래너 그대로(수집·관측용).
    분해 실험: '_G5' = 5 s 지평선판이 같은 틱에 지금 바꾸기를 고를 때만 지금 바꾼다.
      '_MhNN' = 실행 중 모니터(차선 변경·머리 넣기)만 지평선 NN 스텝, 판단은 hNN 그대로.
    RL 위험 값: '_RV@<RV 가중치>' = 판단의 기회 제약·비용에 지평선 뒤 최대 위험 Σ_b p_b V_b(rl/risk_value.py),
      '_RA@<RA 가중치>' = 루트 비용에 Σ_b p_b · 30 · A(잎_b)(지평선 뒤 중단 기댓값, 30 = 기존 복귀 대응 비용),
      둘 다: '_RV_RA@<RV>+<RA>'. '_RVc@' = 비용에만, '_RVg@' = 기회 제약에만(잎 학습판은 루트 0 에만 붙어 비용에 넣으면 기다리기 쪽으로 기운다), 'RVm' 을 더하면 실행 중 모니터의 계속·복귀 위험에도 같은 값을 쓴다(문턱은 기존 2ε).
    남은 시간 값: '_VT@<V_T 가중치>' = 휴리스틱 루트 비용에 Σ_b p_b [대응 비용 + τ + V_T(잎)] 를 모든 루트에 같은 함수로 더한다
      (V_T = 잎에서 이 차선 변경을 마칠 때까지의 J [s], rl/terminal_cost.py). 기다리기 루트는 기록용(선택 후보 아님),
      준비 후보는 차선 끝 전에 설 수 있는 것만. '_VTr@' = 휴리스틱의 지연 항(1.0·t_lc)을 빼고 V_T 가 대신한다.
      '_VTp@' = 준비 후보(t_lc 0.4–2.0 s)를 지평선 맞춤 없이 모두 둔다(실행 시작이 지평선 뒤면 지평선 안은 원 차선 준비, 잎을 V_T 가 평가).
      위험 값과 함께: '_RVg_VT@<RV>+<V_T>' (마지막 경로가 V_T). 수집: '_RVg_TCr@<RV>+<V_T>' = 판단은 RVg 그대로, 잎만 기록."""
    from .tree import TreeConfig
    tr_path, rc_path, rc_th, tc_path, rv_path = None, None, None, None, None
    if "@" in name:
        name, tr_path = name.split("@", 1)
        nf = set(name.split("_"))
        rc_th = next((int(f[2:]) / 100.0 for f in name.split("_") if f.startswith("RC") and f[2:].isdigit()), None)
        if rc_th is not None:
            rc_path, tr_path = tr_path, None
        elif {"TC", "TCb", "TCr", "VT", "VTr", "VTp"} & nf:
            if {"RV", "RVc", "RVg", "RA"} & nf:
                rv_path, tc_path = tr_path.rsplit("+", 1)
            else:
                tc_path = tr_path
            tr_path = None
        elif {"RV", "RVc", "RVg", "RA"} & nf:
            rv_path, tr_path = tr_path, None
    flags = set(name.split("_"))
    cfg = TreeConfig(probability_table="gt", keep_gates=False, decision="tree", monitor=True)
    floor = next((int(f[1:]) / 100.0 for f in flags if f.startswith("f") and f[1:].isdigit()), None)
    factor = next((float(f[1:]) for f in flags if f.startswith("m") and f[1:].isdigit()), None)
    horizon = next((int(f[1:]) for f in flags if f.startswith("h") and f[1:].isdigit()), None)
    mon_h = next((int(f[2:]) for f in flags if f.startswith("Mh") and f[2:].isdigit()), None)
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
        if horizon is not None or relax is not None or mon_h is not None or "VTp" in flags:
            P = copy.copy(P)
            P.prep_beyond_horizon = "VTp" in flags
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
        tp.mon_horizon = mon_h
        if tr_path:
            from .rl.tail_risk import load_tail_fn
            tp.tail_fn = load_tail_fn(tr_path)
        if rv_path:
            from .rl.risk_value import load_rv_fn
            paths = rv_path.split("+")                 # '_RV_RA@<RV>+<RA>', '_RA@<RA>'
            if {"RV", "RVc", "RVg"} & flags:
                tp.rv_fn = load_rv_fn(paths.pop(0))
            if "RA" in flags:
                tp.ra_fn = load_rv_fn(paths.pop(0))
            tp.rv_cost_only = "RVc" in flags
            tp.rv_constraint_only = "RVg" in flags
            tp.rv_monitor = "RVm" in flags
        if tc_path:
            from .rl.terminal_cost import load_tc_fn
            tp.tc_fn = load_tc_fn(tc_path)
            tp.tc_binary = bool({"TCb", "TCr"} & flags)
            tp.tc_observe = "TCr" in flags
            tp.tc_add = bool({"VT", "VTr", "VTp"} & flags)
            tp.tc_nodelay = "VTr" in flags
            tp.prep_beyond = "VTp" in flags
        if rc_path:
            from .rl.risk_critic import load_rc_fn
            tp.rc_fn, tp.rc_th = load_rc_fn(rc_path), rc_th
            tp.rc_fallback = "rcf" in flags
        tp.nudge_depth = probe if probe is not None else nudge
        tp.nudge_go = probe is not None
        tp.nudge_max_speed = nudge_vmax
        tp.nudge_monitor = nudge_mon
        if "G5" in flags:
            # 분해 실험: 같은 플래너의 5 s 지평선판이 같은 틱에 지금 바꾸기를 고를 때만 지금 바꾼다(학습 없음, 비교 기준용)
            g = "_".join(f for f in name.split("_") if f not in ("G5", "TC", "TCb", "TCr")).replace("_h10_", "_h25_")
            tp.gate = make_reactive_planner(g)(P, track)
        return tp
    return factory
