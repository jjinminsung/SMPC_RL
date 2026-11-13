"""위험 가치망 RC(risk critic): 짧은 지평선 트리가 보지 못하는 '지금 차선 변경을 시작하면 그 뒤에 생길 위험'을
실제로 겪은 결과에서 배운다(모델 없는 강화학습의 비용 가치, constrained MDP 의 cost critic).

무엇을 배우나
    상태 s 에서 '지금 바꾸기(CHANGE_NOW)'를 고른 뒤 지금 정책 π(실행 중 모니터 포함)가 그 차선 변경을 끝까지 몰고 갔을 때
    나쁜 결과(중단·충돌·근접)가 생길 확률
        Q_c(s, 바꾸기) = E_π[ c | s, 바꾸기 ],   c = 1 (중단 또는 충돌 또는 TTC < 1 s), 아니면 0.
    차선 변경 하나를 한 에피소드(옵션)로 보고, 그 결과를 몬테카를로로 센다(할인 없음). 리턴이 0/1 이라 회귀 = 이진 분류이고
    손실은 교차 엔트로피(BCE)다.

TR(tail_risk.py)과 다른 점
    TR 은 5 s 트리가 '예측 모델로 계산한' 위험을 흉내 낸다(모델 기반 증류). RC 는 시뮬레이터 NPC 가 실제로 어떻게 움직였는지
    (예측 모델과 다른, MORAI 측정 분포)에서 나온 결과로 배운다. 그래서 예측 모델이 틀리는 곳도 배울 수 있다.

판단에 쓰는 법(정책 개선)
    2 s 트리가 지평선 안 기회 제약(Σ p r ≤ ε)으로 '지금 바꾸기'를 허용해도 RC(s) > θ 이면 막는다(준비 후보·기다리기로 간다).
    θ 는 검증 시드의 J 로 고른다. 새 정책으로 다시 모으고 다시 배우는 것을 반복한다(근사 정책 반복).
    막힌 상태의 추정이 틀려도 고쳐지도록, 수집 때는 막힌 바꾸기를 확률 ε_rc 로 그대로 실행해 본다(탐색).

입력 특징 [DIM]
    features2 의 상태 특징(자차 속도·횡 진행도·차선·차선 끝 거리/시간, 세 차선 앞·뒤차 간격·상대 속도)
    + 지평선 안 최대 위험 log10(peak) + 정지 connector 여부.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features2 import DIM as FDIM, frame_features, target_set_vehicles

RISK_FLOOR = 1e-4
DIM = FDIM + 2


def rc_features(t, track, v_e, peak, connector):
    """판단 노드가 가진 정보(TargetSet)만으로 만든 특징 [DIM]. 학습·판단이 같은 함수를 쓴다."""
    cur = int(t.current_lane_id)
    veh = [(v.s, v.lane_id, max(0.0, v.v_long), v.length) for v in target_set_vehicles(t)]
    f = frame_features(cur, int(t.target_lane_id), t.ego_s_current, track.center_y(cur) + t.ego_d_current, v_e, veh)
    extra = np.array([np.log10(max(float(peak), 0.0) + RISK_FLOOR) / 4.0, float(connector)], dtype=np.float32)
    return np.concatenate([f, extra])


class RCNet(nn.Module):
    """작은 MLP. 출력 = 나쁜 결과의 로짓(sigmoid 하면 확률)."""

    def __init__(self, dim=DIM, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def save_rc(path, nets, meta=None):
    torch.save({"ensemble": [n.state_dict() for n in nets], "meta": meta or {}}, path)


def load_rc_fn(path):
    """입력 [M, DIM] → 나쁜 결과 확률 [M]. 앙상블(부트스트랩) 평균 + κ·표준편차(meta 'kappa', 기본 0):
    데이터가 적은 상태에서 낙관하지 않도록 κ 를 줄 수 있다."""
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    nets = []
    for sd in ck["ensemble"]:
        n = RCNet(dim=sd["net.0.weight"].shape[1], hidden=sd["net.0.weight"].shape[0])
        n.load_state_dict(sd)
        n.eval()
        nets.append(n)
    torch.set_num_threads(1)
    kappa = float(ck.get("meta", {}).get("kappa", 0.0))

    def fn(x):
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            p = torch.sigmoid(torch.stack([n(xt) for n in nets])).numpy()
        return p.mean(axis=0) + kappa * p.std(axis=0)
    fn.meta = ck.get("meta", {})
    return fn
