#!/usr/bin/env python3
"""RL 잎 값 가중치(.pt) → C++ 판단 노드(smpc_decision_node rl/rv_weights, rl/vt_weights)용 텍스트.

    python3 scripts/export_rl_weights.py results/rv/RVT_0.pt models/RVT_0.txt
    python3 scripts/export_rl_weights.py results/vt/VT_iter1.pt models/VT_iter1.txt

머리 한 줄: M in h out link scale leaf lane_target
  link = sigmoid(위험 값: 앙상블 평균 확률) / identity(남은 시간 값: scale·평균, scale = terminal_cost.SCALE)
그다음 앙상블마다 W1[h×in] b1[h] W2[h×h] b2[h] W3[out×h] b3[out] (행 우선, 공백 구분).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from smpc_sim.rl.terminal_cost import SCALE  # noqa: E402


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location="cpu", weights_only=True)
    meta = ck.get("meta", {})
    ens = ck["ensemble"]
    sd = ens[0]
    h, d_in = sd["net.0.weight"].shape
    out = sd["net.4.weight"].shape[0]
    is_rv = "leaf" in meta or meta.get("target") in ("risk", "trace") or "val_auc_gt_eps" in meta
    if meta.get("target") == "abort":
        raise SystemExit("중단 값(RA)은 C++ 에 없음")
    link, scale = ("sigmoid", 1.0) if is_rv else ("identity", SCALE)
    leaf = int(bool(meta.get("leaf", False))) if is_rv else 0
    lane = int(meta.get("target") == "lane") if not is_rv else 0
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        f.write(f"{len(ens)} {d_in} {h} {out} {link} {scale} {leaf} {lane}\n")
        for sd in ens:
            for k in ("net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias"):
                f.write(" ".join(f"{x:.9g}" for x in sd[k].reshape(-1).tolist()) + "\n")
    print(f"{src} -> {dst}: M={len(ens)} in={d_in} h={h} out={out} link={link} scale={scale} leaf={leaf} lane={lane}")


if __name__ == "__main__":
    main()
