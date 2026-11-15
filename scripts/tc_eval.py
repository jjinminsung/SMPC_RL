#!/usr/bin/env python3
"""TC 플래너를 검증 시드(6000–6299)에서 2 s·5 s 와 같은 시드로 비교한다.

    python3 scripts/tc_eval.py --tag v2 rx_..._h10_rf10_TCb@results/tc/TC_iter0v.pt [다른 플래너 ...]
"""
import argparse
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
from rc_iterate import summarize  # noqa: E402
from tc_iterate import REF, TC, VAL0, VALN, collect, log, ref, BASE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("planners", nargs="+")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed0", type=int, default=VAL0)
    ap.add_argument("--n", type=int, default=VALN)
    a = ap.parse_args()
    v2 = ref("val_2s", BASE, VAL0, VALN) if a.seed0 == VAL0 else None
    v5 = ref("val_5s", BASE.replace("_h10_", "_h25_"), VAL0, VALN) if a.seed0 == VAL0 else None
    for j, pl in enumerate(a.planners):
        pl = pl.replace("@results/", f"@{R}/results/")
        ev = collect(pl, a.seed0, a.n, str(TC / "eval" / a.tag / f"p{j}"), [], chunk=a.n)
        out = summarize(ev, v2)
        if v5 is not None:
            out["dJ_vs_5s"] = summarize(ev, v5)["dJ_vs_ref"]
        eps = list(ev.values())
        out["tc_keeps_per_ep"] = sum(e.get("tc_keeps", 0) for e in eps) / len(eps)
        out["tc_declines_per_ep"] = sum(e.get("tc_declines", 0) for e in eps) / len(eps)
        log(f"eval {a.tag} p{j} {pl.split('/')[-1] if '@' in pl else pl}:", json.dumps(out))


if __name__ == "__main__":
    main()
