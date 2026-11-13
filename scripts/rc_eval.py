#!/usr/bin/env python3
"""RC 정책 시험: 같은 시드에서 2 s + RC(+rcf), 2 s, 5 s 를 짝 비교한다.

    python3 scripts/rc_eval.py --model models/RC.pt --theta 0.5 --seed0 3000 --n 1000

결과: results/rc/eval/<이름>_s*.json(에피소드), results/rc/eval/summary_s<seed0>.json(요약·짝 비교 J 차와 95 % 구간).
J = 합류 시각(실패면 120 s) + 60·충돌 + 2·중단.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rc_iterate as ri  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ri.R / "models" / "RC.pt"))
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--no_rcf", action="store_true", help="RC 가 막은 틱의 차선 끝 대체 정지(rcf)를 끈다")
    ap.add_argument("--seed0", type=int, default=3000)
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    out = ri.RC / "eval"
    out.mkdir(parents=True, exist_ok=True)
    th = int(round(a.theta * 100))
    model = str(Path(a.model).resolve())
    arms = {"RC": f"{ri.BASE}{'' if a.no_rcf else '_rcf'}_RC{th}@{model}",
            "2s": ri.BASE,
            "5s": ri.BASE.replace("_h10_", "_h25_")}
    eps = {}
    for name, pol in arms.items():
        tag = f"RC{th}{'' if a.no_rcf else '_rcf'}" if name == "RC" else name
        pre = out / tag
        ri.collect(pol, a.seed0, a.n, str(pre), [])
        eps[name] = ri.episodes_of(sorted(out.glob(f"{tag}_s*.json")))
        eps[name] = {k: v for k, v in eps[name].items() if a.seed0 <= k < a.seed0 + a.n}
    res = {"arms": arms, **{k: ri.summarize(v) for k, v in eps.items()},
           "RC_vs_2s": ri.summarize(eps["RC"], eps["2s"])["dJ_vs_ref"],
           "RC_vs_5s": ri.summarize(eps["RC"], eps["5s"])["dJ_vs_ref"]}
    json.dump(res, open(out / f"summary_s{a.seed0}.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
