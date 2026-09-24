#!/usr/bin/env python3
"""Probability-average ensemble of saved runs, scored with kev's own metrics.

    cd kev && uv run python ../tools/ensemble_rows.py --out ../results/mid/ens-2b-4b ../results/mid/d2b-mix4 ../results/mid/k4b-mix3 [--weights 0.4,0.6]

Each input is a kev_batch_bench output directory (<run>/<suite>/rows.json). Rows are matched by (id, question); the
ensemble probability is the weighted mean of the members' probabilities. Writes <out>/<suite>/report.json + summary.json.
"""
import argparse, json
from pathlib import Path

import numpy as np
from kev.benchmark import summarize
from kev.suite import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="")
    a = ap.parse_args()
    runs = [Path(r) for r in a.runs]
    w = np.array([float(x) for x in a.weights.split(",")] if a.weights else [1.0] * len(runs)); w = w / w.sum()
    suites = sorted(set.intersection(*({p.parent.name for p in r.glob("*/rows.json")} for r in runs)))
    summary = {}
    for s in suites:
        members = [{(x["id"], x["question"]): x for x in json.loads((r / s / "rows.json").read_text())} for r in runs]
        keys = set.intersection(*(set(m) for m in members))   # rows every member answered (rejections differ by model)
        rows = []
        for k in sorted(keys):
            base = dict(members[0][k]); base.pop("logits", None); base.pop("inference_temperature", None)
            base["p"] = (sum(wi * np.array(m[k]["p"]) for wi, m in zip(w, members))).tolist()
            rows.append(base)
        report = summarize(rows)
        (Path(a.out) / s).mkdir(parents=True, exist_ok=True)
        write_json(Path(a.out) / s / "report.json", report); write_json(Path(a.out) / s / "rows.json", rows)
        c = report["clean"]
        summary[s] = {"acc": round(c["acc"], 4), "brier": round(c["brier"], 4), "ece": round(c["ece"], 4), "n": c["n"]}
    summary["_members"] = {"runs": [str(r) for r in runs], "weights": w.tolist()}
    write_json(Path(a.out) / "summary.json", summary)
    v = [x for k, x in summary.items() if not k.startswith("_")]
    print(f"ensemble {len(v)} suites: acc {np.mean([x['acc'] for x in v]):.4f} brier {np.mean([x['brier'] for x in v]):.4f}")


if __name__ == "__main__":
    main()
