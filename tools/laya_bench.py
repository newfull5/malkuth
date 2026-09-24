#!/usr/bin/env python3
"""Score Laya checkpoints on Kev-format suites with kev's own metrics (kev.benchmark.prediction_rows / summarize),
so Laya and Kev reports are directly comparable. Also reports Laya's Router: per record it takes the checkpoint the
router would pick (english / multilingual) from those two runs, which is exactly what Router.predict returns.

    cd kev && uv pip install --no-deps -e ../laya
    cd kev && uv run python ../tools/laya_bench.py --out runs/lite-laya ../benchmarks/lite/*/

Output: <out>/<checkpoint>/<suite>/{report.json, rows.json}, <out>/summary.json.
Questions Laya cannot encode (options exceed its head_max_len) are counted as rejected, as kev.benchmark does.
"""
import argparse, json, time
from pathlib import Path

import laya
import laya.agent
from kev.api import question_keys
from kev.benchmark import prediction_rows, summarize
from kev.suite import load_split, read_manifest, write_json

laya.agent.round = lambda x, ndigits=None: x   # system_one rounds probabilities to 4 places; score the unrounded ones
CHECKPOINTS = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}


def predict(agent, record):
    qs = {qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")} for qid, q in record["questions"].items()}
    ans = agent.system_one(record["state"], qs)["answers"]
    out = {}
    for qid, q in qs.items():
        keys, a = question_keys(q["type"], q.get("criteria")), ans[qid]
        if q["type"] == "noul":
            out[qid] = dict(zip(keys, [1 - a["noul"], a["noul"]]))   # kev noul keys are [false, true]
        else:
            out[qid] = dict(zip(keys, a["probabilities"].values()))
    return {"probabilities": out}


def score(records, preds, manifest, out, seconds):
    rows = [row for i, r in enumerate(records) if i in preds for row in prediction_rows(r, preds[i])]
    report = summarize(rows, 1.0, manifest["holdout_sources"])
    report.update(coverage={"requested_records": len(records), "evaluated_records": len(preds),
                            "rejected_records": len(records) - len(preds),
                            "requested_questions": sum(len(r["questions"]) for r in records), "evaluated_questions": len(rows)},
                  seconds=seconds)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "rows.json", rows); write_json(out / "report.json", report)
    c = report["clean"]
    return {"acc": round(c["acc"], 4), "brier": round(c["brier"], 4), "ece": round(c["ece"], 4), "n": c["n"],
            "rejected": report["coverage"]["rejected_records"], "seconds": round(seconds, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suites", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.out)
    suites = [(Path(s), load_split(s, "development"), read_manifest(s)) for s in a.suites]
    router = laya.Router(max_loaded=1)
    route = {s.name: [router.route(r["state"], r["questions"])["model"] for r in recs] for s, recs, _ in suites}
    all_preds, summary = {}, {}
    for name, sub in CHECKPOINTS.items():
        agent = laya.load("convaiinnovations/laya", device=a.device, subfolder=sub)
        for s, recs, man in suites:
            preds, t = {}, time.time()
            for i, r in enumerate(recs):
                try: preds[i] = predict(agent, r)
                except ValueError: pass   # options exceed head_max_len
            all_preds[(name, s.name)] = preds
            summary.setdefault(name, {})[s.name] = score(recs, preds, man, out / name / s.name, time.time() - t)
            print(f"{name:16s} {s.name:22s} {json.dumps(summary[name][s.name])}", flush=True)
        del agent
    for s, recs, man in suites:   # Router: the checkpoint it routes each record to
        preds = {i: all_preds[(route[s.name][i], s.name)][i] for i in range(len(recs)) if i in all_preds[(route[s.name][i], s.name)]}
        summary.setdefault("router", {})[s.name] = {**score(recs, preds, man, out / "router" / s.name, 0.0),
                                                     "routed_multilingual": sum(m == "multilingual" for m in route[s.name])}
        print(f"{'router':16s} {s.name:22s} {json.dumps(summary['router'][s.name])}", flush=True)
    write_json(out / "summary.json", summary)


if __name__ == "__main__":
    main()
