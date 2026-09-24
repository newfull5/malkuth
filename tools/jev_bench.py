#!/usr/bin/env python3
"""Score TypeSafe Jev (api.typesafe.ai) on Kev-format suites with kev's metrics, so its report sits next to Kev / Laya.

    export TYPESAFE_API_KEY=...        # never pass the key on the command line or write it to a file
    cd kev && uv run python ../tools/jev_bench.py --out ../results/lite/jev ../benchmarks/lite/*/

Output: <out>/<suite>/{report.json, rows.json}, <out>/summary.json (includes token usage and list-price cost).
Requests run concurrently (--workers); failures after retries count as rejected records.
"""
import argparse, json, os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kev.benchmark import prediction_rows, summarize
from kev.predictors import PRICE_PER_MILLION, RemotePredictor
from kev.suite import load_split, read_manifest, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suites", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base-url", default="https://api.typesafe.ai")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--max-usd", type=float, default=2.0, help="stop before list-price input cost would exceed this")
    a = ap.parse_args()
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key: ap.error("set TYPESAFE_API_KEY")
    jev = RemotePredictor(a.base_url, a.model, key, timeout=120, retries=4)
    out, summary, tokens = Path(a.out), {}, 0
    for s in map(Path, a.suites):
        records, manifest = load_split(s, "development"), read_manifest(s)
        if tokens * PRICE_PER_MILLION / 1e6 > a.max_usd:
            print(f"cost cap reached before {s.name}", flush=True); break

        def call(i):
            try: return i, jev(records[i])
            except Exception as e: return i, e
        t = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            results = dict(ex.map(call, range(len(records))))
        preds = {i: p for i, p in results.items() if not isinstance(p, Exception)}
        errors = [str(p)[:200] for p in results.values() if isinstance(p, Exception)]
        used = sum(p.get("input_tokens") or 0 for p in preds.values()); tokens += used
        rows = [row for i in sorted(preds) for row in prediction_rows(records[i], preds[i])]
        report = summarize(rows, 1.0, manifest["holdout_sources"])
        report.update(coverage={"requested_records": len(records), "evaluated_records": len(preds), "rejected_records": len(errors),
                                "requested_questions": sum(len(r["questions"]) for r in records), "evaluated_questions": len(rows)},
                      seconds=time.time() - t, served_model=jev.served_model, input_tokens=used,
                      estimated_usd=used * PRICE_PER_MILLION / 1e6, errors=errors[:5])
        (out / s.name).mkdir(parents=True, exist_ok=True)
        write_json(out / s.name / "rows.json", rows); write_json(out / s.name / "report.json", report)
        c = report["clean"] if rows else {"acc": float("nan"), "brier": float("nan"), "ece": float("nan"), "n": 0}
        summary[s.name] = {"acc": round(c["acc"], 4), "brier": round(c["brier"], 4), "ece": round(c["ece"], 4), "n": c["n"],
                           "rejected": len(errors), "seconds": round(report["seconds"], 1), "input_tokens": used}
        print(f"{s.name:22s} {json.dumps(summary[s.name])}", flush=True)
    summary["_total"] = {"input_tokens": tokens, "estimated_usd": round(tokens * PRICE_PER_MILLION / 1e6, 4),
                         "usd_per_million": PRICE_PER_MILLION, "served_model": jev.served_model}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary["_total"]), flush=True)


if __name__ == "__main__":
    main()
