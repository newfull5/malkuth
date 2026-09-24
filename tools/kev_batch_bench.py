#!/usr/bin/env python3
"""Batched kev.benchmark: many records per forward pass via DecisionModel.forward_batch (the path kev.train uses),
instead of kev.benchmark's one record at a time. Same predictions, same report.json format, one model load for all suites.

    cd kev && uv run python ../tools/kev_batch_bench.py --run jaredpalmer/kev-4b --out runs/lite-4b ../benchmarks/lite/*/

Output: <out>/<suite>/{report.json, rows.json, predictions.jsonl} per suite, plus <out>/summary.json.
Records are sorted by length and grouped into passes of at most --pass-tokens tokens (state + branch per question row),
so padding is small; right padding never changes a real token (kev.model._pad_rows), and parity with the per-record
path is checked by --check.
"""
import argparse, json, math, shutil, sys, time
from pathlib import Path

import torch
import kev.model as km
from kev.api import question_keys
from kev.benchmark import prediction_rows, summarize
from kev.checkpoint import LoadOptions
from kev.data import api_request, materialize
from kev.model import ContextOverflow, rows_of
from kev.predictors import LocalPredictor
from kev.suite import CONTEXT, load_split, read_manifest, record_digest, write_json


def row_tokens(enc):
    """Tokens this record costs in row form: every question is one row of state + its branch."""
    S, _, brs = rows_of(enc)
    return sum(len(S) + len(r["ids"]) for r in brs)


def batches(items, budget):
    """items: (index, enc) sorted by length -> lists whose row tokens stay under budget (at least one record each)."""
    cur, used = [], 0
    for i, enc in items:
        t = row_tokens(enc)
        if cur and used + t > budget:
            yield cur; cur, used = [], 0
        cur.append((i, enc)); used += t
    if cur: yield cur


def to_prediction(record, logits, temperature):
    keys = {qid: question_keys(q["type"], q.get("criteria")) for qid, q in record["questions"].items()}
    return {"probabilities": {qid: dict(zip(keys[qid], torch.softmax(z, -1).cpu().tolist())) for qid, z in zip(keys, logits)},
            "logits": {qid: dict(zip(keys[qid], z.float().cpu().tolist())) for qid, z in zip(keys, logits)},
            "inference_temperature": temperature}


def run_suite(pred, suite, out, pass_tokens):
    records = load_split(suite, "development")
    manifest = read_manifest(suite)
    ctx = manifest.get("context", CONTEXT)
    pred.context = ctx   # the per-record path (--check) must encode under the same limits
    model, tok = pred.model, pred.tok
    encs, rejected = {}, []
    for i, r in enumerate(records):
        try:
            enc = model.encode(tok, materialize(r), max_state=ctx["max_state"], max_branch=ctx["max_branch"], strict=True)
            if len(enc["ids"]) > ctx["max_packed"]: raise ContextOverflow("packed request exceeds the limit")
            encs[i] = enc
        except ContextOverflow as e:
            rejected.append({"id": r["_meta"]["id"], "error": str(e)})
    order = sorted(encs.items(), key=lambda kv: row_tokens(kv[1]))
    preds, t0 = {}, time.time()
    def run(batch):
        """forward_batch with halving on CUDA OOM (a GPU shared with training can run short mid-suite)."""
        try:
            return model.forward_batch([e for _, e in batch])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1: raise
            h = len(batch) // 2
            return run(batch[:h]) + run(batch[h:])
    with torch.no_grad():
        for batch in batches(order, pass_tokens):
            for (i, _), lg in zip(batch, run(batch)):
                preds[i] = to_prediction(records[i], lg, pred.temperature)
    torch.cuda.synchronize()
    secs = time.time() - t0
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for i in sorted(preds):
            r = records[i]
            new = prediction_rows(r, preds[i]); rows += new
            f.write(json.dumps({"request_sha256": record_digest(api_request(r)), "id": r["_meta"]["id"], "prediction": preds[i], "rows": new},
                               allow_nan=False) + "\n")
    write_json(out / "rows.json", rows)
    if rejected: write_json(out / "rejected.json", rejected)
    report = summarize(rows, 1.0, manifest["holdout_sources"])
    report.update(coverage={"requested_records": len(records), "evaluated_records": len(preds), "rejected_records": len(rejected),
                            "requested_questions": sum(len(r["questions"]) for r in records), "evaluated_questions": len(rows)},
                  seconds=secs, batched={"pass_tokens": pass_tokens}, suite=str(suite), run=str(pred.run))
    write_json(out / "report.json", report)
    return report, preds, records


def check(pred, records, preds, n=20):
    """Max |p| difference between batched and per-record (kev.benchmark's LocalPredictor) on n records."""
    worst = 0.0
    for i in list(preds)[:n]:
        ref = pred(records[i])["probabilities"]
        for qid, dist in preds[i]["probabilities"].items():
            worst = max(worst, max(abs(dist[k] - ref[qid][k]) for k in dist))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suites", nargs="+")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pass-tokens", type=int, default=32768, help="row tokens per forward pass (memory knob)")
    ap.add_argument("--check", type=int, default=0, help="compare N records per suite against the per-record path")
    a = ap.parse_args()
    # one forward pass may take --pass-tokens tokens (kev's own default is one serving row, 16,384)
    km.rows_per_pass = lambda rows, prefix_len=0, budget=a.pass_tokens: max(1, budget // (prefix_len + max(len(r) for r in rows)))
    t0 = time.time()
    pred = LocalPredictor(a.run, a.device, LoadOptions.from_env(), context={**CONTEXT, "max_state": 8192, "max_branch": 8192, "max_packed": 16384})
    print(f"loaded {a.run} in {time.time() - t0:.0f}s", flush=True)
    summary = {}
    for s in a.suites:
        s = Path(s)
        if (Path(a.out) / s.name / "report.json").exists():   # resume: a finished suite is kept as is
            r = json.loads((Path(a.out) / s.name / "report.json").read_text()); c = r["clean"]
            summary[s.name] = {"acc": round(c["acc"], 4), "brier": round(c["brier"], 4), "ece": round(c["ece"], 4), "n": c["n"],
                               "seconds": round(r["seconds"], 1), "rejected": r["coverage"]["rejected_records"]}
            continue
        if (Path(a.out) / s.name).exists(): shutil.rmtree(Path(a.out) / s.name)   # partial suite from a crash
        report, preds, records = run_suite(pred, s, Path(a.out) / s.name, a.pass_tokens)
        c = report["clean"]
        line = {"acc": round(c["acc"], 4), "brier": round(c["brier"], 4), "ece": round(c["ece"], 4), "n": c["n"],
                "seconds": round(report["seconds"], 1), "rejected": report["coverage"]["rejected_records"],
                "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        if a.check: line["max_abs_diff_vs_per_record"] = check(pred, records, preds, a.check)
        summary[s.name] = line
        print(f"{s.name:22s} {json.dumps(line)}", flush=True)
    summary["_total_seconds"] = round(time.time() - t0, 1)
    write_json(Path(a.out) / "summary.json", summary)
    print(f"total {summary['_total_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
