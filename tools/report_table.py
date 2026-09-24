#!/usr/bin/env python3
"""The single comparison table in the README, computed from results/full/.

    python3 tools/report_table.py            # markdown table on stdout

Every model answered the same 29 suites, Jev through its API. Columns are unweighted means over suites. "held-out" is
the nine suites in HELD_OUT, the datasets that contributed nothing to the Malkuth training mix; "all" is all 29. `global_mmlu` and `hwu64` are excluded as contaminated.
Reads each suite's report.json rather than summary.json, which a partially finished run can leave stale.
"""
import json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Datasets that contributed nothing to the training mix, and whose source families were not used either. laya_apps
# would belong here but is dropped: it carries Kev's and Laya's own training data, so it is not held out for them.
HELD_OUT = ["sib200", "belebele", "rtp_lx", "polyguard", "multi_eurlex", "goemotions", "financial_phrasebank",
            "ledgar", "kold"]
ROWS = [("Jev API", "Undisclosed", "jev"),
        ("Malkuth-4B (ours)", "Qwen3.5-4B-Base", "k4b-mix4-s4500"),
        ("Malkuth-2B (ours)", "Qwen3.8-2B-Distill", "d2b-mix4-cal"),
        ("Kev-9B", "Qwen3.5-9B-Base", "kev-9b"),
        ("Kev-4B", "Qwen3.5-4B-Base", "kev-4b"),
        ("Kev-0.8B", "Qwen3.5-0.8B-Base", "kev-0.8b"),
        ("Laya Router", "English / multilingual", "laya/router"),
        ("Laya Multilingual", "mmBERT-base", "laya/multilingual"),
        ("Laya Typed Decisions", "ModernBERT-large", "laya/typed-decisions"),
        ("Laya English", "ModernBERT-large", "laya/english")]


def scores(run):
    d = ROOT / "results" / "full" / run
    if not d.is_dir(): return None
    suites = {p.name: json.loads((p / "report.json").read_text())["clean"] for p in d.iterdir()
              if p.is_dir() and (p / "report.json").exists()}
    if not suites: return None
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    held = [s for s in HELD_OUT if s in suites]
    return {"heldout": mean([suites[s]["acc"] for s in held]),
            "se": math.sqrt(sum(suites[s]["acc"] * (1 - suites[s]["acc"]) / suites[s]["n"] for s in held)) / len(held) if held else None,
            "all": mean([v["acc"] for v in suites.values()]),
            "brier": mean([suites[x]["brier"] for x in held]),
            "suites": len(suites)}


def main():
    cell = lambda v: f"{v:.3f}" if v is not None else "–"
    out = ["| Model | Base model | Accuracy (held-out) ↑ | Accuracy (all) ↑ | Brier (held-out) ↓ |",
           "|---|---|---:|---:|---:|"]
    best = {}
    got = [(n, b, scores(r)) for n, b, r in ROWS]
    got = [(n, b, s) for n, b, s in got if s]
    for k in ("heldout", "all"): best[k] = max(s[k] for _, _, s in got if s[k] is not None)
    best["brier"] = min(s["brier"] for _, _, s in got if s["brier"] is not None)
    bold = lambda k, s: f"**{cell(s[k])}**" if s[k] is not None and abs(s[k] - best[k]) < 1e-9 else cell(s[k])
    for name, base, s in got:
        if s["suites"] != 29: print(f"warning: {name} has {s['suites']}/29 suites", file=sys.stderr)
        acc = f"{bold('heldout', s)} ±{s['se']:.4f}" if s["se"] else bold("heldout", s)
        out.append(f"| {name} | {base} | {acc} | {bold('all', s)} | {bold('brier', s)} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
