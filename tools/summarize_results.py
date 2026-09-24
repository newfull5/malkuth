#!/usr/bin/env python3
"""Collect report.json files from kev_batch_bench.py / laya_bench.py runs into one markdown comparison.

    python3 tools/summarize_results.py results/lite > results/lite/RESULTS.md

Layout expected: <root>/<model>/<suite>/report.json (Kev) and <root>/laya/<checkpoint>/<suite>/report.json (Laya).
"""
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
models = {}   # label -> {suite: report}
for rep in sorted(root.glob("*/*/report.json")) + sorted(root.glob("laya/*/*/report.json")):
    parts = rep.relative_to(root).parts
    label = f"laya-{parts[1]}" if parts[0] == "laya" else parts[0]
    if parts[0] == "laya" and len(parts) != 4: continue
    if parts[0] != "laya" and len(parts) != 3: continue
    models.setdefault(label, {})[parts[-2]] = json.loads(rep.read_text())
order = [m for m in ("jev", "kev-0.5b", "kev-0.6b", "kev-0.8b", "kev-4b", "kev-9b", "laya-english", "laya-multilingual", "laya-typed-decisions", "laya-router") if m in models]
suites = sorted({s for m in order for s in models[m]})


def cell(m, s, key="acc"):
    r = models[m].get(s)
    if not r: return "–"
    v = r["clean"][key]
    rej = r["coverage"].get("rejected_records", 0)
    return f"{v:.3f}" + (f" ({rej}✗)" if rej else "")


def table(key, title):
    print(f"\n### {title}\n")
    print("| suite | n | " + " | ".join(order) + " |")
    print("|---|---|" + "---|" * len(order))
    for s in suites:
        n = next((models[m][s]["clean"]["n"] for m in order if s in models[m]), "")
        print(f"| `{s}` | {n} | " + " | ".join(cell(m, s, key) for m in order) + " |")
    means = []
    for m in order:
        vals = [models[m][s]["clean"][key] for s in suites if s in models[m]]
        means.append(f"**{sum(vals) / len(vals):.3f}**" if vals else "–")
    print("| **Mean (unweighted across suites)** | | " + " | ".join(means) + " |")


print(f"# Evaluation results (`{root}`)\n")
print("Question-level metrics follow kev.benchmark. Higher accuracy and lower Brier/ECE are better. `(N✗)` marks records the model could not encode; these are excluded from scores.")
table("acc", "Accuracy")
table("brier", "Brier (lower is better)")
table("ece", "ECE (lower is better)")

# per-language accuracy on multilingual suites: task names end with _<lang>
langs = ["en", "ko", "ja", "zh", "de", "fr", "es", "hi", "ar", "th"]
print("\n### Accuracy by language (aggregated over multilingual suites that include Korean: massive_intent · sib200 · paws_x · belebele · polyguard · rtp_lx)\n")
print("| model | " + " | ".join(langs) + " | ko − en |")
print("|---|" + "---|" * (len(langs) + 1))
for m in order:
    agg = {l: [0, 0] for l in langs}
    for s, r in models[m].items():
        if s.startswith("laya_") or not any("_ko" in task for task in r["tasks"]): continue   # same item set for every language
        for task, t in r["tasks"].items():
            lang = next((x for x in task.split("_")[1:] if x in agg), None)   # e.g. sib200_ko, polyguard_ko_prompt
            if lang in agg and not s.startswith("laya_"):
                agg[lang][0] += t["acc"] * t["n"]; agg[lang][1] += t["n"]
    acc = {l: (a / n if n else None) for l, (a, n) in agg.items()}
    gap = f"{acc['ko'] - acc['en']:+.3f}" if acc["ko"] is not None and acc["en"] is not None else "–"
    print(f"| {m} | " + " | ".join(f"{acc[l]:.3f}" if acc[l] is not None else "–" for l in langs) + f" | {gap} |")
