#!/usr/bin/env python3
"""Laya's published benchmarks, rebuilt with Laya's own code (same seed 13, same questions) and frozen as Kev suites,
so Kev / our models answer the exact questions behind Laya's README numbers.

    cd kev && uv run python ../tools/build_laya_benchmarks.py [suite ...]
    cd kev && uv run python -m kev.benchmark --run jaredpalmer/kev-4b --suite ../benchmarks/laya_xnli --out runs/b-laya_xnli

Sources (laya repo, cloned at ../laya):
  research/scripts/laya_benchmark_colab.ipynb  section 5 -> laya_typed_decisions, laya_massive_intent, laya_massive_scenario,
                                                           laya_xnli, laya_english   (README "Why Route", T4 table)
  research/scripts/bench_apps.py build()       -> laya_apps                           (workflow chart + Jev comparison)
  research/scripts/bench_local.py build_massive -> laya_massive51                     (51-language sweep, 120 per language)
Needs torch + transformers (Laya's scripts import them), hence kev's environment.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAYA = ROOT / "laya"
SCRIPTS = LAYA / "research" / "scripts"
sys.path[:0] = [str(LAYA), str(SCRIPTS), str(Path(__file__).resolve().parent)]
from build_kev_benchmarks import freeze   # noqa: E402


def to_record(state, questions, gold, src, part, **meta):
    """Laya case (state, questions) + gold index per question -> Kev labelled record."""
    qs = {}
    for qid, q in questions.items():
        idx = gold[qid]
        label = list(q["criteria"])[idx] if q["type"] == "choice" else bool(idx) if q["type"] == "noul" else int(idx)
        qs[qid] = {**q, "label": label, "src": src}
    return {"state": state, "questions": qs, "_part": part, **({"_extra": meta} if meta else {})}


def notebook_suites():
    """Exec the notebook's section-5 cells (suite construction only, no model code)."""
    nb = json.loads((SCRIPTS / "laya_benchmark_colab.ipynb").read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    start = next(i for i, c in enumerate(cells) if "SUITES = {}" in c)
    end = next(i for i, c in enumerate(cells) if "suites built:" in c)
    ns = {}
    for c in cells[start:end + 1]:
        exec(c, ns)
    return ns["SUITES"]


def notebook_groups(suites):
    groups = {"laya_typed_decisions": [], "laya_massive_intent": [], "laya_massive_scenario": [], "laya_xnli": [], "laya_english": []}
    for name, s in suites.items():
        family = name.split(".")[0]
        part = name.split(".", 1)[1] if "." in name else None
        src = name.replace(".", "_").replace("-", "_")
        target = {"typed_decisions": "laya_typed_decisions", "massive_intent": "laya_massive_intent",
                  "massive_scenario": "laya_massive_scenario", "xnli": "laya_xnli", "en": "laya_english"}[family]
        for i, ((state, questions), g) in enumerate(zip(s["cases"], s["gold"])):
            if family == "typed_decisions":
                wf = s["meta"]["workflows"][i]
                groups[target].append(to_record(state, questions, {q: v["idx"] for q, v in g.items()}, f"td_{wf}", wf,
                                                soft={q: v["soft"] for q, v in g.items()}))
            else:
                groups[target].append(to_record(state, questions, {q: v["idx"] for q, v in g.items()}, src, part))
    return groups


def apps():
    import bench_apps
    bench_apps.build()
    out = []
    for name, s in bench_apps.SUITES.items():
        src = name.replace(".", "_")
        for (state, questions), gi in zip(s["cases"], s["gold"]):
            (qid,) = questions   # every app case asks exactly one question; gold is its index
            out.append(to_record(state, questions, {qid: gi}, src, name))
    return out


def massive51():
    import bench_local
    suites = bench_local.build_massive(bench_local.massive_languages(), 120)   # bench_local.py --per-lang default
    return [to_record(state, qs, {"intent": gi}, f"massive51_{lang.replace('-', '_')}", lang)
            for lang, (cases, gold, _) in suites.items() for (state, qs), gi in zip(cases, gold)]


SOURCE = "laya repo (NandhaKishorM/laya), seed 13"
RIGHTS = "per upstream dataset; see docs/benchmarks.md"


def main():
    want = set(sys.argv[1:])
    nb_names = {"laya_typed_decisions", "laya_massive_intent", "laya_massive_scenario", "laya_xnli", "laya_english"}
    if not want or want & nb_names:
        for name, records in notebook_groups(notebook_suites()).items():
            if not want or name in want:
                print(f"{name}: {freeze(name, SOURCE + ', laya_benchmark_colab.ipynb', RIGHTS, records, 0, 13)}", flush=True)
    if not want or "laya_apps" in want:
        print(f"laya_apps: {freeze('laya_apps', SOURCE + ', bench_apps.py', RIGHTS, apps(), 0, 13)}", flush=True)
    if not want or "laya_massive51" in want:
        print(f"laya_massive51: {freeze('laya_massive51', SOURCE + ', bench_local.py', RIGHTS, massive51(), 0, 13)}", flush=True)


if __name__ == "__main__":
    main()
