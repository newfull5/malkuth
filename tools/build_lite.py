#!/usr/bin/env python3
"""Lite evaluation set: a subsample of benchmarks/full/ for quick runs -> benchmarks/lite/<suite>/ (same Kev suite layout).

    python3 tools/build_lite.py
    cd kev && uv run python -m kev.benchmark --run jaredpalmer/kev-4b --device cuda --suite ../benchmarks/lite/sib200 --out runs/lite-sib200

Records keep their original _meta (id, group_id), so lite results can be joined back to full runs. Parallel corpora stay
parallel: the same items in every language. Rationale per suite: docs/benchmarks.md "약식 세트".
"""
import hashlib, json, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "benchmarks/full", ROOT / "benchmarks/lite"
SEED = 0

# suite -> records per part (language / task); 0 = keep all. Suites not listed are dropped (duplicates or cost; see docs).
PLAN = {
    # English
    "clinc150": 20, "goemotions": 20, "financial_phrasebank": 20, "ledgar": 20,
    # multilingual (per language)
    "massive_intent": 20, "sib200": 20, "xnli": 20, "paws_x": 20, "belebele": 20, "tweet_sentiment": 20,
    "multi_eurlex": 5,    # 21 questions per long document; the costliest suite (5 x 4 languages = 20)
    # multilingual safety (per language)
    "polyguard": 20, "rtp_lx": 20,
    # Korean (kobest: per task)
    "klue_ynat": 20, "klue_nli": 20, "nsmc": 20, "kold": 20, "kobest": 20,
    # Jev / Laya comparison (per workflow / task)
    "laya_typed_decisions": 20, "laya_apps": 20,
}


def main():
    import argparse
    from build_kev_benchmarks import sample   # the same parallel-aware sampler the full suites were built with
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="override every PLAN size with this per-part count (multi_eurlex keeps its own)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out_root = Path(a.out)
    if out_root.exists(): shutil.rmtree(out_root)
    total_r = total_q = 0
    for name, n in PLAN.items():
        if a.n and name != "multi_eurlex": n = a.n
        records = [json.loads(l) for l in (SRC / name / "development.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in records:
            m = r["_meta"]
            r["_part"] = m.get("part")
            r["_key"] = m["group_id"]   # parallel suites share it across languages; elsewhere unique, so the sampler falls back per part
        kept = sample(records, n, SEED)
        for r in kept: r.pop("_part", None); r.pop("_key", None)
        manifest = json.loads((SRC / name / "manifest.json").read_text(encoding="utf-8"))
        d = out_root / name; d.mkdir(parents=True)
        parts = {"development.jsonl": kept, "train.jsonl": [], "calibration.jsonl": [], "test.jsonl": []}
        for f, rs in parts.items():
            (d / f).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
            manifest["files"][f] = {"sha256": hashlib.sha256((d / f).read_bytes()).hexdigest(), "records": len(rs)}
        counts = {}
        for r in kept: counts[str(r["_meta"].get("part"))] = counts.get(str(r["_meta"].get("part")), 0) + 1
        manifest["parts"] = counts
        manifest["lite"] = {"from": f"benchmarks/full/{name}", "n_per_part": n or None, "seed": SEED}
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        q = sum(len(r["questions"]) for r in kept)
        total_r += len(kept); total_q += q
        print(f"{name:22s} {len(kept):5d} records {q:6d} questions  {counts if len(counts) > 1 else ''}")
    print(f"TOTAL {len(PLAN)} suites, {total_r} records, {total_q} questions -> {out_root}")


if __name__ == "__main__":
    main()
