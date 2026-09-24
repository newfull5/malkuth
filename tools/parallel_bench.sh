#!/usr/bin/env bash
# Evaluate one checkpoint on many suites across several GPUs, then merge the per-suite reports into one summary.json.
#
#   tools/parallel_bench.sh <run> <out_dir> <gpus> <suite_dir>...
#   tools/parallel_bench.sh runs/k4b-mix4/step-4500 ../results/mid/k4b-v4 0,1,2,3 ../benchmarks/mid/*/     # run from kev/
#
# Suites are dealt round-robin to the GPUs (one kev_batch_bench process per GPU, bf16). kev_batch_bench skips suites
# that already have a report.json, so re-running after a crash only redoes the missing ones.
set -euo pipefail
run=$1; out=$2; IFS=, read -ra gpus <<< "$3"; shift 3
suites=("$@"); n=${#gpus[@]}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
for i in "${!gpus[@]}"; do
  shard=()
  for j in "${!suites[@]}"; do (( j % n == i )) && shard+=("${suites[$j]}"); done
  (( ${#shard[@]} )) || continue
  CUDA_VISIBLE_DEVICES=${gpus[$i]} KEV_DTYPE=${KEV_DTYPE:-bf16} uv run python "$here/kev_batch_bench.py" \
    --run "$run" --out "$out" --pass-tokens "${PASS_TOKENS:-16384}" "${shard[@]}" > "$out/gpu${gpus[$i]}.log" 2>&1 &
done
wait
python3 - "$out" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); summary = {}
for rep in sorted(out.glob("*/report.json")):
    r = json.loads(rep.read_text()); c = r["clean"]
    summary[rep.parent.name] = {"acc": round(c["acc"], 4), "brier": round(c["brier"], 4), "ece": round(c["ece"], 4), "n": c["n"],
                                "seconds": round(r.get("seconds", 0), 1), "rejected": r["coverage"]["rejected_records"]}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
v = list(summary.values())
print(f"{len(v)} suites  acc {sum(x['acc'] for x in v) / len(v):.4f}  brier {sum(x['brier'] for x in v) / len(v):.4f}  -> {out}/summary.json")
EOF
