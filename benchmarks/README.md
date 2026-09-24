# Evaluation suites

`lite/`, `mid/` and `val/` are checked in. `full/` is built or downloaded separately.

The repository's Apache-2.0 licence covers code, not this directory. Each suite carries the data of its source dataset
and keeps that dataset's licence; see [../docs/benchmarks.md](../docs/benchmarks.md) for the full list.

## Missing suites

Four suites are **not** checked in, because their sources are non-commercial and RACE forbids redistributing derived
data outright:

| Suite | Source | Licence |
|---|---|---|
| `lite/xnli`, `mid/xnli`, `val/val_xnli` | `facebook/xnli` | CC-BY-NC-4.0 |
| `lite/financial_phrasebank`, `mid/financial_phrasebank` | `mteb/FinancialPhrasebankClassification` | CC-BY-NC-SA-3.0 |
| `val/val_race` | `ehovy/race` | non-commercial research only, no redistribution |

Rebuild them from their sources:

```bash
uv run --no-project --with datasets python tools/build_kev_benchmarks.py   # -> benchmarks/full/
python3 tools/build_lite.py                                               # -> benchmarks/lite/
python3 tools/build_lite.py --n 100 --out benchmarks/mid                  # -> benchmarks/mid/
uv run --no-project --with datasets python tools/build_val.py             # -> benchmarks/val/
```

Scores in the README and [../docs/evaluation.md](../docs/evaluation.md) include these suites. Runs against a checkout
without them cover 19 of the 20 mid suites and will not match.
