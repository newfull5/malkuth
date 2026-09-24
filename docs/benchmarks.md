# Benchmark guide

The suites, the rules for reporting on them, and the commands that build and run them.

## Benchmark sets

| Directory | Purpose | Size |
|---|---|---|
| `benchmarks/val/` | Checkpoint selection and calibration | Validation splits and non-benchmark proxies |
| `benchmarks/lite/` | Smoke checks | 20 suites; 2,020 records / 3,140 questions |
| `benchmarks/mid/` | Same-item model comparisons | 20 suites; 14,020 questions |
| `benchmarks/` | Full evaluation inventory; generated or downloaded separately | 31 suites; 84,522 records / 136,522 questions; about 131 MB |

A record holds a state and one or more questions. Metrics are per question, and aggregate tables are unweighted means
over suites. The full Malkuth runs drop `global_mmlu` and `hwu64`, leaving 29 suites and 128,876 questions.

Every suite uses Kev's eval-only format: `development.jsonl` and `manifest.json`.

Four suites are not distributed with the repository because their sources are non-commercial. See
[../benchmarks/README.md](../benchmarks/README.md) to rebuild them.

## Reporting rules

Pick checkpoints and fit calibration on `benchmarks/val/` only. Lite, mid and full are for reporting.

| Suites | In the Malkuth training mix |
|---|---|
| `massive_intent`, `xnli`, `paws_x`, `tweet_sentiment`, `clinc150`, `klue_nli`, `klue_ynat`, `nsmc`, `kobest`, `laya_typed_decisions` | Yes, their train splits. Evaluated on separate splits. |
| `sib200`, `belebele`, `rtp_lx`, `polyguard`, `multi_eurlex`, `goemotions`, `financial_phrasebank`, `ledgar`, `kold` | No. These nine are the held-out set. |
| `laya_apps` | No, but it carries Kev's and Laya's training data, so it stays out of the held-out mean. |
| `global_mmlu`, `hwu64` | Overlap MMMLU and MASSIVE. Excluded from the full set. |

This describes Malkuth post-training, not the base model's pretraining.

## Full inventory

The core ten languages are English, Korean, Japanese, Chinese, German, French, Spanish, Hindi, Arabic and Thai.
Seventeen suites include Korean. `laya_massive51` covers 51 languages.

| Category | Suites | Records | Questions | Purpose |
|---|---:|---:|---:|---|
| English classification | 6 | 3,000 | 3,500 | Classification with 3–151 options |
| Multilingual classification and comprehension | 10 | 38,040 | 78,040 | Cross-language comparisons, including parallel corpora |
| Multilingual safety | 3 | 13,150 | 23,050 | Harmfulness, toxicity, and jailbreak detection |
| Korean-only tasks | 5 | 4,397 | 4,397 | Native Korean data |
| Laya reproduction | 7 | 25,935 | 27,535 | Tasks generated with Laya's code and sampling rules |
| **Total** | **31** | **84,522** | **136,522** | |

`tools/build_kev_benchmarks.py` builds the first four categories, `tools/build_laya_benchmarks.py` the Laya ones. None
of the first four come from Kev's listed training sources, though some later became Malkuth training sources.

### English classification

`noul ×N` means N questions per record.

| Suite | Task | Type | Options | Records / questions |
|---|---|---|---:|---|
| `clinc150` | Assistant intent and out-of-scope detection | choice | 151 | 500 / 500 |
| `hwu64` | Assistant intent | choice | 64 | 500 / 500 |
| `goemotions` | Fine-grained emotion, single-label examples only | choice | 28 | 500 / 500 |
| `financial_phrasebank` | Financial news sentiment, unanimous annotations | choice | 3 | 500 / 500 |
| `ledgar` | Contract clause category | choice | 100 | 500 / 500 |
| `toxicchat` | Toxicity and jailbreak detection | noul ×2 | 2 | 500 / 1,000 |

### Multilingual classification and comprehension

| Suite | Task | Type | Options | Languages | Records / questions | Parallel |
|---|---|---|---:|---:|---|---|
| `massive_intent` | Assistant intent; all 60 options | choice | 60 | 10 | 5,000 / 5,000 | Yes |
| `massive_scenario` | Assistant scenario | choice | 18 | 10 | 5,000 / 5,000 | Yes |
| `sib200` | Sentence topic; Flores-200 | choice | 7 | 10 | 2,040 / 2,040 | Yes |
| `mtop_intent` | Task-oriented intent | choice | 102 | 6 | 3,000 / 3,000 | Yes |
| `xnli` | Entailment, neutral, or contradiction | choice | 3 | 8 | 4,000 / 4,000 | Yes |
| `paws_x` | Paraphrase detection | noul | 2 | 7 | 3,500 / 3,500 | Yes |
| `belebele` | Reading comprehension | choice | 4 | 10 | 5,000 / 5,000 | Yes |
| `global_mmlu` | Knowledge questions | choice | 4 | 9 | 4,500 / 4,500 | Yes |
| `tweet_sentiment` | Tweet sentiment | choice | 3 | 8 | 4,000 / 4,000 | No |
| `multi_eurlex` | EU legislation topics; 21 EuroVoc labels | noul ×21 | 2 | 4 | 2,000 / 42,000 | Yes |

Laya's own MASSIVE evaluation uses 20 options rather than 60, and its XNLI setup uses the first 300 examples, so Laya's
published numbers are not comparable to these suites.

### Multilingual safety

| Suite | Task | Type | Languages | Records / questions | Notes |
|---|---|---|---:|---|---|
| `polyguard` | Harmful request, harmful response, and refusal | noul ×≤3 | 10 | 5,000 / 14,900 | Parallel; positive and negative labels |
| `rtp_lx` | Human-rated toxicity on five levels | score | 10 | 5,000 / 5,000 | Parallel; levels 0–4 contain 1,449 / 1,163 / 1,266 / 723 / 399 examples |
| `multijail` | Harmful request detection | noul | 10 | 3,150 / 3,150 | Parallel; all requests are harmful, so accuracy equals recall |

### Korean-only tasks

| Suite | Task | Type | Options | Records / questions | Source split |
|---|---|---|---|---|---|
| `klue_ynat` | News headline topic | choice | 7 | 500 / 500 | validation; test is private |
| `klue_nli` | Natural language inference | choice | 3 | 500 / 500 | validation; test is private |
| `nsmc` | Positive movie review detection | noul | 2 | 500 / 500 | test |
| `kobest` | BoolQ, COPA, WiC, HellaSwag, SentiNeg | noul / choice | 2–4 | 2,397 / 2,397 | test |
| `kold` | Offensive comment detection, with article title | noul | 2 | 500 / 500 | complete dataset; no official test split |

### Laya reproduction

| Suite | Task | Type | Options | Languages | Records / questions | Published reference |
|---|---|---|---|---|---|---|
| `laya_typed_decisions` | Four TypeSafe-style workflows; teacher soft labels in `_meta.soft` | choice / noul / score | 2–5 | en | 400 / 2,000 | Laya-td 0.766 / Jev 0.727 |
| `laya_massive_intent` | Correct intent plus 19 random distractors | choice | 20 | 14 | 4,200 / 4,200 | Laya “Why Route” table |
| `laya_massive_scenario` | MASSIVE scenario | choice | 18 | 14 | 4,200 / 4,200 | Laya BENCHMARKS.md |
| `laya_xnli` | XNLI | choice | 3 | 15 | 4,500 / 4,500 | Laya “Why Route” table |
| `laya_english` | SST-5, Emotion, prompt injections, AG News, BoolQ | score / choice / noul | 2–6 | en | 2,516 / 2,516 | Laya English tasks table |
| `laya_apps` | News, emotion, banking intent, ticket routing, spam, phishing, jailbreaks, toxicity, retrieval, routing | choice / noul | 2–77 | en | 3,999 / 3,999 | Workflow chart and Jev comparison |
| `laya_massive51` | MASSIVE intent with 20 options | choice | 20 | 51 | 6,120 / 6,120 | “45 / 51 languages” comparison |

## Language coverage

| Suite | Languages |
|---|---|
| `massive_intent`, `massive_scenario`, `sib200`, `belebele`, `polyguard`, `rtp_lx` | en, ko, ja, zh, de, fr, es, hi, ar, th |
| `mtop_intent` | en, de, fr, es, hi, th |
| `xnli` | en, zh, de, fr, es, hi, ar, th |
| `paws_x` | en, ko, ja, zh, de, fr, es |
| `global_mmlu` | en, ko, ja, zh, de, fr, es, hi, ar |
| `tweet_sentiment` | en, de, fr, es, hi, ar, it, pt |
| `multi_eurlex` | en, de, fr, es |
| `multijail` | en, ko, zh, ar, th, it, vi, bn, sw, jv |
| `laya_massive_intent`, `laya_massive_scenario` | en, ko, ja, zh, de, fr, es, hi, ar, pt, ru, tr, ta, sw |
| `laya_xnli` | en, zh, de, fr, es, hi, ar, th, ru, tr, ur, vi, el, bg, sw |
| `laya_massive51` | 51 languages, including all ten core languages |

Parallel suites keep the same items across languages through `_meta.group_id`. Per-language scores are under
`report.json["tasks"]`, named like `sib200_ko`.

## Lite set

Twenty records per language or task, sampled from the full suites with seed 0. MultiEURLEX uses five per language.
`_meta.id` is preserved and parallel suites keep matched items. Sizes live in `PLAN` in `tools/build_lite.py`.

| Suites | Records | Questions | Sampling |
|---|---:|---:|---|
| `clinc150`, `goemotions`, `financial_phrasebank`, `ledgar` | 20 each | 20 each | English |
| `massive_intent`, `sib200`, `belebele` | 200 each | 200 each | 10 languages × 20 |
| `xnli`, `tweet_sentiment` | 160 each | 160 each | 8 languages × 20 |
| `paws_x` | 140 | 140 | 7 languages × 20 |
| `multi_eurlex` | 20 | 420 | 4 languages × 5; 21 questions per record |
| `polyguard` | 200 | 600 | 10 languages × 20 |
| `rtp_lx` | 200 | 200 | 10 languages × 20 |
| `klue_ynat`, `klue_nli`, `nsmc`, `kold` | 20 each | 20 each | Korean |
| `kobest` | 100 | 100 | 5 tasks × 20 |
| `laya_typed_decisions` | 80 | 400 | 4 workflows × 20 |
| `laya_apps` | 200 | 200 | 10 tasks × 20 |

Questions by language: en 985, ko 340, de/fr/es 305 each, zh/hi/ar 180 each, ja/th 160 each, it/pt 20 each.

| Excluded from lite | Reason |
|---|---|
| `massive_scenario`, `mtop_intent` | Similar to `massive_intent`; MTOP lacks Korean |
| `hwu64` | Similar to CLINC150 assistant intent |
| `toxicchat` | Few positive labels; `laya_apps` includes balanced toxic-chat tasks |
| `global_mmlu` | Measures knowledge rather than classification |
| `multijail` | All-positive data; PolyGuard provides a broader safety check |
| `laya_english` | AG News, BoolQ, and SST-5 were Kev training sources; Emotion overlaps `laya_apps` |
| Other Laya multilingual suites | Overlap the main multilingual tasks; use full suites to reproduce Laya comparisons |

At 20 questions per task a 95% accuracy interval is roughly ±20 points. Lite is for pipeline checks, not for comparing
models.

## Reading the numbers

- **Class imbalance.** MultiEURLEX is about 17% positive, so always answering false scores around 0.83. MultiJail is
  all-positive, and ToxicChat has 38 toxicity and 11 jailbreak positives in 500 records.
- **Context limit.** MultiEURLEX, PolyGuard responses and LEDGAR can exceed Kev's 384-token training context. Records
  past the 8,192-token serving limit are rejected rather than truncated; the count is in `coverage.rejected_records`.

## Running an evaluation

Build the non-Laya suites from the repository root. Default sample size is 500 per language or task, seed 0. `--n 0`
takes the whole source split.

```bash
uv run --no-project --with datasets python tools/build_kev_benchmarks.py --n 500 --seed 0
uv run --no-project --with datasets python tools/build_lite.py
uv run --no-project --with datasets python tools/build_lite.py --n 100 --out benchmarks/mid
```

The Laya suites need the pinned Laya checkout (`NandhaKishorM/laya` at `c752770`,
`uv pip install --no-deps -e ../laya`), then from `kev/`:

```bash
uv run python ../tools/build_laya_benchmarks.py
```

Evaluate from `kev/`:

```bash
uv run python ../tools/kev_batch_bench.py \
  --run dhtocks/malkuth-4b --out ../results/local-full/malkuth-4b \
  ../benchmarks/full/*/
```

Drop `global_mmlu` and `hwu64` for held-out or 29-suite aggregates. Upstream Kev takes `--remote <url>` in place of
`--run` for a compatible endpoint.

## Benchmark sources

| Suite | Source | Source split and size | License |
|---|---|---|---|
| `clinc150` | `clinc/clinc_oos` | test 5,500; 1,000 OOS | CC-BY-3.0 |
| `hwu64` | `DeepPavlov/hwu64` | test 1,076 | CC-BY-4.0 |
| `goemotions` | `google-research-datasets/go_emotions`, simplified | test 4,590 single-label | Apache-2.0 |
| `financial_phrasebank` | `mteb/FinancialPhrasebankClassification` | allagree 2,264 | CC-BY-NC-SA-3.0 |
| `ledgar` | `coastalcph/lex_glue`, ledgar | test 10,000 | CC-BY-4.0 |
| `toxicchat` | `lmsys/toxic-chat`, toxicchat0124 | test 5,083 | CC-BY-NC-4.0 |
| `massive_intent`, `massive_scenario` | `mteb/amazon_massive_intent`, `mteb/amazon_massive_scenario` | test 2,974 per language | CC-BY-4.0 |
| `sib200` | `Davlan/sib200` | test 204 per language | CC-BY-SA-4.0 |
| `mtop_intent` | `mteb/mtop_intent`, JSONL | test; varies by language | CC-BY-SA-4.0 |
| `xnli` | `facebook/xnli` | test 5,010 per language | CC-BY-NC-4.0 |
| `paws_x` | `google-research-datasets/paws-x` | test 2,000 per language | free for any purpose (Google) |
| `belebele` | `facebook/belebele` | test 900 per language | CC-BY-SA-4.0 |
| `global_mmlu` | `CohereLabs/Global-MMLU` | test 14,042 per language | Apache-2.0 |
| `tweet_sentiment` | `cardiffnlp/tweet_sentiment_multilingual`, Parquet | test 870 per language | CC-BY-3.0 |
| `multi_eurlex` | `mteb/eurlex-multilingual`, level 1 | test 5,000 per language | CC-BY-SA-4.0 |
| `polyguard` | `ToxicityPrompts/PolyGuardPrompts` | test 1,725 per language | CC-BY-4.0 |
| `rtp_lx` | `ToxicityPrompts/RTP-LX` | test approximately 1,000 per language | RTP-LX license |
| `multijail` | `DAMO-NLP-SG/MultiJail` | 315 per language | MIT |
| `klue_ynat`, `klue_nli` | `klue/klue` | validation 9,107 / 3,000 | CC-BY-SA-4.0 |
| `nsmc` | `e9t/nsmc`, ratings_test.txt | test 50,000 | CC0-1.0 |
| `kobest` | `skt/kobest_v1` | test 1,404 / 1,000 / 1,260 / 500 / 397 | CC-BY-SA-4.0 |
| `kold` | `nayohan/KOLD` | complete dataset, 40,429 | CC-BY-SA-4.0 |
| `laya_*` | Sources referenced by Laya builders | First N records or seed-13 samples, per upstream code | Source-specific |

Laya sources include typed-decisions, MASSIVE, XNLI, SST-5, Emotion, deepset/prompt-injections, AG News, BoolQ,
mteb/banking77, customer-support-tickets, enron_spam, phishing-email-dataset, toxic-chat, MS MARCO, GSM8K and MBPP.

## Training sources

The released checkpoints were trained on `train_mix_v4`, 89,791 records built by `tools/build_train_mix.py`.

| Source | Dataset | Records | Licence |
|---|---|---:|---|
| `massive_intent` | `mteb/amazon_massive_intent` | 10,000 | Apache-2.0 |
| `massive_scenario` | `mteb/amazon_massive_scenario` | 3,000 | Apache-2.0 |
| `xquad_mc` | `google/xquad` | 8,800 | CC-BY-SA-4.0 |
| `xnli` | `facebook/xnli` | 6,400 | **CC-BY-NC-4.0** |
| `tweet_sentiment` | `cardiffnlp/tweet_sentiment_multilingual` | 6,400 | *not stated* |
| `klue` | `klue/klue` | 6,000 | CC-BY-SA-4.0 |
| `civil_toxicity` | `google/civil_comments` | 5,001 | CC0-1.0 |
| `kobest` | `skt/kobest_v1` | 5,000 | CC-BY-SA-4.0 |
| `toxicity` | `textdetox/multilingual_toxicity_dataset` | 5,000 | OpenRAIL++ |
| `cuad` | `dvgodoy/CUAD_v1_Contract_Understanding_clause_classification` | 5,000 | CC-BY-4.0 |
| `paws_x` | `google-research-datasets/paws-x` | 4,900 | free for any purpose (Google) |
| `mmmlu` | `openai/MMMLU` | 4,500 | MIT |
| `race` | `ehovy/race` | 4,000 | **non-commercial research only** |
| `apeach` | `jason9693/APEACH` | 4,000 | CC-BY-SA-4.0 |
| `clinc150` | `clinc/clinc_oos` | 3,000 | CC-BY-3.0 |
| `nsmc` | `e9t/nsmc` | 3,000 | CC0-1.0 |
| `jailbreak` | `jackhhao/jailbreak-classification` + `deepset/prompt-injections` | 1,590 | Apache-2.0 (both) |
| `arc` | `allenai/ai2_arc` | 1,500 | CC-BY-SA-4.0 |
| `spam` | `ucirvine/sms_spam` | 1,500 | *not stated* |
| `typed_decisions` | `LocalLLaMA/typed-decisions` | 1,200 | Apache-2.0 |

Two sources carry non-commercial terms, which is why the weights are research use only. XNLI is CC-BY-NC-4.0 in
`facebookresearch/XNLI`, though the Hugging Face mirror has no licence tag, and RACE is "available for non-commercial
research purpose only". Two more state no licence.

`aegis` and `beavertails` arrived in v5, which the released checkpoints do not use.

## Results

Saved runs are in [../results/](../results/). The comparison tables are in the
[README](../README.md) and [evaluation.md](evaluation.md).
