# Evaluation

Results behind the summary table in the [README](../README.md).

## Evaluation sets

| Set | Suites | Questions | Purpose |
|---|---:|---:|---|
| `benchmarks/val/` | 9 | 2,750 | **Selection only.** Checkpoint choice and temperature fitting. Never reported as a result. |
| `benchmarks/lite/` | 20 | 3,140 | Smoke checks. Many suites have 20 questions; unsuitable for close comparisons. |
| `benchmarks/mid/` | 20 | 14,020 | A subsample of the full set, kept for quick reruns. Superseded by the full set for reporting. |
| `benchmarks/` (full) | 29 | ~129,000 | **Everything reported here.** Excludes `global_mmlu` and `hwu64` as contaminated. |

Checkpoints and temperature calibration were chosen on `benchmarks/val/` only. The reported sets were not used for
selection.

The mid set has substantially higher variance: four of its nine held-out suites contain 100 questions, putting the
standard error of the held-out mean at ±0.0097 against ±0.0044 on the full set.

## Same-item comparison with Jev

29 suites, about 129,000 questions. Every model answered identical items. Jev was called through its API.

| Model | Base model | Accuracy (held-out) ↑ | Accuracy (all) ↑ | Brier (held-out) ↓ |
|---|---|---:|---:|---:|
| Jev API | Undisclosed | **0.754** ±0.0042 | 0.790 | **0.355** |
| Malkuth-4B (ours) | Qwen3.5-4B-Base | 0.724 ±0.0044 | **0.794** | 0.385 |
| Malkuth-2B (ours) | Qwen3.8-2B-Distill | 0.703 ±0.0044 | 0.765 | 0.412 |
| Kev-9B | Qwen3.5-9B-Base | 0.700 ±0.0046 | 0.737 | 0.407 |
| Kev-4B | Qwen3.5-4B-Base | 0.686 ±0.0046 | 0.720 | 0.407 |
| Kev-0.8B | Qwen3.5-0.8B-Base | 0.589 ±0.0050 | 0.584 | 0.539 |
| Laya Router | English / multilingual | 0.544 ±0.0044 | 0.507 | 0.654 |
| Laya Multilingual | mmBERT-base | 0.533 ±0.0046 | 0.508 | 0.651 |
| Laya Typed Decisions | ModernBERT-large | 0.510 ±0.0045 | 0.463 | 0.625 |
| Laya English | ModernBERT-large | 0.502 ±0.0044 | 0.442 | 0.718 |

### Calibration

Restricted to the nine held-out suites:

| Model | Accuracy ↑ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|
| Jev API | **0.7539** | **0.3546** | 0.1002 |
| Malkuth-4B | 0.7241 | 0.3849 | 0.0859 |
| Malkuth-2B | 0.7031 | 0.4116 | **0.0852** |
| Kev-4B | 0.6859 | 0.4071 | 0.0786 |

ECE here is averaged across suites, so errors of opposite sign cancel. It also does not track accuracy: Kev-4B has the
lowest ECE of these four while being the least accurate.

Held-out questions bucketed by the probability of the selected option. Jev n=70,562, Malkuth n=68,294; Jev answers a
few long records that the 8,192-token serving context rejects.

| Confidence | Jev observed | Malkuth-4B observed | Malkuth-2B observed |
|---|---:|---:|---:|
| ~0.35 | 0.281 (−0.066) | 0.262 (−0.057) | 0.280 (−0.036) |
| ~0.53 | 0.451 (−0.075) | 0.436 (−0.090) | 0.465 (−0.062) |
| ~0.70 | 0.629 (−0.071) | 0.572 (−0.132) | 0.633 (−0.072) |
| ~0.85 | 0.764 (−0.088) | 0.716 (−0.139) | 0.756 (−0.098) |
| ~0.97 | 0.888 (−0.077) | 0.869 (−0.113) | 0.871 (−0.111) |

Accuracy rises with confidence for all three, and all three are overconfident. Jev's shortfall stays between 0.066 and
0.088. Malkuth-4B's reaches 0.139 in the middle bands. Malkuth-2B is better calibrated than Malkuth-4B despite being
less accurate.

### Per-dataset scores

`M-4B` and `M-2B` are Malkuth, `Laya-R` is Laya's Router. The second column says whether the suite's dataset went into
the Malkuth training mix. Training data for Jev and Laya is unknown.

| Suite | In Malkuth's mix? | n | Jev | M-4B | M-2B | Kev-9B | Kev-4B | Laya-R |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `belebele` | no (held out) | 5,000 | **0.939** | 0.866 | 0.788 | 0.800 | 0.740 | 0.313 |
| `clinc150` | yes, train split | 500 | **0.926** | 0.898 | 0.886 | 0.814 | 0.832 | 0.396 |
| `financial_phrasebank` | no (held out) | 500 | 0.908 | 0.908 | 0.912 | 0.928 | 0.884 | **0.962** |
| `goemotions` | no (held out) | 500 | 0.342 | **0.444** | 0.344 | 0.410 | 0.376 | 0.412 |
| `klue_nli` | yes, train split | 500 | **0.936** | 0.912 | 0.832 | 0.882 | 0.866 | 0.562 |
| `klue_ynat` | yes, train split | 500 | 0.724 | **0.852** | 0.824 | 0.758 | 0.728 | 0.386 |
| `kobest` | yes, train split | 2,397 | 0.903 | **0.904** | 0.819 | 0.850 | 0.809 | 0.542 |
| `kold` | no (held out) | 500 | 0.806 | **0.820** | 0.768 | 0.602 | 0.654 | 0.576 |
| `laya_apps` | related sources | 3,999 | **0.774** | 0.735 | 0.714 | 0.741 | 0.729 | 0.695 |
| `laya_english` | related sources | 2,516 | 0.733 | **0.748** | 0.712 | 0.743 | 0.731 | 0.679 |
| `laya_massive51` | related sources | 6,120 | **0.886** | 0.844 | 0.767 | 0.771 | 0.718 | 0.383 |
| `laya_massive_intent` | related sources | 4,200 | **0.903** | 0.881 | 0.853 | 0.810 | 0.784 | 0.462 |
| `laya_massive_scenario` | related sources | 4,200 | 0.706 | **0.843** | 0.833 | 0.667 | 0.651 | 0.432 |
| `laya_typed_decisions` | yes, train split | 2,000 | 0.736 | **0.759** | 0.723 | 0.722 | 0.653 | 0.360 |
| `laya_xnli` | related sources | 4,500 | 0.746 | **0.821** | 0.752 | 0.786 | 0.767 | 0.727 |
| `ledgar` | no (held out) | 500 | **0.750** | 0.628 | 0.682 | 0.642 | 0.638 | 0.194 |
| `massive_intent` | yes, train split | 5,000 | 0.773 | **0.839** | 0.806 | 0.657 | 0.651 | 0.337 |
| `massive_scenario` | yes, train split | 5,000 | 0.685 | **0.877** | 0.865 | 0.668 | 0.641 | 0.380 |
| `mtop_intent` | related sources | 3,000 | **0.856** | 0.801 | 0.758 | 0.773 | 0.714 | 0.267 |
| `multi_eurlex` | no (held out) | 41,622 | 0.788 | 0.782 | **0.797** | 0.788 | 0.791 | 0.645 |
| `multijail` | related sources | 3,150 | **0.722** | 0.602 | 0.618 | 0.424 | 0.479 | 0.129 |
| `nsmc` | yes, train split | 500 | 0.850 | **0.878** | 0.862 | 0.856 | 0.806 | 0.540 |
| `paws_x` | yes, train split | 3,500 | 0.792 | **0.876** | 0.851 | 0.737 | 0.745 | 0.639 |
| `polyguard` | no (held out) | 14,900 | **0.912** | 0.861 | 0.839 | 0.884 | 0.862 | 0.713 |
| `rtp_lx` | no (held out) | 5,000 | **0.448** | 0.361 | 0.369 | 0.406 | 0.371 | 0.315 |
| `sib200` | no (held out) | 2,040 | **0.893** | 0.845 | 0.828 | 0.840 | 0.857 | 0.764 |
| `toxicchat` | related sources | 1,000 | **0.970** | 0.935 | 0.955 | 0.964 | 0.962 | 0.872 |
| `tweet_sentiment` | yes, train split | 4,000 | **0.699** | 0.696 | 0.669 | 0.651 | 0.660 | 0.518 |
| `xnli` | yes, train split | 4,000 | 0.792 | **0.808** | 0.748 | 0.789 | 0.769 | 0.506 |

Jev's lead concentrates in `ledgar` (+0.122 over Malkuth-4B), `rtp_lx` (+0.087), `belebele` (+0.072) and `polyguard`
(+0.050). LEDGAR is 100-way legal clause classification and RTP-LX is an ordinal toxicity rating, neither of which the
training mix covers. PolyGuard asks whether an assistant's response is harmful, an axis the mix does not contain.

Malkuth-4B's wins sit mostly in suites whose train splits it used. Its held-out wins are `goemotions` (+0.102) and
`kold` (+0.014), both on 500 questions.

## Third-party suites

Test sets built by other projects, with live Jev results published alongside them and shipped in Kev's
`evals/external/`. Malkuth had no part in their construction.

| Suite | Questions | Jev | Malkuth-4B | Malkuth-2B |
|---|---:|---:|---:|---:|
| `semif-v1` (SemIf authored decisions) | 144 | **0.9653** | 0.8681 | 0.8333 |
| `wanli-v1` (WANLI NLI) | 256 | **0.7578** | 0.7422 | 0.7305 |
| `scienthoon-v1` (support tickets) | 873 | **0.7526** | 0.7056 | 0.6174 |

Jev leads all three. On `wanli-v1` Malkuth-4B (0.7422) is ahead of the released Kev-4B (0.695, Kev's own figure).

### TypeSafe public cases

TypeSafe publishes probability distributions for 102 rows over 20 cases, together with two general-purpose models.
Scored with SemIf's protocol (`scripts/compare_typesafe.py`): agreement with the reference answer, and total-variation
distance to the reference distribution, averaged within each case and then over cases.

| System | Rows answered | Agreement ↑ | TVD ↓ |
|---|---:|---:|---:|
| Claude Opus 5 | 102 | **0.9123** | **0.1013** |
| GPT-5.6 Sol | 101 | 0.9065 | 0.1030 |
| Jev (API) | 102 | 0.8915 | 0.1246 |
| Jev (published) | 102 | 0.8831 | 0.1268 |
| Released Kev-4B | 89 | 0.8560 | 0.2230 |
| Malkuth-4B | 89 | 0.8285 | **0.1981** |
| Malkuth-2B | 89 | 0.7579 | 0.2787 |

These are long documents. Kev-4B and Malkuth-4B both answer 89 of 102 rows; the rest exceed the 8,192-token serving
context. Against Kev-4B, Malkuth-4B is 0.028 lower on agreement, inside the noise for 89 rows, and 0.025 better on TVD.
The set covers 20 cases, so it resolves little.

## Reading the aggregates

- **Held-out** is the unweighted mean over nine suites: SIB-200, Belebele, RTP-LX, PolyGuard, MultiEURLEX, GoEmotions,
  Financial PhraseBank, LEDGAR, KOLD. It refers to Malkuth post-training exposure, not the base model's pretraining.
- **Ten suites contributed their train splits** and are evaluated on their test splits: MASSIVE intent, XNLI, PAWS-X,
  tweet sentiment, CLINC150, KLUE-NLI, KLUE-YNAT, NSMC, KoBEST, typed decisions. All-suite numbers include them.
- **`laya_apps`** is excluded from the held-out mean because it carries Kev's and Laya's own training data.
- **`global_mmlu` and `hwu64`** overlap MMMLU and MASSIVE and are excluded from the full set.
- **Aggregates are unweighted means over suites**, so a 100-question suite counts as much as a 15,000-question one.
- **Rejected records** exceed the serving context, produce no prediction and are excluded. Each report states the count.

Dataset inventory and per-source licences: [benchmarks.md](benchmarks.md).
