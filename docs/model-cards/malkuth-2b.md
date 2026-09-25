---
base_model: empero-ai/Qwen3.8-2B-Distill
base_model_relation: adapter
library_name: peft
pipeline_tag: zero-shot-classification
license: cc-by-nc-4.0
language: [en, ko, ja, zh, de, fr, es, hi, ar, th]
tags: [decision-model, classification, calibration, kev, system-one, lora, pointer-head]
---

# Malkuth-2B

A small multilingual decision model for classification, with a focus on Korean.

Given a text, a question, and a set of criteria, Malkuth returns probabilities over the possible choices instead of
generating text.

Malkuth-2B is a LoRA adapter trained on [Kev](https://github.com/jaredpalmer/kev) and built on a frozen
`empero-ai/Qwen3.8-2B-Distill`. It has 17.9M trainable parameters and a 65 MB adapter.

It supports three question types:

- `choice`: select one of up to 255 options
- `noul`: yes or no
- `score`: select an ordered level

Multiple questions can be evaluated against the same input. Questions are evaluated independently, so they cannot see
each other's answers. Probabilities are temperature-calibrated on a held-out validation split.

Malkuth is served through Kev and works with the TypeSafe System One API. The companion model is
[dhtocks/malkuth-4b](https://huggingface.co/dhtocks/malkuth-4b) (4B). Code, training mixes and evaluation live in
[newfull5/malkuth](https://github.com/newfull5/malkuth).

## Usage

```bash
git clone https://github.com/jaredpalmer/kev.git kev && cd kev && uv sync --extra serve
uv run --extra serve python -m kev.serve --run dhtocks/malkuth-2b --port 8009
```

The official `typesafe-sdk` works against the local server with a `base_url` change.

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009", model="kev-latest")
r = client.system_one(
    state="Shoes arrived two weeks late and in the wrong size. I also see two charges on my card.",
    questions={
        "team": Choice(instructions="Which team should handle this?",
                       criteria={"billing": "Charges and invoices", "shipping": "Delivery problems",
                                 "returns": "Exchanges and wrong items"}),
        "escalate": Noul(instructions="Does this need urgent human attention?"),
        "urgency": Score(instructions="How urgent is this ticket?", criteria=["low", "normal", "high"]),
    },
)
print(r.choices["team"].choice, r.choices["team"].probabilities)
```

## Evaluation

29 suites and about 129,000 questions. Every model answered the same items. Jev was measured through its API.

- **held-out**: the nine datasets we did not train on
- **all**: all 29 suites, ten of which we trained on
- **Brier**: quality of the probabilities, lower is better

| Model | Accuracy (held-out) | Accuracy (all) | Brier (held-out) |
|---|---:|---:|---:|
| Jev API | **0.754** ±0.0042 | 0.790 | **0.355** |
| Malkuth-4B | 0.724 ±0.0044 | **0.794** | 0.385 |
| Malkuth-2B | 0.703 ±0.0044 | 0.765 | 0.412 |
| Kev-9B | 0.700 ±0.0046 | 0.737 | 0.407 |
| Kev-4B | 0.686 ±0.0046 | 0.720 | 0.407 |
| Kev-0.8B | 0.589 ±0.0050 | 0.584 | 0.539 |
| Laya (best of four) | 0.544 ±0.0044 | 0.507 | 0.654 |

On the five Korean suites:

| Suite | Jev | Malkuth-4B | Malkuth-2B | Kev-9B | Kev-4B |
|---|---:|---:|---:|---:|---:|
| `klue_nli` | **0.936** | 0.912 | 0.832 | 0.882 | 0.866 |
| `klue_ynat` | 0.724 | **0.852** | 0.824 | 0.758 | 0.728 |
| `nsmc` | 0.850 | **0.878** | 0.862 | 0.856 | 0.806 |
| `kobest` | 0.903 | **0.904** | 0.819 | 0.850 | 0.809 |
| `kold` | 0.806 | **0.820** | 0.768 | 0.602 | 0.654 |
| mean | 0.844 | **0.873** | 0.821 | 0.790 | 0.773 |

Four of these five contributed their train splits to our mix, so this table is in-distribution for Malkuth.

Latency on one A100 80GB in bf16, median of 20 requests. The second number is a repeated input served from the prefix
cache. Six questions take 23 ms on a new input and 16 ms on a repeated one.

Per-dataset scores and calibration analysis are in
[docs/evaluation.md](https://github.com/newfull5/malkuth/blob/main/docs/evaluation.md).

## Training data

89,791 labelled requests from 20 public sources. 62,400 `choice`, 24,790 `noul` and 7,401 `score` questions, across ten
core languages with five Korean-only sources.

Built by [`tools/build_train_mix.py`](https://github.com/newfull5/malkuth/blob/main/tools/build_train_mix.py).
Per-source licences are in
[docs/benchmarks.md](https://github.com/newfull5/malkuth/blob/main/docs/benchmarks.md).

## License

- Weights: research use only. Two of the twenty training sources are non-commercial: XNLI (CC-BY-NC-4.0) and
  RACE ("non-commercial research purpose only"). `sms_spam` and `tweet_sentiment_multilingual` state no licence.
- Code: Apache-2.0, same as Kev and the Qwen bases.

## Citation

```bibtex
@misc{malkuth2026,
  title={Malkuth: multilingual System One decision models},
  author={Saechan Oh},
  year={2026}
}
```
