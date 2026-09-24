# Malkuth

Malkuth is a small multilingual decision model for classification, with a focus on Korean. It returns probabilities
over a fixed set of choices rather than generating text. Two sizes:
[Malkuth-4B](https://huggingface.co/dhtocks/malkuth-4b) and [Malkuth-2B](https://huggingface.co/dhtocks/malkuth-2b).


Malkuth supports three question types:

- `choice`: one of up to 255 options
- `noul`: yes or no
- `score`: an ordered level

Each request carries its own criteria, and can contain several independent questions over the same input.

The models are post-trained from [Kev](https://github.com/jaredpalmer/kev). They are served with Kev and used through
the TypeSafe System One API.

<sub>The name is motivated by my favorite game.</sub>

## Quick start

Install Kev and [uv](https://docs.astral.sh/uv/). The weights download from the Hub.

```bash
git clone https://github.com/jaredpalmer/kev.git kev && cd kev
uv sync --extra serve
uv run --extra serve python -m kev.serve --run dhtocks/malkuth-4b --port 8009
```

Use `dhtocks/malkuth-2b` for the smaller model. The first load also downloads the base model. To reproduce the scores
below, pin `git checkout 557598f`, the Kev revision the evaluations were run on.

In another terminal, send a request:

```bash
curl http://127.0.0.1:8009/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "I was charged twice for the same order. Please refund the duplicate payment.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
          "billing": "Charges, invoices, and payment problems",
          "shipping": "Delivery status, delays, and lost packages",
          "returns": "Exchanges and damaged items"
        }
      }
    }
  }'
```

```json
{
  "department": {
    "type": "choice",
    "choice": "billing",
    "confidence": 0.9917,
    "probabilities": {"billing": 0.9945, "shipping": 0.0018, "returns": 0.0037}
  }
}
```

## Evaluation

29 suites and about 129,000 questions. Every model answered the same items. Jev was measured through its API.

- **held-out**: the nine datasets we did not train on
- **all**: all 29 suites, ten of which we trained on
- **Brier**: quality of the probabilities, lower is better

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

Latency on one A100 80GB in bf16, median of 20 requests. The second number is a repeated input served from the prefix
cache.

| Shape | Malkuth-4B | Malkuth-2B |
|---|---:|---:|
| 2 questions | 37 / 21 ms | 18 / 10 ms |
| 6 questions | 48 / 32 ms | 23 / 16 ms |
| 5 questions, ~370-token state | 58 / 30 ms | 25 / 14 ms |
| 5 questions, ~2,200-token state | 145 / 33 ms | 58 / 15 ms |

Per-dataset scores, calibration analysis and third-party test sets are in [docs/evaluation.md](docs/evaluation.md).

## License

- **Code**: [Apache-2.0](LICENSE), same as Kev and the Qwen bases.
- **Weights**: research use only. The training mix includes XNLI (CC-BY-NC-4.0) and RACE (non-commercial research
  only).
- **Data under `benchmarks/`**: each suite keeps its source licence, and Apache-2.0 does not apply there. Four suites
  from non-commercial sources are not checked in; [benchmarks/README.md](benchmarks/README.md) rebuilds them.

## Citation

```bibtex
@misc{malkuth2026,
  title={Malkuth: multilingual System One decision models},
  author={Saechan Oh},
  year={2026}
}
```
