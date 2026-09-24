#!/usr/bin/env python3
"""Validation set for checkpoint / data-version selection -> benchmarks/val/<suite>/ (Kev suite layout).

    uv run --no-project --with datasets python tools/build_val.py

Selection happens here; benchmarks/lite, benchmarks/mid and benchmarks/full are for final reporting only.
Sources never overlap the evaluated splits:
  GROUP_A proxies  the benchmark's own *validation* split (eval items come from test; training from train)
  GROUP_B proxies  held-out splits of the non-benchmark datasets that target those skills (RACE -> belebele,
                   civil_comments -> rtp_lx, APEACH -> kold); no GROUP_B benchmark data is touched
Same instruction wording and option names as the benchmark suites, so scores are comparable in kind.
"""
import random, sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_kev_benchmarks as bkb
from build_kev_benchmarks import ABCD, FLORES, INTENT, choice, human, noul, nli

bkb.OUT = Path(__file__).resolve().parent.parent / "benchmarks/val"
PARTS = 50


def massive_intent():
    en = load_dataset("mteb/amazon_massive_intent", "en")
    keys = sorted(set(en["train"]["label"]) | set(en["test"]["label"]))
    codes = {**{k: k for k in FLORES}, "zh": "zh-CN"}
    return "mteb/amazon_massive_intent:validation", "cc-by-4.0", [
        choice(ex["text"], INTENT, keys, ex["label"], f"val_massive_{lang}", _part=lang, _key=ex["id"])
        for lang, code in codes.items() for ex in load_dataset("mteb/amazon_massive_intent", code, split="validation")]


def xnli():
    out = []
    for lang in ("en", "zh", "de", "fr", "es", "hi", "ar", "th"):
        ds = load_dataset("facebook/xnli", lang, split="validation"); names = ds.features["label"].names
        out += [nli(ex["premise"], ex["hypothesis"], names[ex["label"]], f"val_xnli_{lang}", _part=lang, _key=i) for i, ex in enumerate(ds)]
    return "facebook/xnli:validation", "cc-by-nc-4.0", out


def paws_x():
    crit = {"true": "Same meaning, possibly reworded", "false": "Different meaning, even if most words match"}
    return "google-research-datasets/paws-x:validation", "other", [
        {"state": ex["sentence1"], "questions": {"paraphrase": noul(f'Does this sentence mean the same thing: "{ex["sentence2"]}"', ex["label"] == 1,
                                                                   f"val_paws_{lang}", crit)}, "_part": lang, "_key": ex["id"]}
        for lang in ("en", "ko", "ja", "zh", "de", "fr", "es") for ex in load_dataset("google-research-datasets/paws-x", lang, split="validation")]


def clinc150():
    ds = load_dataset("clinc/clinc_oos", "plus", split="validation"); names = ds.features["intent"].names
    return "clinc/clinc_oos:plus/validation", "cc-by-3.0", [
        choice(ex["text"], INTENT, names, names[ex["intent"]], "val_clinc150", {"oos": "The request matches none of the other intents"}) for ex in ds]


def tweet_sentiment():
    names = ["negative", "neutral", "positive"]
    langs = {"en": "english", "de": "german", "fr": "french", "es": "spanish", "hi": "hindi", "ar": "arabic", "it": "italian", "pt": "portuguese"}
    return "cardiffnlp/tweet_sentiment_multilingual:validation", "cc-by-3.0", [
        choice(ex["text"], "What is the sentiment of this tweet?", names, names[ex["label"]], f"val_tweet_{lang}", _part=lang)
        for lang, name in langs.items()
        for ex in load_dataset("parquet", split="train", data_files=f"hf://datasets/cardiffnlp/tweet_sentiment_multilingual@refs%2Fconvert%2Fparquet/{name}/validation/*.parquet")]


def kobest():
    out = []
    for ex in load_dataset("skt/kobest_v1", "boolq", split="validation"):
        out.append({"state": ex["paragraph"], "_part": "boolq", "questions": {"answer": noul(f"Based on the passage, is the answer to this question yes: {ex['question']}", ex["label"] == 1, "val_kobest_boolq")}})
    for ex in load_dataset("skt/kobest_v1", "copa", split="validation"):
        kind = "cause" if ex["question"] == "원인" else "effect"
        out.append(choice(ex["premise"], f"Which option is the more plausible {kind} of the situation?", ["a", "b"], "ab"[ex["label"]], "val_kobest_copa",
                          {"a": ex["alternative_1"], "b": ex["alternative_2"]}, _part="copa"))
    for ex in load_dataset("skt/kobest_v1", "wic", split="validation"):
        out.append({"state": {"sentence 1": ex["context_1"], "sentence 2": ex["context_2"]}, "_part": "wic",
                    "questions": {"same": noul(f'Is the word "{ex["word"]}" used with the same meaning in both sentences?', ex["label"] == 1, "val_kobest_wic")}})
    for ex in load_dataset("skt/kobest_v1", "sentineg", split="validation"):
        out.append({"state": ex["sentence"], "_part": "sentineg", "questions": {"positive": noul("Is the sentiment of this sentence positive?", ex["label"] == 1, "val_kobest_sentineg")}})
    return "skt/kobest_v1:validation", "cc-by-sa-4.0", out


def race():   # belebele proxy
    return "ehovy/race:validation", "other", [
        choice({"passage": ex["article"], "question": ex["question"]}, "Which option correctly answers the question, based on the passage?", ABCD,
               ex["answer"].lower(), "val_race", dict(zip(ABCD, ex["options"]))) for ex in load_dataset("ehovy/race", "all", split="validation")]


def civil_toxicity():   # rtp_lx proxy: same 5-level Score as the training records and the rtp_lx suite
    levels = ["not toxic", "slightly toxic", "moderately toxic", "very toxic", "extremely toxic"]
    cuts, per, buckets = [0.1, 0.3, 0.5, 0.7], 60, {i: [] for i in range(5)}
    for ex in load_dataset("google/civil_comments", split="validation"):
        lvl = sum(ex["toxicity"] >= c for c in cuts)
        if len(buckets[lvl]) < per and ex["text"].strip(): buckets[lvl].append(ex["text"][:2000])
    return "google/civil_comments:validation", "cc0-1.0", [
        {"state": t, "questions": {"toxicity": {"type": "score", "instructions": "How toxic is this text?", "criteria": levels, "label": lvl, "src": "val_civil"}}}
        for lvl, ts in buckets.items() for t in ts]


def apeach():   # kold proxy
    return "jason9693/APEACH:test", "cc-by-sa-4.0", [
        {"state": ex["text"], "questions": {"offensive": noul("Is this comment offensive?", ex["class"] == 1, "val_apeach")}}
        for ex in load_dataset("jason9693/APEACH", split="test")]


SUITES = {"val_massive_intent": massive_intent, "val_xnli": xnli, "val_paws_x": paws_x, "val_clinc150": clinc150,
          "val_tweet_sentiment": tweet_sentiment, "val_kobest": kobest, "val_race": race, "val_civil_toxicity": civil_toxicity,
          "val_apeach": apeach}
SIZES = {"val_clinc150": 200, "val_race": 200, "val_civil_toxicity": 0, "val_apeach": 200}


def main():
    for name, fn in SUITES.items():
        try: source, rights, records = fn()
        except Exception as e: print(f"SKIP {name}: {type(e).__name__}: {str(e)[:160]}", flush=True); continue
        print(f"{name}: {bkb.freeze(name, source, rights, records, SIZES.get(name, PARTS), 0)}", flush=True)


if __name__ == "__main__":
    main()
