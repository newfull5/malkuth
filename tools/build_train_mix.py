#!/usr/bin/env python3
"""Training mix v1: multilingual, benchmark-aware, in Kev's --data format (labelled System One requests).

    uv run --no-project --with datasets python tools/build_train_mix.py        # -> data/train_mix_v1/train.jsonl
    cd kev && uv run python -m kev.train --data ../data/train_mix_v1/train.jsonl --suite evals/v7/decision-v7 --replay 6000 ...

Benchmark policy (decided 2026-09-23: "use half"): the lite suites are split in two groups.
  GROUP_A  train splits may be used (never the split the eval items were drawn from)
  GROUP_B  held out entirely: no data from these datasets, measured as generalisation
Every source below is either a GROUP_A train split or a dataset that is in no benchmark at all.
Instructions: half the records use the benchmark's own wording, half a paraphrase, so the model learns the task and not
one template. Many-label intent questions show the full label set most of the time and a random subset otherwise.
"""
import csv, io, json, random, sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_kev_benchmarks import ABCD, FLORES, INTENT, MNLI, human, lines   # same option names / wording as the suites

GROUP_A = ["massive_intent", "xnli", "paws_x", "tweet_sentiment", "clinc150", "klue_nli", "klue_ynat", "nsmc", "kobest",
           "laya_typed_decisions"]
GROUP_B = ["sib200", "belebele", "rtp_lx", "polyguard", "multi_eurlex", "goemotions", "financial_phrasebank", "ledgar", "kold",
           "laya_apps"]
OUT = Path(__file__).resolve().parent.parent / "data" / "train_mix_v1"
R = random.Random(0)


def pick(eval_wording, paraphrases):
    return eval_wording if R.random() < 0.5 else R.choice(paraphrases)


def sample(ds, n):
    idx = list(range(len(ds))); R.shuffle(idx)
    return [ds[i] for i in idx[:n]]


def choice_q(instr, keys, label, src, desc=None, full=True):
    """keys: option names (already human). full=False -> gold + a random subset of the others, shuffled."""
    if not full:
        others = [k for k in keys if k != label]
        keys = [label] + R.sample(others, min(len(others), R.randint(4, 19))); R.shuffle(keys)
    return {"type": "choice", "instructions": instr, "criteria": {k: (desc or {}).get(k) for k in keys}, "label": label, "src": src}


def noul_q(instr, label, src, criteria=None):
    return {"type": "noul", "instructions": instr, "label": bool(label), "src": src, **({"criteria": criteria} if criteria else {})}


def rec(state, q, source, qid="label"):
    return {"state": state, "questions": {qid: q}, "_meta": {"source": source}}


INTENT_P = ["What does the user want?", "Classify the intent of this request.", "Which intent matches this utterance?",
            "Pick the assistant intent for this message."]
NLI_P = ['Premise and hypothesis: does the premise support the hypothesis "{h}"?', 'How does this statement relate to the text: "{h}"',
         'Hypothesis: "{h}" What is its relation to the premise?']


# ---------- GROUP_A train splits ----------

def massive_intent(n=1000):
    codes = {**{k: k for k in FLORES}, "zh": "zh-CN"}
    en = load_dataset("mteb/amazon_massive_intent", "en")
    keys = sorted({human(x) for x in set(en["train"]["label"]) | set(en["test"]["label"])})
    out = []
    for lang, code in codes.items():
        for ex in sample(load_dataset("mteb/amazon_massive_intent", code, split="train"), n):
            out.append(rec(ex["text"], choice_q(pick(INTENT, INTENT_P), keys, human(ex["label"]), f"mix_massive_{lang}", full=R.random() < 0.7),
                           f"massive_intent/{lang}"))
    return out


def massive_scenario(n=300):   # scenario is not a lite suite; topic-like signal for the held-out sib200
    codes = {**{k: k for k in FLORES}, "zh": "zh-CN"}
    en = load_dataset("mteb/amazon_massive_scenario", "en")
    keys = sorted({human(x) for x in set(en["train"]["label"])})
    return [rec(ex["text"], choice_q(pick("Which scenario does this user request belong to?", ["What domain is this request about?", "Which area does this request concern?"]),
                                     keys, human(ex["label"]), f"mix_scenario_{lang}"), f"massive_scenario/{lang}")
            for lang, code in codes.items() for ex in sample(load_dataset("mteb/amazon_massive_scenario", code, split="train"), n)]


def xnli(n=800):
    out = []
    for lang in ("en", "zh", "de", "fr", "es", "hi", "ar", "th"):
        ds = load_dataset("facebook/xnli", lang, split="train")
        names = ds.features["label"].names
        for ex in sample(ds, n):
            h = ex["hypothesis"]
            instr = f'Hypothesis: "{h}" How does it relate to the premise?' if R.random() < 0.5 else R.choice(NLI_P).format(h=h)
            out.append(rec(ex["premise"], choice_q(instr, list(MNLI), names[ex["label"]], f"mix_xnli_{lang}", MNLI), f"xnli/{lang}"))
    return out


def paws_x(n=700):
    crit = {"true": "Same meaning, possibly reworded", "false": "Different meaning, even if most words match"}
    out = []
    for lang in ("en", "ko", "ja", "zh", "de", "fr", "es"):
        for ex in sample(load_dataset("google-research-datasets/paws-x", lang, split="train"), n):
            s2 = ex["sentence2"]
            instr = pick(f'Does this sentence mean the same thing: "{s2}"', [f'Is "{s2}" a paraphrase of this sentence?', f'Do these say the same: "{s2}"'])
            out.append(rec(ex["sentence1"], noul_q(instr, ex["label"] == 1, f"mix_paws_{lang}", crit if R.random() < 0.5 else None), f"paws_x/{lang}"))
    return out


def tweet_sentiment(n=800):
    names = ["negative", "neutral", "positive"]
    langs = {"en": "english", "de": "german", "fr": "french", "es": "spanish", "hi": "hindi", "ar": "arabic", "it": "italian", "pt": "portuguese"}
    out = []
    for lang, name in langs.items():
        ds = load_dataset("parquet", split="train", data_files=f"hf://datasets/cardiffnlp/tweet_sentiment_multilingual@refs%2Fconvert%2Fparquet/{name}/train/*.parquet")
        for ex in sample(ds, n):
            out.append(rec(ex["text"], choice_q(pick("What is the sentiment of this tweet?", ["How does the author feel?", "Classify the sentiment of this post."]),
                                                names, names[ex["label"]], f"mix_tweet_{lang}"), f"tweet_sentiment/{lang}"))
    return out


def clinc150(n=3000):
    ds = load_dataset("clinc/clinc_oos", "plus", split="train")
    names = [human(x) for x in ds.features["intent"].names]
    desc = {"oos": "The request matches none of the other intents"}
    return [rec(ex["text"], choice_q(pick(INTENT, INTENT_P), names, names[ex["intent"]], "mix_clinc150", desc, full=R.random() < 0.7), "clinc150")
            for ex in sample(ds, n)]


def klue(n=3000):
    ynat = load_dataset("klue/klue", "ynat", split="train"); yn = ynat.features["label"].names
    nli = load_dataset("klue/klue", "nli", split="train"); nn = nli.features["label"].names
    out = [rec(ex["title"], choice_q(pick("What is the topic of this news headline?", ["Which news section does this headline belong to?"]), yn, yn[ex["label"]], "mix_klue_ynat"), "klue_ynat")
           for ex in sample(ynat, n)]
    for ex in sample(nli, n):
        h = ex["hypothesis"]
        instr = f'Hypothesis: "{h}" How does it relate to the premise?' if R.random() < 0.5 else R.choice(NLI_P).format(h=h)
        out.append(rec(ex["premise"], choice_q(instr, list(MNLI), nn[ex["label"]], "mix_klue_nli", MNLI), "klue_nli"))
    return out


def nsmc(n=3000):
    rows = [r for r in csv.DictReader(io.StringIO(lines("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt")), delimiter="\t", quoting=csv.QUOTE_NONE) if r["document"]]
    R.shuffle(rows)
    return [rec(r["document"], noul_q(pick("Is this movie review positive?", ["Does the reviewer like the movie?", "Is the sentiment of this review positive?"]),
                                      r["label"] == "1", "mix_nsmc"), "nsmc") for r in rows[:n]]


def kobest(n=1000):
    out = []
    for ex in sample(load_dataset("skt/kobest_v1", "boolq", split="train"), n):
        out.append(rec(ex["paragraph"], noul_q(f"Based on the passage, is the answer to this question yes: {ex['question']}", ex["label"] == 1, "mix_kobest_boolq"), "kobest/boolq"))
    for ex in sample(load_dataset("skt/kobest_v1", "copa", split="train"), n):
        kind = "cause" if ex["question"] == "원인" else "effect"
        out.append(rec(ex["premise"], choice_q(f"Which option is the more plausible {kind} of the situation?", ["a", "b"], "ab"[ex["label"]], "mix_kobest_copa",
                                               {"a": ex["alternative_1"], "b": ex["alternative_2"]}), "kobest/copa"))
    for ex in sample(load_dataset("skt/kobest_v1", "wic", split="train"), n):
        out.append(rec({"sentence 1": ex["context_1"], "sentence 2": ex["context_2"]},
                       noul_q(f'Is the word "{ex["word"]}" used with the same meaning in both sentences?', ex["label"] == 1, "mix_kobest_wic"), "kobest/wic"))
    for ex in sample(load_dataset("skt/kobest_v1", "hellaswag", split="train"), n):
        out.append(rec(ex["context"], choice_q("Which ending most plausibly continues the text?", ABCD, ABCD[ex["label"]], "mix_kobest_hellaswag",
                                               {k: ex[f"ending_{i + 1}"] for i, k in enumerate(ABCD)}), "kobest/hellaswag"))
    for ex in sample(load_dataset("skt/kobest_v1", "sentineg", split="train"), n):
        out.append(rec(ex["sentence"], noul_q("Is the sentiment of this sentence positive?", ex["label"] == 1, "mix_kobest_sentineg"), "kobest/sentineg"))
    return out


def typed_decisions():
    """LocalLLaMA/typed-decisions train (1,200 cases): teacher labels + soft distributions, target = ½ label + ½ teacher."""
    out = []
    for r in load_dataset("LocalLLaMA/typed-decisions", "all", split="train"):
        qs = json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
        gold = json.loads(r["gold"]) if isinstance(r["gold"], str) else r["gold"]
        state = r["state"]
        try: state = json.loads(state)
        except Exception: pass
        questions = {}
        for qid, q in qs.items():
            g, t = gold[qid], q["type"]
            probs = g.get("probabilities") or {}
            if t == "choice":
                label = str(g["label"]); keys = list(q["criteria"])
            elif t == "noul":
                label = str(g["label"]).lower() == "true"; keys = ["false", "true"]
                probs = {"true": float(probs.get("true", g.get("noul", 0.5)))}; probs["false"] = 1 - probs["true"]
            else:
                label = int(g["label"]); keys = [str(i) for i in range(len(q["criteria"]))]
            hard = {k: float(k == (str(label).lower() if t == "noul" else str(label))) for k in keys}
            target = {k: 0.5 * hard[k] + 0.5 * float(probs.get(k, 0.0)) for k in keys} if probs else None
            questions[qid] = {**q, "label": label, "src": f"mix_td_{r['workflow']}", **({"target": target} if target else {})}
        out.append({"state": state, "questions": questions, "_meta": {"source": f"typed_decisions/{r['workflow']}"}})
    return out


# ---------- sources in no benchmark (skills behind GROUP_B suites) ----------

def race(n=4000):
    ds = load_dataset("ehovy/race", "all", split="train")
    return [rec({"passage": ex["article"], "question": ex["question"]},
                choice_q(pick("Which option correctly answers the question, based on the passage?", ["Choose the answer supported by the passage.", "Answer the question using the passage."]),
                         ABCD, ex["answer"].lower(), "mix_race", dict(zip(ABCD, ex["options"]))), "race") for ex in sample(ds, n)]


def arc(n=1500):
    out = []
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        for ex in sample(load_dataset("allenai/ai2_arc", cfg, split="train"), n // 2):
            labels, texts = ex["choices"]["label"], ex["choices"]["text"]
            if ex["answerKey"] not in labels or len(labels) > 5: continue
            keys = [l.lower() for l in labels]
            out.append(rec({"question": ex["question"]}, choice_q("Which option correctly answers the question?", keys, ex["answerKey"].lower(), "mix_arc",
                                                                  dict(zip(keys, texts))), "arc"))
    return out


def toxicity(n=500):
    langs = ("en", "de", "es", "zh", "ar", "hi", "fr", "ja", "it", "ru")
    d = load_dataset("textdetox/multilingual_toxicity_dataset")
    out = []
    for lang in langs:
        for ex in sample(d[lang], n):
            instr = pick("Is this text toxic, harmful, or offensive?", ["Does this message contain insults, hate, or abuse?", "Is this post toxic?"])
            out.append(rec(ex["text"], noul_q(instr, ex["toxic"] == 1, f"mix_toxic_{lang}"), f"textdetox/{lang}"))
    return out


# ---------- v2 additions (sources in no lite benchmark) ----------

def mmmlu(n=500):
    """Multilingual 4-way MC (MMLU translated by OpenAI): the answer-from-options format belebele needs, in 9 languages.
    Note: shares questions with global_mmlu (a full-set suite, not in lite/mid), which is contaminated from v2 on."""
    langs = {"ar": "AR_XY", "de": "DE_DE", "es": "ES_LA", "fr": "FR_FR", "hi": "HI_IN", "ja": "JA_JP", "ko": "KO_KR", "zh": "ZH_CN", "it": "IT_IT"}
    return [rec({"subject": human(ex["Subject"]), "question": ex["Question"]},
                choice_q("Which option correctly answers the question?", ABCD, ex["Answer"].lower(), f"mix_mmmlu_{lang}", {k: ex[k.upper()] for k in ABCD}),
                f"mmmlu/{lang}") for lang, cfg in langs.items() for ex in sample(load_dataset("openai/MMMLU", cfg, split="test"), n)]


def spam(n=1500):
    return [rec(ex["sms"], noul_q(pick("Is this message spam?", ["Is this unsolicited spam or a scam?", "Is this bulk marketing or spam?"]), ex["label"] == 1, "mix_spam"), "sms_spam")
            for ex in sample(load_dataset("ucirvine/sms_spam", split="train"), n)]


def jailbreak():
    out = [rec(ex["prompt"][:3000], noul_q(pick("Is this prompt trying to jailbreak an AI assistant?", ["Does this prompt try to make an AI ignore its rules?"]),
                                           ex["type"] == "jailbreak", "mix_jailbreak"), "jailbreak_classification")
           for ex in load_dataset("jackhhao/jailbreak-classification", split="train")]
    out += [rec(ex["text"], noul_q(pick("Does this text try to inject or override instructions given to an AI system?", ["Is this a prompt injection attempt?"]),
                                   ex["label"] == 1, "mix_injection"), "prompt_injections") for ex in load_dataset("deepset/prompt-injections", split="train")]
    return out


V2 = {"mmmlu": mmmlu, "spam": spam, "jailbreak": jailbreak}


# ---------- v3 additions: the held-out GROUP_B gaps (ledgar, rtp_lx, kold), from datasets in no benchmark ----------

TOX_LEVELS = ["not toxic", "slightly toxic", "moderately toxic", "very toxic", "extremely toxic"]   # the rtp_lx scale wording


def civil_toxicity(total=5000):
    """civil_comments' annotator toxicity fraction bucketed into five ordinal levels (Score). Levels are sampled in
    proportion to sqrt(natural frequency): most real text is not toxic (a uniform mix taught v3 models to over-rate
    toxicity, +0.35 levels on average), while the rare high levels still get enough examples."""
    ds = load_dataset("google/civil_comments", split="train[:400000]")
    cuts = [0.1, 0.3, 0.5, 0.7]
    buckets = {i: [] for i in range(5)}
    for ex in ds:
        if ex["text"].strip(): buckets[sum(ex["toxicity"] >= c for c in cuts)].append(ex["text"])
    weight = {l: len(t) ** 0.5 for l, t in buckets.items()}
    quota = {l: round(total * w / sum(weight.values())) for l, w in weight.items()}
    print(f"civil_toxicity natural {dict((l, len(t)) for l, t in buckets.items())} -> quota {quota}", flush=True)
    out = []
    for lvl, texts in buckets.items():
        R.shuffle(texts)
        for t in texts[:quota[lvl]]:
            q = {"type": "score", "instructions": pick("How toxic is this text?", ["Rate how offensive this comment is.", "How toxic is this comment?"]),
                 "criteria": TOX_LEVELS, "label": lvl, "src": "mix_civil_toxicity"}
            out.append(rec(t[:2000], q, "civil_comments"))
    return out


def apeach(n=4000):
    return [rec(ex["text"], noul_q(pick("Is this comment offensive?", ["Is this text hateful or insulting?", "Does this comment contain hate speech?"]), ex["class"] == 1,
                                   "mix_apeach"), "apeach") for ex in sample(load_dataset("jason9693/APEACH", split="train"), n)]


def cuad(n=5000):
    ds = load_dataset("dvgodoy/CUAD_v1_Contract_Understanding_clause_classification", split="train")
    names = sorted(set(ds["label"]))
    return [rec(ex["clause"][:3000], choice_q(pick("Which contract provision type is this clause?", ["What kind of contract clause is this?", "Classify this contract clause."]),
                                              names, ex["label"], "mix_cuad", full=R.random() < 0.7), "cuad") for ex in sample(ds, n)]


V3 = {"civil_toxicity": civil_toxicity, "apeach": apeach, "cuad": cuad}


# ---------- v4: multilingual reading comprehension as 4-way choice (belebele's format; XQuAD is in no benchmark) ----------

def xquad_mc(per_lang=800):
    """Extractive QA -> 4 options: the gold span plus three answers to other questions about the same paragraph, so
    every distractor is a plausible span of the passage. Paragraphs with fewer than four distinct answers are skipped.
    KorQuAD (CC-BY-ND) is deliberately not used: converting it would be an adaptation."""
    out = []
    for lang in ("en", "de", "es", "ar", "hi", "th", "zh", "vi", "ru", "tr", "el"):
        ds = load_dataset("google/xquad", f"xquad.{lang}", split="validation")
        by_ctx = {}
        for ex in ds: by_ctx.setdefault(ex["context"], []).append(ex)
        items = []
        for ctx, qs in by_ctx.items():
            answers = list(dict.fromkeys(q["answers"]["text"][0].strip() for q in qs))
            if len(answers) < 4: continue
            for q in qs:
                gold = q["answers"]["text"][0].strip()
                opts = [gold] + R.sample([a for a in answers if a != gold], 3); R.shuffle(opts)
                items.append(rec({"passage": ctx, "question": q["question"]},
                                 choice_q(pick("Which option correctly answers the question, based on the passage?", ["Choose the answer supported by the passage.", "Answer the question using the passage."]),
                                          ABCD, ABCD[opts.index(gold)], f"mix_xquad_{lang}", dict(zip(ABCD, opts))), f"xquad/{lang}"))
        R.shuffle(items); out += items[:per_lang]
    return out


V4 = {"xquad_mc": xquad_mc}


# ---------- v5: response-level safety (polyguard asks about the assistant's reply; nothing before covered it) ----------
# PolyGuard's own training mix / WildGuard train are the polyguard benchmark's source family and are not used.

def guard(state_user, state_assistant, qs, source):
    state = {"user": state_user, "assistant": state_assistant} if state_assistant else state_user
    return {"state": state, "questions": qs, "_meta": {"source": source}}


def aegis(n=6000):
    out = []
    for ex in sample(load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train"), n):
        qs = {}
        if ex["prompt_label"] in ("safe", "unsafe"):
            qs["prompt_harmful"] = noul_q("Is the user's request harmful?", ex["prompt_label"] == "unsafe", "mix_aegis_prompt")
        resp = ex["response"] if ex["response"] and ex["response_label"] in ("safe", "unsafe") else None
        if resp: qs["response_harmful"] = noul_q("Is the assistant's response harmful?", ex["response_label"] == "unsafe", "mix_aegis_response")
        if qs: out.append(guard(ex["prompt"][:3000], resp[:3000] if resp else None, qs, "aegis"))
    return out


def beavertails(n=4000):
    """Research use only (CC-BY-NC-4.0)."""
    return [guard(ex["prompt"][:3000], ex["response"][:3000],
                  {"response_harmful": noul_q("Is the assistant's response harmful?", not ex["is_safe"], "mix_beavertails")}, "beavertails")
            for ex in sample(load_dataset("PKU-Alignment/BeaverTails", split="30k_train"), n)]


V5 = {"aegis": aegis, "beavertails": beavertails}

SOURCES = {"massive_intent": massive_intent, "massive_scenario": massive_scenario, "xnli": xnli, "paws_x": paws_x,
           "tweet_sentiment": tweet_sentiment, "clinc150": clinc150, "klue": klue, "nsmc": nsmc, "kobest": kobest,
           "typed_decisions": typed_decisions, "race": race, "arc": arc, "toxicity": toxicity}


def main():
    global OUT
    version = sys.argv[1] if len(sys.argv) > 1 else "v1"
    v5 = {k: f for k, f in {**SOURCES, **V2, **V3, **V4, **V5}.items() if k != "cuad"}   # CUAD's 41 clause types hurt ledgar (0.65 -> 0.63 -> 0.59)
    sources = {"v1": SOURCES, "v2": {**SOURCES, **V2}, "v3": {**SOURCES, **V2, **V3}, "v4": {**SOURCES, **V2, **V3, **V4}, "v5": v5}[version]
    OUT = OUT.parent / f"train_mix_{version}"
    OUT.mkdir(parents=True, exist_ok=True)
    records, counts = [], {}
    for name, fn in sources.items():
        try: rs = fn()
        except Exception as e: print(f"SKIP {name}: {type(e).__name__}: {str(e)[:160]}", flush=True); continue
        records += rs; counts[name] = len(rs)
        print(f"{name:18s} {len(rs):6d}", flush=True)
    R.shuffle(records)
    with (OUT / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=True) + "\n")   # kev.data.load_records uses splitlines(), which also splits on U+2028 inside strings
    qtypes = Counter(q["type"] for r in records for q in r["questions"].values())
    (OUT / "manifest.json").write_text(json.dumps({"records": len(records), "by_source": counts, "question_types": qtypes,
                                                   "group_a_train_splits_used": GROUP_A, "group_b_held_out": GROUP_B, "seed": 0},
                                                  indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"TOTAL {len(records)} records, {sum(qtypes.values())} questions {dict(qtypes)} -> {OUT / 'train.jsonl'}")


if __name__ == "__main__":
    main()
