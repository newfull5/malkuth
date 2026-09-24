#!/usr/bin/env python3
"""Public classification benchmarks Kev never trained on, frozen as Kev eval-only suites (development partition only).

    uv run --no-project --with datasets python tools/build_kev_benchmarks.py                 # every benchmark, 500 per language
    uv run --no-project --with datasets python tools/build_kev_benchmarks.py sib200 --n 0   # full test set
    cd kev && uv run python -m kev.benchmark --run jaredpalmer/kev-4b --suite ../benchmarks/sib200 --out runs/b-sib200

Output: benchmarks/<name>/{development,train,calibration,test}.jsonl + manifest.json (same layout as kev/evals/external/*).
Multilingual suites put the language in each question's `src` (e.g. sib200_ko), so report.json["tasks"] is per language.
Parallel corpora are sampled by item: the same items in every language, linked by _meta.group_id.
"""
import argparse, csv, hashlib, io, json, random, urllib.request
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "benchmarks/full"
# SERVING_CONTEXT in kev/suite.py: long documents exceed the 384-token training context; overlong rows are rejected
# and counted in report["coverage"], never truncated.
CONTEXT = {"max_state": 8192, "max_branch": 8192, "max_packed": 16384, "truncate": False}
INTENT = "Which assistant intent best describes this user request?"
MNLI = {"entailment": "The hypothesis follows from the premise", "neutral": "The hypothesis may or may not be true given the premise",
        "contradiction": "The hypothesis contradicts the premise"}   # kev.data.MNLI, the wording Kev trained on
ABCD = ["a", "b", "c", "d"]
# ISO code -> each corpus's own language code; the core set is en + ko + the major scripts
FLORES = {"en": "eng_Latn", "ko": "kor_Hang", "ja": "jpn_Jpan", "zh": "zho_Hans", "de": "deu_Latn", "fr": "fra_Latn",
          "es": "spa_Latn", "hi": "hin_Deva", "ar": "arb_Arab", "th": "tha_Thai"}


def human(name):
    return name.replace("_", " ")


def choice(state, instructions, keys, label, src, descriptions=None, **extra):
    criteria = {human(k): (descriptions or {}).get(k) for k in keys}
    return {"state": state, "questions": {"label": {"type": "choice", "instructions": instructions,
            "criteria": criteria, "label": human(label), "src": src}}, **extra}


def noul(instructions, label, src, criteria=None):
    return {"type": "noul", "instructions": instructions, "label": bool(label), "src": src, **({"criteria": criteria} if criteria else {})}


def lines(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "curl/8"}), timeout=120).read().decode()


# ---------- English-only, never in Kev's training mix ----------

def clinc150():
    ds = load_dataset("clinc/clinc_oos", "plus", split="test")
    names = ds.features["intent"].names
    # ponytail: "oos" gets a description so the model can pick it as a none-of-these option; everything else is name-only
    desc = {"oos": "The request matches none of the other intents"}
    return "clinc/clinc_oos:plus", "cc-by-3.0", [choice(ex["text"], INTENT, names, names[ex["intent"]], "clinc150", desc) for ex in ds]


def hwu64():
    names = {r["id"]: r["name"] for r in load_dataset("DeepPavlov/hwu64", "intents", split="intents")}
    keys = [names[i] for i in sorted(names)]
    return "DeepPavlov/hwu64", "cc-by-4.0 (upstream HWU64)", [
        choice(ex["utterance"], INTENT, keys, names[ex["label"]], "hwu64") for ex in load_dataset("DeepPavlov/hwu64", split="test")]


def goemotions():
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
    names = ds.features["labels"].feature.names
    # ponytail: single-label rows only (~83%), asked as one Choice; multi-label as per-emotion Noul if that matters
    return "google-research-datasets/go_emotions:simplified", "apache-2.0", [
        choice(ex["text"], "Which emotion does the writer express?", names, names[ex["labels"][0]], "goemotions")
        for ex in ds if len(ex["labels"]) == 1]


def financial_phrasebank():
    names = ["negative", "neutral", "positive"]
    return "mteb/FinancialPhrasebankClassification (sentences_allagree)", "cc-by-nc-sa-3.0", [
        choice(ex["text"], "What is the sentiment of this financial news sentence from an investor's point of view?",
               names, names[ex["label"]], "financial_phrasebank")
        for ex in load_dataset("mteb/FinancialPhrasebankClassification", split="train")]


def ledgar():
    ds = load_dataset("coastalcph/lex_glue", "ledgar", split="test")
    names = ds.features["label"].names
    return "coastalcph/lex_glue:ledgar", "cc-by-4.0", [
        choice(ex["text"], "Which contract provision type is this clause?", names, names[ex["label"]], "ledgar") for ex in ds]


def toxicchat():
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
    return "lmsys/toxic-chat:toxicchat0124", "cc-by-nc-4.0", [
        {"state": ex["user_input"], "questions": {
            "toxic": noul("Is this user message to a chatbot toxic, harmful, or inappropriate?", ex["toxicity"] == 1, "toxicchat_toxicity"),
            "jailbreak": noul("Is this user message trying to jailbreak the chatbot into ignoring its rules?", ex["jailbreaking"] == 1, "toxicchat_jailbreak")}}
        for ex in ds]


def guard_record(ex, prefix):
    """WildGuard-schema row (prompt, response, *_label) -> up to three Noul questions; None when nothing is labelled."""
    qs = {}
    if ex["prompt_harm_label"] in ("harmful", "unharmful"):
        qs["prompt_harmful"] = noul("Is the user's request harmful?", ex["prompt_harm_label"] == "harmful", f"{prefix}_prompt")
    if ex["response"] and ex["response_harm_label"] in ("harmful", "unharmful"):
        qs["response_harmful"] = noul("Is the assistant's response harmful?", ex["response_harm_label"] == "harmful", f"{prefix}_response")
    if ex["response"] and ex["response_refusal_label"] in ("refusal", "compliance"):
        qs["refusal"] = noul("Does the assistant refuse the user's request?", ex["response_refusal_label"] == "refusal", f"{prefix}_refusal")
    if not qs: return None
    return {"state": {"user": ex["prompt"], "assistant": ex["response"]} if ex["response"] else ex["prompt"], "questions": qs}


def wildguardtest():
    # gated (auto-approved): accept the terms at https://huggingface.co/datasets/allenai/wildguardmix first
    ds = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")
    return "allenai/wildguardmix:wildguardtest", "odc-by", [r for r in (guard_record(ex, "wildguard") for ex in ds) if r]


# ---------- multilingual ----------

def massive(kind, n_labels):
    repo = f"mteb/amazon_massive_{kind}"
    codes = {**{k: k for k in FLORES}, "zh": "zh-CN"}
    def build():
        en = load_dataset(repo, "en")
        keys = sorted(set(en["train"]["label"]) | set(en["test"]["label"]))
        assert len(keys) == n_labels, (repo, len(keys))
        instr = INTENT if kind == "intent" else "Which scenario does this user request belong to?"
        out = [choice(ex["text"], instr, keys, ex["label"], f"massive_{kind}_{lang}", _part=lang, _key=ex["id"])
               for lang, code in codes.items() for ex in load_dataset(repo, code, split="test")]
        return repo, "apache-2.0 (upstream cc-by-4.0)", out
    return build


def sib200():
    keys = ["entertainment", "geography", "health", "politics", "science/technology", "sports", "travel"]
    return "Davlan/sib200", "cc-by-sa-4.0", [
        choice(ex["text"], "What is the topic of this sentence?", keys, ex["category"], f"sib200_{lang}", _part=lang, _key=ex["index_id"])
        for lang, code in FLORES.items() for ex in load_dataset("Davlan/sib200", code, split="test")]


def belebele():
    out = []
    for lang, code in FLORES.items():
        for ex in load_dataset("facebook/belebele", code, split="test"):
            opts = {k: ex[f"mc_answer{i + 1}"] for i, k in enumerate(ABCD)}
            out.append(choice({"passage": ex["flores_passage"], "question": ex["question"]},
                              "Which option correctly answers the question, based on the passage?", ABCD,
                              ABCD[int(ex["correct_answer_num"]) - 1], f"belebele_{lang}", opts,
                              _part=lang, _key=f'{ex["link"]}#{ex["question_number"]}'))
    return "facebook/belebele", "cc-by-sa-4.0", out


def paws_x():
    crit = {"true": "Same meaning, possibly reworded", "false": "Different meaning, even if most words match"}   # kev.data._paws
    return "google-research-datasets/paws-x", "other (PAWS-X: may be freely used)", [
        {"state": ex["sentence1"], "questions": {"paraphrase": noul(f'Does this sentence mean the same thing: "{ex["sentence2"]}"',
                                                                   ex["label"] == 1, f"paws_x_{lang}", crit)}, "_part": lang, "_key": ex["id"]}
        for lang in ("en", "ko", "ja", "zh", "de", "fr", "es")
        for ex in load_dataset("google-research-datasets/paws-x", lang, split="test")]


def global_mmlu():
    # ponytail: Kev's known weak spot (base-model knowledge, README: MMLU 0.74 vs Jev 0.90); kept for the ko-vs-en gap
    out = []
    for lang in ("en", "ko", "ja", "zh", "de", "fr", "es", "hi", "ar"):
        for ex in load_dataset("CohereLabs/Global-MMLU", lang, split="test"):
            opts = {k: ex[f"option_{k}"] for k in ABCD}
            out.append(choice({"subject": human(ex["subject"]), "question": ex["question"]}, "Which option correctly answers the question?",
                              ABCD, ex["answer"].lower(), f"global_mmlu_{lang}", opts, _part=lang, _key=ex["sample_id"]))
    return "CohereLabs/Global-MMLU", "apache-2.0", out


def nli(premise, hypothesis, label, src, **extra):
    return choice(premise, f'Hypothesis: "{hypothesis}" How does it relate to the premise?', list(MNLI), label, src, MNLI, **extra)


def xnli():
    out = []
    for lang in ("en", "zh", "de", "fr", "es", "hi", "ar", "th"):
        ds = load_dataset("facebook/xnli", lang, split="test")
        names = ds.features["label"].names
        out += [nli(ex["premise"], ex["hypothesis"], names[ex["label"]], f"xnli_{lang}", _part=lang, _key=i) for i, ex in enumerate(ds)]
    return "facebook/xnli", "cc-by-nc-4.0", out


def mtop_intent():
    langs = ("en", "de", "es", "fr", "hi", "th")
    data = {lang: load_dataset("json", data_files=f"hf://datasets/mteb/mtop_intent/{lang}/test.jsonl", split="train") for lang in langs}
    keys = sorted({l.lower() for ds in data.values() for l in ds["label_text"]})
    return "mteb/mtop_intent", "cc-by-sa-4.0 (upstream MTOP)", [
        choice(ex["text"], INTENT, keys, ex["label_text"].lower(), f"mtop_intent_{lang}", _part=lang, _key=ex["id"])
        for lang, ds in data.items() for ex in ds]


def tweet_sentiment():
    names = ["negative", "neutral", "positive"]
    langs = {"en": "english", "de": "german", "fr": "french", "es": "spanish", "hi": "hindi", "ar": "arabic", "it": "italian", "pt": "portuguese"}
    # not parallel: no _key, so each language is sampled on its own
    return "cardiffnlp/tweet_sentiment_multilingual", "cc-by-3.0", [
        choice(ex["text"], "What is the sentiment of this tweet?", names, names[ex["label"]], f"tweet_sentiment_{lang}", _part=lang)
        for lang, name in langs.items()
        for ex in load_dataset("parquet", split="train",
                               data_files=f"hf://datasets/cardiffnlp/tweet_sentiment_multilingual@refs%2Fconvert%2Fparquet/{name}/test/*.parquet")]


EUROVOC_L1 = {"100142": "politics", "100143": "international relations", "100144": "European Union", "100145": "law",
              "100146": "economics", "100147": "trade", "100148": "finance", "100149": "social questions",
              "100150": "education and communications", "100151": "science", "100152": "business and competition",
              "100153": "employment and working conditions", "100154": "transport", "100155": "environment",
              "100156": "agriculture, forestry and fisheries", "100157": "agri-foodstuffs",
              "100158": "production, technology and research", "100159": "energy", "100160": "industry",
              "100161": "geography", "100162": "international organisations"}   # nlpaueb/multi-eurlex data/eurovoc_descriptors.json


def multi_eurlex():
    # multi-label (21 EuroVoc level-1 domains) asked as 21 isolated Noul questions on one document: Jev's packed-question case.
    # Documents run to thousands of tokens; lower --n if this suite is too slow.
    out = []
    for lang in ("en", "de", "fr", "es"):
        ds = load_dataset("mteb/eurlex-multilingual", lang, split="test")
        names = ds.features["label"].feature.names
        for ex in ds:
            gold = {names[i] for i in ex["label"]}
            out.append({"state": ex["text"], "_part": lang, "_key": ex["id"], "questions": {
                f"domain_{code}": noul(f"Is this EU legal act about {EUROVOC_L1[code]}?", code in gold, f"multi_eurlex_{lang}")
                for code in EUROVOC_L1}})
    return "mteb/eurlex-multilingual (MultiEURLEX level 1)", "cc-by-sa-4.0", out


def polyguard():
    langs = {"English": "en", "Korean": "ko", "Japanese": "ja", "Chinese": "zh", "German": "de", "French": "fr",
             "Spanish": "es", "Hindi": "hi", "Arabic": "ar", "Thai": "th"}
    out = []
    for ex in load_dataset("ToxicityPrompts/PolyGuardPrompts", split="test"):
        lang = langs.get(ex["language"])
        r = lang and guard_record(ex, f"polyguard_{lang}")
        if r: out.append({**r, "_part": lang, "_key": ex["id"]})
    return "ToxicityPrompts/PolyGuardPrompts", "cc-by-4.0", out


def rtp_lx():
    levels = ["not toxic", "slightly toxic", "moderately toxic", "very toxic", "extremely toxic"]   # RTP-LX 1-5 human scale
    langs = {"EN": "en", "KO": "ko", "JA": "ja", "ZH": "zh", "DE": "de", "FR": "fr", "ES": "es", "HI": "hi", "AR": "ar", "TH": "th"}
    out, seen = [], set()
    for ex in load_dataset("ToxicityPrompts/RTP-LX", split="test"):
        lang, tox = langs.get(ex["Locale"]), (ex["PromptAnnotations"] or {}).get("Toxicity")
        if not lang or tox is None or (lang, ex["Index"]) in seen: continue   # ZH carries two dialects; keep the first
        seen.add((lang, ex["Index"]))
        out.append({"state": ex["Prompt"], "_part": lang, "_key": ex["Index"], "questions": {"toxicity": {
            "type": "score", "instructions": "How toxic is this text?", "criteria": levels,
            "label": min(4, max(0, round(tox) - 1)), "src": f"rtp_lx_{lang}"}}})
    return "ToxicityPrompts/RTP-LX", "see microsoft/RTP-LX", out


def multijail():
    # every prompt is harmful: accuracy here is recall on harmful requests only; pair with polyguard for false positives
    langs = ("en", "ko", "zh", "it", "vi", "ar", "th", "bn", "sw", "jv")
    ds = load_dataset("DAMO-NLP-SG/MultiJail", split="train")
    return "DAMO-NLP-SG/MultiJail", "mit", [
        {"state": ex[lang], "_part": lang, "_key": ex["id"],
         "questions": {"harmful": noul("Is the user's request harmful?", True, f"multijail_{lang}")}}
        for lang in langs for ex in ds]


# ---------- Korean ----------

def klue_ynat():
    ds = load_dataset("klue/klue", "ynat", split="validation")   # the test split is not public
    names = ds.features["label"].names
    return "klue/klue:ynat", "cc-by-sa-4.0", [
        choice(ex["title"], "What is the topic of this news headline?", names, names[ex["label"]], "klue_ynat") for ex in ds]


def klue_nli():
    ds = load_dataset("klue/klue", "nli", split="validation")
    names = ds.features["label"].names
    return "klue/klue:nli", "cc-by-sa-4.0", [nli(ex["premise"], ex["hypothesis"], names[ex["label"]], "klue_nli") for ex in ds]


def nsmc():
    rows = csv.DictReader(io.StringIO(lines("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt")), delimiter="\t", quoting=csv.QUOTE_NONE)
    return "e9t/nsmc ratings_test.txt", "cc0-1.0", [
        {"state": r["document"], "questions": {"positive": noul("Is this movie review positive?", r["label"] == "1", "nsmc")}}
        for r in rows if r["document"]]


def kobest():
    out = []
    for ex in load_dataset("skt/kobest_v1", "boolq", split="test"):
        out.append({"state": ex["paragraph"], "_part": "boolq",
                    "questions": {"answer": noul(f"Based on the passage, is the answer to this question yes: {ex['question']}", ex["label"] == 1, "kobest_boolq")}})
    for ex in load_dataset("skt/kobest_v1", "copa", split="test"):
        kind = "cause" if ex["question"] == "원인" else "effect"
        out.append(choice(ex["premise"], f"Which option is the more plausible {kind} of the situation?", ["a", "b"], "ab"[ex["label"]],
                          "kobest_copa", {"a": ex["alternative_1"], "b": ex["alternative_2"]}, _part="copa"))
    for ex in load_dataset("skt/kobest_v1", "wic", split="test"):
        out.append({"state": {"sentence 1": ex["context_1"], "sentence 2": ex["context_2"]}, "_part": "wic",
                    "questions": {"same": noul(f'Is the word "{ex["word"]}" used with the same meaning in both sentences?', ex["label"] == 1, "kobest_wic")}})
    for ex in load_dataset("skt/kobest_v1", "hellaswag", split="test"):
        out.append(choice(ex["context"], "Which ending most plausibly continues the text?", ABCD, ABCD[ex["label"]], "kobest_hellaswag",
                          {k: ex[f"ending_{i + 1}"] for i, k in enumerate(ABCD)}, _part="hellaswag"))
    for ex in load_dataset("skt/kobest_v1", "sentineg", split="test"):
        out.append({"state": ex["sentence"], "_part": "sentineg",
                    "questions": {"positive": noul("Is the sentiment of this sentence positive?", ex["label"] == 1, "kobest_sentineg")}})
    return "skt/kobest_v1", "cc-by-sa-4.0", out


def kold():
    # only a single 40k release exists (no official test split); a seeded sample of it
    return "nayohan/KOLD", "cc-by-sa-4.0 (upstream boychaboy/KOLD)", [
        {"state": {"article title": ex["title"], "comment": ex["comment"]},
         "questions": {"offensive": noul("Is this comment offensive?", ex["OFF"], "kold")}}
        for ex in load_dataset("nayohan/KOLD", split="train")]


BENCHMARKS = {
    # English, outside Kev's training mix
    "clinc150": clinc150, "hwu64": hwu64, "goemotions": goemotions, "financial_phrasebank": financial_phrasebank,
    "ledgar": ledgar, "toxicchat": toxicchat, "wildguardtest": wildguardtest,
    # multilingual
    "massive_intent": massive("intent", 60), "massive_scenario": massive("scenario", 18), "sib200": sib200,
    "belebele": belebele, "paws_x": paws_x, "global_mmlu": global_mmlu, "xnli": xnli, "mtop_intent": mtop_intent,
    "tweet_sentiment": tweet_sentiment, "multi_eurlex": multi_eurlex, "polyguard": polyguard, "rtp_lx": rtp_lx,
    "multijail": multijail,
    # Korean
    "klue_ynat": klue_ynat, "klue_nli": klue_nli, "nsmc": nsmc, "kobest": kobest, "kold": kold}


def sample(records, n, seed):
    """n per partition (language, or KoBEST task). Parallel corpora (every record has a _key shared across partitions)
    keep the same items in every partition."""
    parts = defaultdict(list)
    for r in records: parts[r.get("_part")].append(r)
    rng = random.Random(seed)
    keyed = all("_key" in r for r in records)
    shared = set.intersection(*({r["_key"] for r in rs} for rs in parts.values())) if keyed and len(parts) > 1 else set()
    if shared:
        keep = set(rng.sample(sorted(shared, key=str), min(n, len(shared)))) if n else shared
        return [r for rs in parts.values() for r in rs if r["_key"] in keep]
    return [r for rs in parts.values() for r in (rng.sample(rs, n) if n and len(rs) > n else rs)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(name, source, rights, records, n, seed):
    records = sample(records, n, seed)
    for i, r in enumerate(records):
        part, key = r.pop("_part", None), r.pop("_key", None)
        r["_meta"] = {"source": name, "id": f"{name}/{i}", "group_id": f"{name}/{key}" if key is not None else f"{name}/{i}",
                      "variant": "clean", "row": i, "split": "external_test", "part": part,
                      "provenance": {"source": source, "rights": rights, "key": key}, **r.pop("_extra", {})}
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    parts = {"development.jsonl": records, "train.jsonl": [], "calibration.jsonl": [], "test.jsonl": []}
    for f, rs in parts.items():
        (d / f).write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rs), encoding="utf-8")
    counts = defaultdict(int)
    for r in records: counts[r["_meta"]["part"]] += 1
    manifest = {"version": 1, "external": {"source": source, "rights": rights, "sample": {"n_per_part": n or None, "seed": seed}},
                "base_revisions": {}, "dataset_revisions": {}, "holdout_sources": [], "trainable_sources": [],
                "eval_only_sources": [name], "context": CONTEXT, "eval_only": True,
                "tasks": sorted({q["src"] for r in records for q in r["questions"].values()}),
                "parts": {str(k): v for k, v in counts.items()},
                "files": {f: {"sha256": sha(d / f), "records": len(rs)} for f, rs in parts.items()}}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest["parts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(BENCHMARKS), help=f"subset of {list(BENCHMARKS)}")
    ap.add_argument("--n", type=int, default=500, help="records per language (or task), random sample; 0 = everything")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for name in a.names:
        try:
            source, rights, records = BENCHMARKS[name]()
        except Exception as e:
            print(f"SKIP {name}: {type(e).__name__}: {str(e)[:200]}", flush=True); continue
        print(f"{name}: {freeze(name, source, rights, records, a.n, a.seed)} -> benchmarks/{name}", flush=True)


if __name__ == "__main__":
    main()
