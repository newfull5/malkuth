#!/usr/bin/env python3
"""Stream a running kev.train log into Weights & Biases, and benchmark every checkpoint it writes, without touching
the training process.

    uv run --no-project --with wandb python tools/log2wandb.py --log kev/runs/q38-2b-distill.log --name q38-2b-distill --gpu 0

Train metrics come from kev.train's progress lines ("ep0 step 120/394 loss 0.912 ... 0.412s/rec"). Each
"checkpoint step N -> <dir>" line (kev.train --save_every) and the final "saved <dir>" line starts a lite-benchmark run
(tools/kev_batch_bench.py on --bench, default benchmarks/val/, bf16 for speed) whose scores are logged against eval/step.
"""
import argparse, json, os, re, subprocess, threading, time
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parent.parent
STEP = re.compile(r"ep(\d+) step (\d+)/(\d+) loss ([\d.]+) kl ([\d.]+) anchor ([\d.]+) ([\d.]+)s/rec")
CKPT = re.compile(r"checkpoint step (\d+) -> (\S+)")
LANGS = ["en", "ko", "ja", "zh", "de", "fr", "es", "hi", "ar", "th"]


def training_cmd(name):
    out = subprocess.run(["pgrep", "-af", f"kev.train .*--out runs/{name}"], capture_output=True, text=True).stdout
    return next((l.split(" ", 1)[1] for l in out.splitlines() if "uv run" not in l), "")


def evaluate(name, step, ckpt, gpu, lock, bench="benchmarks/val"):
    """Lite benchmark of one checkpoint -> flat metrics dict (None if the run failed)."""
    out = ROOT / "results" / "train" / name / bench / f"step-{step}"
    suites = sorted(str(p) for p in (ROOT / bench).iterdir() if p.is_dir())
    ckpt = Path(ckpt) if Path(ckpt).is_absolute() else ROOT / "kev" / ckpt
    with lock:   # one eval at a time per run
        r = subprocess.run(["uv", "run", "python", str(ROOT / "tools" / "kev_batch_bench.py"), "--run", str(ckpt), "--out", str(out),
                            "--pass-tokens", "16384", *suites], cwd=ROOT / "kev", capture_output=True, text=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "KEV_DTYPE": "bf16"})
    if r.returncode: return {"eval/error": r.stderr[-400:]}
    summary = json.loads((out / "summary.json").read_text())
    suites = {k: v for k, v in summary.items() if not k.startswith("_")}
    m = {"eval/step": step, "eval/acc_mean": sum(v["acc"] for v in suites.values()) / len(suites),
         "eval/brier_mean": sum(v["brier"] for v in suites.values()) / len(suites),
         "eval/ece_mean": sum(v["ece"] for v in suites.values()) / len(suites), "eval/seconds": summary["_total_seconds"]}
    m.update({f"eval_suite/{k}": v["acc"] for k, v in suites.items()})
    agg = {l: [0.0, 0] for l in LANGS}   # per-language accuracy over suites that contain Korean (same items per language)
    for s in suites:
        tasks = json.loads((out / s / "report.json").read_text())["tasks"]
        if s.startswith("laya_") or not any("_ko" in t for t in tasks): continue
        for t, v in tasks.items():
            lang = next((x for x in t.split("_")[1:] if x in agg), None)
            if lang: agg[lang][0] += v["acc"] * v["n"]; agg[lang][1] += v["n"]
    m.update({f"eval_lang/{l}": a / n for l, (a, n) in agg.items() if n})
    if agg["ko"][1] and agg["en"][1]: m["eval/ko_minus_en"] = m["eval_lang/ko"] - m["eval_lang/en"]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default="system-one")
    ap.add_argument("--entity", default="newfull5", help="wandb entity; pinned so curves land with the existing runs no matter which account is logged in")
    ap.add_argument("--gpu", default="0", help="GPU for checkpoint evals (the training GPU is fine: evals are short)")
    ap.add_argument("--bench", default="benchmarks/val", help="suite directory for checkpoint evals: selection uses validation, never the reported benchmarks")
    a = ap.parse_args()
    run = wandb.init(entity=a.entity, project=a.project, name=a.name, config={"cmd": training_cmd(a.name), "log": a.log})
    wandb.define_metric("eval/step")
    for prefix in ("eval/*", "eval_suite/*", "eval_lang/*"): wandb.define_metric(prefix, step_metric="eval/step")
    lock, threads, last_step = threading.Lock(), [], 0

    def launch(step, ckpt):
        def work():
            m = evaluate(a.name, step, ckpt, a.gpu, lock, a.bench)
            wandb.log(m); print(f"eval step {step}: {json.dumps({k: round(v, 4) for k, v in m.items() if k.startswith('eval/') and isinstance(v, float)})}", flush=True)
        t = threading.Thread(target=work); t.start(); threads.append(t)

    with Path(a.log).open(errors="replace") as f:
        idle = 0
        while True:
            line = f.readline()
            if not line:
                idle += 1
                if idle % 60 == 0 and not training_cmd(a.name): break   # process gone without a "saved" line
                time.sleep(1); continue
            idle = 0
            if m := STEP.search(line):
                ep, step, steps, loss, kl, anchor, spr = m.groups(); last_step = int(steps)
                wandb.log({"train/loss": float(loss), "train/kl": float(kl), "train/epoch": int(ep),
                           "train/progress": int(step) / int(steps), "train/sec_per_record": float(spr)}, step=int(step))
            elif m := CKPT.search(line):
                launch(int(m.group(1)), m.group(2))
            elif "training requests" in line:
                run.summary["data"] = line.strip()[:500]
            elif line.startswith("saved"):
                run.summary["saved"] = line.strip()
                launch(last_step, line.split(" ", 1)[1].strip()); break
            elif "Traceback" in line or "Error" in line:
                run.summary["error"] = line.strip()[:500]
    for t in threads: t.join()
    run.finish()


if __name__ == "__main__":
    main()
