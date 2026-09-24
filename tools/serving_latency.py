#!/usr/bin/env python3
"""Median model time per request against a running kev.serve endpoint, in Kev's README shapes.

    python3 -m kev.serve --run <ckpt> --port 8009 &     # give it a GPU to itself
    python3 tools/serving_latency.py --base-url http://127.0.0.1:8009 --name malkuth-4b

Reports `latency_ms` from the API (model time, not wall time) as median of --n requests, as new state / repeated
state: a new state is the normal call, a repeated one is served from the server's prefix cache. Each row is warmed
once first, so CUDA-graph capture is not counted.
"""
import argparse, json, statistics, time, urllib.request

CRITERIA = {"billing": "Charges, invoices, payment problems", "shipping": "Delivery status, delays, lost packages",
            "returns": "Exchanges, refunds, wrong or damaged items"}
FILLER = ("The customer contacted support about an order placed earlier this month. "
          "Details from the account history follow. ")


def questions(n):
    qs = {}
    for i in range(n):
        if i % 3 == 0:   qs[f"q{i}"] = {"type": "choice", "instructions": "Which team should handle this?", "criteria": CRITERIA}
        elif i % 3 == 1: qs[f"q{i}"] = {"type": "noul", "instructions": "Does this need urgent human attention?"}
        else:            qs[f"q{i}"] = {"type": "score", "instructions": "How frustrated is the customer?",
                                        "criteria": ["Calm", "Frustrated", "Very angry"]}
    return qs


def call(url, state, qs):
    body = json.dumps({"state": state, "model": "kev-latest", "questions": qs}).encode()
    req = urllib.request.Request(url + "/v1/systemone", body, {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["latency_ms"]


def row(url, n_q, state, n):
    call(url, state, questions(n_q))                                          # warm: captures graphs, fills the cache
    fresh = [call(url, state + f" ref {i}", questions(n_q)) for i in range(n)]  # new state each time
    repeat = [call(url, state, questions(n_q)) for _ in range(n)]              # prefix cache hit
    return statistics.median(fresh), statistics.median(repeat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8009")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    short = "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card."
    rows = [("2 questions", 2, short), ("6 questions", 6, short),
            ("5 questions, ~370-token state", 5, FILLER * 12), ("5 questions, ~2,200-token state", 5, FILLER * 75)]
    print(f"{a.name}  (median of {a.n}, model time as reported by the API)")
    print(f"| Shape | New state | Repeated state |")
    print(f"|---|---:|---:|")
    for label, n_q, state in rows:
        f, r = row(a.base_url, n_q, state, a.n)
        print(f"| {label} | {f:.0f} ms | {r:.0f} ms |")


if __name__ == "__main__":
    main()
