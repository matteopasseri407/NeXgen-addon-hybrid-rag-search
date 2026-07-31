#!/usr/bin/env python3
"""Score a running instance against a gold set: hits@1, hits@5, MRR, latency.

    python eval/run_eval.py --url http://127.0.0.1:8089 --queries eval/queries.json

The gold set is a list of {"query": ..., "expect": "<relative/path.md>"}.
`expect` may also be a list, in which case any of the paths counts as correct.

Why this file exists: a retrieval change that "feels better" usually isn't.
The reranker in this repo measurably HURT results on its first attempt --
hits@5 fell from 15/20 to 12/20 because the cross-encoder was fed bare
snippets without the title/filename prefix. Nothing but a gold set catches
that; it looks like a working reranker either way.

Compare two configurations by pointing this at the same instance twice, with
RERANK_ENABLED flipped between runs.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.parse
import urllib.request


def search(base_url, query, k):
    url = (f"{base_url.rstrip('/')}/search?"
           + urllib.parse.urlencode({"q": query, "k": k}))
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=30) as r:
        payload = json.load(r)
    return payload.get("results", []), (time.perf_counter() - t0) * 1000


def rank_of(results, expected):
    """1-indexed rank of the first correct hit, or None."""
    wanted = {expected} if isinstance(expected, str) else set(expected)
    for i, hit in enumerate(results, start=1):
        if hit.get("path") in wanted:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8089")
    ap.add_argument("--queries", default="eval/queries.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    gold = json.load(open(args.queries, encoding="utf-8"))
    if not gold:
        sys.exit("gold set is empty")

    try:
        with urllib.request.urlopen(
                f"{args.url.rstrip('/')}/health", timeout=10) as r:
            health = json.load(r)
    except Exception as e:
        sys.exit(f"cannot reach {args.url}: {e!r}")

    print(f"instance: {args.url}")
    print(f"  reranker={health.get('reranker')} "
          f"chunks={health.get('chunks')} files={health.get('files')}")
    print(f"gold set: {len(gold)} queries, k={args.k}\n")

    ranks, latencies, misses = [], [], []
    for case in gold:
        results, ms = search(args.url, case["query"], args.k)
        r = rank_of(results, case["expect"])
        ranks.append(r)
        latencies.append(ms)
        if r is None:
            misses.append(case["query"])
        if args.verbose:
            mark = f"@{r}" if r else "MISS"
            print(f"  {mark:5} {ms:6.0f}ms  {case['query'][:58]}")
            if r is None and results:
                print(f"        got: {results[0].get('path')}")

    n = len(ranks)
    hits1 = sum(1 for r in ranks if r == 1)
    hitsk = sum(1 for r in ranks if r is not None)
    mrr = sum((1.0 / r) for r in ranks if r) / n

    print(f"\nhits@1   {hits1}/{n}  ({hits1 / n:.0%})")
    print(f"hits@{args.k}   {hitsk}/{n}  ({hitsk / n:.0%})")
    print(f"MRR      {mrr:.3f}")
    print(f"latency  p50 {statistics.median(latencies):.0f}ms  "
          f"max {max(latencies):.0f}ms")

    if misses:
        print(f"\nmissed ({len(misses)}):")
        for q in misses:
            print(f"  - {q}")


if __name__ == "__main__":
    main()
