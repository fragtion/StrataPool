#!/usr/bin/env python3
"""StrataPool's A/B suite: the same requests against one server, the engine's own timings saved per request.

    .venv\\Scripts\\python tools\\pool_bench.py --label desktop-alone                 (this PC's server)
    .venv\\Scripts\\python tools\\pool_bench.py --label laptop-alone --url http://192.168.10.81:8080

Every request is greedy (temperature 0) with the shipped config's penalties (presence 1.0 over the last 512 tokens)
and no thinking, so a setup sees the same text as any other (up to near-ties where a GPU and a CPU round an expert
differently).  Every run starts its prompts with its own id, so nothing a previous run left in a cache is reused;
inside a run, the agent test reuses on purpose.  The results go to bench-results/<label>.json and a table.

Tests (about 4-6 minutes in all):
  code      a Python module from a short prompt, 1200 tokens                       decode speed
  prose     how TCP works, 1000 tokens (predictable text: long accepted drafts)     decode speed
  long      an ~8K-token document and a question about it, 300 tokens            prompt read + decode
  short     fresh prompts of ~500, ~2,000 and ~2,800 tokens, 24 tokens each       prompt read (CPU assist)
  agent     two sibling conversations on one ~3K-token system prompt, turns that switch between them:
            what each turn reads again (parked conversations, borrowing) and how long it waits for its first token
  pair      the code and prose requests at the same time                          both together (two chats)
`--only code,prose` runs a part; `--skip pair` leaves one out.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import threading
import time
import urllib.request
import uuid

ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLING = {"temperature": 0, "presence_penalty": 1.0, "penalty_last_n": 512, "reasoning_effort": "none"}

CODE = ("Write a complete Python module `ttl_cache.py` implementing a thread-safe LRU cache with per-entry time-to-live: "
        "get/set/delete, a maximum size, an optional loader callback for misses, statistics (hits, misses, evictions, "
        "expirations), and a decorator that memoizes a function with it. Include docstrings and a unittest test class "
        "covering eviction order, expiry with a fake clock, thread safety and the decorator.")
# in the pair, the code request once answered with a tool call after ~100 tokens (no tools are offered): asked plainly
PAIR_NOTE = " Write the whole module in your reply as one code block; do not call any tools."
PROSE = ("Explain in detail how TCP works: the three-way handshake, sequence and acknowledgement numbers, the sliding "
         "window, retransmission and timeouts, slow start and congestion avoidance, fast retransmit and recovery, and "
         "connection teardown. Use headings and keep it accurate.")


def corpus(chars: int, offset: int = 0) -> str:
    """Plain text from this repository's docs, deterministic: `chars` characters starting `offset` into it."""
    text = ""
    for name in ("docs/DETAILS.md", "docs/POOL.md", "docs/MULTI_GPU.md", "docs/HOW_IT_WORKS.md", "docs/INSTALL.md",
                 "docs/MODELS.md", "docs/TROUBLESHOOTING.md", "docs/BATCHING.md"):
        p = ROOT / name
        if p.is_file():
            text += p.read_text(encoding="utf-8", errors="replace") + "\n\n"
    while len(text) < offset + chars:
        text += text
    return text[offset:offset + chars]


def post(url: str, body: dict, timeout: float = 1800) -> tuple[dict, float]:
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer bench"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.perf_counter() - t0


def ask(url: str, messages: list, max_tokens: int, name: str, results: list, lock=None) -> dict:
    body = {"model": "strata", "messages": messages, "max_tokens": max_tokens, **SAMPLING}
    out, wall = post(url, body)
    tm = out.get("timings") or {}
    usage = out.get("usage") or {}
    rec = {"test": name, "wall_s": round(wall, 2), "prompt_tokens": usage.get("prompt_tokens"),
           "reused": tm.get("cache_n"), "read": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
           "prompt_tok_s": tm.get("prompt_per_second"), "generated": usage.get("completion_tokens"),
           "decode_ms": tm.get("predicted_ms"), "decode_tok_s": tm.get("predicted_per_second"),
           "drafts": tm.get("draft_n"), "accepted": tm.get("draft_n_accepted"),
           "finish": (out.get("choices") or [{}])[0].get("finish_reason")}
    if lock:
        with lock:
            results.append(rec)
    else:
        results.append(rec)
    if rec["finish"] not in ("length", "stop"):
        print(f"  ! {name}: finished with {rec['finish']!r} - not a full answer; its speed is not comparable")
    print(f"  {name:<22} prompt {rec['prompt_tokens']} (reused {rec['reused']}, read {rec['read']} at "
          f"{rec['prompt_tok_s']} tok/s), {rec['generated']} generated at {rec['decode_tok_s']} tok/s, "
          f"{rec['wall_s']} s", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="what this run is (desktop-alone, split-default, ...)")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--only", default="", help="comma-separated tests to run")
    ap.add_argument("--skip", default="", help="comma-separated tests to leave out")
    ap.add_argument("--no-warmup", action="store_true", help="skip the unrecorded first request")
    a = ap.parse_args()
    tests = ["code", "prose", "long", "short", "agent", "pair"]
    if a.only:
        tests = [t for t in tests if t in a.only.split(",")]
    tests = [t for t in tests if t not in a.skip.split(",")]
    run = uuid.uuid4().hex[:8]
    tag = f"[bench run {run}] "
    results: list[dict] = []
    print(f"{a.label}: run {run} against {a.url}: {', '.join(tests)}", flush=True)
    t_all = time.perf_counter()
    if not a.no_warmup:   # the first request after a start pays one-time costs (graphs, buffers): not recorded
        ask(a.url, [{"role": "user", "content": tag + "Say hello in one short sentence."}], 32, "warmup", [])

    if "code" in tests:
        ask(a.url, [{"role": "user", "content": tag + CODE}], 1200, "code", results)
    if "prose" in tests:
        ask(a.url, [{"role": "user", "content": tag + PROSE}], 1000, "prose", results)
    if "long" in tests:
        doc = corpus(30000)
        ask(a.url, [{"role": "user", "content": tag + "Here is a document:\n\n" + doc +
                     "\n\nList the ten most important settings this document describes and what each one does."}],
            300, "long (~8K)", results)
    if "short" in tests:
        for n, chars in ((500, 1900), (2000, 7600), (2800, 10600)):
            ask(a.url, [{"role": "user", "content": tag + f"({n}) Summarize this in one sentence:\n\n" +
                         corpus(chars, offset=40000 + chars)}], 24, f"short ~{n}", results)
    if "agent" in tests:
        system = (tag + "You are a coding agent working in a repository. The project's documentation follows; use it "
                  "to answer.\n\n" + corpus(11500, offset=80000))
        conv = {"A": [{"role": "system", "content": system},
                      {"role": "user", "content": "Task A: explain in five bullet points how a layer split divides the "
                                                  "work between PCs."}],
                "B": [{"role": "system", "content": system},
                      {"role": "user", "content": "Task B: list the settings that change how much VRAM the expert "
                                                  "cache gets, one line each."}]}
        follow = {"A": ["Now say which of those points matter most on a slow network, and why.",
                        "Summarize everything so far in three sentences."],
                  "B": ["Which of those would you change first on a 12 GB card? One paragraph."]}
        order = ["A", "B", "A", "B", "A"]          # A new, B a sibling, A back, B back, A back
        turn = {"A": 0, "B": 0}
        for i, k in enumerate(order):
            if turn[k] > 0:
                conv[k].append({"role": "user", "content": follow[k][turn[k] - 1]})
            out = ask(a.url, conv[k], 200, f"agent {i + 1}: {k} turn {turn[k] + 1}", results)
            conv[k].append({"role": "assistant", "content": (out["choices"][0]["message"].get("content") or "")})
            turn[k] += 1
    if "pair" in tests:
        lock = threading.Lock()
        t0 = time.perf_counter()
        th = [threading.Thread(target=ask, args=(a.url, [{"role": "user", "content": tag + "(pair) " + p}], m,
                                                 f"pair: {n}", results, lock))
              for n, p, m in (("code", CODE + PAIR_NOTE, 1200), ("prose", PROSE, 1000))]
        for t in th:
            t.start()
        for t in th:
            t.join()
        wall = time.perf_counter() - t0
        gen = sum(r["generated"] or 0 for r in results if r["test"].startswith("pair:"))
        results.append({"test": "pair: both", "wall_s": round(wall, 2), "generated": gen,
                        "decode_tok_s": round(gen / wall, 1) if wall > 0 else None})
        print(f"  pair: both together    {gen} tokens in {wall:.1f} s = {gen / wall:.1f} tok/s (wall clock)", flush=True)

    out_dir = ROOT / "bench-results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{a.label}.json"
    path.write_text(json.dumps({"label": a.label, "run": run, "url": a.url, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "sampling": SAMPLING, "total_s": round(time.perf_counter() - t_all, 1),
                                "results": results}, indent=1), encoding="utf-8")
    print(f"saved {path} ({time.perf_counter() - t_all:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
