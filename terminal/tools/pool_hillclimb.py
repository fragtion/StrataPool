#!/usr/bin/env python3
"""StrataPool's tuning bench: one setting per run, medians over a fixed panel, controls on both sides, a retrieval check.

    (in terminal\\, with the server running in the setup to measure)
    ..\\..\\strata\\.venv\\Scripts\\python tools\\pool_hillclimb.py run ctl-1  --note "pool split 25, spec 4"
    ... run spec3-1 --note "--spec 3"                  (change ONE thing, restart, run)
    ... run ctl-2                                       (back to the control, restart, run)
    ... compare ctl-1 spec3-1 ctl-2                    (the verdict)
    ... show                                            (every run so far)

The method (from sybil-solutions/inference-optimisation-skill's ground rules, MIT): run-to-run noise here is about
8%, so one run proves nothing.  A run takes --reps passes over a fixed panel of six prompts (each prompt once per
pass, every prompt made unique so no cache answers it) and keeps the median.  A candidate is accepted only if its
median beats BOTH neighbouring controls on the target metric and no other metric regresses by more than 3%; a
bootstrap 95% interval of median(candidate) / median(controls) is printed beside it.  A setting that changes the
output (stand-ins, tail skip, int8 rows) also has to pass the retrieval check (--needle): three codes hidden at 10%,
50% and 90% of a long document, answered exactly.

Requests use the server's own sampling (the config's temperature, top_p, top_k, penalties): greedy runs mislead
(78 tok/s greedy vs ~50 sampled on 2026-11-03).  Thinking stays as configured unless --no-think.  Speeds come from
the engine's own clock (the reply's `timings`), never from counting streamed chunks; tokens per window are
generated / (generated - accepted drafts).  The API key is read from STRATA_API_KEY or from run-coder-iq1_m.bat,
and never printed.  Results append to bench-results/hillclimb.jsonl (ignored by git).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import statistics
import string
import sys
import time
import urllib.error
import urllib.request
import uuid

TERMINAL = pathlib.Path(__file__).resolve().parents[1]
ROOT = TERMINAL.parent
LEDGER = ROOT / "bench-results" / "hillclimb.jsonl"
DEFAULT_URL = "http://192.168.10.50:8080"     # StrataPool's server (127.0.0.1:8080 may be another engine)

# A fixed panel: acceptance and speed depend on the kind of text, so these never change between runs.
PANEL = [
    ("code", "Write a Python function that parses an ISO 8601 duration string such as 'P3DT4H12M' into a "
             "datetime.timedelta, with a docstring, input validation and five unittest cases."),
    ("code", "Write a Rust function that merges overlapping intervals in a Vec<(i64, i64)> and returns them sorted. "
             "Explain its complexity in two sentences."),
    ("code", "Refactor this JavaScript into async/await with proper error handling and explain each change:\n"
             "function load(u, cb) { fetch(u).then(r => r.json()).then(d => cb(null, d)).catch(e => cb(e)); }"),
    ("tool", "Return a JSON object (no prose) describing three HTTP endpoints of a todo-list API: method, path, "
             "request body schema and response schema for each."),
    ("explain", "Explain how a B-tree keeps itself balanced on insert and delete, with a small worked example."),
    ("prose", "Write a 300-word story about a ship's cat who keeps the night watch."),
]
FILLER = ("Strata reads a mixture-of-experts model whose experts live in RAM and in a VRAM cache. The router picks ten "
          "experts per token and layer; the ones the GPU holds run there, the rest on the CPU or over PCIe. A pool "
          "splits the layers across PCs over a local network and hands rows from one PC to the next. ")


def api_key() -> str:
    k = os.environ.get("STRATA_API_KEY", "")
    if k:
        return k
    bat = TERMINAL / "run-coder-iq1_m.bat"
    if bat.is_file():
        m = re.search(r'set\s+"?STRATA_API_KEY=([^"\r\n]*)', bat.read_text(encoding="utf-8", errors="replace"), re.I)
        if m:
            return m.group(1).strip()
    return ""


def post(url: str, body: dict, key: str, timeout: float = 3600) -> tuple[dict, float]:
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + (key or "none")})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.perf_counter() - t0


def looping(text: str, n: int = 8, thresh: float = 0.5) -> bool:
    """A reply whose 8-word shingles repeat more than half the time is a loop, not a fast answer."""
    w = text.split()
    if len(w) < 4 * n:
        return False
    sh = [" ".join(w[i:i + n]) for i in range(len(w) - n)]
    return 1 - len(set(sh)) / len(sh) > thresh


def one(url, key, content, max_tokens, think, run_id):
    body = {"model": "strata", "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": f"[{run_id}] {content}"}]}
    if not think:
        body["reasoning_effort"] = "none"
    try:
        out, wall = post(url, body, key)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return {"error": str(e)[:200]}
    tm, us = out.get("timings") or {}, out.get("usage") or {}
    msg = ((out.get("choices") or [{}])[0].get("message") or {})
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    gen, acc = tm.get("predicted_n") or us.get("completion_tokens"), tm.get("draft_n_accepted")
    return {"wall_s": round(wall, 2), "prompt_tokens": us.get("prompt_tokens"), "generated": gen,
            "decode_tok_s": tm.get("predicted_per_second"), "prompt_tok_s": tm.get("prompt_per_second"),
            "read": tm.get("prompt_n"), "drafts": tm.get("draft_n"), "accepted": acc,
            "tok_per_window": round(gen / (gen - acc), 3) if gen and acc is not None and gen > acc else None,
            "loop": looping(text), "text": text}


def document(tokens: int, codes: dict) -> str:
    """~`tokens` tokens of filler (~4 characters a token) with each code sentence at its depth."""
    body = (FILLER * (tokens * 4 // len(FILLER) + 1))[:tokens * 4]
    out, last = [], 0
    for depth, (name, code) in sorted((d, nc) for nc, d in codes.items()):
        cut = body.rfind(". ", 0, int(len(body) * depth)) + 2
        out += [body[last:cut], f"The access code for the {name} vault is {code}. "]
        last = cut
    return "".join(out) + body[last:]


def needle(url, key, tokens, run_id):
    rnd = random.Random(f"{run_id}-{tokens}")
    names = rnd.sample(["amber", "cobalt", "saffron", "jade", "crimson", "ivory", "onyx", "teal"], 3)
    codes = {(n, "".join(rnd.choices(string.ascii_uppercase + string.digits, k=8))): d
             for n, d in zip(names, (0.1, 0.5, 0.9))}
    q = ("Above is a long document. It contains three vault access codes. Reply with only a JSON object mapping each "
         "vault name to its code, e.g. {\"name\": \"CODE\"}.")
    r = one(url, key, document(tokens, codes) + "\n\n" + q, 120, False, run_id)
    found = sum(1 for (n, c) in codes if c in (r.get("text") or ""))
    return {"tokens": r.get("prompt_tokens"), "found": found, "of": 3, "prompt_tok_s": r.get("prompt_tok_s"),
            "error": r.get("error")}


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 2) if xs else None


def cmd_run(a):
    key = api_key()
    run_id = uuid.uuid4().hex[:8]
    rec = {"label": a.label, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "url": a.url, "note": a.note,
           "reps": a.reps, "think": not a.no_think, "max_tokens": a.max_tokens, "samples": [], "prefill": [],
           "needle": [], "flags": []}
    print(f"{rec['time']} {a.label}: {a.reps} passes x {len(PANEL)} prompts on {a.url}"
          + ("" if key else " (no API key found)"))
    if not a.skip_warm:   # the first request after a start warms caches and the expert profile: not counted
        one(a.url, key, "Say hello in five words.", 16, False, run_id + "-warm")
    for rep in range(a.reps):
        for i, (kind, prompt) in enumerate(PANEL):
            r = one(a.url, key, prompt, a.max_tokens, not a.no_think, f"{run_id}-{rep}-{i}")
            r.update(kind=kind, rep=rep, prompt=i)
            r.pop("text", None)
            rec["samples"].append(r)
            print(f"  pass {rep + 1} {kind:7s} " + (f"ERROR {r['error']}" if "error" in r else
                  f"{r['decode_tok_s']} tok/s, {r['generated']} tokens, {r['tok_per_window']} per window"
                  + (" LOOP" if r["loop"] else "")), flush=True)
    for n in a.prefill:
        r = one(a.url, key, (FILLER * (n * 4 // len(FILLER) + 1))[:n * 4] + "\n\nSummarise this in one line.", 16,
                False, f"{run_id}-pf{n}")
        r.pop("text", None)
        rec["prefill"].append({"target": n, **r})
        print(f"  prompt ~{n}: {r.get('prompt_tokens')} tokens read at {r.get('prompt_tok_s')} tok/s")
    for n in a.needle:
        r = needle(a.url, key, n, run_id)
        rec["needle"].append(r)
        print(f"  needle ~{n}: {r['found']}/3 found ({r['tokens']} tokens)" + (f" ERROR {r['error']}" if r["error"] else ""))
    s = rec["samples"]
    if any("error" in x for x in s):
        rec["flags"].append("REQ_FAIL")
    if any(x.get("loop") for x in s):
        rec["flags"].append("LOOP")
    if any(r["found"] < 3 for r in rec["needle"]):
        rec["flags"].append("NEEDLE_MISS")
    rec["summary"] = summary(rec)
    LEDGER.parent.mkdir(exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print_rows([rec])
    return 0


def summary(rec):
    s = [x for x in rec["samples"] if "error" not in x]
    out = {"decode": med([x["decode_tok_s"] for x in s]), "tok_per_window": med([x["tok_per_window"] for x in s]),
           "n": len(s)}
    for kind in sorted({x["kind"] for x in s}):
        out["decode_" + kind] = med([x["decode_tok_s"] for x in s if x["kind"] == kind])
    for p in rec.get("prefill", []):
        out[f"prefill_{p['target']}"] = p.get("prompt_tok_s")
    return out


def load():
    if not LEDGER.is_file():
        return []
    return [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]


def find(runs, label):
    hits = [r for r in runs if r["label"] == label]
    if not hits:
        sys.exit(f"no run labelled {label!r} in {LEDGER}")
    return hits[-1]


def print_rows(runs):
    keys = sorted({k for r in runs for k in r["summary"]} - {"n"}, key=lambda k: (k != "decode", k))
    print(f"\n{'label':16s} {'time':16s} {'n':>3} " + " ".join(f"{k[:14]:>14s}" for k in keys) + "  needle  flags  note")
    for r in runs:
        nd = " ".join(f"{x['found']}/3" for x in r.get("needle", [])) or "-"
        print(f"{r['label'][:16]:16s} {r['time'][5:16]:16s} {r['summary'].get('n', 0):>3} "
              + " ".join(f"{(r['summary'].get(k) if r['summary'].get(k) is not None else '-'):>14}" for k in keys)
              + f"  {nd:6s}  {','.join(r['flags']) or '-'}  {r.get('note', '')}")


def cmd_show(a):
    runs = load()
    print_rows(runs[-a.last:] if a.last else runs)
    return 0


def boot_ratio(cand, ctl, n=4000, seed=1):
    rnd = random.Random(seed)
    rs = []
    for _ in range(n):
        c = statistics.median(rnd.choices(cand, k=len(cand)))
        b = statistics.median(rnd.choices(ctl, k=len(ctl)))
        rs.append(c / b)
    rs.sort()
    return rs[int(0.025 * n)], rs[int(0.975 * n)]


def cmd_compare(a):
    runs = load()
    c1, cand, c2 = find(runs, a.ctl1), find(runs, a.cand), find(runs, a.ctl2)
    print_rows([c1, cand, c2])
    m = a.metric
    v1, vc, v2 = (r["summary"].get(m) for r in (c1, cand, c2))
    if None in (v1, vc, v2):
        sys.exit(f"metric {m!r} is missing from one of the runs")
    beats = vc > v1 and vc > v2
    regress = []
    for k in sorted(set(cand["summary"]) & set(c1["summary"]) & set(c2["summary"]) - {"n", m, "tok_per_window"}):
        ref = (c1["summary"][k] + c2["summary"][k]) / 2 if None not in (c1["summary"][k], c2["summary"][k]) else None
        if ref and cand["summary"][k] is not None and cand["summary"][k] < ref * 0.97:
            regress.append(f"{k} {cand['summary'][k]} vs {ref:.1f}")
    print(f"\n{m}: control {v1} | candidate {vc} | control {v2}  ({(vc / ((v1 + v2) / 2) - 1) * 100:+.1f}% vs the controls' mean)")
    if m == "decode":
        cs = [x["decode_tok_s"] for x in cand["samples"] if x.get("decode_tok_s")]
        bs = [x["decode_tok_s"] for r in (c1, c2) for x in r["samples"] if x.get("decode_tok_s")]
        if cs and bs:
            lo, hi = boot_ratio(cs, bs)
            print(f"bootstrap 95% interval of median(candidate)/median(controls): {lo:.3f} .. {hi:.3f}"
                  + ("  (excludes 1.0)" if lo > 1 or hi < 1 else "  (includes 1.0: within noise)"))
    flags = cand["flags"]
    ok = beats and not regress and not flags
    print("verdict:", "ACCEPT" if ok else "REJECT",
          "" if beats else "- does not beat both controls;", "; ".join(regress) and "regresses: " + "; ".join(regress),
          ("flags: " + ",".join(flags)) if flags else "")
    if ok and not cand.get("needle"):
        print("note: no retrieval check in the candidate run; a setting that changes the output needs --needle.")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="measure the server as it is now")
    r.add_argument("label")
    r.add_argument("--url", default=DEFAULT_URL)
    r.add_argument("--reps", type=int, default=6, help="passes over the panel (median of reps x 6 replies)")
    r.add_argument("--max-tokens", type=int, default=800)
    r.add_argument("--no-think", action="store_true", help="reasoning_effort none (the config's thinking otherwise)")
    r.add_argument("--prefill", type=lambda s: [int(x) for x in s.split(",") if x], default=[],
                   help="cold prompt reads, e.g. 4000,25000 (approximate tokens)")
    r.add_argument("--needle", type=lambda s: [int(x) for x in s.split(",") if x], default=[],
                   help="retrieval checks, e.g. 32000,120000 (approximate tokens)")
    r.add_argument("--skip-warm", action="store_true")
    r.add_argument("--note", default="", help="what this run's setting is (one change from the control)")
    s = sub.add_parser("show", help="the ledger as a table")
    s.add_argument("--last", type=int, default=0)
    c = sub.add_parser("compare", help="control, candidate, control: accept or reject")
    c.add_argument("ctl1")
    c.add_argument("cand")
    c.add_argument("ctl2")
    c.add_argument("--metric", default="decode")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "show": cmd_show, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
