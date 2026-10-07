# StrataPool against upstream Strata, each PC alone and pooled (2026-10-06)

Every file here is one run of `tools/pool_bench.py` (its docstring says what each test sends): greedy decoding with
presence penalty 1.0 over 512 tokens, thinking off, max tokens fixed per test, a fresh id at the start of every run's
prompts. The `timings` in each record are the engine's own clock (prompt read, decode, drafts).

- Desktop: RTX 3060 12 GB (PCIe 4.0 x16), Core i5-12600KF, 64 GB, Windows 11. Laptop: RTX 5060 Laptop GPU 8 GB,
  Core i7-14650HX, 32 GB, Windows 11. Gigabit LAN, 0.6-0.8 ms round trip.
- Qwen3.8-Flash-Next coder IQ1_M; desktop 256K context, laptop 192K; K/V int8 streamed from RAM (`--kv-resident`).
- Both PCs calibrated on upstream (`tools/calibrate.py`): desktop `--pcie-frac 0.35 --spec-min-p 0.70` (5 workers),
  laptop `--pcie-frac 0.20 --spec-min-p 0.30 --pool-workers 7`, laptop `--expert-cache auto`. The same config and
  learned expert profile for upstream and StrataPool on each PC.
- `upstream-*`: Strata 0.1.40.1 (82f46a8), compiled like StrataPool (MSVC 2022, CUDA 13.0, sm 75/86/89/120,
  `STRATA_PORTABLE=ON`). `sp-*`: StrataPool `pool-support` 019a350 (main 25d6e89).

| File | Setup |
| --- | --- |
| `upstream-desktop-alone.json` | **discarded**: the first run after calibration, 18-37% slower in every test than `upstream-desktop-alone-2` and the same engine's `upstream-desktop-pair` |
| `upstream-desktop-alone-2.json`, `upstream-desktop-pair.json` | the desktop alone, upstream |
| `upstream-laptop-alone.json`, `upstream-laptop-pair.json` | the laptop alone, upstream (its pair run in `-alone` ended early in a tool call; `-pair` is the fixed prompt) |
| `sp-desktop-alone.json`, `sp-desktop-alone-2.json` | the desktop alone, StrataPool (CPU help with chunks staged up to 3,072 tokens) |
| `sp-desktop-noassist.json`, `sp-desktop-noassist-2.json` | the same, `STRATA_PREFILL_CPU=0` |
| `sp-laptop-alone.json` | the laptop alone, StrataPool |
| `sp-split.json` | split layers (desktop 0-28, laptop 29-47), Several chats Off, Overlap Off, Drafts Off |
| `sp-split-overlap.json` | + Overlap the PCs |
| `sp-split-several.json`, `sp-split-several-drafts.json` | Several chats at once (2 slots); + Drafts in several chats |
| `sp-share.json` | Share requests (desktop primary) |

In the several-chats runs the pair's per-request `reused` / `read` are not right (a chat in a batch slot reports the
slot's tokens); its `generated` and `wall_s` are.

## The controlled A/B (`ab-*`)

Upstream and StrataPool alternated twice each on each PC (`up-1`, `sp-1`, `up-2`, `sp-2`), with `--adapt-every 0
--pcie-frac 0` and the same frozen expert profile (`ab-profile-coder.bin`) in both configs; the laptop with
`--expert-cache 300`. `ms_per_window` = decode time / (generated - accepted drafts): the cost of a window apart from
how many drafts its text let it accept. `-lp`: the desktop's engines started "as administrator" (large pages were
still refused, error 1314, with the privilege held and enabled in the token).

| Means of two runs | Code tok/s (ms/window) | Prose | Document answer |
| --- | --- | --- | --- |
| Desktop, upstream | 35.0 (82.5) | 30.1 (64.4) | 27.7 (85.3) |
| Desktop, StrataPool | 36.8 (81.7) | 30.6 (65.6) | 28.0 (86.6) |
| Laptop, upstream | 25.8 (125.4) | 20.5 (118.4) | 23.9 (122.5) |
| Laptop, StrataPool | 25.8 (122.8) | 22.1 (114.2) | 23.8 (111.7) |

The replies' hashes differ between runs of the same engine too: even with swaps and the PCIe share off, the CPU
pool's threads add in an order that varies, and a near-tie flips (`STRATA_IQ_MT_MIN=1` would make it bitwise). So a
single run's speed moves by its text; compare means of alternated runs.
