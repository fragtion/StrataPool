<h1 align="center">StrataPool</h1>

<p align="center"><b>Strata, pooled: several PCs on your network as one</b></p>

> **This is StrataPool's README, not Strata's.** StrataPool is a fork of [Strata](https://github.com/Niko1221/Strata)
> by Niko1221, kept in step with its releases. For everything about Strata itself (the model and its sizes, which
> graphics cards and how much RAM, the app, the API, troubleshooting), read
> **[Strata's own README](https://github.com/Niko1221/Strata#readme)**. It all applies to StrataPool unchanged.

StrataPool adds a **Pool** tab that lets Strata use more than one PC:

- **Split layers:** the PCs divide one model's layers. Each holds its layers' experts in its own VRAM and RAM and its
  share of the context, so together they cache far more of the model than one PC can.
- **Share requests:** every PC runs the whole model. A chat runs on the PC that holds its conversation, else on an
  idle one, so two chats (or an agent's subtasks) run on two PCs at once.

## What to expect

A pool helps most when one PC cannot cache the model well: a mixture-of-experts model whose experts do not fit in
one card's VRAM. Our pair is a desktop (RTX 3060 12 GB, Core i5-12600KF, 64 GB of RAM) and a gaming laptop (RTX 5060
Laptop GPU 8 GB, Core i7-14650HX, 32 GB), both on Windows 11 on a wired 1 Gbps LAN, running Qwen3.8-Flash-Next
**Coder IQ1_M**. Against the desktop alone:

| | What to expect |
| --- | --- |
| **One chat, writing** (Split layers, the defaults) | about **20-30% faster**, the most on prose |
| **+ Overlap the PCs** | a few percent more on code, about the same on prose |
| **Two chats** | about **25-30% more** tokens per second in total: Share requests runs one on each PC; the split runs them one after the other, each at the split's speed |
| **Switching agent conversations** | as quick as on one PC: each PC restores its part of a parked conversation instead of reading it again |
| **Reading a long prompt** | about **a third slower**: every chunk also passes through the slower PC's layers |
| **Several chats at once** (batch slots in a split) | slower on a two-PC home pool; it is for pools that serve many agents at once |

The ranges come from our runs on that pair with [tools/pool_bench.py](tools/pool_bench.py) (greedy decoding, the
same text for every setup) in October 2026, on StrataPool versions from Strata 0.1.40 to 0.1.45. In long agent
sessions on StrataPool with Strata 0.1.45 (100-170K tokens of context, temperature 0.6), the split wrote about 45-75
tokens/s and read new prompt text at about 1,100 tokens/s. Your PCs will differ: `tools/pool_bench.py` sends the same
requests to any server, so run it against each PC alone, the split and Share requests to measure your own pool.

**Where the speed comes from:** memory, not compute. A 12 GB card holds a few thousand of the model's experts and
the CPU computes the rest. Split, each PC caches the experts of its own layers, so more of every word runs on a GPU.
A pool never adds the GPUs together on one word: each word passes through the layers in turn, so the PCs take turns.
That is why the gain is largest when a model is too big for one PC to cache well, and why Share requests adds chats,
not speed per chat.

**Getting the most from it:**

- **Wired network.** Wi-Fi works, but every word crosses it. The `f16` wire format (Pool tab) sends half the bytes
  of `f32`.
- **Keep VRAM free on each PC at long contexts** (`--vram-reserve-mib` in the model's config; about 1,400-1,600 MiB
  on a 12 GB card). A card that fills up during a long prompt makes Windows move GPU memory to RAM: our desktop, with
  770 MiB free, read a 168K-token prompt at about 500 tokens/s instead of about 1,100, and once it reached 0 MiB a
  window stalled.
- **Defaults first.** Split layers with Several chats off is the fastest setup on a two-PC pool; turn on Overlap
  the PCs if your replies are mostly code.

## What StrataPool adds

- **The Pool tab:** this PC's role (alone, Share requests, split coordinator or worker), the workers found on the
  network, the layer map with each PC's experts in VRAM, and the switches below. Every setting is in
  [docs/POOL.md](docs/POOL.md).
- **Split layers over the network:** a coordinator runs the first layers, the head, sampling and the draft layer;
  workers run the rest. Each PC loads only its own layers' weights and experts, keeps its own share of the context,
  and learns its own expert ranking. The split is placed automatically from each PC's VRAM and RAM (including what
  Windows lets a program pin) and refines itself from the timings it measures.
- **Share requests:** each chat runs on the PC that holds its conversation, else on an idle one; a PC that goes away
  is skipped.
- **Parked conversations across the split:** switching between chats or subagents restores each PC's part of the
  conversation instead of reading it again, and a sibling subagent borrows the shared start of a parked one.
- **Overlap the PCs** (one chat): the coordinator starts the next window while the workers finish the current one.
- **Several chats at once / Drafts in several chats:** batch slots pipelined across the PCs, each chat with its own
  repetition penalties.
- **For each PC alone too:** faster prompt tokenizing on long conversations, IQ1_S models on the GPU and the CPU, and
  the fixes listed under Credits.

## What you need

- Two or more PCs on the same local network, each able to run Strata
  ([what Strata needs](https://github.com/Niko1221/Strata#what-you-need)). For Split layers, each needs an NVIDIA
  card (RTX 20 or newer, 8 GB or more) and **the same model and size** installed.
- A wired network is best. Wi-Fi works, but slower.
- DeepSeek-V4.1-Flash runs on its own engine, which has no layer split: in a pool it can use Share requests (each PC
  runs the whole model), not Split layers.

## Install

[Download StrataPool](https://github.com/fragtion/StrataPool/archive/refs/heads/main.zip) and unzip it (or
`git clone` it) on every PC. **Windows:** double-click **`START-HERE.bat`**. **Linux:** run **`./setup.sh`**. Setup
asks the same questions as Strata's and finds the models an existing Strata install already downloaded.

The first start compiles StrataPool's engine (10-20 minutes, once; setup installs the compiler and the CUDA toolkit
itself), because Strata's ready-made engines do not include the layer split. Share requests works with any engine.

Then run `POOL-FIREWALL.bat` as administrator on each PC (and `POOL-INTEGRITY.bat` after each new engine, see
[large pages](docs/POOL.md#windows-large-pages-and-the-low-integrity-label)), open the app's **Pool** tab and pick a mode. The steps for
each mode, every setting, what each PC holds and what is not supported yet: **[docs/POOL.md](docs/POOL.md)**.
`POOL-SELFTEST.bat` checks the whole split on one PC first.

## Branches

`main` is Strata's history plus one commit holding all of StrataPool's changes; that commit is replaced at each sync,
so `main` is force-pushed. Every change one by one is on
**[`pool-support`](https://github.com/fragtion/StrataPool/tree/pool-support)**, which has the same files as `main` and
only moves forward. When Strata rewrote its own history (October 2026), `pool-support`'s commits were replayed onto it
once; both branches from before that are kept as the tags `archive/main-before-upstream-rewrite` and
`archive/pool-support-before-upstream-rewrite`.

## Credits and license

The engine, the installer, the app and almost everything else is [Strata](https://github.com/Niko1221/Strata) by
Niko1221 and its contributors ([support Strata](https://buymeacoffee.com/strataengine)). The model is
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team; Strata's README lists the
compressed versions and every credit. StrataPool is open source under the [MIT License](LICENSE), like Strata. A few
parts and every model have their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).

**Code StrataPool carries that is not (yet) in Strata's own releases.** Thanks to everyone below; each item is credited
in its commit on `pool-support`. Where we measured an effect, it is given; the others are fixes or were not
measured on their own.

| Who | What | What it does for StrataPool |
| --- | --- | --- |
| [architectds](https://github.com/architectds/Strata) | pipelined decode: the verified window served first, the prompt lookup carried across windows, the gate at 0.10 | **Overlap the PCs**, measured 6% slower than the defaults before, now code +8%, prose -3%, the document answer +3% |
| BlueKingMuch - [#1163](https://github.com/Niko1221/Strata/pull/1163), [#1164](https://github.com/Niko1221/Strata/pull/1164) (in Strata since 0.1.45) | a parked conversation's reserve kept small | the split's version of #1164's sibling borrow: each PC borrows its own part, so sibling subagents stop re-reading each other's history |
| Vadim / tirex999 ([DACAN](https://github.com/tirex999/DACAN)) | parked conversations with KV streaming (the idea) | led to parked conversations across the split (above) |
| AncientMystic - [#855](https://github.com/Niko1221/Strata/pull/855) | the PLE page cache without its 45x read amplification | the lookup table's page cache no longer reads the same pages many times over |
| Anon - [#652](https://github.com/Niko1221/Strata/pull/652) | only the tokens a verify window hands out are committed | |
| ATIVX928 - [#1019](https://github.com/Niko1221/Strata/pull/1019) | the int8 K/V gather decodes 8 values per thread | |
| Maxim Shipko - [#1064](https://github.com/Niko1221/Strata/pull/1064) | commit limits applied after an allocation failure | |
| sergiywith - [#1231](https://github.com/Niko1221/Strata/pull/1231), [#1284](https://github.com/Niko1221/Strata/pull/1284) | the elastic K/V without a lend; `--adapt-decay` checked | |
| Yxmura - [#1374](https://github.com/Niko1221/Strata/pull/1374) | IQ1_S on the GPU and on AVX2; IQ1_S and IQ1_M on AVX-512 | Unsloth's UD-IQ1_S models run, on every PC of a pool. Since Strata 0.1.49 the AVX2 IQ1_M rows are Strata's own (about 20-30% faster per expert in our kernel test), and StrataPool sends Qwen's IQ1_M experts to them |
| anon761 - [#1247](https://github.com/Niko1221/Strata/pull/1247) | a layer split inside one PC: `--prefill-help` | for a pool PC with two cards |
| Jeremiah Ritchey - [#1290](https://github.com/Niko1221/Strata/pull/1290) | batch-MTP drafters share their weights fully; a slot's draft row only where the context and its answer leave room | **Drafts in several chats** holds less VRAM |
| lask3802 - [#1748](https://github.com/Niko1221/Strata/pull/1748) (the idea) | a layer split across PCs over TCP ("remote stage"), with every measurement written down | the Windows clock finding behind `POOL-CLOCKS.bat`; workers take the coordinator's `STRATA_ROUTE_TAIL_SKIP` |
| Xiaomi2008 - [#1911](https://github.com/Niko1221/Strata/pull/1911), [#1912](https://github.com/Niko1221/Strata/pull/1912) (in Strata since 0.1.45) | free VRAM through NVML on Windows (opt-in) | `--pool-affinity measure` times a pool PC's own layers |
| xrip - [#1823](https://github.com/Niko1221/Strata/pull/1823) | whole messages remembered by the tokenizer | |
| Saber - [#1362](https://github.com/Niko1221/Strata/pull/1362), Arthur031221 - [#1314](https://github.com/Niko1221/Strata/pull/1314) | two tool-call text fixes | |

Strata 0.1.40.2 took in [#851](https://github.com/Niko1221/Strata/pull/851), [#789](https://github.com/Niko1221/Strata/pull/789),
[#813](https://github.com/Niko1221/Strata/pull/813), [#1063](https://github.com/Niko1221/Strata/pull/1063),
[#1201](https://github.com/Niko1221/Strata/pull/1201), [#1185](https://github.com/Niko1221/Strata/pull/1185) and
[#1101](https://github.com/Niko1221/Strata/pull/1101), which StrataPool carried before - thanks to Francesco Albano,
InB4DevOps, Fringe210, imanu, win10ogod, wOvAN, Chen Cheng and uncle daddy. Strata 0.1.41-0.1.43 took in [#1237](https://github.com/Niko1221/Strata/pull/1237), [#1332](https://github.com/Niko1221/Strata/pull/1332), [#1359](https://github.com/Niko1221/Strata/pull/1359), [#1372](https://github.com/Niko1221/Strata/pull/1372), [#1370](https://github.com/Niko1221/Strata/pull/1370), [#1251](https://github.com/Niko1221/Strata/pull/1251),
[#1190](https://github.com/Niko1221/Strata/pull/1190) and [#1283](https://github.com/Niko1221/Strata/pull/1283) (Zhong Uncle, Ethan, Alekperov Fuad, sergiywith, Thig, Evan), and its own CPU prefill share
replaced architectds' CPU assist. Strata 0.1.45 took in [#567](https://github.com/Niko1221/Strata/pull/567), [#1164](https://github.com/Niko1221/Strata/pull/1164), [#1166](https://github.com/Niko1221/Strata/pull/1166), [#1368](https://github.com/Niko1221/Strata/pull/1368), [#1367](https://github.com/Niko1221/Strata/pull/1367), [#1120](https://github.com/Niko1221/Strata/pull/1120), [#1912](https://github.com/Niko1221/Strata/pull/1912), [#1888](https://github.com/Niko1221/Strata/pull/1888), [#1949](https://github.com/Niko1221/Strata/pull/1949) and [#1305](https://github.com/Niko1221/Strata/pull/1305) (Guillaume Putier and benoit lange, BlueKingMuch, Francesco Albano, sergiywith, bsvinay, Xiaomi2008, KarlGrier, Alex Sadleir, Zack).

## Contributing

Pull requests, forks, issues and suggestions are all welcome.

---

## Support

If StrataPool has been useful to you, consider buying me a coffee:

**PayPal:** [![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?business=2CGE77L7BZS3S&no_recurring=0)  
**BTC:** `1Q4QkBn2Rx4hxFBgHEwRJXYHJjtfusnYfy`  
**XMR:** `4AfeGxGR4JqDxwVGWPTZHtX5QnQ3dTzwzMWLBFvysa6FTpTbz8Juqs25XuysVfowQoSYGdMESqnvrEQ969nR9Q7mEgpA5Zm`
