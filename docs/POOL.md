# StrataPool: several PCs as one

StrataPool has two ways to use more than one PC. Pick one in the **Pool** tab, on each PC.

| | **Share requests** (routing) | **Split layers** (coordinator + workers) |
| --- | --- | --- |
| Each PC holds | the whole model, as it would alone | only its own layers (experts, dense weights, context) |
| One chat | as fast as one PC | often faster than the stronger PC alone (more experts cached), never the sum |
| Two chats at once | both run at full speed, one per PC | one waits for the other |
| What it adds | throughput, and a spare when a PC is off | room: a model, context or expert cache too big for one PC |
| Network | only the request and the answer | every token crosses it; wired is best |
| A PC goes away | the others carry on alone | the pool stops until it is back |

**Neither adds up the PCs' compute for one chat.** Each token passes through the 48 layers in order and needs the
token before it. Two GPUs can work on one token only by taking turns (the layer split) or by exchanging results at
every layer, which over a network costs more time than it saves. A split can still be faster than its stronger PC
alone, because the two GPUs together cache far more of the model's experts: on an RTX 3060 desktop + RTX 5060 Laptop
pool, long agent prompts went from 21-33 tok/s on the desktop alone to 34-40 split (the README has the full table).

## Share requests (routing mode)

Every PC runs the whole model as an ordinary Strata server, with its own conversation cache. A chat request that
reaches **any** of them runs on the PC that suits it:

- **The PC that holds the conversation.** A PC keeps a chat's context (its prompt cache and parked chats), so the
  next turn runs there and reads only the new part. Moving a chat means reading it all again on the other PC, which
  takes minutes for a long one. The router recognises a conversation from its messages (a chain of hashes; the
  prompt-caching markers clients move from turn to turn are ignored).
- **If that PC is busy** and another is idle, the chat moves only when the other PC would read at most *Move a busy
  PC's conversation* tokens again (8,000 by default, about 10 seconds of reading). A long chat waits for its own PC.
- **A new conversation** goes to an idle PC: the **primary** first, then the PC it arrived at, then any other. When
  every PC is busy, it joins the shortest queue.
- **A PC whose context window is too small for the request is skipped.** Each PC keeps its own window (set in its own
  config) and tells its own clients that number. A PC with a 192K window never gets a 230K-token chat; it can still
  send one to a PC with a 256K window. You do not need to make the windows equal.
- **When the other PCs cannot be reached**, each PC answers its own requests and checks for the others every few
  seconds. A request in flight to a PC that stops answering before it starts is run on this PC instead.

Switching between PCs never loses context: the API is stateless, so every request carries the whole conversation.
The answers are the same model's. The sampling defaults (temperature, thinking budget, ...) are those of the PC that
runs the request, so keep the configs alike; the Pool tab marks a PC whose defaults differ.

Setting it up, on **each** PC: choose **Share requests**, give every PC the same **pool secret**, and add the other
PCs' web app addresses (`192.168.1.20` or `laptop:8080`; they also appear under *Found on your network*). Turn on
**This PC is the primary** on the PC new conversations should prefer. Apply. The model keeps running (switching
between *Off* and *Share requests* does not restart it). Point your apps at any PC; the response header
`X-Strata-Pool-Ran-On` names the PC that answered a routed request.

The PCs prove themselves to each other with the pool secret: a routed request carries an HMAC over the time, a nonce,
the path and the body. A routed request skips the receiving PC's own API key; nothing else does.

## Split layers (coordinator + workers)

The model is cut **by layers**. Each PC runs a contiguous range of the 48 layers on its own GPU, and for those layers
it keeps:

- **their experts**, in its own RAM and in its own VRAM cache, and
- **their share of the context**: the KV cache and recurrent state of those layers for every token.

So the pool holds more experts on GPUs than any one of its PCs could, and a long context is divided between them. A
PC whose RAM could not hold the whole model can still take part: it only holds its own layers' experts.

```
 Coordinator (your main PC)              Worker (another PC)
 ┌──────────────────────────┐            ┌──────────────────────────┐
 │ layers 0..K-1            │  rows ──►  │ layers K..47             │
 │ + output head, sampler   │  ◄── rows  │ its experts: RAM + VRAM  │
 │ + draft layer (MTP)      │            │ its KV / GDN state       │
 │ + the server you chat to │            └──────────────────────────┘
 └──────────────────────────┘
```

The **coordinator** is the PC you chat with. It runs the first layers, the output head, the sampler and the draft
layer (speculative decoding), and every decision of the serve loop: checkpoints, the conversation cache, sampling.
A **worker** is a plain function of what it receives. It gets each verify window's residual rows (12,804 floats per
token, about 300 KB for a 6-token window), runs its layers, and sends the rows back. It does the same with each prompt
chunk. It keeps its own layers' state and checkpoints under the ids the coordinator gives it.

This is pipeline parallelism, the same cut as Strata's [multi-GPU layer split](MULTI_GPU.md), with a network in
place of pinned RAM. A token crosses the network once out and once back per verify window, not once per layer. The
experts of one layer are never split across PCs: that would cost a network round trip at each of the 48 layers for
every token.

### What you need

- Two or more PCs on the same local network, each with a supported NVIDIA card (RTX 20 or newer, 8 GB or more).
- **The same model and size installed on every PC** (for example the Coder IQ1_M on both). Each PC reads its own
  layers from its own copy. The coordinator refuses a worker whose model differs, and says how.
- A wired network is best. At 1 Gbit/s a decode window adds about 3-5 ms (two crossings of ~300 KB plus latency).
  A 6,144-token prompt chunk is ~250 MB out in exact mode (about 2 s at 1 Gbit/s) and ~125 MB back. The next chunk goes
  out while the workers still read one, so the network, the coordinator and the workers all work at once
  (`STRATA_POOL_PF_PIPE=0` sends one chunk at a time). Prompt precision in the Pool tab is Exact by default (the
  answers match one PC's); f16 halves the bytes out for a slow network such as Wi-Fi, and answers can then differ very
  slightly. Wi-Fi works, but slower.
- StrataPool's engine on each PC. `START-HERE.bat` compiles it the first time (10-20 minutes, once; it installs the
  compiler and the CUDA toolkit itself): Strata's ready-made engines do not include the layer split. An engine
  without it says so in the Pool tab. Share requests (routing) works with any engine: only the server takes part.
- The **experimental speed projection** (a control vector, `--control-vector-scaled`) works in a split (engines
  built from 2026-11-17 on): set it in the **coordinator's** model config. The coordinator sends each worker the vector's
  rows for its own layers, and switches it per request with the coordinator (`"experimental_speed_projection": false`
  in a request's or the config's sampling turns it off). A worker's own vector arguments are ignored. Every worker needs
  an engine that knows this: the coordinator refuses an older one and says which PC to update.

### Setting it up

1. **On every PC:** install StrataPool with the same model (`START-HERE.bat`; it finds the model files an existing
   Strata install keeps in `Strata-data`). Then run `POOL-FIREWALL.bat` as administrator, once. It opens TCP
   7701 (the engines), UDP 7702 (the PCs finding each other) and TCP 8080 (the app) on private networks.
2. **On the worker PC:** open the app, go to **Pool**, choose **Worker**, and click **Apply**. The chat on that PC
   turns off and its GPU waits for a coordinator.
3. **On the coordinator PC:** open **Pool** and copy the **pool secret**. Paste it into the worker's Pool tab and
   Apply there too. Every PC of a pool must have the same secret.
4. **On the coordinator PC:** choose **Coordinator**. Add the worker: it shows up under *Found on your network*, or type
   its address (`192.168.1.20`, or `laptop:7701`). Click **Apply**. The model restarts. The coordinator connects to
   the worker and places the split. Both PCs then load their layers at the same time, which takes a minute or two.
   The **Layer map** shows which PC runs which layers, and how many experts each holds in VRAM.

Then chat on the coordinator, or point your apps at it, as before (`http://<coordinator>:8080/v1`).

With more than one worker, the order in the Workers table is the layer order. Use the arrows to change it.

### Windows: keep the GPUs at full clocks

In a split every PC's GPU waits while the other PCs run their layers, and the Windows driver lowers its clocks in those
gaps (the Linux driver does not). In Strata PR #1748, which runs a layer split across PCs too, a worker RTX 3070 spent
94% of a decode run below full clocks and its part of a window took 12-22 ms instead of 3.7-4.1 ms, decode 33.8
against 41.8 tok/s with its clocks locked. `POOL-CLOCKS.bat` (as administrator, on each PC) holds the graphics clock
between 75% of its maximum and the maximum and the memory clock at its maximum through `nvidia-smi -lgc` / `-lmc`;
`POOL-CLOCKS.bat reset` gives them back to the driver, as a reboot does. Not measured on StrataPool's own PCs yet: run
`pool_bench` with and without it. A laptop GPU or driver may refuse the memory lock (it says so); the graphics lock is
the one that matters most.

### Windows: large pages and the Low integrity label

Windows starts a program at the lower of its parent's integrity and its file's mandatory label. A folder shared with a
sandbox (Claude's Cowork, some sync tools) can carry a **Low** label that every file created in it inherits, and an
engine running at Low has "Lock pages in memory" taken out of its token: large pages fail with error 1314, and the
working-set lock that lets Windows pin more of the experts fails too. `POOL-INTEGRITY.bat` (as administrator, on each
PC) relabels `terminal\engine\*.exe`, the image generator (`terminal\engine-image\`), `terminal\.venv`'s Python and
the Python that venv runs to Medium (an install from before Strata 0.2.16 keeps them at the top: those too), the same
for a Strata install next to the folder
(`..\strata`) and any venv, folder or `.exe` named on its command line; removes the downloaded-file mark from an
engine copied from another PC; and says whether the account holds "Lock pages in memory" (granted in `secpol.msc`,
effective at the next logon). A rebuilt, copied or updated engine takes the folder's label again, so run it after
each one. The folders themselves are left alone. A Low program also cannot write outside Low-labelled folders,
which is why the image generator could not save its pictures.

## Testing it on one PC first

`POOL-SELFTEST.bat` checks the whole pool path on a single PC. Close Strata first, so the GPU is free.
The test runs the model alone. Then it runs the same model split between a coordinator and a worker on this same GPU,
connected over `127.0.0.1`, and uses the same prompts. It prints both answers side by side, how many tokens agree,
and the decode speed of each run.

Both engines share one GPU, so each gets a small expert cache and a short context (`--cache 400 --ctx 8192` by
default). The answers need not match token for token. A GPU-computed expert rounds slightly differently from a
CPU-computed one, and the pair caches different experts than the single engine. The answers should still make sense
and mostly agree.

```
POOL-SELFTEST.bat                         (or, in terminal\: .venv\Scripts\python tools\pool_selftest.py strata-<model>.json)
POOL-SELFTEST.bat --split 30 --wire f16   other options: --ctx, --cache, --chunk, --max-new, --skip-single
```

## Settings (Pool tab)

| Setting | What it does |
| --- | --- |
| **Role** | *Off*: this PC runs the model alone, as Strata. *Share requests*: routing mode (above). *Split: coordinator*: it uses the workers listed. *Split: worker*: it lends its GPU and RAM to a coordinator. |
| **Pool secret** | Proves to a worker that a coordinator belongs to the pool, and the other way round (HMAC-SHA256 challenge, both ways). The secret itself never crosses the network. |
| **Worker port** | The TCP port a worker's engine listens on (7701). |
| **Split** | *Automatic* (the default) weighs each PC's free VRAM after its dense weights, its free RAM for its layers' experts, and its GPU's speed. It uses the same cost model as the multi-GPU split, plus a network hop per PC. *Manual*: the first layer of each worker, e.g. `30`, or `16,32` for two workers. The coordinator keeps at least layers 0 and 1. |
| **Network precision** | *Exact* (f32) sends 32-bit rows, so the answers match one PC's exactly. *f16* / *bf16* halve the bytes on a slow network, with a small rounding difference. The last worker's prompt rows feed only the draft layer, so they always travel as f16. Drafts are only guesses that the model checks, so this never changes the output. |

The settings live in `strata-<model>.pool.json` next to the model's config. The role is kept between starts.
`terminal\run-<model>.bat --role off` (or `coordinator` / `worker`) overrides it for one start.

## What each PC holds

Each PC loads only its own layers: their experts in RAM, their dense weights and their part of the expert cache in
VRAM, and their part of the context (KV and state). A worker leaves out the embeddings and the output head, and the
coordinator leaves out the workers' layers. So the VRAM that the other layers' dense weights would take goes to
experts instead, and the pool as a whole caches more of them than any one PC.

With `--kv-resident` (KV streaming) each PC also keeps a full copy of its layers' K/V in pinned RAM, one per session:
the main one, plus one per chat slot when **Several chats at once** is on. The automatic split counts those copies
next to the experts when it checks a PC's RAM. On Windows it also keeps a PC's experts and K/V copies under about half
its RAM, less 2 GiB, because that is roughly what Windows lets a program pin (a 32 GB laptop failed at about 14 GiB
and ran at 12.7). Set `STRATA_POOL_PIN_GIB=<n>` in a PC's config `env` to give its own limit, or `=0` for none.

### Parked conversations

With `--conversation-cache-mib` in the coordinator's config (setup suggests it when there is RAM to spare), a split
parks a conversation it switches away from, as one PC does: an agent's subtasks, a second client, or a title request
no longer make the main conversation read again everything after the part they share (usually the system prompt) when
it comes back. Each PC keeps its own part in its own RAM: the coordinator its layers and the draft layer, under the
budget and slot count of its config; each worker its layers, as long as that leaves
`--conversation-cache-min-free-mib` (default 2,560 MiB) of its RAM free. A worker that cannot park its part says so in
the coordinator's log (`skip parking`), and that conversation is read again when it comes back, as before. A new
conversation that starts like a parked one (a sibling subagent with the same system prompt and tools) borrows that
start on every PC and leaves the parked one in place (Strata PR #1164). The worker's log shows each park, restore and
borrow, and how much it holds. Every PC must run an engine with this (an older worker turns it off, with a note in the
coordinator's log).

### Reading short prompts

Every PC's CPU pool computes part of a short prompt chunk's least-routed experts from RAM while its GPU copies in
the rest: Strata's own CPU prefill share, on by default since 0.1.41 on one NVIDIA GPU for chunks below 1,024 tokens
(`STRATA_PREFILL_CPU_SHARE=0` in a PC's config `env` turns it off, `STRATA_PREFILL_CPU_SHARE_MAX=3072` extends it to
larger chunks). In a pool each PC runs its own layers on its own CPU pool, so every PC takes it for its layers. It is
not bitwise (mean KL about 0.004, Strata's measurement). Strata measured 512-token prompts 28-34% and 1,000-token
prompts 20-26% faster on an RTX 3060; StrataPool's earlier CPU assist (from architectds' fork, which this replaces)
measured 11% at ~560 tokens on the desktop below.

## The split learns from its timings

After each request the coordinator's log splits a decode window into its parts:

```
strata pool: <N> windows, <W> ms each = this PC <C> ms + the workers' layers <L> ms + the network <X> ms ...
strata pool:   this PC: its layers <A> ms + the head, sampling and the drafter <B> ms
```

The head, the sampling and the draft layer always run on the coordinator, so no split moves that part. The
coordinator compares each PC's measured layer time with what the built-in estimate predicted for that PC's layers,
and keeps the ratio per pool in `pool-split-measured.txt`, in the engine's working folder (`terminal\`). One line per pool: the model, the
context, the KV format, the window, and the PCs' names. After 200 windows, the next **automatic** split predicts with
the measured numbers instead of the estimate. One split's numbers cannot tell a PC's own layer time from what its
cache misses cost, so the search moves at most 3 layers from a split it measured; each split it tries is measured in
turn. It moves workers that still hold their layers only when the measured numbers say the new split is at least 5%
faster (they reload once, a minute or two). If a worker cannot load a split the measured numbers chose (it ran
out of memory on the way), the file notes it (`failed=`), and later automatic splits give that worker less than it
had there. Delete the file to
start again from the estimate, or set `STRATA_POOL_CALIB=0` in the config's `env` to turn it off
(`STRATA_POOL_CALIB=<file>` keeps it elsewhere).

## What each PC's expert cache learns

With `"expert_profile_save"` in the model's config, each PC saves which experts its requests used, so the next
start fills the cache with them (Strata #477). In a pool, a PC sees only its own layers, so it re-ranks only those
layers' experts and keeps the order it started with for every other layer. The saved file is still a ranking for the
whole model: the same PC can start alone, share requests, or take another split without a lopsided cache. A worker
saves between requests (every `expert_profile_save_every` minutes, 10 by default) and when its coordinator leaves.

## Several chats at once in a split

**Several chats at once** in the Pool tab (coordinator, off by default) gives the split one batch slot per PC
(Strata's `--batch`, docs/BATCHING.md; fewer when they do not fit). More would only share the same speed among more
chats: with two PCs, two chats already keep both busy. A chat alone still runs as before, with drafts. When a second one arrives, both continue
in slots: one word per step each, without drafts, and the steps are pipelined across the PCs. While the laptop runs
chat A's layers, the desktop runs chat B's, so neither PC waits for the other.

Measured on the desktop (RTX 3060) + laptop (RTX 5060 Laptop) pool (October 2026, the README's table): two chats
at once wrote 36-39 words/s together, against 44 for the same two chats one after the other, and a chat alone ran
35-39 instead of 44. On that pool it is a loss; it can pay where one PC's step is much shorter than the window (a
fast worker, or more PCs).

What it costs:

- Every slot has its own state on every PC (its layers' KV and recurrent state), in VRAM the expert cache would
  otherwise use, so a chat alone runs a little slower than with one slot. With KV streaming (`--kv-resident`) each
  slot's whole context also takes pinned RAM.
- Each chat in a slot keeps its own repetition penalties: the head runs on the coordinator, which keeps every
  slot's recent tokens and applies them to that slot's rows. (One PC's batch windows still apply none.)
- The PCs' slot counts must match: the coordinator uses as many as every worker could carve.
- Leave it off if you run one chat at a time: the slots then cost speed and memory for nothing.
- For two chats at once, *Share requests* runs each at one PC's full speed (each PC holds the whole model). The
  split's slots suit a pool that only works as a split (a model too big for one PC, or one fast PC with a slower
  helper).

## Overlap the PCs (one chat)

**Overlap the PCs** in the Pool tab (coordinator; Strata's `--pipeline-windows 2`) lets the coordinator start the next
decode window on a guess while the workers still run the current one. The draft layer guesses that the current window
is accepted whole and what its next word is; when the verdict says otherwise, the coordinator puts its layers' state
back and runs the right window. The workers only ever get verified windows, in order, so nothing changes on them (a
worker started with an older engine works too). It helps most when the workers' part and the network are a large share
of a window and the drafts are usually right. Upstream measured +14-16% on two GPUs in one PC; on the desktop + laptop
pool it was about 6% slower (41-42 against 44 tok/s, 41 against 38 ms a window: 1,907 of 1,908 windows ran overlapped,
but too many guessed windows were thrown away). It needs the draft layer and is off while several chats run (the log
says so). Requests with repetition penalties (`penalty_last_n`) overlap too: the head on the coordinator sees each
window after the one before it. Since that measurement the coordinator serves the verified window's layers before a
guessed one's (a guessed window one layer at a time while the workers run the verified one), guesses more often (the
gate went from 0.20 to 0.10), and keeps copying from the prompt lookup across windows when the text repeats an earlier
stretch (all three from architectds' fork, which measured +8.9% on a two-GPU PC). Measured on the same pool with
tools/pool_bench.py (greedy, 2026-10-06), against the defaults: code 55.4 against 51.3 tok/s (+8%), prose 50.5 against
52.2 (-3%), the answer to a 9.8K-token document 50.1 against 48.8 (+3%). Keep it on if your replies are mostly code;
compare on your own PCs.

## Which switches pay

Measured on 2026-11-03 (engine 0.1.56, desktop RTX 3060 coordinator + laptop RTX 5060 Laptop worker, one chat, a
1,000-word answer about TCP with thinking on, the shipped sampling: temperature 0.6, top_p 0.95, top_k 20, presence
1.0 over 512 tokens; three seeds each, decode tok/s):

| Pool tab | Split | Decode | Why |
| --- | --- | --- | --- |
| Several chats + Overlap (with `STRATA_PW_WITH_BATCH=1`) | 29 | 39.1 | as below, plus the overlap's cost |
| Several chats | 29 | 42.4 | each slot's K/V copy of a 262K context takes pinned RAM on the laptop, so it can only hold layers 29-47; the desktop's 29 layers then share a smaller cache (78% hits) |
| Overlap | 25 | 49.1 | two thirds of the guessed windows were thrown away (sampled text), and a window the loop does not guess takes 36 ms against 31.5 serially |
| **neither** | 25 | **51.2** | desktop 17.5 ms + laptop 12 ms + network 2 ms a window, 94% hits on the desktop |
| (the desktop alone) | - | 42.6 | |

So for one chat at a time the plain split is the fastest: +20% over the desktop alone. A stricter guess gate for the
overlap (`STRATA_PIPELINE_DEBUG=1 STRATA_PIPELINE_THETA=0.5` or `0.8`) was slower still (44-46). With greedy decoding
nearly every draft is accepted and all of these run much faster (78 tok/s with both switches on), so compare with the
sampling you use. Turn Several chats on only when two chats must really run at once (or lower the context, which
shrinks each slot's K/V copy).

Long prompts, measured on 2026-11-16 (engine 0.1.61, the same two PCs over 1 Gbit/s, split 28, a 25,723-token prompt
read from scratch, two runs each; prompt tok/s):

| Prompt chunks | Prompt | Why |
| --- | --- | --- |
| one at a time, exact rows (`STRATA_POOL_PF_PIPE=0`, Prompt precision Exact) | 928 / 923 | each chunk waits for its rows to cross the link both ways and for the laptop's layers |
| **overlapped, exact rows** (the default: Prompt precision Exact) | **1,236 / 1,250** | the next chunk crosses while the laptop reads one: +34% |
| overlapped, f16 rows (Prompt precision f16) | 1,243 / 1,280 | half the bytes out; the link is no longer the slow part, so only 1-2% more (within noise), and not exact |
| (the desktop alone, earlier the same day) | 1,276 | |

So a long prompt now reads as fast as on the desktop alone (it was 30% slower), and the decode keeps the split's gain.
Decode was unchanged by the prompt switches (about 41 tok/s in both runs; that day the desktop shared its card with
another engine). Greedy replies differ from run to run even on one setting (drafts and the GPU's rounding), so
the overlap's exactness is checked in `strata-pool-link-test` instead: with exact rows the same bytes go out and
come back, in order.

### Tuning your own pool

Two tools in `terminal\tools\` help with tuning. Their method comes from
[inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill) (MIT):

- **`pool_ceiling.py` estimates how fast a decode window could be.** It reads the model's own files (the GGUF's tensor
  sizes, the pack's expert blobs, the drafter and its draft vocabulary) and counts the bytes each PC must move per
  window: dense weights, the cached experts and K/V on the GPU, the missed experts from RAM or over PCIe, and the rows
  over the link. It divides each by that memory's speed, for each draft length.
  - Set your PCs with `--pc name:gpu=,ram=,pcie=,hit=`, then `--split`, `--context`, and `--accepted` (measured
    tokens per window).
  - On the two PCs above at split 25 it gives 17.3 ms for the coordinator, 12.3 ms for the worker and 2.1 ms for the
    link. The measured figures are 18.4, 12.2 and 2.2 ms. So this split is bound by memory bandwidth, and what pays now
    is fewer bytes per window: fewer drafts, smaller dense weights, or more cache hits.
  - The tool assumes perfect overlap and leaves out the engine's own overheads. A measured window well above its
    figure means the time goes to overheads, not to bytes.
- **`pool_hillclimb.py` is an A/B bench.** It uses a fixed panel of six prompts with the config's own sampling, and
  6 passes by default; each result is a median.
  - Run a control, then the candidate (one change), then the control again, and `compare` them. A candidate is
    accepted only if it beats both controls and nothing else drops by more than 3%. The comparison prints a bootstrap
    95% interval.
  - `--needle 32000,120000` hides three codes in a long document and checks that all three come back exactly. Use it
    for any setting that changes the output.
  - Results go to `bench-results\hillclimb.jsonl`.

## Drafts in several chats

**Drafts in several chats** (shown when Several chats at once is on; Strata's `--batch-mtp`) gives each chat's window
its draft too: a window then holds two rows per chat, and the head keeps the draft where it agrees, so a chat can get two
words a window. The PCs then run one window over every chat (the coordinator's layers, then the workers', then the head
here) and commit after the verdict, instead of the chats taking turns on the PCs. On the desktop + laptop pool it was no
faster: about 33 words/s for two chats together, against 36-39 without the drafts.

## Which PCs make good workers

The pool runs its layers one after the other, so a token waits for every PC's part in turn. A worker helps when its
layers run about as fast as the coordinator's would, and when its RAM and VRAM let the pool cache experts that would
otherwise miss. A much slower GPU adds its slowness to every token, and the network adds about 1.5 ms each way.

For example, a GTX 1050 (4 GB, Pascal) takes around 10 ms per layer, where an RTX 3060 takes about 1.5 ms. Even one
layer on it costs more time than the experts it could cache save. Its prompt processing is slow as well, because
Pascal has no tensor cores. So the engine is built for Turing (RTX 20xx) and newer, and a GPU like that is better left
out of the pool. The automatic split would give it as few layers as it can.

## When a worker stops, or you change something

- A worker that loses its coordinator **keeps its layers loaded**. When the coordinator comes back with the same
  settings (it restarted, or it was unloaded while idle), the worker continues at once. If the coordinator asks for
  other layers, the worker restarts to load them and the coordinator waits for it.
- If the coordinator cannot reach a worker, the Pool tab says why: not reachable, a different model, a different
  secret, or busy with another coordinator. **Restart engine** tries again.
- If a worker fails during a request, the request ends with an error and the next request starts the pool again.

## What is not supported yet

- **Images** (vision) and the **low-RAM modes**. The Pool tab says when your setup has one of them on.
- **Several GPUs in one pool PC.** Each pool PC uses one GPU, and the pool is the split.
- From Strata 0.1.40: **`--kv-grow`** (the elastic K/V; off with a note - it also needs the whole K/V in VRAM, which a
  long context with `--kv-resident` does not have) and **session files** (`/slots/0?action=save|restore` answers that
  a pool does not support them yet: the other PCs hold their layers' part of the conversation).

## Engine flags

`terminal\run-<model>.bat` gets these from the Pool tab, so you do not normally type them.

```
--pool-peers H:P[,H:P..]     coordinator: the workers, in layer order (default port 7701)
--pool-split auto|K1[,K2..]  where each worker's layers start
--pool-listen [H:]P          worker: wait for a coordinator here
--pool-secret S              the shared secret (also STRATA_POOL_SECRET; the app passes it in the environment)
--pool-wire f32|f16|bf16     the rows on the network (f32: exact, the default)
--pool-draft-wire F          the last worker's prompt rows (default f16; they feed only the draft layer)
--pool-prefill-wire F        the prompt rows to the workers (default: as --pool-wire; the Pool tab passes its Prompt precision, Exact = f32 by default)
--pool-timeout-s S           a worker's reply (default 300)
--pool-wait-s S              connecting to and loading the workers at start (default 900)
```

## Opt-in switches for a split

Some of Strata's engine switches are off by default and can be tried on a pool. Put them in the coordinator's
`<config>.pool.json` under `"env"` (or in the model config's own `"env"`), for example:

```json
"env": {"STRATA_PARK_ASYNC": "1", "STRATA_CKPT_ASYNC": "1", "STRATA_PREFILL_COPY_EVENT_EVERY": "4"}
```

The coordinator hands the pool-wide ones below to every worker when the split starts (a worker that sets one itself
keeps its own; its log says `strata pool: from the coordinator: ...`). A change restarts the coordinator (Apply in the
Pool tab); a worker that kept its layers loaded from the last coordinator sees the other switches and reloads itself
(engine 0.1.62 StrataPool builds from 2026-11-20 on; an older worker keeps the switches it started with until it is
restarted by hand). Most of them have not been measured on a pool yet, so compare a few replies or prompts with each one
on and off.

| Switch | What it does | Where |
| --- | --- | --- |
| `STRATA_PARK_ASYNC=1` | parking a conversation copies its K/V to RAM in the background (the copy into a GPU buffer is done before the park returns, so the next request cannot overwrite it) | pool-wide |
| `STRATA_CKPT_ASYNC=1`, `STRATA_CKPT_GATHER=1` | a conversation checkpoint's running state is copied in the background; its small pieces in one copy | pool-wide |
| `STRATA_PREFILL_COPY_EVENT_EVERY=N` (try 4-8), `STRATA_PREFILL_COPY_WAIT_DEDUPE=1` | while a prompt streams experts: one copy event every N experts instead of one each, and one wait per release (#4005). Aimed at Windows, where each event record or wait is a submission of its own (about 80 / 45 microseconds on Windows 11); the same bytes | pool-wide |
| `STRATA_PREFILL_ROUTED_STREAM=1` | a prompt chunk does not copy the streamed experts no token of it routes to (#4283, by Stefano Sinden; RTX 3090: a 1,790-token follow-up read 18% faster, 2K / 4K / 32K reads 20% / 8% / 5%; the same bits; in Strata since 0.2.10). It stays off where `STRATA_PREFILL_COPY_EVENT_EVERY` is above 1 or `STRATA_PREFILL_COPY_WAIT_DEDUPE=1` (the engine says so once) | pool-wide |
| `STRATA_PF_GROUP_SCRATCH=1`, `STRATA_PF_HC_REGION=1` | smaller prompt scratch, so `--prefill auto` picks bigger chunks (#4039, by merbanan; RTX 2060S 8 GB: prompts +18% / +24% / +21% at 4K / 16K / 32K; the same bits). The pool's chunk is the smallest any PC can take, so the 8 GB worker's gain counts most | pool-wide |
| `STRATA_STAGE_RESIDENT=0` | back to the old prompt staging: since #2968 (by sergiywith) a PC with more than 10% of its experts unpinned (a worker that may pin only half its RAM) keeps the full streamed ring and gets 8 copy threads (RTX 5090, Windows: 32K prompts +13%, 2K 1-2.5% slower) | pool-wide |
| `STRATA_PREFILL_CPU_SHARE_BATCH=1` | with Several chats: a prompt read uses the CPU expert share while no slot decodes (#3382, by Vop3p; 2x RTX 3090: follow-ups 19-49% faster) | coordinator only (a worker does not know which slots decode) |
| `STRATA_ROUTE_PRIOR=1.5` | residency-biased routing (faster on 12 GB cards; off by default since Strata 0.2.8, #4402, because it was tied to repetition loops in long agent sessions): on every PC | pool-wide |
| `STRATA_ROUTE_RESIDENT=M`, `STRATA_ROUTE_RESIDENT_RANKS=lo-hi`, `STRATA_ROUTE_RESIDENT_BUDGET=B` | stand-ins: after the router's pick, an expert at ranks lo..hi that is not in this PC's VRAM is replaced by the best one that is, if its logit is within M (`1e30` = any distance), and with a budget, only if the replaced pick's weight x logit gap is at most B (Strata-HotRod's rule; without it, ranks 3-9 at any distance gave malformed JSON / code there). Each PC swaps against its own cache, so a worker's layers use the worker's VRAM. StrataPool runs it on 512 x 10 (Flash-Next), 256 x 10 (the Coder) and 256 x 8; upstream only on 512 x 10. It changes the output | pool-wide |
| `STRATA_L3_PF=1` (`STRATA_L3_PF_TOP=4`, `STRATA_POOL_STATIC=1`) | after each layer's CPU experts, each worker thread reads its fixed row slice of the next layer's likely CPU experts (that layer's router on this layer's input, top 4 by votes) into the CPU caches while the GPU works; a node predicts only its own layers. From upstream PR #6550 (by gcp; in Strata since 0.2.20, where it is on by default on a card of 13 GiB or less with 8+ CPU workers and fewer than half of the node's experts cached, so a laptop worker may take it by itself): +6% decode on an RTX 3090 + Ryzen 9 5950X, output unchanged. On this pool's desktop (5 CPU workers) it was slower: see the A/B notes. `STRATA_L3_PF=0` in the pool env turns it off on every PC | pool-wide |
| `STRATA_PCIE_EARLY=1` | the PCIe share's fetch kernel (the missed experts read over PCIe, `--pcie-frac`) runs on its own stream beside the layer's VRAM experts instead of after them. From Strata-HotRod 07f52d929 (where it measured about 7% off the decode window on an RTX 3060 that streams experts from a drive; on by default there). Not measured on a pool yet | pool-wide |
| `STRATA_SM86_DOWN_WS=6` | sm_86 only (RTX 30): decode down kernels without stack spills for IQ4_NL / Q2_0 down experts (the Coder's), bit-exact. From upstream PR #6581 (by nmqx; in Strata since 0.2.20): +3.6% decode on 2x RTX 3090. Set it in the coordinator's model config, not the pool env: untested on other cards | this PC |
| `STRATA_PW_WITH_BATCH=1` | with Several chats: a chat that is alone keeps **Overlap the PCs** (the pipelined windows); the batch windows of several chats run without it (Vop3p's #3095, in Strata since 0.2.10; 2x RTX A6000: a lone request +14% on code, +3% on a long decode). On the desktop + laptop pool it was slower (a lone chat 39 against 42 tok/s, see "Which switches pay"), so the Pool tab does not set it. Without it, Several chats turns Overlap off | coordinator |
| `STRATA_PIPE_ADAPT_EVERY=N` | with Several chats: the expert cache's swap round every N group windows (the pipeline empties for it; the workers already swap between the windows they serve) | coordinator |
| `STRATA_PARK_VRAM_MIB=N`, `STRATA_CKPT_VRAM_SLOTS=N` | parked K/V and checkpoints kept in VRAM in front of the RAM copy: costs expert-cache room | each PC, in its own config |
| `STRATA_DEFAULTS=026` | every default Strata 0.2.7 changed goes back to 0.2.6 | each PC (a worker reads it before the coordinator's settings arrive) |

Layer-major prompt reading (`STRATA_PREFILL_LAYER_MAJOR`, Strata #3976) stays off on a pool PC: its walk has no
hand-off to other PCs yet. For the same reason Strata 0.2.10's small-card prompt rule (unfused experts plus the
layer-major walk below an 8K chunk) is not applied on a pool PC.

What a pool PC does not take from Strata 0.2.10 yet: the DFlash2 drafter (a pool starts with the MTP drafter; the
engine refuses `--draft dflash2` on a pool PC, because the hand-off carries the MTP's rows, not DFlash2's taps; Coder
has no DFlash2 drafter anyway), prompt-lookup windows past 8 rows (`--lookup-max-t`, which a layer split does not take
either), and the all-cards split engine (`strata-split`: every card computes a share of every layer and the cards
exchange sums twice per layer, which needs the cards in one PC).

## How it is built

| Part | Where |
| --- | --- |
| Frames, control messages, row codecs (f32/f16/bf16) | `engine/include/strata/pool/protocol.hpp`, `engine/src/pool/protocol.cpp` |
| Sockets, SHA-256 / HMAC handshake | `engine/include/strata/pool/net.hpp`, `engine/src/pool/net.cpp` |
| The worker's handshake and its "busy" answer | `engine/src/pool/worker.cpp` |
| The split search | `engine/src/pool/split.cpp` |
| The coordinator's link: workers, the head, prompt rows | `engine/src/pool/link.cpp` |
| The verifier's and prompt path's hand-off to the network | `Verifier::set_link` / `set_headless`, `Prefill::remote_next` / `set_headless` |
| Each node's experts in RAM for its range only | `ArenaExpertSource::set_layer_range` |
| The worker's serve loop, and the coordinator's hooks | `engine/src/program/generate.cpp` (search for `POOL`) |
| The server's side: roles, the worker supervisor, discovery | `terminal/serve/pool.py`; the Pool tab: `terminal/serve/web/pool.js` |

Tests (no GPU needed): `strata-pool-test` covers hashing, the wire formats, frames, the handshake and the split
search. `strata-pool-link-test` runs the coordinator against scripted workers: the secret, the model check,
reloading, busy workers, prompt rows through two workers, and the lazily acknowledged commits.
`python -m unittest serve.test_pool` covers the config, the supervisor, discovery, and switching a running server
between roles.

The protocol in one paragraph: one TCP connection per worker. The worker sends `HELLO` (its model, its GPU, its free
VRAM and RAM, and a nonce). The coordinator answers `AUTH` (an HMAC of the nonce under the secret, plus its own nonce),
and the worker answers `AUTH_OK` the same way. Then `CONFIG` gives the worker its layer range and the settings that
must match (context, KV format, window size, rope), and the worker loads and answers `READY`. After that each request
gets exactly one reply, in order: `VERIFY` gets `VERIFY_ROWS`, and `PREFILL` gets `PREFILL_ROWS`. `COMMIT`, `RESET`,
`CKPT_SAVE`, `CKPT_RETAIN` and `END_REQUEST` are acknowledged lazily. The coordinator reads those acknowledgements
before its next request, so a commit costs no round trip of its own.
