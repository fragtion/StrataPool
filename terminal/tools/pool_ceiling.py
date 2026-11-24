#!/usr/bin/env python3
"""StrataPool's speed ceilings: how fast a decode window could be on each PC and across the pool.

    (in terminal\\)
    ..\\..\\strata\\.venv\\Scripts\\python tools\\pool_ceiling.py                        (the desktop + laptop defaults)
    ... --split 25 --context 100000 --accepted 1.7
    ... --pc desktop:gpu=360,ram=57.6,pcie=31.5,hit=0.84 --pc laptop:gpu=384,ram=89.6,pcie=31.5,hit=0.946

Decode reads memory more than it computes, so the floor of a window is the bytes it must move divided by the speed of
the memory they come from.  This counts the bytes from the model's own files (the GGUF's tensor sizes, the pack's
expert blobs, the MTP drafter and its draft vocabulary), never parameter counts, and splits them by where they live:

  GPU    the dense weights of this PC's layers, the experts the VRAM cache holds (hit rate), K/V and the GDN state,
         and on the coordinator the output head and the drafter (MTP layer + draft head, once per draft)
  CPU    the experts the cache misses that the CPU computes, read from RAM
  PCIe   the misses that are streamed to the GPU instead (--pcie-frac's share)
  link   the hand-off rows between the PCs (a split), plus a fixed latency

A PC's window takes at least max(GPU, CPU, PCIe) (perfect overlap); a split adds its PCs and the link (they run one
after the other unless Overlap is on).  Two routing bounds are shown: every row of a window picks the same experts
(the fewest bytes; the true ceiling) and every row picks independently (the most).  Real routing is in between.

Tokens per window come from a per-draft acceptance p fitted to --accepted (measured tokens per window at --spec):
a(spec) = 1 + p + ... + p^spec.  The sweep then shows which draft length the bytes favour.  This is a model, not a
measurement: the engine's own overheads (launches, syncs, the host's share) are not in it, so a measured window
far above the ceiling says the time goes to overheads, not to bytes.

Bandwidths default to spec-sheet values times practical efficiency (GPU 0.85, RAM 0.75, PCIe 0.8).  Measure them to
refine (a device copy for the GPU, a multi-threaded copy for RAM).  The laptop's values are assumptions: check its
RAM with `Get-CimInstance Win32_PhysicalMemory` and pass --pc laptop:ram=...
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import struct
import sys

DATA = r"D:\ai\Strata-data"
DEF_GGUF = DATA + r"\models\coder-IQ1_M\Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
DEF_PACK = DATA + r"\packs\coder-iq1_m"
DEF_MTP = DATA + r"\mtp\rt"
HERE = os.path.dirname(os.path.abspath(__file__))
DEF_VOCAB = os.path.join(HERE, "..", "data", "draft_vocab_en.bin")

# spec-sheet GB/s: desktop RTX 3060 12 GB (192-bit, 15 Gbps), DDR4-3600 x2, PCIe 4.0 x16;
# laptop RTX 5060 Laptop (128-bit GDDR7, 24 Gbps: check), DDR5-5600 x2 (assumed), PCIe 5.0 x8.
# hit = share of the routed experts the VRAM cache holds (measured 2026-11-20 with stand-ins: desktop alone 84.2%,
# pool worker 94.6%; the coordinator's own share in the pool is assumed equal to its alone figure).
PCS = {
    "desktop": {"gpu": 360.0, "ram": 57.6, "pcie": 31.5, "hit": 0.842, "pcie_frac": 0.35},
    "laptop": {"gpu": 384.0, "ram": 89.6, "pcie": 31.5, "hit": 0.946, "pcie_frac": 0.35},
}
EFF = {"gpu": 0.85, "ram": 0.75, "pcie": 0.80}
GG = 1e9


# ---- GGUF header (tensor names, shapes, types, byte sizes from offsets) ------------------------------------------
def _rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def _str(f):
    (n,) = _rd(f, "Q")
    return f.read(n).decode("utf-8", "replace")


_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _val(f, t):
    if t == 8:
        return _str(f)
    if t == 9:
        et, n = _rd(f, "IQ")
        return [_val(f, et) for _ in range(n)]
    return _rd(f, _SCALAR[t])[0]


def read_gguf(path):
    """(metadata, {name: bytes}) - each tensor's size is the distance to the next tensor's data (padding included)."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit(f"{path}: not a GGUF file")
        _ver, nt, nk = _rd(f, "IQQ")
        kv = {}
        for _ in range(nk):
            k = _str(f)
            (t,) = _rd(f, "I")
            kv[k] = _val(f, t)
        ts = []
        for _ in range(nt):
            name = _str(f)
            (nd,) = _rd(f, "I")
            _rd(f, "Q" * nd)
            _ty, off = _rd(f, "IQ")
            ts.append((off, name))
        al = int(kv.get("general.alignment", 32))
        data = (f.tell() + al - 1) // al * al
    end = os.path.getsize(path) - data
    ts.sort()
    sizes = {name: (ts[i + 1][0] if i + 1 < len(ts) else end) - off for i, (off, name) in enumerate(ts)}
    return kv, sizes


# ---- the model's bytes --------------------------------------------------------------------------------------------
class Model:
    def __init__(self, gguf, pack, mtp, vocab):
        kv, sizes = read_gguf(gguf)
        arch = kv["general.architecture"]
        g = lambda k, d=None: kv.get(f"{arch}.{k}", d)
        self.n_layer = int(g("block_count"))
        self.n_embd = int(g("embedding_length"))
        self.n_exp = int(g("expert_count"))
        self.k = int(g("expert_used_count"))
        self.hc = int(g("hyper_connection.count", 1))
        self.fa_every = int(g("full_attention_interval", 1))
        self.kv_heads = int(g("attention.head_count_kv"))
        self.head_dim = int(g("attention.key_length"))
        self.idx_dim = int(g("attention.indexer.key_length", 0))
        self.idx_topk = int(g("attention.indexer.top_k", 0))
        self.ssm_state = int(g("ssm.state_size", 0))
        self.ssm_heads = int(g("ssm.time_step_rank", 0))
        self.ssm_inner = int(g("ssm.inner_size", 0))
        self.dense = [0] * self.n_layer      # per layer, everything but the routed experts
        self.exp_layer = [0] * self.n_layer  # per layer, all routed experts
        self.head = 0
        self.hc_bytes = 0                    # the hyper-connection mixers (bf16 unless STRATA_HC_REQ8=1)
        for name, b in sizes.items():
            if ".hc_" in name:
                self.hc_bytes += b
            m = re.match(r"blk\.(\d+)\.(.*)", name)
            if m:
                layer, rest = int(m.group(1)), m.group(2)
                (self.exp_layer if "_exps" in rest else self.dense)[layer] += b
            elif name.startswith("output.") or name.startswith("output_hc"):
                self.head += b            # the output head (verify reads it once per window) and its mixer
            # token_embd: one row per token, not read whole
        self.exp_blob = [e / self.n_exp for e in self.exp_layer]
        nat = os.path.join(pack, "native_experts.txt") if pack else ""
        if nat and os.path.isfile(nat):   # the pack's own blob sizes (what the arena holds), when there is one
            for line in open(nat, encoding="utf-8"):
                p = line.split()
                if p and not line.startswith("#") and len(p) >= 5:
                    self.exp_blob[int(p[0])] = int(p[4])
        # the drafter: MTP dense + its routed experts (k of n), and the draft head over the draft vocabulary
        self.mtp_dense = os.path.getsize(os.path.join(mtp, "dense.bin")) if mtp and os.path.isfile(os.path.join(mtp, "dense.bin")) else 0
        mtp_exp = os.path.getsize(os.path.join(mtp, "experts.bin")) if mtp and os.path.isfile(os.path.join(mtp, "experts.bin")) else 0
        mtp_n = 512 if mtp_exp else 1
        dt = os.path.join(mtp, "dense.txt") if mtp else ""
        if dt and os.path.isfile(dt):
            for line in open(dt, encoding="utf-8"):
                p = line.split()
                if p and p[0] == "mlp.gate.weight":
                    mtp_n = int(p[2])
        self.mtp_exp_blob = mtp_exp / mtp_n
        out_bytes_per_row = sizes.get("output.weight", 0) / max(1, self._vocab(kv, arch))
        self.n_dvocab = os.path.getsize(vocab) // 4 if vocab and os.path.isfile(vocab) else self._vocab(kv, arch)
        self.draft_head = self.n_dvocab * out_bytes_per_row

    @staticmethod
    def _vocab(kv, arch):
        t = kv.get("tokenizer.ggml.tokens")
        return len(t) if isinstance(t, list) else int(kv.get(f"{arch}.vocab_size", 248320))

    def full_attn(self, layer):
        return (layer + 1) % self.fa_every == 0

    def union(self, rows, shared):
        """Distinct experts a layer reads for `rows` rows."""
        if shared:
            return float(self.k)
        return self.n_exp * (1.0 - (1.0 - self.k / self.n_exp) ** rows)


# ---- one PC's window ----------------------------------------------------------------------------------------------
def pc_window(m, pc, lb, le, rows, shared, ctx, kv_bytes, coordinator, spec):
    """Milliseconds a window needs at least on one PC for layers [lb, le), and the bytes by memory."""
    gpu = cpu = pcie = 0.0
    for l in range(lb, le):
        u = m.union(rows, shared) * m.exp_blob[l]
        gpu += m.dense[l] + pc["hit"] * u
        miss = (1.0 - pc["hit"]) * u
        pcie += pc["pcie_frac"] * miss
        cpu += (1.0 - pc["pcie_frac"]) * miss
        if m.full_attn(l):
            # sparse attention: the indexer's keys over the whole context, then K/V of its top-k tokens
            gpu += ctx * m.idx_dim * 2 + min(ctx, m.idx_topk or ctx) * 2 * m.kv_heads * m.head_dim * kv_bytes
        elif m.ssm_state:
            gpu += 2 * m.ssm_heads * m.ssm_state * (m.ssm_inner // max(1, m.ssm_heads)) * 4   # GDN state, read + write (f32)
    if coordinator:
        gpu += m.head + spec * (m.mtp_dense + m.k * m.mtp_exp_blob + m.draft_head)
    bw = {k: pc[k] * EFF[k] * GG for k in EFF}
    t = {"gpu": gpu / bw["gpu"], "cpu": cpu / bw["ram"], "pcie": pcie / bw["pcie"]}
    return 1e3 * max(t.values()), {k: v * 1e3 for k, v in t.items()}, {"gpu": gpu, "cpu": cpu, "pcie": pcie}


def link_ms(m, rows, wire_bytes, gbps, latency_ms):
    return rows * m.hc * m.n_embd * wire_bytes * 8 / (gbps * 1e9) * 1e3 + latency_ms


def fit_p(accepted, spec):
    """Per-draft acceptance p with 1 + p + ... + p^spec = accepted (bisection)."""
    lo, hi = 0.0, 0.9999
    for _ in range(60):
        mid = (lo + hi) / 2
        a = sum(mid ** i for i in range(spec + 1))
        lo, hi = (mid, hi) if a < accepted else (lo, mid)
    return lo


def parse_pc(spec_list):
    pcs = {k: dict(v) for k, v in PCS.items()}
    for s in spec_list or []:
        name, _, rest = s.partition(":")
        pcs.setdefault(name, dict(PCS["desktop"]))
        for kv in filter(None, rest.split(",")):
            k, _, v = kv.partition("=")
            pcs[name][k] = float(v)
    return pcs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", default=DEF_GGUF)
    ap.add_argument("--pack", default=DEF_PACK)
    ap.add_argument("--mtp", default=DEF_MTP)
    ap.add_argument("--draft-vocab", default=DEF_VOCAB)
    ap.add_argument("--pc", action="append", help="name:key=value,... (gpu, ram, pcie GB/s spec; hit; pcie_frac)")
    ap.add_argument("--coordinator", default="desktop")
    ap.add_argument("--worker", default="laptop")
    ap.add_argument("--split", type=int, default=25, help="the coordinator runs layers 0..split-1")
    ap.add_argument("--spec", type=int, default=4, help="drafts per window the --accepted figure was measured at")
    ap.add_argument("--accepted", type=float, default=1.7, help="measured tokens per window at --spec")
    ap.add_argument("--context", type=int, default=8000)
    ap.add_argument("--kv-bytes", type=float, default=1.0, help="bytes per K/V element (int8 = 1)")
    ap.add_argument("--wire-bytes", type=float, default=4.0, help="hand-off bytes per value (f32 = 4, f16 = 2)")
    ap.add_argument("--link-gbps", type=float, default=1.0)
    ap.add_argument("--link-latency-ms", type=float, default=0.5)
    ap.add_argument("--max-spec", type=int, default=7, help="the pool link holds 8 rows")
    a = ap.parse_args(argv)

    m = Model(a.gguf, a.pack, a.mtp, a.draft_vocab)
    pcs = parse_pc(a.pc)
    co, wk = pcs[a.coordinator], pcs[a.worker]
    p = fit_p(a.accepted, a.spec)
    L = m.n_layer

    print(f"model: {L} layers, {m.n_exp} experts x top-{m.k}, expert {sum(m.exp_blob) / L / 2**20:.2f} MiB avg, "
          f"dense {sum(m.dense) / GG:.2f} GB (hyper-connections {m.hc_bytes / GG:.2f}), head {m.head / GG:.2f} GB, drafter {m.mtp_dense / 1e6:.0f} MB dense + "
          f"{m.mtp_exp_blob / 1e6:.2f} MB/expert + draft head {m.draft_head / 1e6:.0f} MB ({m.n_dvocab} tokens)")
    print(f"per token, one row: experts {m.k * sum(m.exp_blob) / GG:.2f} GB + dense {sum(m.dense) / GG:.2f} GB + head "
          f"{m.head / GG:.2f} GB;  context {a.context}, K/V {a.kv_bytes:g} B, p = {p:.3f} per draft "
          f"(fitted to {a.accepted} tokens per window at --spec {a.spec})")
    for name, pc in ((a.coordinator, co), (a.worker, wk)):
        print(f"  {name}: GPU {pc['gpu']:g} RAM {pc['ram']:g} PCIe {pc['pcie']:g} GB/s spec (x{EFF['gpu']}/{EFF['ram']}/"
              f"{EFF['pcie']}), hit {pc['hit']:.3f}, pcie_frac {pc['pcie_frac']:g}")
    print()

    for shared in (True, False):
        print("routing:", "every row picks the same experts (ceiling)" if shared else "rows pick independently")
        hdr = (f"{'spec':>4} {'tok/win':>7} | {'alone ' + a.coordinator:>22} | {'alone ' + a.worker:>22} | "
               f"{'pool split ' + str(a.split):>34}")
        print(hdr)
        print(f"{'':>4} {'':>7} | {'ms':>6} {'tok/s':>6} {'bound':>8} | {'ms':>6} {'tok/s':>6} {'bound':>8} | "
              f"{'co ms':>6} {'wk ms':>6} {'link':>5} {'tok/s':>6} {'bound':>7}")
        for spec in range(1, a.max_spec + 1):
            rows = spec + 1
            acc = sum(p ** i for i in range(spec + 1))
            cells = []
            for pc in (co, wk):
                ms, t, _ = pc_window(m, pc, 0, L, rows, shared, a.context, a.kv_bytes, True, spec)
                cells.append((ms, acc / ms * 1e3, max(t, key=t.get)))
            cms, ct, _ = pc_window(m, co, 0, a.split, rows, shared, a.context, a.kv_bytes, True, spec)
            wms, wt, _ = pc_window(m, wk, a.split, L, rows, shared, a.context, a.kv_bytes, False, spec)
            lk = link_ms(m, rows, a.wire_bytes, a.link_gbps, a.link_latency_ms)
            pool_ms = cms + wms + lk
            parts = {"co": cms, "wk": wms, "link": lk}
            print(f"{spec:>4} {acc:>7.2f} | " + " | ".join(f"{ms:>6.1f} {ts:>6.1f} {b:>8}" for ms, ts, b in cells) +
                  f" | {cms:>6.1f} {wms:>6.1f} {lk:>5.1f} {acc / pool_ms * 1e3:>6.1f} {max(parts, key=parts.get):>7}")
        print()

    # where a window's bytes go at the measured spec (independent routing), for the coordinator's share of the pool
    _, t, b = pc_window(m, co, 0, a.split, a.spec + 1, False, a.context, a.kv_bytes, True, a.spec)
    _, tw, bw = pc_window(m, wk, a.split, L, a.spec + 1, False, a.context, a.kv_bytes, False, a.spec)
    print(f"pool at --spec {a.spec}, independent routing, bytes per window:")
    print(f"  {a.coordinator} (layers 0-{a.split - 1} + head + drafter): GPU {b['gpu'] / GG:.2f} GB ({t['gpu']:.1f} ms), "
          f"CPU {b['cpu'] / GG:.2f} GB ({t['cpu']:.1f} ms), PCIe {b['pcie'] / GG:.2f} GB ({t['pcie']:.1f} ms)")
    print(f"  {a.worker} (layers {a.split}-{L - 1}): GPU {bw['gpu'] / GG:.2f} GB ({tw['gpu']:.1f} ms), "
          f"CPU {bw['cpu'] / GG:.2f} GB ({tw['cpu']:.1f} ms), PCIe {bw['pcie'] / GG:.2f} GB ({tw['pcie']:.1f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
