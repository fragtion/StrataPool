# V100 / Swift IQ2_XS / 32k / --kv int8: the KV read path

Test bed: 2x Tesla V100-SXM2-16GB (SM 7.0), CUDA 12.8, one card. No full-engine run; kernel microbench only.

## 1. Audit - where int8 KV is decoded, and whether it is vectorized

The default decode path is the fused split-K attention (`g_fast_attn = true`, `src/core/layer.cpp:42`, chosen at
`layer.cpp:950-952`), which reads the pools directly - no gather:

- K decode: `load8_q8`, `src/kernels/cuda/qsa_decode_attn.cu:46-54`. One lane reads 8 int8 as a `uint2` (8 B) and
  one fp16 scale per 64-group, then scalar `(float)c * sc` x8 (I2F + FMUL each). The *load* is vectorized, the
  *decode* is scalar.
- V decode: `attn_chunk_kernel`, `qsa_decode_attn.cu:158-179`. Thread `t` owns dimension `t` for all 12 heads:
  one int8 load + one fp16 scale + one I2F+FMUL per cell, then 12 FMAs. 1 byte/thread/cell, coalesced to 256 B.
- MTP draft/verify use the same kernel by batch: `src/core/mtp.cpp:503`, `src/core/verify.cpp:647`
  (`qsa_decode_attn_batch`).

The non-fast fallback gathers int8 into an fp16 scratch first, then attends over it:

- `kv_gather_q8_kernel`, `src/kernels/cuda/kv_q8.cu:69-99`: `char4` load (4 int8) + 2 fp16 scales, 8 scalar
  converts, `ushort4` fp16 store. 4 values/thread.
- `qsa_attend_kernel` then reads the fp16 scratch: `qsa.cu:462-467` (K dot), `qsa.cu:502-505` (V), 2 B/value.

The KV-streaming copy is already wide: `copy_kernel`, `src/kernels/cuda/kv_stream.cu:161-171`, `uint4` (16 B).

## 2. Vectorization

TBD

## 3. int8 vs q4_0 vs k8v4 at 32k

TBD
