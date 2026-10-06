// include/strata/pool/link.hpp - the pool's coordinator side: the workers, and the head that runs here.
//
// The coordinator's own verifier and prompt path run layers [0, K1) and stop there (`Verifier::set_link`,
// `Prefill::remote_next`).  PoolLink carries their hand-off to each worker in turn (a star: coordinator -> worker 1
// -> coordinator -> worker 2 ...; with the usual two PCs that is one hop out and one back) and brings the final
// residual home, where it runs the output head and the sampler (the same kernels the verify window's head records)
// and keeps the final rows where the draft layer (MTP) reads them.  The drafter, the sampler and every decision of
// the serve loop therefore stay on the coordinator, exactly as with one GPU; a worker is a pure function of the
// rows it gets and of its own layers' state.
#pragma once

#include "strata/core/layer.hpp"
#include "strata/core/verify.hpp"
#include "strata/pool/protocol.hpp"
#include "strata/pool/split.hpp"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core { class NativeHead; }

namespace strata::pool {

struct WorkerInfo {
    std::string addr;        ///< as configured (host:port)
    KV hello;                ///< what it said about itself
    int64_t lb = 0, le = 0;  ///< its layers once configured
    KV ready;                ///< READY: slots, cache_mib, chunk, vram_free_mib, arena_mib
    KV req;                  ///< its own numbers for the last request (the END_REQUEST ack): windows, ms_verify, ...
    double ms_rtt = 0;       ///< that ack's round trip (the network's latency, near enough)
};

/// What the coordinator knows about the model, to compare with each worker's HELLO.
struct ModelFingerprint {
    int64_t n_layers = 0, n_expert = 0, k = 0, n_embd = 0, hc = 0;
    uint64_t experts_bytes = 0, dense_bytes = 0;
    int native = 0;
    std::string pack;        ///< the pack folder's name (a hint in messages only)
    std::string engine;      ///< engine version
    KV to_kv() const;
    /// "" when `w` describes the same model, else what differs
    std::string mismatch(const KV& w) const;
};

class PoolLink final : public core::StageLink, public core::PipeRemote {
public:
    PoolLink() = default;
    ~PoolLink() override;
    PoolLink(const PoolLink&) = delete;
    PoolLink& operator=(const PoolLink&) = delete;

    /// Connects to every worker (retrying for up to `wait_s` while they start), reads its HELLO and proves the shared
    /// secret both ways.  `timeout_s` bounds every later reply.
    bool connect(const std::vector<std::string>& peers, const std::string& secret, const ModelFingerprint& fp,
                 double wait_s, double timeout_s, std::string& err);
    std::vector<WorkerInfo>& workers() { return workers_; }
    size_t size() const { return workers_.size(); }

    /// CONFIG to every worker: worker i runs [at[i], at[i + 1]) (the last to n_layers).  `common` carries the
    /// settings every worker must share (context, KV format, window, rope, wire).
    bool configure(const std::vector<int64_t>& at, int64_t n_layers, const KV& common, Wire wire, Wire draft_wire,
                   std::string& err);
    /// Waits for every READY (each worker loads its share meanwhile; minutes for a big range).
    bool wait_ready(double timeout_s, std::string& err);
    /// the smallest prompt chunk any worker can take (0 when none said)
    int64_t max_chunk() const;

    /// The device side, on the coordinator's GPU: the buffers the head writes into, the window's hand-off (host
    /// pointer of the mapped buffer the coordinator's verifier writes), and the prompt rows.  After `wait_ready`.
    bool init_device(const core::WeightTable& wt, const core::ModelGeometry& g, const core::BlockBuffers& block,
                     const core::NativeHead* head, int max_t, int64_t n_vocab, int64_t max_chunk,
                     const float* handoff_host, std::string& err);

    // ---- core::StageLink: the rest of a verify window
    bool run(int T, const int32_t* tokens, int64_t pos0, int32_t* out, std::string& err) override;
    bool commit(int n_keep, std::string& err) override;
    const float* final_R_all() const override { return R_dev_; }
    void set_sampling(const strata::kernels::SamplerParams& sp) override { sampling_ = sp; }
    void set_history(const int32_t* history, int history_len) override { hist_d_ = history; hist_len_ = history_len; }
    void set_head_sampling(bool on) override { head_sampling_ = on; }
    void stage_ms(double ms) override { ms_here_layers += ms; }

    /// The rest of a prompt chunk: `rows` (host, T x hc*n_embd float32) through every worker; `*out` then points at
    /// the final rows (host, float32, valid until the next call) for the draft layer.  `segment` is the length of the
    /// batched part this chunk belongs to (a worker lends its cache slots for that); `ckpt` != 0 asks every worker to
    /// keep its state after this chunk under that id.
    bool prefill(const float* rows, const int64_t* tokens, int64_t T, int64_t p0, int64_t segment, uint64_t ckpt,
                 const float** out, std::string& err);

    // ---- batch slots (several conversations at once): each worker is one stage of the pipeline that
    // --batch-groups runs (generate.cpp's pump).  A group's window: this PC's verifier runs its layers (batch_launch),
    // then batch_send(0) / batch_poll(0) on the first worker, ... and the last worker's batch_poll runs the head and
    // the picks here (batch_out).  Rows of the group [base, base + S) are rows [base, base + S) of the hand-off.
    static constexpr int kMaxBatchRows = 8;
    bool batch_send(size_t w, int base, int S, const int32_t* tokens, const int64_t* pos, std::string& err);
    /// 1 = worker w's rows are back (the last worker's: the picks are in batch_out), 0 = not yet, -1 = an error
    int batch_poll(size_t w, std::string& err);
    const int32_t* batch_out() const { return batch_out_; }
    /// --batch-mtp (core::StageLink): one window over every live slot's rows through the workers, the head and each
    /// row's pick here (its slot's sampling at its position); the workers keep nothing until commit_rows
    bool run_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos, int32_t* out,
                  std::string& err) override;
    bool commit_rows(const int* keep, int n, std::string& err) override;
    /// a slot's sampling for its batch rows (greedy until set; penalties are not applied in batch windows)
    void set_slot_sampling(int slot, const strata::kernels::SamplerParams& sp);
    /// every worker copies its main session's first ids.size() tokens into slot `slot` (load) or the slot's back
    /// into its main session (!load), as this PC does for its own layers; returns when all are done
    bool slot_copy(bool load, int slot, const std::vector<int32_t>& ids, std::string& err);
    /// the slot sessions every worker carved (READY's batch_slots; 0 when one has none)
    int64_t batch_slots() const;
    /// tests: the hand-off rows (hb floats each) without a device side; `head` false leaves the last worker's rows in
    /// last_batch_rows() instead of running the head
    void set_batch_host(const float* hand, int64_t hb, bool head) { hand_host_ = hand; hb_ = hb; batch_head_ = head; }
    const std::vector<float>& last_batch_rows() const { return batch_.back().rows; }
    double ms_batch_net = 0;    ///< batch groups: from the send to a worker to its rows back, summed
    int64_t batch_windows = 0;  ///< batch groups through the head here

    // ---- --pipeline-windows: the later stage of a pipelined decode (core::PipeRemote).  This PC's two first-stage
    // verifiers (one per window parity) each write their own hand-off; `pipe_init` gives the link both, and a second
    // buffer for the final rows (the drafter reads one window's rows while the next window's arrive).
    bool pipe_init(const float* handoff_host_odd, std::string& err);
    bool pl_launch(int parity, int T, const int32_t* tokens, int64_t pos0, std::string& err) override;
    int pl_poll(int parity, std::string& err) override;
    bool pl_finish(int parity, int32_t* out, std::string& err) override;
    bool pl_commit(int parity, int n_keep, std::string& err) override;
    bool pl_in_flight(int parity) const override { (void) parity; return pl_phase_ != 0; }
    const float* pl_final_R(int parity, int t) const override {
        return ((parity & 1) ? R_odd_ : R_dev_) + (size_t) t * (size_t) d_;
    }
    cudaStream_t pl_stream() const override { return (cudaStream_t) stream_; }
    int64_t pl_windows = 0;     ///< windows decoded pipelined (their timings overlap: no split calibration from them)
    /// tests: the pipelined later stage without a device side - the last worker's rows stay in last_pl_rows()
    const std::vector<float>& last_pl_rows() const { return pl_rows_; }

    /// the prompt rows' width and longest chunk without a device side (init_device sets both; the tests use this)
    void set_rows_geometry(int64_t d, int64_t max_chunk);

    bool reset(std::string& err);                        ///< every worker's session to position 0
    bool ckpt_save(uint64_t id, std::string& err);       ///< every worker keeps its running state under id
    bool ckpt_restore(uint64_t id, std::string& err);    ///< ... and puts it back
    bool ckpt_retain(const std::vector<uint64_t>& ids, std::string& err);   ///< drop every other id
    bool end_request(std::string& err);                  ///< a request ended (workers refill, log)
    uint64_t new_ckpt_id() { return ++ckpt_seq_; }
    void bye();                                          ///< tell the workers we are leaving (they stay loaded)

    /// one line for the log / INFO: per worker its range, cache and traffic
    std::string summary() const;
    int64_t worker_slots() const;
    /// After end_request: one log line per request that splits a window's time into this PC, each worker's layers
    /// and the network (from the workers' own timings), then resets the counters.  `decode_ms`: the request's decode.
    void report_request(double decode_ms);
    /// The measured split (SplitCalib): the file and this pool's key, the built-in model's ms per window for each
    /// node's range as configured, and what the file held at the start.  report_request then adds each request's
    /// numbers and writes the file.  Without it nothing is kept.
    void set_calibration(const std::string& path, const std::string& key, const std::vector<double>& node_pred,
                         const SplitCalib& loaded);
    double ms_here_layers = 0;  ///< this PC's own layers (the verifier's stage_ms), summed over the windows
    double ms_net = 0;          ///< wall time spent waiting on workers (verify windows)
    double ms_prefill_net = 0;  ///< ... (prompt chunks)
    int64_t windows = 0;

private:
    bool connect_one(const std::string& addr, Channel& out, KV& hello, std::string& err);
    KV config_for(size_t i) const;
    bool forward(Msg type, const std::vector<uint8_t>& prefix_tmpl, int64_t T, int64_t floats_per_row,
                 const uint8_t* first, Wire last_wire, std::vector<uint8_t>& last, std::string& err);
    std::vector<WorkerInfo> workers_;
    std::vector<Channel> ch_;
    std::mutex mu_;
    Wire wire_ = Wire::F32, draft_wire_ = Wire::F16;
    double timeout_s_ = 300.0, wait_s_ = 900.0;
    std::string secret_;
    ModelFingerprint fp_;
    KV common_;
    uint64_t ckpt_seq_ = 0;
    uint64_t req_bytes0_ = 0;   ///< bytes on all channels when the request's counters were last reset
    struct BatchStage {
        bool busy = false;
        int base = 0, S = 0;
        int64_t pos[kMaxBatchRows] = {};
        std::chrono::steady_clock::time_point t0;
        std::vector<uint8_t> enc, in;
        std::vector<float> rows;   ///< the worker's reply, decoded (the next worker's input)
    };
    std::vector<BatchStage> batch_;
    int32_t batch_out_[kMaxBatchRows] = {};
    bool batch_head_ = true;
    std::vector<strata::kernels::SamplerParams> slot_sp_;
    bool head_logits(int T, std::string& err, const float* R = nullptr);
    /// the picks of S rows of logits_dev_ (head_logits done): greedy, then each sampled slot's row again (slot[r], pos[r])
    void pick_rows(int S, const int* slot, const int64_t* pos);
    // --pipeline-windows: the window in flight on the workers (one at a time)
    const float* pl_src_[2] = {nullptr, nullptr};   ///< each parity's hand-off (mapped host)
    float* R_odd_ = nullptr;                        ///< the odd windows' final rows
    void* pl_ev_ = nullptr;                         ///< cudaEvent_t after the head of the window in flight
    int pl_phase_ = 0;                              ///< 0 idle, 1 on worker pl_w_, 2 the head runs here, 3 picks ready
    int pl_par_ = 0, pl_T_ = 0;
    int64_t pl_pos_ = 0;
    size_t pl_w_ = 0;
    std::chrono::steady_clock::time_point pl_t0_;
    std::vector<uint8_t> pl_prefix_, pl_in_;
    std::vector<float> pl_rows_;
    std::string calib_path_, calib_key_;
    std::vector<double> calib_pred_;
    SplitCalib calib_;

    // the window and the head (device)
    const core::WeightTable* wt_ = nullptr;
    const core::ModelGeometry* g_ = nullptr;
    core::BlockBuffers block_{};
    const core::NativeHead* head_ = nullptr;
    int max_t_ = 0;
    int64_t n_vocab_ = 0, hb_ = 0, d_ = 0, max_chunk_ = 0;
    const float* hand_host_ = nullptr;   ///< the coordinator verifier's hand-off (mapped host)
    float* R_dev_ = nullptr;             ///< (max_t, hc*n_embd): the final residual of the last window
    float* mixed_dev_ = nullptr;
    uint8_t* xq_dev_ = nullptr;
    float* logits_dev_ = nullptr;
    int32_t* out_host_ = nullptr;        ///< mapped
    int32_t* out_dev_ = nullptr;
    float* rows_host_ = nullptr;         ///< pinned: decoded window rows (max_t x hb)
    float* pf_host_ = nullptr;           ///< pinned: decoded prompt rows (max_chunk x d)
    void* stream_ = nullptr;
    std::vector<uint8_t> enc_, buf_a_, buf_b_, fin_;
    std::vector<float> pf_vec_;
    strata::kernels::SamplerParams sampling_ = [] {
        strata::kernels::SamplerParams s;
        s.greedy = true;
        s.temperature = 0.0f;
        return s;
    }();
    const int32_t* hist_d_ = nullptr;
    int hist_len_ = 0;
    bool head_sampling_ = true;
};

}  // namespace strata::pool
