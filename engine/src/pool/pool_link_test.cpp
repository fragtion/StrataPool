// src/pool/pool_link_test.cpp - the coordinator's PoolLink against scripted workers, without a GPU: the handshake
// (secret, model check, busy, reload), CONFIG / READY, a prompt chunk through two workers in both wire formats, the
// lazily acknowledged commits and resets, and checkpoints.  (A verify window needs the GPU head: not here.)
#include "strata/pool/link.hpp"
#include "strata/pool/worker.hpp"

#include <atomic>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <thread>

using namespace strata::pool;

static int failures = 0;
#define CHECK(c)                                                                     \
    do {                                                                             \
        if (!(c)) {                                                                  \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);        \
            ++failures;                                                              \
        }                                                                            \
    } while (0)

static ModelFingerprint fp() {
    ModelFingerprint f;
    f.n_layers = 48; f.n_expert = 256; f.k = 10; f.n_embd = 2560; f.hc = 4;
    f.experts_bytes = 123456789; f.dense_bytes = 42; f.native = 1; f.pack = "coder-iq1_m"; f.engine = "0.1.30";
    return f;
}

// A worker: adds `add` to every row value it gets (so the coordinator can tell who ran), keeps checkpoints by id.
struct FakeWorker {
    Listener l;
    std::thread t;
    std::string secret;
    float add;
    std::atomic<int> runs{0}, bcommits{0}, last_keep0{0}, verifies{0}, commits{0}, resets{0}, prefills{0}, sessions{0}, batches{0}, slot_loads{0}, slot_stores{0};
    std::atomic<int> last_base{-1}, last_slot{-1};
    std::set<uint64_t> ckpts;
    std::set<uint64_t> convs;   // parked conversations
    std::mutex mu;
    bool reload_first = false;     // answer the first CONFIG with "reload" (it had another range)
    bool wrong_model = false;
    bool knows_prefill_wire = true;   // false: an engine from before CONFIG prefill_wire (prompt rows on the decode wire)
    int prefill_ms = 0;               // each prompt chunk takes this long (the coordinator's sends run ahead)
    int fail_prefill_at = -1;         // this prompt chunk (0-based, per session) is answered with an Err
    bool knows_cvec = true;           // false: an engine from before the control vector in a pool (no HELLO cvec)
    std::atomic<int> cvecs{0}, cvec_on{-1};   // CVEC switches received, and the last one
    KV last_cfg;                      // the last CONFIG (under mu)
    int port = 0;
    std::atomic<bool> stop{false};

    FakeWorker(std::string s, float a) : secret(std::move(s)), add(a) {
        std::string e;
        l.open("127.0.0.1", 0, e);
        port = l.port();
    }
    void run(int sessions_to_serve) {
        t = std::thread([this, sessions_to_serve] {
            for (int sess = 0; sess < sessions_to_serve; ++sess) {
                Channel ch;
                KV cfg;
                auto hello = [&] {
                    KV h = fp().to_kv();
                    if (wrong_model) h.set("n_expert", (int64_t) 512);
                    h.set("proto", (int64_t) kProtocol);
                    if (knows_prefill_wire) h.set("prefill_wire", (int64_t) 1);
                    if (knows_cvec) h.set("cvec", (int64_t) 1);
                    h.set("name", std::string("fake"));
                    h.set("vram_free", (int64_t) 8 << 30);
                    return h;
                };
                if (!accept_coordinator(l, secret, hello, ch, cfg, [](const std::string&) {}, &stop)) return;
                ++sessions;
                { std::lock_guard<std::mutex> g(mu); last_cfg = cfg; }
                if (reload_first) {
                    reload_first = false;
                    KV no;
                    no.set("reload", (int64_t) 1);
                    no.set("message", std::string("reloading"));
                    std::string e;
                    ch.send(Msg::Err, no, e);
                    ch.close();
                    continue;   // "restarts" and takes the connection again
                }
                const Wire wire = cfg.str("wire") == "f16" ? Wire::F16 : Wire::F32;
                Wire pwire = wire;   // the prompt rows' wire
                if (knows_prefill_wire) parse_wire(cfg.str("prefill_wire", cfg.str("wire", "f32")), pwire);
                const Wire dwire = cfg.i64("last") ? (cfg.str("draft_wire") == "bf16" ? Wire::BF16 : cfg.str("draft_wire") == "f16" ? Wire::F16 : Wire::F32) : pwire;
                KV ready;
                ready.set("slots", (int64_t) 1000);
                ready.set("chunk", (int64_t) 64);
                ready.set("cache_mib", (int64_t) 2048);
                ready.set("conv_park", (int64_t) 1);
                std::string e;
                ch.send(Msg::Ready, ready, e);
                std::vector<uint8_t> p;
                for (;;) {
                    Msg t;
                    if (!ch.recv(t, p, e)) break;
                    Unpacker u(p.data(), p.size());
                    if (t == Msg::Prefill) {
                        int64_t p0, T, seg;
                        uint64_t ck;
                        u.get(p0); u.get(T); u.get(seg); u.get(ck);
                        u.skip((size_t) T * 8);
                        const size_t n = u.left() / wire_bytes(pwire);
                        std::vector<float> rows(n);
                        decode_rows(pwire, u.here(), n, rows.data());
                        for (float& x : rows) x += add + (float) p0;   // + p0: which chunk this is, for the order checks
                        std::vector<uint8_t> out(n * wire_bytes(dwire));
                        encode_rows(dwire, rows.data(), n, out.data());
                        if (prefill_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(prefill_ms));
                        if (prefills.load() == fail_prefill_at) {
                            ++prefills;
                            KV no;
                            no.set("message", std::string("a prompt chunk failed"));
                            ch.send(Msg::Err, no, e);
                            continue;
                        }
                        if (ck) { std::lock_guard<std::mutex> g(mu); ckpts.insert(ck); }
                        ++prefills;
                        ch.send(Msg::PrefillRows, out, e);
                    } else if (t == Msg::BatchVerify) {
                        // i32 base, i32 S, i32 tokens[S], i64 pos[S], rows: back with `add` on every value
                        int32_t base = 0, S = 0;
                        std::memcpy(&base, p.data(), 4);
                        std::memcpy(&S, p.data() + 4, 4);
                        const size_t off = 8 + (size_t) S * 12;
                        std::vector<float> r((p.size() - off) / 4);
                        std::memcpy(r.data(), p.data() + off, r.size() * 4);
                        for (float& v : r) v += add;
                        batches++;
                        last_base = base;
                        ch.send(Msg::BatchRows, r.data(), r.size() * 4, nullptr, 0, e);
                    } else if (t == Msg::BatchRun) {
                        // i32 S, i32 slot[S], i32 tokens[S], i64 pos[S], rows (the wire): back with `add`
                        int32_t S = 0;
                        std::memcpy(&S, p.data(), 4);
                        const size_t off = 4 + (size_t) S * 16;
                        const size_t n = (p.size() - off) / wire_bytes(wire);
                        std::vector<float> r(n);
                        decode_rows(wire, p.data() + off, n, r.data());
                        for (float& v : r) v += add;
                        std::vector<uint8_t> out(n * wire_bytes(wire));
                        encode_rows(wire, r.data(), n, out.data());
                        runs++;
                        ch.send(Msg::BatchRows, out, e);
                    } else if (t == Msg::BatchCommit) {
                        int32_t n = 0, k0 = 0;
                        std::memcpy(&n, p.data(), 4);
                        if (n > 0) std::memcpy(&k0, p.data() + 4, 4);
                        last_keep0 = k0;
                        bcommits++;
                        ch.send(Msg::Ack, e);
                    } else if (t == Msg::Verify) {
                        // i32 T, i64 pos0, i32 tokens[T], rows (the wire): back with `add` on every value
                        int32_t T = 0;
                        std::memcpy(&T, p.data(), 4);
                        const size_t off = 12 + (size_t) T * 4;
                        const size_t n = (p.size() - off) / wire_bytes(wire);
                        std::vector<float> r(n);
                        decode_rows(wire, p.data() + off, n, r.data());
                        for (float& v : r) v += add;
                        std::vector<uint8_t> out(n * wire_bytes(wire));
                        encode_rows(wire, r.data(), n, out.data());
                        verifies++;
                        ch.send(Msg::VerifyRows, out, e);
                    } else if (t == Msg::SlotLoad || t == Msg::SlotStore) {
                        int32_t slot = 0;
                        std::memcpy(&slot, p.data(), 4);
                        if (t == Msg::SlotLoad) slot_loads++;
                        else slot_stores++;
                        last_slot = slot;
                        ch.send(Msg::Ack, e);
                    } else if (t == Msg::Commit) {
                        ++commits;
                        ch.send(Msg::Ack, e);
                    } else if (t == Msg::Cvec) {
                        int32_t on = -1; u.get(on);
                        cvec_on = on;
                        ++cvecs;
                        ch.send(Msg::Ack, e);
                    } else if (t == Msg::Reset) {
                        ++resets;
                        ch.send(Msg::Ack, e);
                    } else if (t == Msg::CkptSave) {
                        uint64_t id; u.get(id);
                        { std::lock_guard<std::mutex> g(mu); ckpts.insert(id); }
                        ch.send(Msg::Ack, e);
                    } else if (t == Msg::CkptRestore) {
                        uint64_t id; u.get(id);
                        bool have;
                        { std::lock_guard<std::mutex> g(mu); have = ckpts.count(id) != 0; }
                        if (have) ch.send(Msg::Ack, e);
                        else { KV no; no.set("message", "no checkpoint " + std::to_string(id)); ch.send(Msg::Err, no, e); }
                    } else if (t == Msg::CkptRetain) {
                        uint32_t n; u.get(n);
                        std::set<uint64_t> keep;
                        for (uint32_t i = 0; i < n; ++i) { uint64_t id; u.get(id); keep.insert(id); }
                        { std::lock_guard<std::mutex> g(mu); std::set<uint64_t> k2; for (auto id : ckpts) if (keep.count(id)) k2.insert(id); ckpts = k2; }
                        ch.send(Msg::Ack, e);
                    } else if (t == Msg::ConvPark) {
                        uint64_t id; int64_t n; u.get(id); u.get(n);
                        if (n >= 1000) { KV no; no.set("message", std::string("parking: too big")); ch.send(Msg::Err, no, e); }
                        else { { std::lock_guard<std::mutex> g(mu); convs.insert(id); } ch.send(Msg::Ack, e); }
                    } else if (t == Msg::ConvRestore) {
                        uint64_t id; u.get(id);
                        bool have;
                        { std::lock_guard<std::mutex> g(mu); have = convs.count(id) != 0; }
                        if (have) ch.send(Msg::Ack, e);
                        else { KV no; no.set("message", "no parked conversation " + std::to_string(id)); ch.send(Msg::Err, no, e); }
                    } else if (t == Msg::ConvBorrow) {
                        uint64_t id; int64_t n; u.get(id); u.get(n);
                        bool have;
                        { std::lock_guard<std::mutex> g(mu); have = convs.count(id) != 0; }
                        if (have) ch.send(Msg::Ack, e);
                        else { KV no; no.set("message", "no parked conversation " + std::to_string(id)); ch.send(Msg::Err, no, e); }
                    } else if (t == Msg::ConvRetain) {
                        uint32_t n; u.get(n);
                        std::set<uint64_t> keep;
                        for (uint32_t i = 0; i < n; ++i) { uint64_t id; u.get(id); keep.insert(id); }
                        { std::lock_guard<std::mutex> g(mu); std::set<uint64_t> k2; for (auto id : convs) if (keep.count(id)) k2.insert(id); convs = k2; }
                        ch.send(Msg::Ack, e);
                    } else if (t == Msg::EndRequest) {
                        KV st;
                        st.set("windows", (int64_t) 4);
                        st.set("ms_verify", 10.0);
                        st.set("hits", (int64_t) 3);
                        st.set("lookups", (int64_t) 4);
                        ch.send(Msg::Ack, st, e);
                    } else if (t == Msg::Bye) {
                        break;
                    }
                }
            }
        });
    }
    ~FakeWorker() {
        stop = true;
        if (t.joinable()) t.join();
    }
};

static void test_two_workers(Wire wire) {
    FakeWorker a("pw", 1.0f), b("pw", 10.0f);
    a.run(1);
    b.run(1);
    PoolLink link;
    std::string err;
    const std::vector<std::string> peers = {"127.0.0.1:" + std::to_string(a.port), "127.0.0.1:" + std::to_string(b.port)};
    CHECK(link.connect(peers, "pw", fp(), 10.0, 10.0, err));
    CHECK(link.size() == 2);
    KV common;
    common.set("max_context", (int64_t) 4096);
    CHECK(link.configure({20, 34}, 48, common, wire, Wire::F32, err));
    CHECK(link.workers()[0].lb == 20 && link.workers()[0].le == 34 && link.workers()[1].le == 48);
    CHECK(link.wait_ready(10.0, err));
    CHECK(link.max_chunk() == 64 && link.worker_slots() == 2000);
    // a prompt chunk: 3 rows of 8 floats through both workers (+1 then +10)
    link.set_rows_geometry(8, 64);
    std::vector<float> rows(24);
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = (float) i * 0.25f;
    const int64_t toks[3] = {1, 2, 3};
    const float* out = nullptr;
    CHECK(link.prefill(rows.data(), toks, 3, 0, 3, 7, &out, err));
    for (size_t i = 0; out && i < rows.size(); ++i) CHECK(std::fabs(out[i] - (rows[i] + 11.0f)) < 1e-2f);
    // commits are acknowledged lazily: many in a row, then a synchronous restore drains them all
    for (int i = 0; i < 25; ++i) CHECK(link.commit(1, err));
    CHECK(link.reset(err));
    CHECK(link.ckpt_save(99, err));
    CHECK(link.ckpt_restore(99, err));
    CHECK(link.ckpt_restore(7, err));                       // saved with the prompt chunk
    CHECK(link.ckpt_retain({99}, err));
    CHECK(!link.ckpt_restore(7, err) && err.find("no checkpoint 7") != std::string::npos);
    err.clear();
    // parked conversations: every worker parks, restores and drops its part; a refusal reaches the caller
    CHECK(link.all_ready("conv_park"));
    CHECK(!link.all_ready("no_such_key"));
    CHECK(link.conv_park(41, 500, err));
    CHECK(link.conv_park(42, 600, err));
    CHECK(!link.conv_park(43, 5000, err) && err.find("too big") != std::string::npos);
    err.clear();
    CHECK(link.conv_restore(41, err));
    CHECK(link.conv_retain({42}, err));
    CHECK(!link.conv_restore(41, err) && err.find("no parked conversation 41") != std::string::npos);
    err.clear();
    CHECK(link.conv_restore(42, err));
    CHECK(link.conv_borrow(42, 100, err));                  // the image stays: borrowed again, then restored
    CHECK(link.conv_borrow(42, 50, err));
    CHECK(link.conv_restore(42, err));
    CHECK(!link.conv_borrow(41, 10, err) && err.find("no parked conversation 41") != std::string::npos);
    err.clear();
    // a pipelined batch group (slots 1-2) through both workers, without the head; slot copies on both (the fake
    // workers read and write f32 rows)
    if (wire == Wire::F32) {
        const int64_t hb = 5;
        std::vector<float> hand(8 * hb, 0.0f);
        for (int i = 0; i < 2 * hb; ++i) hand[(size_t) (hb + i)] = (float) i;   // rows 1..2
        link.set_batch_host(hand.data(), hb, false);
        const int32_t bt[2] = {11, 12};
        const int64_t bp[2] = {100, 200};
        CHECK(link.batch_send(0, 1, 2, bt, bp, err));
        CHECK(!link.batch_send(0, 1, 2, bt, bp, err));          // one group at a time per worker
        int r = 0;
        for (int spin = 0; spin < 2000 && (r = link.batch_poll(0, err)) == 0; ++spin)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        CHECK(r == 1);
        CHECK(link.batch_send(1, 1, 2, bt, bp, err));
        for (int spin = 0; spin < 2000 && (r = link.batch_poll(1, err)) == 0; ++spin)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        CHECK(r == 1);
        const std::vector<float>& back = link.last_batch_rows();
        CHECK(back.size() == (size_t) (2 * hb));
        bool exact = back.size() == (size_t) (2 * hb);
        for (size_t i = 0; exact && i < back.size(); ++i) exact = std::fabs(back[i] - ((float) i + 1.0f + 10.0f)) < 1e-2f;
        CHECK(exact);                                            // worker a added 1, worker b 10
        CHECK(a.batches == 1 && b.batches == 1 && a.last_base == 1 && b.last_base == 1);
        CHECK(link.slot_copy(true, 1, {5, 6, 7}, err) && link.slot_copy(false, 1, {5, 6, 7}, err));
        CHECK(a.slot_loads == 1 && b.slot_stores == 1 && b.last_slot == 1);
        // --pipeline-windows: an odd window (its own hand-off) through both workers without blocking, then its commit
        std::vector<float> odd(8 * hb, 0.0f);
        for (int i = 0; i < 3 * hb; ++i) odd[(size_t) i] = 100.0f + (float) i;
        CHECK(link.pipe_init(odd.data(), err));
        const int32_t pt[3] = {7, 8, 9};
        CHECK(link.pl_launch(1, 3, pt, 300, err));
        CHECK(link.pl_in_flight(1) && !link.pl_launch(0, 3, pt, 303, err));   // one window on the workers at a time
        err.clear();
        for (int spin = 0; spin < 2000 && (r = link.pl_poll(1, err)) == 0; ++spin)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        CHECK(r == 1 && link.pl_poll(1, err) == 1);
        int32_t picks[3] = {};
        CHECK(link.pl_finish(1, picks, err) && !link.pl_in_flight(1));
        const std::vector<float>& pr = link.last_pl_rows();
        exact = pr.size() == (size_t) (3 * hb);
        for (size_t i = 0; exact && i < pr.size(); ++i) exact = std::fabs(pr[i] - (100.0f + (float) i + 11.0f)) < 1e-2f;
        CHECK(exact);
        CHECK(a.verifies == 1 && b.verifies == 1 && link.pl_windows == 1);
        CHECK(link.pl_commit(1, 2, err));
        // --batch-mtp: two slots with a draft row each (slot 0 rows 0-1, slot 2 rows 2-3), then each slot's prefix
        for (int i = 0; i < 4 * hb; ++i) hand[(size_t) i] = 50.0f + (float) i;
        link.set_batch_host(hand.data(), hb, false);
        const int brows[4] = {0, 0, 2, 2};
        const int32_t btok[4] = {1, 2, 3, 4};
        const int64_t bpos[4] = {10, 11, 20, 21};
        int32_t bout[4] = {9, 9, 9, 9};
        CHECK(link.run_rows(brows, 4, btok, bpos, bout, err));
        const std::vector<float>& rr = link.last_pl_rows();
        exact = rr.size() == (size_t) (4 * hb);
        for (size_t i = 0; exact && i < rr.size(); ++i) exact = std::fabs(rr[i] - (50.0f + (float) i + 11.0f)) < 1e-2f;
        CHECK(exact);
        CHECK(a.runs == 1 && b.runs == 1 && link.batch_windows == 1);
        const int keep[3] = {2, 0, 1};
        CHECK(link.commit_rows(keep, 3, err));
        CHECK(link.end_request(err));
        CHECK(a.bcommits == 1 && b.bcommits == 1 && a.last_keep0 == 2);
    }
    CHECK(link.end_request(err));
    CHECK(link.workers()[0].req.i64("windows") == 4 && link.workers()[1].req.f64("ms_verify") == 10.0);
    link.windows = 4;
    link.ms_net = 60.0;
    link.report_request(400.0);
    link.bye();
    const int want_commits = wire == Wire::F32 ? 26 : 25;
    CHECK(a.commits == want_commits && b.commits == want_commits && a.resets == 1 && a.prefills == 1 && b.prefills == 1);
}

static void test_refusals() {
    std::string err;
    {   // the wrong secret
        FakeWorker w("right", 0);
        w.run(1);
        PoolLink link;
        CHECK(!link.connect({"127.0.0.1:" + std::to_string(w.port)}, "wrong", fp(), 5.0, 5.0, err));
        CHECK(err.find("secret") != std::string::npos);
        // let the fake worker's accept end: a coordinator with the right secret that leaves at once
        PoolLink ok;
        std::string e2;
        CHECK(ok.connect({"127.0.0.1:" + std::to_string(w.port)}, "right", fp(), 5.0, 5.0, e2));
        KV c;
        CHECK(ok.configure({24}, 48, c, Wire::F32, Wire::F16, e2) && ok.wait_ready(5.0, e2));
    }
    {   // another model on the worker
        FakeWorker w("s", 0);
        w.wrong_model = true;
        w.run(1);
        PoolLink link;
        err.clear();
        CHECK(!link.connect({"127.0.0.1:" + std::to_string(w.port)}, "s", fp(), 5.0, 5.0, err));
        CHECK(err.find("different model") != std::string::npos);
        // unblock the fake worker's accept (it is waiting for a CONFIG on that connection: closing ends it)
    }
    {   // nobody there: a clear error after the wait
        Listener tmp;
        tmp.open("127.0.0.1", 0, err);
        const int dead = tmp.port();
        tmp.close();
        PoolLink link;
        err.clear();
        CHECK(!link.connect({"127.0.0.1:" + std::to_string(dead)}, "s", fp(), 1.0, 1.0, err));
        CHECK(err.find("not reachable") != std::string::npos);
    }
}

static void test_reload_and_busy() {
    std::string err;
    FakeWorker w("s", 0);
    w.reload_first = true;
    w.run(2);   // the first session answers "reload"; the second serves
    PoolLink link;
    CHECK(link.connect({"127.0.0.1:" + std::to_string(w.port)}, "s", fp(), 10.0, 10.0, err));
    KV c;
    CHECK(link.configure({30}, 48, c, Wire::F32, Wire::F16, err));
    CHECK(link.wait_ready(10.0, err));
    CHECK(w.sessions == 2);
    // while it serves, a second coordinator hears "busy" and gives up after its wait
    {
        BusyResponder busy(w.l, "test");
        PoolLink other;
        std::string e2;
        CHECK(!other.connect({"127.0.0.1:" + std::to_string(w.port)}, "s", fp(), 2.0, 2.0, e2));
        CHECK(e2.find("serving") != std::string::npos);
    }
    link.bye();
}

// Prompt chunks pipelined (prefill_send / prefill_recv): the decode wire stays as configured, the prompt rows go as
// `pwire`; chunks are sent ahead of their replies by one thread and received in order by another.  `old_second`: the
// second worker predates CONFIG prefill_wire, so the link falls back to the decode wire for prompt rows.
static void test_prefill_pipelined(Wire wire, Wire pwire, bool old_second) {
    FakeWorker a("pw", 1.0f), b("pw", 10.0f);
    b.knows_prefill_wire = !old_second;
    a.run(1);
    b.run(1);
    PoolLink link;
    std::string err;
    CHECK(link.connect({"127.0.0.1:" + std::to_string(a.port), "127.0.0.1:" + std::to_string(b.port)}, "pw", fp(), 10.0,
                       10.0, err));
    KV common;
    common.set("max_context", (int64_t) 4096);
    link.set_prefill_wire(pwire);
    CHECK(link.configure({20, 34}, 48, common, wire, Wire::F32, err));
    CHECK(link.wait_ready(10.0, err));
    CHECK(link.prefill_wire() == (old_second ? wire : pwire));
    link.set_rows_geometry(8, 64);
    // chunks of different lengths, so a reply taken for the wrong chunk shows (its size and its + p0)
    const int64_t Ts[6] = {3, 5, 2, 7, 1, 4};
    std::vector<std::vector<float>> rows(6);
    std::vector<int64_t> p0s(6);
    int64_t p = 0;
    for (int c = 0; c < 6; ++c) {
        p0s[(size_t) c] = p;
        rows[(size_t) c].resize((size_t) Ts[c] * 8);
        for (size_t i = 0; i < rows[(size_t) c].size(); ++i) rows[(size_t) c][i] = (float) i * 0.25f;
        p += Ts[c];
    }
    std::vector<int64_t> toks(64, 5);
    std::atomic<bool> send_ok{true};
    std::atomic<int> sent{0}, received{0};
    std::thread sender([&] {
        for (int c = 0; c < 6; ++c) {
            // ahead of the receiver by up to three chunks (the link holds four in flight)
            while (c - received.load() >= 3) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            std::string e;
            if (!link.prefill_send(rows[(size_t) c].data(), toks.data(), Ts[c], p0s[(size_t) c], 32, (uint64_t) (100 + c), e)) {
                std::fprintf(stderr, "send %d: %s\n", c, e.c_str());
                send_ok = false;
                return;
            }
            ++sent;
        }
    });
    for (int c = 0; c < 6; ++c) {
        while (send_ok && sent.load() <= c) std::this_thread::sleep_for(std::chrono::milliseconds(1));   // its send first
        const float* out = nullptr;
        uint64_t ck = 0;
        std::string e;
        const bool ok = link.prefill_recv(Ts[c], &out, &ck, e);
        CHECK(ok);
        if (!ok) { std::fprintf(stderr, "recv %d: %s\n", c, e.c_str()); break; }
        CHECK(ck == (uint64_t) (100 + c));
        for (size_t i = 0; out && i < rows[(size_t) c].size(); ++i)
            CHECK(std::fabs(out[i] - (rows[(size_t) c][i] + 11.0f + 2.0f * (float) p0s[(size_t) c])) < 2e-2f);
        ++received;
    }
    sender.join();
    CHECK(send_ok);
    CHECK(a.prefills == 6 && b.prefills == 6);
    // a chunk received with none in flight, and one received under the wrong length, are refused
    const float* out = nullptr;
    uint64_t ck = 0;
    CHECK(!link.prefill_recv(3, &out, &ck, err) && err.find("no prompt chunk") != std::string::npos);
    CHECK(link.prefill_send(rows[0].data(), toks.data(), 3, 0, 3, 0, err));
    CHECK(!link.prefill_recv(4, &out, &ck, err) && err.find("out of order") != std::string::npos);
    link.bye();
}

// The receiving side keeps a conversation checkpoint between chunks, as generate.cpp's checkpoint_at does, while the
// sending side is chunks ahead: its retain lists must not drop the checkpoints of chunks still in flight, must not
// break the stream (the ACKs come back between the chunks' rows), and must reach every worker.  Then a prompt cut
// part-way: prefill_abort reads past what is still on the way, and the link answers the next call in step.
static void test_prefill_retain_midway() {
    FakeWorker a("pr", 1.0f), b("pr", 10.0f);
    a.prefill_ms = 15;   // the first worker is the slow one: the sends queue up in front of it
    a.run(1);
    b.run(1);
    PoolLink link;
    std::string err;
    CHECK(link.connect({"127.0.0.1:" + std::to_string(a.port), "127.0.0.1:" + std::to_string(b.port)}, "pr", fp(), 10.0,
                       10.0, err));
    KV common;
    common.set("max_context", (int64_t) 4096);
    link.set_prefill_wire(Wire::F16);
    CHECK(link.configure({20, 34}, 48, common, Wire::F32, Wire::F32, err));
    CHECK(link.wait_ready(10.0, err));
    link.set_rows_geometry(8, 64);
    CHECK(link.reset(err));               // an ACK owed before the prompt (the first send reads it)
    CHECK(link.conv_retain({7}, err));
    constexpr int kChunks = 10;
    std::vector<float> rows(4 * 8, 0.5f);
    std::vector<int64_t> toks(64, 5);
    std::atomic<bool> send_ok{true};
    std::atomic<int> sent{0}, received{0};
    std::thread sender([&] {
        for (int c = 0; c < kChunks; ++c) {   // as fast as it can: the link holds it to four chunks on the way
            std::string e;
            if (!link.prefill_send(rows.data(), toks.data(), 4, 4 * c, 64, (uint64_t) (500 + c), e)) {
                std::fprintf(stderr, "send %d: %s\n", c, e.c_str());
                send_ok = false;
                return;
            }
            ++sent;
        }
    });
    std::vector<uint64_t> kept;   // the coordinator keeps every other chunk's checkpoint (a list as checkpoint_at sends)
    for (int c = 0; c < kChunks; ++c) {
        while (send_ok && sent.load() <= c) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const float* out = nullptr;
        uint64_t ck = 0;
        std::string e;
        const bool ok = link.prefill_recv(4, &out, &ck, e);
        CHECK(ok);
        if (!ok) { std::fprintf(stderr, "recv %d: %s\n", c, e.c_str()); break; }
        CHECK(ck == (uint64_t) (500 + c));
        for (int i = 0; out && i < 32; ++i) CHECK(std::fabs(out[i] - (0.5f + 11.0f + 8.0f * (float) c)) < 2e-2f);
        if (c % 2 == 0) kept.push_back(ck);
        CHECK(link.ckpt_retain(kept, e));
        CHECK(link.conv_retain({7}, e));
        ++received;
    }
    sender.join();
    CHECK(send_ok);
    CHECK(a.prefills == kChunks && b.prefills == kChunks);
    // every kept checkpoint is still on both workers (none dropped while its chunk was in flight), the rest are gone
    const std::set<uint64_t> want(kept.begin(), kept.end());
    CHECK(link.ckpt_restore(kept.back(), err));   // in step: the last list's ACKs are read first
    {
        std::lock_guard<std::mutex> g(a.mu);
        CHECK(a.ckpts == want);
        CHECK(a.convs.empty() || a.convs == std::set<uint64_t>{7});
    }
    {
        std::lock_guard<std::mutex> g(b.mu);
        CHECK(b.ckpts == want);
    }
    // a prompt cut part-way: three chunks out, one back, then the receiving side gives up
    for (int c = 0; c < 3; ++c) CHECK(link.prefill_send(rows.data(), toks.data(), 4, 4 * c, 64, (uint64_t) (600 + c), err));
    const float* out = nullptr;
    uint64_t ck = 0;
    CHECK(link.prefill_recv(4, &out, &ck, err) && ck == 600);
    CHECK(link.ckpt_retain(kept, err));   // waits: two chunks are still in flight
    CHECK(link.prefill_abort(err));
    CHECK(link.ckpt_restore(kept.front(), err));   // the next call is answered in step
    CHECK(link.prefill(rows.data(), toks.data(), 4, 0, 4, 0, &out, err));
    for (int i = 0; out && i < 32; ++i) CHECK(std::fabs(out[i] - (0.5f + 11.0f)) < 2e-2f);
    {
        std::lock_guard<std::mutex> g(a.mu);
        CHECK(a.ckpts == want);   // the cut prompt's checkpoints were dropped by the list sent after the abort
    }
    link.bye();
}

// A worker answers a chunk with an Err while the sender, faster than the worker, waits for room: the waiting send
// gives up instead of hanging, prefill_abort reads past the rest, and the link answers the next calls in step.
static void test_prefill_worker_error() {
    FakeWorker a("pe", 1.0f);
    a.prefill_ms = 10;
    a.fail_prefill_at = 2;
    a.run(1);
    PoolLink link;
    std::string err;
    CHECK(link.connect({"127.0.0.1:" + std::to_string(a.port)}, "pe", fp(), 10.0, 10.0, err));
    KV common;
    common.set("max_context", (int64_t) 4096);
    CHECK(link.configure({24}, 48, common, Wire::F32, Wire::F32, err));
    CHECK(link.wait_ready(10.0, err));
    link.set_rows_geometry(8, 64);
    std::vector<float> rows(4 * 8, 0.5f);
    std::vector<int64_t> toks(64, 5);
    std::atomic<int> sent{0}, sends_done{0};
    std::thread sender([&] {
        for (int c = 0; c < 9; ++c) {
            std::string e;
            if (!link.prefill_send(rows.data(), toks.data(), 4, 4 * c, 64, 0, e)) break;
            ++sent;
        }
        sends_done = 1;
    });
    int got = 0;
    for (int c = 0; c < 9; ++c) {
        while (sent.load() <= c && !sends_done.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (sent.load() <= c) break;
        const float* out = nullptr;
        uint64_t ck = 0;
        std::string e;
        if (!link.prefill_recv(4, &out, &ck, e)) {
            CHECK(c == 2 && e.find("a prompt chunk failed") != std::string::npos);
            break;
        }
        ++got;
    }
    sender.join();   // a send waiting for room ends: the failed receive woke it
    CHECK(got == 2);
    CHECK(sent.load() < 9);
    CHECK(link.prefill_abort(err));
    const float* out = nullptr;
    CHECK(link.prefill(rows.data(), toks.data(), 4, 0, 4, 0, &out, err));   // the next prompt is answered in step
    for (int i = 0; out && i < 32; ++i) CHECK(std::fabs(out[i] - (0.5f + 1.0f)) < 2e-2f);
    CHECK(link.reset(err));
    CHECK(link.ckpt_retain({}, err));
    CHECK(link.prefill(rows.data(), toks.data(), 4, 4, 4, 0, &out, err));   // the ACKs owed are read first
    link.bye();
}

// A control vector (the speed projection) in a split: each worker's CONFIG carries the rows of its layers and of the
// layer before them, exactly; the per-request switch goes out only when it changes and is acknowledged in order; a
// worker whose engine does not take a vector is refused at configure.
static void test_cvec() {
    std::string err;
    const int64_t L = 48, N = 8;
    std::vector<float> dir((size_t) (L * N), 0.0f), sc((size_t) L, 0.0f);
    for (int64_t l = 4; l <= 44; ++l) {
        sc[(size_t) l] = 1.0f + 0.25f * (float) l;
        for (int64_t j = 0; j < N; ++j) dir[(size_t) (l * N + j)] = (float) l * 100.0f + (float) j * 0.125f - 3.0f;
    }
    {
        FakeWorker a("s", 0), b("s", 0);
        a.run(1);
        b.run(1);
        PoolLink link;
        CHECK(link.connect({"127.0.0.1:" + std::to_string(a.port), "127.0.0.1:" + std::to_string(b.port)}, "s", fp(),
                           5.0, 5.0, err));
        link.set_cvec(dir, sc, 0, 4, 44, N, 0x1234abcd5678ef90ull);
        CHECK(link.has_cvec());
        KV c;
        CHECK(link.configure({20, 36}, L, c, Wire::F32, Wire::F16, err) && link.wait_ready(5.0, err));
        auto check_rows = [&](FakeWorker& w, int64_t lb, int64_t le) {
            KV k;
            { std::lock_guard<std::mutex> g(w.mu); k = w.last_cfg; }
            CHECK(k.i64("cvec") == 1 && k.i64("cvec_mode", -1) == 0 && k.i64("cvec_first") == 4 && k.i64("cvec_last") == 44);
            CHECK(k.i64("cvec_n_embd") == N && k.i64("cvec_on") == 1);
            CHECK(k.i64("cvec_lb") == lb - 1 && k.i64("cvec_le") == le);   // (the layer before its range too)
            CHECK(std::strtoull(k.str("cvec_id").c_str(), nullptr, 16) != 0);
            std::vector<uint8_t> db, sb;
            CHECK(from_base64(k.str("cvec_dir"), db) && from_base64(k.str("cvec_s"), sb));
            CHECK(db.size() == (size_t) ((le - lb + 1) * N) * 4 && sb.size() == (size_t) (le - lb + 1) * 4);
            if (db.size() == (size_t) ((le - lb + 1) * N) * 4)
                CHECK(std::memcmp(db.data(), dir.data() + (size_t) ((lb - 1) * N), db.size()) == 0);
            if (sb.size() == (size_t) (le - lb + 1) * 4)
                CHECK(std::memcmp(sb.data(), sc.data() + (size_t) (lb - 1), sb.size()) == 0);
        };
        check_rows(a, 20, 36);
        check_rows(b, 36, 48);
        CHECK(link.cvec_set(true, err));    // already on: nothing goes out
        CHECK(link.cvec_set(false, err));
        CHECK(link.cvec_set(false, err));   // (unchanged)
        CHECK(link.end_request(err));       // drains the lazy ACKs
        CHECK(a.cvecs == 1 && b.cvecs == 1 && a.cvec_on == 0 && b.cvec_on == 0);
        CHECK(link.cvec_set(true, err) && link.reset(err) && link.end_request(err));
        CHECK(a.cvecs == 2 && b.cvecs == 2 && a.cvec_on == 1 && b.cvec_on == 1);
        link.bye();
    }
    {   // without a vector nothing about it is sent
        FakeWorker w("s", 0);
        w.run(1);
        PoolLink link;
        CHECK(link.connect({"127.0.0.1:" + std::to_string(w.port)}, "s", fp(), 5.0, 5.0, err));
        KV c;
        CHECK(link.configure({24}, L, c, Wire::F32, Wire::F16, err) && link.wait_ready(5.0, err));
        CHECK(link.cvec_set(false, err) && link.end_request(err));
        KV k;
        { std::lock_guard<std::mutex> g(w.mu); k = w.last_cfg; }
        CHECK(!k.has("cvec") && !k.has("cvec_dir") && w.cvecs == 0);
        link.bye();
    }
    {   // a worker whose engine predates it: refused, naming the fix
        FakeWorker w("s", 0);
        w.knows_cvec = false;
        w.run(1);
        PoolLink link;
        CHECK(link.connect({"127.0.0.1:" + std::to_string(w.port)}, "s", fp(), 5.0, 5.0, err));
        link.set_cvec(dir, sc, 0, 4, 44, N, 7);
        KV c;
        err.clear();
        CHECK(!link.configure({24}, L, c, Wire::F32, Wire::F16, err));
        CHECK(err.find("update that PC's engine") != std::string::npos);
    }
    // base64 round trips of every length mod 3, and a malformed string
    for (size_t n = 0; n < 10; ++n) {
        std::vector<uint8_t> in(n), back;
        for (size_t i = 0; i < n; ++i) in[i] = (uint8_t) (i * 77 + 251);
        CHECK(from_base64(to_base64(in.data(), n), back) && back == in);
    }
    std::vector<uint8_t> junk;
    CHECK(!from_base64("ab=c", junk) && !from_base64("abc", junk) && !from_base64("a*cd", junk));
}

int main() {
    net_init();
    test_cvec();
    test_two_workers(Wire::F32);
    test_two_workers(Wire::F16);
    test_prefill_pipelined(Wire::F32, Wire::F16, false);
    test_prefill_pipelined(Wire::F32, Wire::F32, false);
    test_prefill_pipelined(Wire::F32, Wire::F16, true);
    test_prefill_retain_midway();
    test_prefill_worker_error();
    test_refusals();
    test_reload_and_busy();
    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::printf("strata-pool-link-test: all checks passed\n");
    return 0;
}
