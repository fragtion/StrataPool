// include/strata/pool/split.hpp - where the pool cuts the model between its PCs.
//
// The same cost model as the multi-GPU "auto" split (generate.cpp, measured on the 5080 + 3090 rig): a decode
// window costs every layer its GPU's per-layer time, plus ~190 ms per unit of routed mass that no GPU cache holds
// (the CPU pool computes those experts).  A placement gives each node a contiguous range; that node's cache takes
// its range's profiled pairs, hottest first, until its free VRAM (less the session of its range) is used.
//
// What the pool adds: each node's RAM must hold the experts of its own range (the arena is per range here, which is
// what lets two 32-64 GB PCs run a model neither could alone), and every stage boundary costs a network hop.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace strata::pool {

struct NodeCap {
    std::string name;
    int64_t vram_room = 0;   ///< bytes free for the session of its range + its expert cache
    int64_t ram_room = 0;    ///< bytes its RAM can give the expert arena of its range (<= 0: no limit)
    double layer_ms = 0.5;   ///< GPU time per layer per window (from SMs x clock)
    /// --kv-resident streaming: the pinned RAM one QSA layer's K/V copy takes on this node (0: its K/V stays in VRAM).
    /// Each session of its range (the main one and one per batch slot) holds such a copy beside the experts.
    int64_t kv_host_layer = 0;
    /// the most its experts and those K/V copies may pin together (0: no limit).  Windows pins about half of the
    /// RAM in all; a worker's whole arena is pinned, so this is what runs out first there.
    int64_t pin_room = 0;
};

/// What the pool measured with a split (the coordinator, after each request): per node how its layers' time compared
/// with the model's prediction for them (`scale`), and the window's parts no split moves - the coordinator's own work
/// outside its layers (head, sampling, drafter, the serve loop: `fixed_ms`) and the network (`net_ms`).  The
/// automatic split then predicts node i's time as scale[i] x the model's prediction for its range, plus those two.
/// Kept per pool (`key`: the model, its settings and the nodes' names, in order) in a small text file.
struct SplitCalib {
    static constexpr int64_t kMinWindows = 200;   ///< below this the built-in model is used
    static constexpr int64_t kMemory = 4000;      ///< the past counts at most this many windows
    static constexpr int64_t kTrust = 3;          ///< the search moves at most this many layers from a measured split
    std::vector<double> scale;
    /// the splits these numbers were measured with ("33", "16,32"): one measurement cannot tell a node's per-layer
    /// time from its misses' cost, so the scaled estimate is only trusted near them (kTrust layers per split point)
    std::vector<std::string> seen;
    bool near(const std::vector<int64_t>& at) const;
    double fixed_ms = 0, net_ms = 0;
    int64_t windows = 0;
    /// splits a worker failed to load (it ran out of memory on the way): the automatic split no longer gives any
    /// worker as much as or more than it had in one of them (`excludes`).  Delete the file to try them again.
    std::vector<std::string> failed;
    bool excludes(const std::vector<int64_t>& at, int64_t n_layers) const;
    void add_failed(const std::vector<int64_t>& at);
    bool usable(size_t nodes) const;
    /// one request's numbers: per node its measured and its predicted ms per window (the prediction for the split it
    /// ran with), the coordinator's time outside its layers and the network's, over `w` windows
    void add(const std::vector<double>& measured, const std::vector<double>& predicted, double fixed, double net,
             int64_t w, const std::vector<int64_t>& at = {});
    std::string encode() const;
    bool decode(const std::string& text);
};
/// The entry for `key` in `path` (false: none, or no file)
bool load_calib(const std::string& path, const std::string& key, SplitCalib& c);
/// Writes (replaces) the entry for `key`, keeping the file's other pools.  An empty path writes nothing.
bool save_calib(const std::string& path, const std::string& key, const SplitCalib& c, std::string& err);

struct SplitModel {
    int64_t n_layers = 48;
    std::vector<std::pair<int32_t, int32_t>> profile;     ///< (layer, expert), hottest first
    std::function<int64_t(int64_t layer)> slot_bytes;     ///< VRAM per cached expert of a layer
    std::vector<int64_t> layer_arena_bytes;               ///< RAM for all experts of a layer
    std::function<int64_t(int64_t lb, int64_t le)> session_bytes;
    std::vector<uint8_t> layer_kv;   ///< 1 for a layer whose K/V a node keeps a pinned copy of (NodeCap::kv_host_layer)
    int64_t kv_copies = 1;           ///< sessions per node that each hold that copy (1 + --batch)
    double miss_ms = 190.0;   ///< one unit of routed mass on the CPU pool, per window
    double hop_ms = 1.5;      ///< one network crossing per window (constant per extra node)
    int64_t first_min = 2;    ///< the coordinator keeps at least layers 0 and 1 (the PLE runs at layer 1)
    const SplitCalib* calib = nullptr;   ///< measured timings of this pool: used when `usable` for these nodes
    const SplitCalib* failures = nullptr;   ///< its `failed` splits are skipped by auto_split (usable or not)
};

struct SplitPlan {
    bool ok = false;
    std::string why;                 ///< when !ok: what does not fit
    std::vector<int64_t> at;         ///< the first layer of each later node (size = nodes - 1)
    double ms = 0;                   ///< predicted ms per decode window
    double mass = 0;                 ///< the routed mass the caches hold (0..1)
    int64_t held = 0;                ///< profiled pairs cached across the pool
    std::vector<int64_t> node_slots; ///< predicted cached experts per node
    std::vector<int64_t> node_arena; ///< RAM bytes of each node's range (its experts)
    std::vector<int64_t> node_kv;    ///< pinned RAM of each node's K/V copies (all its sessions)
    std::vector<double> node_pred;   ///< the built-in model's ms per window for each node's range
    std::vector<double> node_ms;     ///< ... as predicted for this plan (scaled by the measurements when `measured`)
    bool measured = false;           ///< `ms` rests on this pool's measured timings
};

/// The best placement over `nodes` (coordinator first).  Every placement for 2 or 3 nodes; beyond that, layers in
/// proportion to speed, then nudged to fit RAM.
SplitPlan auto_split(const SplitModel& m, const std::vector<NodeCap>& nodes);

/// The prediction for a given placement (`at` as above); `ok` false with `why` when a node's RAM cannot hold its
/// range or a range is empty.
SplitPlan evaluate_split(const SplitModel& m, const std::vector<NodeCap>& nodes, const std::vector<int64_t>& at);

/// "18" / "16,32" -> the split points; false on anything else.
bool parse_split(const std::string& s, std::vector<int64_t>& at);

/// The learned expert ranking a pool node saves (--expert-profile-save).  A node only learns about its own layers
/// [lb, le): `learned` ranks those first (its VRAM held only them) and every other layer after them.  Saved as it is,
/// the next start of that PC as one whole model (or with another split) would fill its cache with this range first.
/// Instead the result keeps `prior`'s order and only re-ranks the pairs of [lb, le) among the places `prior` gave
/// them: the hottest learned pair of the range takes the range's best place in `prior`, and so on.  Pairs of other
/// layers keep their places.  Any pair `prior` lacks follows at the end, in `learned`'s order.  An empty `prior`, or a
/// range covering every layer, returns `learned` unchanged.
std::vector<std::pair<int32_t, int32_t>> rank_for_range(const std::vector<std::pair<int32_t, int32_t>>& learned,
                                                        const std::vector<std::pair<int32_t, int32_t>>& prior,
                                                        int64_t lb, int64_t le, int64_t n_layers, int64_t n_expert);

}  // namespace strata::pool
