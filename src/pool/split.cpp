// src/pool/split.cpp - the pool's placement search (see split.hpp for the cost model).
#include "strata/pool/split.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::pool {

SplitPlan evaluate_split(const SplitModel& m, const std::vector<NodeCap>& nodes, const std::vector<int64_t>& at) {
    SplitPlan p;
    p.at = at;
    const int ns = (int) nodes.size();
    if (ns < 1 || (int) at.size() != ns - 1) { p.why = "the split needs one point per worker"; return p; }
    auto lb_of = [&](int i) { return i == 0 ? (int64_t) 0 : at[(size_t) i - 1]; };
    auto le_of = [&](int i) { return i + 1 < ns ? at[(size_t) i] : m.n_layers; };
    for (int i = 0; i < ns; ++i) {
        const int64_t lb = lb_of(i), le = le_of(i);
        if (le <= lb || (i == 0 && le < m.first_min) || le > m.n_layers) {
            p.why = "node " + nodes[(size_t) i].name + " would get no layers (" + std::to_string(lb) + "-" +
                    std::to_string(le - 1) + ")";
            return p;
        }
    }
    p.node_slots.assign((size_t) ns, 0);
    p.node_arena.assign((size_t) ns, 0);
    std::vector<int64_t> room((size_t) ns), used((size_t) ns, 0);
    for (int i = 0; i < ns; ++i) {
        const int64_t lb = lb_of(i), le = le_of(i);
        int64_t arena = 0;
        for (int64_t l = lb; l < le; ++l)
            arena += (size_t) l < m.layer_arena_bytes.size() ? m.layer_arena_bytes[(size_t) l] : 0;
        p.node_arena[(size_t) i] = arena;
        const NodeCap& n = nodes[(size_t) i];
        if (n.ram_room > 0 && arena > n.ram_room) {
            char b[256];
            std::snprintf(b, sizeof b, "%s's RAM cannot hold the experts of layers %lld-%lld (%.1f GiB, %.1f GiB free)",
                          n.name.c_str(), (long long) lb, (long long) (le - 1), (double) arena / 1073741824.0,
                          (double) n.ram_room / 1073741824.0);
            p.why = b;
            return p;
        }
        room[(size_t) i] = n.vram_room - (m.session_bytes ? m.session_bytes(lb, le) : 0);
    }
    double total = 0, held_mass = 0;
    std::vector<double> node_miss((size_t) ns, 0.0);   // each node's routed mass that its cache does not hold
    std::vector<bool> full((size_t) ns, false);
    for (size_t r = 0; r < m.profile.size(); ++r) {
        const double w = std::pow((double) r + 1.0, -1.2);
        total += w;
        const int64_t l = m.profile[r].first;
        int st = 0;
        while (st + 1 < ns && l >= at[(size_t) st]) ++st;
        const int64_t c = m.slot_bytes ? m.slot_bytes(l) : 1;
        if (full[(size_t) st] || used[(size_t) st] + c > room[(size_t) st]) {
            full[(size_t) st] = true;
            node_miss[(size_t) st] += w;
            continue;
        }
        used[(size_t) st] += c;
        ++p.node_slots[(size_t) st];
        ++p.held;
        held_mass += w;
    }
    p.mass = total > 0 ? held_mass / total : 0.0;
    // per node: its layers' GPU time and the CPU time of its misses (the model), then what was measured (calib)
    p.node_pred.assign((size_t) ns, 0.0);
    p.node_ms.assign((size_t) ns, 0.0);
    const bool cal = m.calib != nullptr && m.calib->usable((size_t) ns);
    p.ms = cal ? m.calib->fixed_ms + m.calib->net_ms : m.hop_ms * (double) (ns - 1) * 2.0;
    for (int i = 0; i < ns; ++i) {
        const double pred = (double) (le_of(i) - lb_of(i)) * nodes[(size_t) i].layer_ms +
                            (total > 0 ? m.miss_ms * node_miss[(size_t) i] / total : 0.0);
        p.node_pred[(size_t) i] = pred;
        p.node_ms[(size_t) i] = cal ? pred * m.calib->scale[(size_t) i] : pred;
        p.ms += p.node_ms[(size_t) i];
    }
    p.measured = cal;
    p.ok = true;
    return p;
}

// ------------------------------------------------------------------------------------------------ measured timings
bool SplitCalib::usable(size_t nodes) const {
    // (an entry without the splits it was measured with - written before they were kept - is not used)
    if (windows < kMinWindows || scale.size() != nodes || seen.empty()) return false;
    for (const double s : scale)
        if (!(s > 0.0) || !std::isfinite(s)) return false;
    return std::isfinite(fixed_ms) && std::isfinite(net_ms) && fixed_ms >= 0 && net_ms >= 0;
}

bool SplitCalib::near(const std::vector<int64_t>& at) const {
    if (seen.empty()) return true;
    for (const std::string& s : seen) {
        std::vector<int64_t> p;
        if (!parse_split(s, p) || p.size() != at.size()) continue;
        bool ok = true;
        for (size_t i = 0; i < p.size() && ok; ++i) ok = std::llabs(p[i] - at[i]) <= kTrust;
        if (ok) return true;
    }
    return false;
}

void SplitCalib::add(const std::vector<double>& measured, const std::vector<double>& predicted, double fixed,
                     double net, int64_t w, const std::vector<int64_t>& at) {
    if (w <= 0 || measured.size() != predicted.size() || measured.empty()) return;
    for (size_t i = 0; i < measured.size(); ++i)
        if (!(measured[i] > 0.0) || !(predicted[i] > 0.0)) return;   // a node without its own timing: not this time
    if (scale.size() != measured.size()) { scale.assign(measured.size(), 0.0); windows = 0; }
    // windows-weighted, the past capped at kMemory windows so a change (a driver, other programs, a new split) shows
    // within a few long requests
    const double old = (double) std::min<int64_t>(windows, kMemory), nw = (double) w, tot = old + nw;
    for (size_t i = 0; i < measured.size(); ++i) {
        const double s = std::clamp(measured[i] / predicted[i], 0.02, 50.0);
        scale[i] = (scale[i] * old + s * nw) / tot;
    }
    if (!at.empty()) {
        std::string k;
        for (size_t i = 0; i < at.size(); ++i) k += (i ? "," : "") + std::to_string(at[i]);
        if (std::find(seen.begin(), seen.end(), k) == seen.end()) seen.push_back(k);
        if (seen.size() > 8) seen.erase(seen.begin());
    }
    fixed_ms = (fixed_ms * old + std::max(fixed, 0.0) * nw) / tot;
    net_ms = (net_ms * old + std::max(net, 0.0) * nw) / tot;
    windows = std::min<int64_t>(windows, kMemory) + w;
}

std::string SplitCalib::encode() const {
    std::string sc;
    char b[64];
    for (size_t i = 0; i < scale.size(); ++i) {
        std::snprintf(b, sizeof b, "%s%.5g", i ? "," : "", scale[i]);
        sc += b;
    }
    char line[256];
    std::snprintf(line, sizeof line, "windows=%lld fixed_ms=%.3f net_ms=%.3f scale=", (long long) windows, fixed_ms,
                  net_ms);
    std::string out = line + sc;
    if (!seen.empty()) {
        out += " seen=";
        for (size_t i = 0; i < seen.size(); ++i) out += (i ? ";" : "") + seen[i];
    }
    return out;
}

bool SplitCalib::decode(const std::string& text) {
    *this = SplitCalib();
    size_t a = 0;
    while (a < text.size()) {
        size_t b = text.find(' ', a);
        if (b == std::string::npos) b = text.size();
        const std::string kv = text.substr(a, b - a);
        a = b + 1;
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        if (k == "windows") windows = std::atoll(v.c_str());
        else if (k == "fixed_ms") fixed_ms = std::atof(v.c_str());
        else if (k == "net_ms") net_ms = std::atof(v.c_str());
        else if (k == "seen") {
            size_t c = 0;
            while (c < v.size()) {
                size_t d = v.find(';', c);
                if (d == std::string::npos) d = v.size();
                if (d > c) seen.push_back(v.substr(c, d - c));
                c = d + 1;
            }
        } else if (k == "scale") {
            size_t c = 0;
            while (c < v.size()) {
                size_t d = v.find(',', c);
                if (d == std::string::npos) d = v.size();
                scale.push_back(std::atof(v.substr(c, d - c).c_str()));
                c = d + 1;
            }
        }
    }
    return !scale.empty();
}

bool load_calib(const std::string& path, const std::string& key, SplitCalib& c) {
    c = SplitCalib();
    std::FILE* f = path.empty() ? nullptr : std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    std::string all;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) all.append(buf, n);
    std::fclose(f);
    size_t a = 0;
    bool found = false;
    while (a < all.size()) {
        size_t b = all.find('\n', a);
        if (b == std::string::npos) b = all.size();
        std::string line = all.substr(a, b - a);
        a = b + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t tab = line.find('\t');
        if (line.empty() || line[0] == '#' || tab == std::string::npos || line.substr(0, tab) != key) continue;
        found = c.decode(line.substr(tab + 1));
    }
    return found;
}

bool save_calib(const std::string& path, const std::string& key, const SplitCalib& c, std::string& err) {
    if (path.empty()) return true;
    std::string keep = "# StrataPool: the split's measured timings per pool (written by the engine after each request;\n"
                       "# the automatic split uses them).  Delete this file to start again from the built-in estimate.\n";
    if (std::FILE* f = std::fopen(path.c_str(), "rb")) {
        std::string all;
        char buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) all.append(buf, n);
        std::fclose(f);
        size_t a = 0;
        while (a < all.size()) {
            size_t b = all.find('\n', a);
            if (b == std::string::npos) b = all.size();
            std::string line = all.substr(a, b - a);
            a = b + 1;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t tab = line.find('\t');
            if (line.empty() || line[0] == '#' || tab == std::string::npos || line.substr(0, tab) == key) continue;
            keep += line + "\n";
        }
    }
    keep += key + "\t" + c.encode() + "\n";
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) { err = "cannot write " + tmp; return false; }
    const bool ok = std::fwrite(keep.data(), 1, keep.size(), f) == keep.size();
    if (std::fclose(f) != 0 || !ok) { err = "cannot write " + tmp; std::remove(tmp.c_str()); return false; }
    std::remove(path.c_str());   // Windows: rename does not replace
    if (std::rename(tmp.c_str(), path.c_str()) != 0) { err = "cannot replace " + path; return false; }
    return true;
}

SplitPlan auto_split(const SplitModel& m, const std::vector<NodeCap>& nodes) {
    const int ns = (int) nodes.size();
    SplitPlan best;
    best.why = "no placement fits";
    if (ns < 1) return best;
    if (ns == 1) return evaluate_split(m, nodes, {});
    const int64_t L = m.n_layers;
    std::string last_why;
    // with measured timings: only near a split they were measured with (SplitCalib::near); the plain estimate
    // searches everywhere (and takes over when nothing near a measured split fits any more)
    const bool trust = m.calib != nullptr && m.calib->usable(nodes.size());
    auto consider = [&](const std::vector<int64_t>& at) {
        if (trust && !m.calib->near(at)) return;
        SplitPlan p = evaluate_split(m, nodes, at);
        if (!p.ok) { last_why = p.why; return; }
        // ties (within 0.1%): the placement whose busiest node holds the fewest layers above its share
        if (!best.ok || p.ms < best.ms * 0.999) best = p;
    };
    std::vector<int64_t> at((size_t) ns - 1);
    if (ns == 2) {
        for (int64_t k = m.first_min; k < L; ++k) { at[0] = k; consider(at); }
    } else if (ns == 3) {
        for (int64_t k1 = m.first_min; k1 + 1 < L; ++k1)
            for (int64_t k2 = k1 + 1; k2 < L; ++k2) { at[0] = k1; at[1] = k2; consider(at); }
    } else {
        double tot = 0;
        for (const NodeCap& n : nodes) tot += 1.0 / std::max(n.layer_ms, 1e-3);
        double acc = 0;
        for (int i = 0; i + 1 < ns; ++i) {
            acc += 1.0 / std::max(nodes[(size_t) i].layer_ms, 1e-3);
            at[(size_t) i] = std::clamp<int64_t>((int64_t) std::llround(acc / tot * (double) L),
                                                 i == 0 ? m.first_min : at[(size_t) i - 1] + 1, L - (ns - 1 - i));
        }
        consider(at);
        // nudge each point either way while it improves (and so that RAM fits)
        for (int pass = 0; pass < 4; ++pass)
            for (int i = 0; i + 1 < ns; ++i)
                for (const int d : {-1, 1}) {
                    std::vector<int64_t> t = best.ok ? best.at : at;
                    t[(size_t) i] += d;
                    consider(t);
                }
    }
    if (!best.ok && trust) {   // nothing near a measured split fits now: the built-in estimate, everywhere
        SplitModel plain = m;
        plain.calib = nullptr;
        return auto_split(plain, nodes);
    }
    if (!best.ok) best.why = last_why.empty() ? "no placement fits" : last_why;
    return best;
}

bool parse_split(const std::string& s, std::vector<int64_t>& at) {
    at.clear();
    size_t a = 0;
    while (a < s.size()) {
        size_t b = s.find(',', a);
        if (b == std::string::npos) b = s.size();
        const std::string t = s.substr(a, b - a);
        if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos) return false;
        at.push_back(std::atoll(t.c_str()));
        a = b + 1;
    }
    for (size_t i = 1; i < at.size(); ++i)
        if (at[i] <= at[i - 1]) return false;
    return !at.empty();
}

std::vector<std::pair<int32_t, int32_t>> rank_for_range(const std::vector<std::pair<int32_t, int32_t>>& learned,
                                                        const std::vector<std::pair<int32_t, int32_t>>& prior,
                                                        int64_t lb, int64_t le, int64_t n_layers, int64_t n_expert) {
    if (prior.empty() || (lb <= 0 && le >= n_layers) || n_layers <= 0 || n_expert <= 0) return learned;
    const size_t n = (size_t) (n_layers * n_expert);
    auto idx = [&](const std::pair<int32_t, int32_t>& p) -> int64_t {
        return p.first >= 0 && p.first < n_layers && p.second >= 0 && p.second < n_expert
                   ? (int64_t) p.first * n_expert + p.second : -1;
    };
    auto in_range = [&](const std::pair<int32_t, int32_t>& p) { return p.first >= lb && p.first < le; };
    // this range's pairs in the learned order (each once)
    std::vector<std::pair<int32_t, int32_t>> mine;
    std::vector<uint8_t> seen(n, 0);
    for (const auto& p : learned) {
        const int64_t i = idx(p);
        if (i < 0 || seen[(size_t) i] || !in_range(p)) continue;
        seen[(size_t) i] = 1;
        mine.push_back(p);
    }
    std::vector<std::pair<int32_t, int32_t>> out;
    out.reserve(n);
    std::vector<uint8_t> placed(n, 0);
    size_t next = 0;
    std::vector<uint8_t> prior_seen(n, 0);   // prior's own duplicates count once
    for (const auto& p : prior) {
        const int64_t i = idx(p);
        if (i < 0 || prior_seen[(size_t) i]) continue;
        prior_seen[(size_t) i] = 1;
        if (in_range(p)) {   // a place of this range (whichever of its pairs was here): the next learned one takes it
            // the range's next learned pair takes this place (one not yet placed)
            while (next < mine.size() && placed[(size_t) idx(mine[next])]) ++next;
            if (next >= mine.size()) continue;   // prior has more of this range than learned: dropped here, see below
            const auto q = mine[next++];
            placed[(size_t) idx(q)] = 1;
            out.push_back(q);
        } else if (!placed[(size_t) i]) {
            placed[(size_t) i] = 1;
            out.push_back(p);
        }
    }
    // what prior lacked: the rest of the range in the learned order, then every other pair in the learned order
    for (; next < mine.size(); ++next)
        if (!placed[(size_t) idx(mine[next])]) { placed[(size_t) idx(mine[next])] = 1; out.push_back(mine[next]); }
    for (const auto& p : learned) {
        const int64_t i = idx(p);
        if (i >= 0 && !placed[(size_t) i]) { placed[(size_t) i] = 1; out.push_back(p); }
    }
    return out;
}

}  // namespace strata::pool
