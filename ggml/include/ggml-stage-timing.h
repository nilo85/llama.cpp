// Temporary investigation harness: per-stage wall-clock timing to locate the
// decode bottleneck on SYCL. Compile-time gated by -DGGML_STAGE_TIMING (default off).
//   OFF  -> production build: no-op stubs, zero instrumentation code (safe for benching).
//   ON   -> instrumentation build: full harness; runtime env (GGML_STAGE_TIMING /
//            GGML_STAGE_SYNC_PASS / GGML_STAGE_OP_TIMING) selects sub-features.
// NEVER report throughput from an instrumentation build; use the production build for numbers.
#pragma once

#include "ggml.h"
#include <cstdint>

#ifdef GGML_STAGE_TIMING
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace gstage {

enum Stage : int {
    ST_SET_TENSOR,        // SYCL host->device upload (synchronous path, whole call)
    ST_SET_TENSOR_DRAIN,  //   part: queue drain (queues_wait_and_throw)
    ST_SET_TENSOR_MEMCPY, //   part: H2D memcpy + wait
    ST_CPY_TENSOR,        // SYCL synchronous copy (buffer_cpy_tensor)
    ST_GRAPH_COMPUTE,     // SYCL GPU compute (graph_compute, launch->return)
    ST_GET_ROWS,          // CPU PLE gather (ggml_compute_forward_get_rows)
    ST_PREFETCH,          // llama_prefetch_rows (madvise WILLNEED)
    ST_GET_TENSOR,        // SYCL device->host readback incl. wait (get_tensor)
    ST_GRAPH_BUILD,       // CPU graph build (model.build_graph)
    ST_SCHED,             // ggml_backend_sched_graph_compute_async incl. alloc+split
    ST_SCHED_SYNC,        // host waits inside compute_splits (sync/event_synchronize)
    ST_SPLIT_COMPUTE,     // ggml_backend_graph_compute_async per split (CPU+SYCL)
    ST_SET_TENSOR_ASYNC,  // SYCL async H2D launch (set_tensor_async, incl. MoE expert copies)
    ST_SET_INPUTS,        // res->set_inputs (input cache update incl. PLE rows) in process_ubatch
    ST_DECODE_TOTAL,      // whole llama_context::decode (batch -> kv_update)
    ST_SYNC_CTX,          // llama_context::synchronize (full sched sync, all backends)
    ST_SAMPLE,            // common_sampler_sample (incl. leading sync)
    ST_SLOT_PRE,          // server pre_decode (batch building)
    ST_SLOT_POST,         // server post_decode
    ST_RENDER,            // server batch.render
    ST_MOE_FUSED,         // SYCL mul_mat_id fused MMVQ path (no host wait)
    ST_MOE_WAIT,          // SYCL mul_mat_id generic path (D2H ids + blocking wait)
    ST_SYNC_API,          // public llama_synchronize (with per-caller return address)
    ST_N
};

inline uint64_t now_us() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline const char * stage_name(Stage s) {
    switch (s) {
        case ST_SET_TENSOR:        return "set_tensor";
        case ST_SET_TENSOR_DRAIN:  return "  .drain";
        case ST_SET_TENSOR_MEMCPY: return "  .memcpy";
        case ST_CPY_TENSOR:        return "cpy_tensor";
        case ST_GRAPH_COMPUTE:     return "graph_compute";
        case ST_GET_ROWS:          return "get_rows";
        case ST_PREFETCH:          return "prefetch";
        case ST_GET_TENSOR:        return "get_tensor";
        case ST_GRAPH_BUILD:       return "graph_build";
        case ST_SCHED:             return "sched_compute";
        case ST_SCHED_SYNC:        return "  sched_sync";
        case ST_SPLIT_COMPUTE:     return "  split_compute";
        case ST_SET_TENSOR_ASYNC:  return "set_tensor_async";
        case ST_SET_INPUTS:        return "  set_inputs";
        case ST_DECODE_TOTAL:      return "decode_total";
        case ST_SYNC_CTX:          return "sync_ctx";
        case ST_SAMPLE:            return "sample";
        case ST_SLOT_PRE:          return "slot_pre_decode";
        case ST_SLOT_POST:         return "slot_post_decode";
        case ST_RENDER:            return "batch_render";
        case ST_MOE_FUSED:         return "  moe_fused";
        case ST_MOE_WAIT:          return "  moe_wait";
        case ST_SYNC_API:          return "sync_api";
        default:                   return "?";
    }
}

struct State {
    bool     enabled        = false;
    bool     sync_pass      = false;   // GGML_STAGE_SYNC_PASS=1: wait after each split submit
    bool     op_timing      = false;   // GGML_STAGE_OP_TIMING=1: per-op device time via events
    uint64_t ns[ST_N]       = {0};
    uint64_t n [ST_N]       = {0};
    uint64_t last_ns[ST_N]  = {0};
    uint64_t last_n [ST_N]  = {0};
    uint64_t last_gc        = 0;
    uint64_t last_t         = 0;
    uint64_t gc             = 0;
    uint64_t t_start        = 0;
    bool     reset_done     = false;
    int64_t  print_every_us = 2000000; // 2 s
    int64_t  min_calls      = 32;

    // per-op device time (ns) from barrier events, indexed by ggml_op
    uint64_t op_ns[GGML_OP_COUNT]  = {0};
    uint64_t op_n [GGML_OP_COUNT]  = {0};
    uint64_t l_op_ns[GGML_OP_COUNT] = {0};
    uint64_t l_op_n [GGML_OP_COUNT] = {0};

    // per-split submit/wait (diagnostic: launch overhead vs queue backpressure)
    struct SplitStat { uint64_t submit_ns = 0, wait_ns = 0, n = 0, l_submit = 0, l_wait = 0, l_n = 0; };
    SplitStat splits[16];

    // per-caller sync breakdown (return address -> time)
    struct SyncCaller { void * addr = nullptr; uint64_t ns = 0, n = 0, l_ns = 0, l_n = 0; };
    SyncCaller syncs[ST_N][8];
};

inline State & state() {
    static State st;
    if (st.t_start == 0) {
        st.t_start = now_us();
        const char * e = getenv("GGML_STAGE_TIMING");
        st.enabled = e && e[0] != '\0' && e[0] != '0';
        const char * p = getenv("GGML_STAGE_SYNC_PASS");
        st.sync_pass = p && p[0] == '1';
        const char * o = getenv("GGML_STAGE_OP_TIMING");
        st.op_timing = o && o[0] == '1';
    }
    return st;
}

// per-op device time recording (barrier events, only when GGML_STAGE_OP_TIMING=1)
inline void record_op(int op, uint64_t ns) {
    State & st = state();
    if (!st.enabled || op < 0 || op >= GGML_OP_COUNT) return;
    st.op_ns[op] += ns;
    st.op_n[op]++;
}

// per-split submit/wait recording (only when GGML_STAGE_SYNC_PASS=1)
inline void record_split(int idx, uint64_t submit_us, uint64_t wait_us) {
    State & st = state();
    if (!st.enabled || idx < 0 || idx >= 16) return;
    st.splits[idx].submit_ns += submit_us * 1000;
    st.splits[idx].wait_ns   += wait_us * 1000;
    st.splits[idx].n++;
}

// per-caller sync recording: ret = __builtin_return_address(0) at the sync site
inline void record_sync(Stage s, void * ret, uint64_t us) {
    State & st = state();
    if (!st.enabled) return;
    for (int i = 0; i < 8; i++) {
        if (st.syncs[s][i].addr == ret) { st.syncs[s][i].ns += us * 1000; st.syncs[s][i].n++; return; }
    }
    for (int i = 0; i < 8; i++) {
        if (st.syncs[s][i].addr == nullptr) { st.syncs[s][i].addr = ret; st.syncs[s][i].ns = us * 1000; st.syncs[s][i].n = 1; return; }
    }
}

// RAII timer. No-op when disabled.
struct Timer {
    State & st;
    Stage   s;
    uint64_t t0;
    bool     on;
    explicit Timer(Stage stage) : st(state()), s(stage), t0(0), on(st.enabled) {
        if (on) t0 = now_us();
    }
    ~Timer() {
        if (!on) return;
        st.ns[s] += now_us() - t0;
        st.n[s]++;
    }
};

// Call once per graph_compute. Resets the load-time accumulators on the first
// compute, then prints the delta when due.
inline void on_graph_compute() {
    State & st = state();
    if (!st.enabled) return;
    st.gc++;
    if (!st.reset_done) {
        // clear the one-time model-load uploads so reports reflect steady state
        for (int i = 0; i < ST_N; i++) { st.ns[i] = 0; st.n[i] = 0; st.last_ns[i] = 0; st.last_n[i] = 0; }
        for (int i = 0; i < 16; i++) { st.splits[i] = State::SplitStat(); }
        for (int i = 0; i < ST_N; i++) { for (int c = 0; c < 8; c++) { st.syncs[i][c] = State::SyncCaller(); } }
        for (int i = 0; i < GGML_OP_COUNT; i++) {
            st.op_ns[i] = 0; st.op_n[i] = 0; st.l_op_ns[i] = 0; st.l_op_n[i] = 0;
        }
        st.last_gc = 0;
        st.last_t  = now_us();
        st.reset_done = true;
    }
    const uint64_t t = now_us();
    const bool due = (t - st.last_t >= (uint64_t) st.print_every_us) &&
                     (st.gc - st.last_gc >= (uint64_t) st.min_calls);
    if (!due) return;
    const uint64_t dgc = st.gc - st.last_gc;
    fprintf(stderr, "\n[STAGE] ===== report t=%.2fs gc=%llu dgc=%llu =====\n",
            (double) (t - st.t_start) / 1e6, (unsigned long long) st.gc, (unsigned long long) dgc);
    fprintf(stderr, "[STAGE] %-16s %12s %10s %12s %12s\n",
            "stage", "delta_ms", "calls", "avg_us", "us_per_gc");
    for (int i = 0; i < ST_N; i++) {
        const uint64_t dns = st.ns[i] - st.last_ns[i];
        const uint64_t dn  = st.n[i]  - st.last_n[i];
        const double dms   = (double) dns / 1e3;
        const double avg   = dn  ? (double) dns / dn : 0.0;         // us per call
        const double usgc  = dgc ? (double) dns / dgc : 0.0;        // us per graph_compute
        fprintf(stderr, "[STAGE] %-16s %12.2f %10llu %12.1f %12.1f\n",
                stage_name((Stage) i), dms, (unsigned long long) dn, avg, usgc);
        st.last_ns[i] = st.ns[i];
        st.last_n[i]  = st.n[i];
        // per-caller breakdown for sync stages
        for (int c = 0; c < 8; c++) {
            State::SyncCaller & sc = st.syncs[i][c];
            if (sc.addr == nullptr) continue;
            const uint64_t dns_c = sc.ns - sc.l_ns;
            const uint64_t dn_c  = sc.n  - sc.l_n;
            sc.l_ns = sc.ns;
            sc.l_n  = sc.n;
            if (dn_c == 0) continue;
            fprintf(stderr, "[STAGE]   %-14s @%p %10llu %12.1f\n",
                    stage_name((Stage) i), sc.addr, (unsigned long long) dn_c, (double) dns_c / dn_c / 1000.0);
        }
    }
    // per-split submit vs wait (diagnostic)
    for (int i = 0; i < 16; i++) {
        State::SplitStat & ss = st.splits[i];
        if (ss.n == 0) continue;
        const uint64_t dsub = ss.submit_ns - ss.l_submit;
        const uint64_t dwait = ss.wait_ns - ss.l_wait;
        const uint64_t dn = ss.n - ss.l_n;
        ss.l_submit = ss.submit_ns;
        ss.l_wait   = ss.wait_ns;
        ss.l_n      = ss.n;
        if (dn == 0) continue;
        fprintf(stderr, "[STAGE]   split#%-3d submit %12.1f us  wait %12.1f us  (n=%llu)\n",
                i, (double) dsub / dn / 1000.0, (double) dwait / dn / 1000.0, (unsigned long long) dn);
    }
    // per-op device time (top 12 by delta)
    if (st.op_timing) {
        fprintf(stderr, "[STAGE] %-16s %12s %10s %12s %12s\n",
                "op", "delta_ms", "calls", "avg_us", "us_per_gc");
        int      idx[12] = {0};
        uint64_t val[12] = {0};
        uint64_t cnt[12] = {0};
        for (int i = 0; i < GGML_OP_COUNT; i++) {
            const uint64_t dns = st.op_ns[i] - st.l_op_ns[i];
            const uint64_t dn  = st.op_n[i]  - st.l_op_n[i];
            st.l_op_ns[i] = st.op_ns[i];
            st.l_op_n[i]  = st.op_n[i];
            if (dns == 0) continue;
            int pos = 12;
            for (int k = 0; k < 12; k++) { if (dns > val[k]) { pos = k; break; } }
            if (pos >= 12) continue;
            for (int k = 11; k > pos; k--) { val[k] = val[k-1]; cnt[k] = cnt[k-1]; idx[k] = idx[k-1]; }
            val[pos] = dns; cnt[pos] = dn; idx[pos] = i;
        }
        for (int k = 0; k < 12; k++) {
            if (val[k] == 0) break;
            fprintf(stderr, "[STAGE] %-16s %12.2f %10llu %12.1f %12.1f\n",
                    ggml_op_name((enum ggml_op) idx[k]), (double) val[k] / 1e6,
                    (unsigned long long) cnt[k], (double) val[k] / cnt[k] / 1000.0,
                    dgc ? (double) val[k] / dgc / 1000.0 : 0.0);
        }
    }
    st.last_gc = st.gc;
    st.last_t  = t;
    fprintf(stderr, "[STAGE] ===== end report =====\n\n");
}

} // namespace gstage

#else
// Production build: instrumentation compiled out. No-op stubs so the call sites
// (gstage::Timer / record_* / on_graph_compute / now_us / state()) still compile
// to nothing; the optimizer drops the constant-false branches.
namespace gstage {

enum Stage : int {
    ST_SET_TENSOR, ST_SET_TENSOR_DRAIN, ST_SET_TENSOR_MEMCPY, ST_CPY_TENSOR,
    ST_GRAPH_COMPUTE, ST_GET_ROWS, ST_PREFETCH, ST_GET_TENSOR, ST_GRAPH_BUILD,
    ST_SCHED, ST_SCHED_SYNC, ST_SPLIT_COMPUTE, ST_SET_TENSOR_ASYNC, ST_SET_INPUTS,
    ST_DECODE_TOTAL, ST_SYNC_CTX, ST_SAMPLE, ST_SLOT_PRE, ST_SLOT_POST, ST_RENDER,
    ST_MOE_FUSED, ST_MOE_WAIT, ST_SYNC_API, ST_N
};

inline uint64_t now_us() { return 0; }

struct State { bool enabled = false; bool sync_pass = false; bool op_timing = false; };
inline State & state() { static State s; return s; }

struct Timer { uint64_t t0 = 0; explicit Timer(Stage) {} ~Timer() {} };

inline void record_op(int, uint64_t) {}
inline void record_split(int, uint64_t, uint64_t) {}
inline void record_sync(Stage, void *, uint64_t) {}
inline void on_graph_compute() {}

} // namespace gstage
#endif
