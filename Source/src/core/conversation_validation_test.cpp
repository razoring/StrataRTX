// Host-only validation fixtures. CUDA is linked but must never be initialized:
// invalid restores must return before the first CUDA call or destination write.
#include "strata/core/conversation_snapshot.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <cstring>
#include <utility>
#include <vector>

#if defined(CONVERSATION_TEST_TRANSFERS)
// GNU/ELF link wrapping exercises the actual restore control flow without
// initializing CUDA or deliberately poisoning a real device context. This is
// host-backend fault injection, not a claim of hardware-failure recovery.
namespace {
int copy_calls = 0, sync_calls = 0, fail_copy = 0, fail_sync = 0;
size_t copied_bytes = 0;
int current_device = 0;                        // the device OnDevice has made current
int drained_on_device[4] = {0};                // drains counted on each device
struct CopyEvent { int device; int drains; const void* dst; const void* src; };
std::vector<CopyEvent> copy_order;   // what the wrapping backend saw at each copy
void reset_counters() {
    copy_calls = sync_calls = fail_copy = fail_sync = 0;
    copied_bytes = 0; current_device = 0;
    for (int& n : drained_on_device) n = 0;
    copy_order.clear();
}
}
extern "C" cudaError_t __wrap_cudaGetDevice(int* p) { *p = current_device; return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaSetDevice(int d) { current_device = d; return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    if (++copy_calls == fail_copy) return cudaErrorInvalidValue;
    copied_bytes += n;
    copy_order.push_back({current_device, drained_on_device[current_device], dst, src});
    std::memcpy(dst, src, n);
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaDeviceSynchronize() {
    ++drained_on_device[current_device];
    return ++sync_calls == fail_sync ? cudaErrorUnknown : cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaGetLastError() { return cudaSuccess; }
#endif

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
struct Pools {
    QsaState st;
    std::array<std::vector<uint8_t>, 8> data;
    Pools(const ModelGeometry& g, int format) {
        st.max_cells = 96; st.n_pages = st.n_slots = 24; st.idx_pooled_rows = 26;
        st.kv_int8 = format == 1; st.kv_q4 = format == 2;
        const size_t per = format == 2 ? strata::kernels::kv_q4_bytes_per_head((int) g.head_dim)
                                      : g.head_dim * (format == 1 ? 1 : 2);
        data[0].resize(96 * g.n_head_kv * per, 0xa5); data[1] = data[0];
        data[2].resize(format == 1 ? 96 * g.n_head_kv * (g.head_dim / 64) * 2 : 0, 0xa5); data[3] = data[2];
        data[4].resize(26 * g.idx_key_dim * 4, 0xa5);
        data[5].resize(3 * g.idx_key_dim * 4, 0xa5);
        data[6].resize(g.idx_key_dim * 4, 0xa5); data[7].resize(4, 0xa5);
        if (format == 2) { st.k_q4 = data[0].data(); st.v_q4 = data[1].data(); }
        else if (format == 1) {
            st.k_q = (int8_t*) data[0].data(); st.v_q = (int8_t*) data[1].data();
            st.k_scale = (uint16_t*) data[2].data(); st.v_scale = (uint16_t*) data[3].data();
        } else { st.k_pool = (uint16_t*) data[0].data(); st.v_pool = (uint16_t*) data[1].data(); }
        st.idx_pooled = (float*) data[4].data(); st.idx_tail = (float*) data[5].data();
        st.idx_dead = (float*) data[6].data(); st.idx_block_pos = (int32_t*) data[7].data();
    }
    ConversationKv image(const ModelGeometry& g, bool index) const {
        ConversationKv k;
        k.format = qsa_kv_format(st); k.cells = 12; k.page_size = 4;
        k.heads = g.n_head_kv; k.head_dim = g.head_dim; k.idx_dim = g.idx_key_dim;
        k.pooled_rows = index ? 3 : 0;
        k.k.resize(data[0].size() / 8, 13); k.v = k.k;
        k.k_scale.resize(data[2].size() / 8, 13); k.v_scale = k.k_scale;
        k.pooled.resize(k.pooled_rows * g.idx_key_dim * 4, 13);
        return k;
    }
};

void fixture(int format, int experts, bool zero_qsa, bool ple) {
    ModelGeometry g;
    g.n_layers = zero_qsa ? 3 : 8; g.n_expert = experts;
    g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    g.n_head_kv = 1; g.head_dim = 64; g.idx_key_dim = 8;
    std::string error;
    ConversationStateSizes z;
    check(conversation_state_sizes(g, z, error), "checked canonical/pruned/zero-QSA sizing");
    Pools first(g, format), last(g, format), draft(g, format);
    std::array<QsaState, 2> layers{first.st, last.st};
    std::vector<uint8_t> gdn(z.gdn, 0xa5), history(ple ? z.ple : 0, 0xa5);
    SessionState ss;
    ss.max_cells = 96; ss.gdn_state = (float*) gdn.data();
    // the whole-model carve, as session_init leaves a one-GPU session
    ss.layer_hi = g.n_layers; ss.gdn_alloc = g.n_gdn_layers(); ss.qsa_alloc = g.n_qsa_layers();
    ss.ple_hist = ple ? (float*) history.data() : nullptr;
    ss.qsa_states = zero_qsa ? nullptr : layers.data();
    ConversationStages stages{{&ss, -1}};
    SavedConversation image;
    image.geometry = {g.n_embd,g.n_layers,g.qsa_interval,g.ssm_state_size,g.ssm_k_heads,g.ssm_v_heads,
                      g.ssm_d_conv,g.ssm_conv_channels,g.ssm_value_dim,g.n_head,g.n_head_kv,g.head_dim,
                      g.idx_q_heads,g.idx_key_dim,g.hc,g.hc_lr,g.n_expert,g.n_ff};
    image.layer_hi = g.n_layers;
    auto checkpoint = [&](size_t tokens) {
        ConversationCheckpoint c;
        c.ids.resize(tokens);
        for (size_t i = 0; i < tokens; ++i) c.ids[i] = (int32_t) i + 1;
        c.imgs = {{1, 44}};
        c.layer_lo = ss.layer_lo; c.layer_hi = ss.layer_hi;
        c.gdn.resize(z.gdn, 13); c.ple.resize(ple ? z.ple : 0, 13);
        c.tails.resize(g.n_qsa_layers() * z.tail, 13); c.dead.resize(g.n_qsa_layers() * z.dead, 13);
        c.block_pos.resize(g.n_qsa_layers() * z.block_pos, 13);
        return c;
    };
    image.live = checkpoint(9); image.checkpoints.push_back(checkpoint(5));
    image.kv.reserve((size_t) g.n_qsa_layers() + 1);
    if (!zero_qsa) { image.kv.push_back(first.image(g, true)); image.kv.push_back(last.image(g, true)); }
    image.kv.push_back(draft.image(g, false));
    check(conversation_snapshot_validate(image, stages, g, draft.st, error), "complete image validates without CUDA");
    size_t estimate = 0;
    check(conversation_snapshot_bytes({image.live.ids,image.live.imgs,image.checkpoints,true},stages,g,draft.st,estimate,error),
          "capture estimate works without CUDA");
    check(estimate == image.bytes(), "estimate covers checkpoint/indexer/spare payloads");
    auto unchanged = [&] {
        auto pristine = [](const auto& bytes) { return std::all_of(bytes.begin(),bytes.end(),[](uint8_t b){return b==0xa5;}); };
        if (!pristine(gdn) || !pristine(history) || ss.ple_prev[0] != -1 || ss.ple_prev[1] != -1) return false;
        for (const auto* p : {&first,&last,&draft}) for (const auto& bytes : p->data) if (!pristine(bytes)) return false;
        return true;
    };
    auto reject = [&](const std::function<void(SavedConversation&)>& mutate, const char* label) {
        auto bad = image; mutate(bad);
        check(conversation_snapshot_restore(bad,stages,g,draft.st,error) == ConversationRestore::invalid,label);
        check(unchanged(), "invalid restore did not touch any layer or running-state buffer");
    };
    reject([](auto& s){s.kv.back().k.pop_back();}, "late draft corruption rejected before first layer write");
    reject([](auto& s){s.geometry[16] = 128;}, "different expert geometry rejected");
    reject([](auto& s){s.live.stage_parts.emplace_back();}, "stage part count beyond the single stage rejected");
    reject([](auto& s){s.checkpoints[0].stage_parts.emplace_back();}, "checkpoint part count beyond the single stage rejected");
    reject([](auto& s){s.kv.pop_back();}, "missing KV layer rejected");
    reject([](auto& s){s.live.gdn.pop_back();}, "bad live recurrence rejected");
    reject([](auto& s){s.checkpoints.back().gdn.pop_back();}, "bad retained checkpoint rejected");
    reject([](auto& s){s.checkpoints.back().ids[0] = 99;}, "foreign checkpoint prefix rejected");
    reject([](auto& s){s.checkpoints.back().imgs[0].hash++;}, "foreign checkpoint image rejected");
    reject([](auto& s){s.live.imgs[0].start = -1;}, "negative image position rejected");
    reject([](auto& s){s.live.ids[0] = -1;}, "negative live token rejected");
    reject([](auto& s){s.live.ple.push_back(0);}, "PLE size/enabled mismatch rejected");
    reject([](auto& s){s.live.tails.push_back(0);}, "tail size/zero-QSA mismatch rejected");
    if (!zero_qsa) {
        reject([](auto& s){s.kv[1].pooled.pop_back();}, "late indexer spare row corruption rejected");
        reject([](auto& s){s.live.dead.pop_back();}, "missing indexer spare key rejected");
        reject([](auto& s){s.live.block_pos.pop_back();}, "missing indexer metadata rejected");
        ss.qsa_states = nullptr;
        check(!conversation_snapshot_validate(image,stages,g,draft.st,error), "missing QSA targets rejected");
        ss.qsa_states = layers.data();
        reject([](auto& s){s.layer_lo = 4;}, "snapshot from another layer range rejected");
        // #216's carve: a split stage owning layers [4, 8) holds GDN rows 3..5 and QSA ordinal 1 only
        SessionState stage = ss;
        stage.layer_lo = 4; stage.gdn_alloc = 3; stage.qsa_ord0 = 1; stage.qsa_alloc = 1;
        ConversationStateSizes zs;
        check(conversation_session_sizes(g,stage,zs,error) && zs.gdn == z.gdn / 6 * 3 && zs.tail == z.tail,
              "carved session sizes cover the owned rows only");
        check(!conversation_checkpoint_validate(image.live,stage,g,error), "whole-model checkpoint rejected by a carve");
        ConversationCheckpoint part = image.live;
        part.layer_lo = stage.layer_lo; part.layer_hi = stage.layer_hi;
        part.gdn.resize(zs.gdn); part.tails.resize(zs.tail); part.dead.resize(zs.dead);
        part.block_pos.resize(zs.block_pos);
        check(conversation_checkpoint_validate(part,stage,g,error), "carve-sized checkpoint validates");
        ConversationStages stage_stages{{&stage, -1}};
        check(!conversation_snapshot_validate(image,stage_stages,g,draft.st,error), "whole-model snapshot rejected by a carve");
        stage.qsa_alloc = 2;
        check(!conversation_session_sizes(g,stage,zs,error), "carve past the last QSA ordinal rejected");
        stage.qsa_alloc = 1; stage.gdn_alloc = 7;
        check(!conversation_session_sizes(g,stage,zs,error), "carve past the GDN rows rejected");
    }
    auto bad_geometry = g; bad_geometry.ssm_state_size = std::numeric_limits<int64_t>::max();
    check(!conversation_state_sizes(bad_geometry,z,error), "running-state arithmetic overflow rejected");
    bad_geometry = g; bad_geometry.qsa_interval = 0;
    check(!conversation_state_sizes(bad_geometry,z,error), "zero layer interval rejected before division");
    bad_geometry = g; bad_geometry.n_head_kv = std::numeric_limits<int64_t>::max();
    check(conversation_kv_bytes(draft.st,bad_geometry,9,false)==0, "KV byte overflow rejected");
#if defined(CONVERSATION_TEST_TRANSFERS)
    auto reset = [&] {
        for (auto* p : {&first,&last,&draft}) for (auto& bytes : p->data) std::fill(bytes.begin(),bytes.end(),0xa5);
        std::fill(gdn.begin(),gdn.end(),0xa5); std::fill(history.begin(),history.end(),0xa5);
        ss.ple_prev[0] = ss.ple_prev[1] = -1;
        reset_counters();
    };
    reset(); fail_sync = 1;
    check(conversation_snapshot_restore(image,stages,g,draft.st,error)==ConversationRestore::transfer_failed,
          "pre-transfer CUDA synchronization failure is fatal");
    check(copy_calls==0 && unchanged(), "pre-sync failure did not mutate state");
    reset(); fail_copy = 1;
    check(conversation_snapshot_restore(image,stages,g,draft.st,error)==ConversationRestore::transfer_failed,
          "first CUDA transfer failure is fatal");
    check(unchanged(), "first-copy failure did not mutate state");
    reset(); fail_copy = 2;
    check(conversation_snapshot_restore(image,stages,g,draft.st,error)==ConversationRestore::transfer_failed,
          "partial CUDA transfer failure is fatal, not an invalid-image fallback");
    check(!unchanged(), "fault fixture genuinely produced partial state");
    reset(); fail_sync = 2;
    check(conversation_snapshot_restore(image,stages,g,draft.st,error)==ConversationRestore::transfer_failed,
          "post-transfer CUDA synchronization failure is fatal");
    check(copy_calls>0 && !unchanged(), "post-sync failure occurred after state writes");
    reset();
    check(conversation_snapshot_restore(image,stages,g,draft.st,error)==ConversationRestore::restored,
          "injected host transfer backend completes a valid restore");
    check(gdn==image.live.gdn && history==image.live.ple && ss.ple_prev[0]==8 && ss.ple_prev[1]==9,
          "successful restore publishes correct running state and PLE window");
    uint64_t fingerprint = 0;
    check(conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back verifies complete draft payload");
    draft.data[0][0] ^= 1;
    check(!conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back detects corrupted draft byte");
    draft.data[0][0] ^= 1;
    fail_copy = copy_calls + 1;
    check(!conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back transfer failure is not a successful fingerprint");
    reset(); copied_bytes = 0;
    const ConversationView view{image.live.ids,image.live.imgs,image.checkpoints,true};
    SavedConversation full,incremental;
    check(conversation_snapshot_save(full,view,stages,g,draft.st,error), "full capture through host transfer backend");
    const size_t full_copies = copied_bytes;
    ConversationKvReuse reuse{full.kv,9,9};
    size_t peak = 0, reused = 0;
    check(conversation_snapshot_capture_bytes(reuse,view,stages,g,draft.st,peak,error), "incremental capture admission without transfers");
    copied_bytes = 0;
    const bool saved_incrementally = conversation_snapshot_save(incremental,view,stages,g,draft.st,error,std::move(reuse),&reused);
    if (!saved_incrementally) std::fprintf(stderr,"capture error: %s\n",error.c_str());
    check(saved_incrementally, "incremental capture through host backend");
    check(reused > 0 && copied_bytes + reused == full_copies, "reused byte count measures transfers actually omitted");
    check(incremental.bytes() <= peak && incremental.live.gdn == full.live.gdn, "running state refreshed within admitted allocation");
    for (size_t i=0;i<full.kv.size();++i)
        check(incremental.kv[i].k==full.kv[i].k && incremental.kv[i].v==full.kv[i].v &&
              incremental.kv[i].k_scale==full.kv[i].k_scale && incremental.kv[i].v_scale==full.kv[i].v_scale &&
              incremental.kv[i].pooled==full.kv[i].pooled, "incremental and full capture agree across every payload");
    reuse = {full.kv,9,9};
    reuse.kv.back().k.pop_back();
    copy_calls = sync_calls = 0;
    check(!conversation_snapshot_capture_bytes(reuse,view,stages,g,draft.st,peak,error), "malformed retained draft rejected before admission");
    check(!copy_calls && !sync_calls, "invalid retained storage performs no CUDA calls");
    reuse = {full.kv,9,9};
    fail_copy = copy_calls + 1;
    SavedConversation unpublished;
    check(!conversation_snapshot_save(unpublished,view,stages,g,draft.st,error,std::move(reuse)), "incremental capture copy failure is reported");
    check(unpublished.kv.empty() && unpublished.live.ids.empty(), "failed incremental capture cannot publish a partial image");
#endif
}

// A two-stage split: stage 0 owns layers [0,4) (GDN rows 0..2, QSA ordinal 0), stage 1 owns [4,8)
// (rows 3..5, ordinal 1). `dev` stays -1 so no CUDA is touched; the per-device OnDevice scoping itself
// is checked on the build box, not here.
void split_fixture(int format) {
    ModelGeometry g;
    g.n_layers = 8; g.n_expert = 256;
    g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    g.n_head_kv = 1; g.head_dim = 64; g.idx_key_dim = 8;
    std::string error;
    ConversationStateSizes z;
    check(conversation_state_sizes(g, z, error), "split fixture sizing");
    Pools first(g, format), second(g, format), draft(g, format);
    std::vector<uint8_t> gdn0(z.gdn / 6 * 3, 0xa5), hist0(z.ple, 0xa5);
    std::vector<uint8_t> gdn1(z.gdn / 6 * 3, 0xa5), hist1(z.ple, 0xa5);
    SessionState stage0, stage1;
    stage0.max_cells = stage1.max_cells = 96;
    // The QSA table is whole-model: each stage owns a slice of it by ordinal, not its own array.
    std::array<QsaState, 2> qsa{first.st, second.st};
    stage0.gdn_state = (float*) gdn0.data(); stage0.ple_hist = (float*) hist0.data(); stage0.qsa_states = qsa.data();
    stage1.gdn_state = (float*) gdn1.data(); stage1.ple_hist = (float*) hist1.data(); stage1.qsa_states = qsa.data();
    stage0.layer_hi = 4; stage0.gdn_alloc = 3; stage0.qsa_alloc = 1;                            // QSA ordinal 0
    stage1.layer_lo = 4; stage1.layer_hi = 8; stage1.gdn_alloc = 3; stage1.qsa_ord0 = 1; stage1.qsa_alloc = 1;   // ordinal 1
    ConversationStages stages{{&stage0, -1}, {&stage1, -1}};
    SavedConversation image;
    image.geometry = {g.n_embd,g.n_layers,g.qsa_interval,g.ssm_state_size,g.ssm_k_heads,g.ssm_v_heads,
                      g.ssm_d_conv,g.ssm_conv_channels,g.ssm_value_dim,g.n_head,g.n_head_kv,g.head_dim,
                      g.idx_q_heads,g.idx_key_dim,g.hc,g.hc_lr,g.n_expert,g.n_ff};
    image.layer_lo = 0; image.layer_hi = 4;   // stage 0's carve decides the image-level range
    image.kv.reserve(3);   // one owned layer per stage plus the drafter, as capture would size it
    auto part = [&](SessionState& s, size_t tokens) {
        ConversationCheckpoint c;
        c.ids.resize(tokens);
        for (size_t i = 0; i < tokens; ++i) c.ids[i] = (int32_t) i + 1;
        c.imgs = {{1, 44}};
        c.layer_lo = s.layer_lo; c.layer_hi = s.layer_hi;
        ConversationStateSizes zs;
        check(conversation_session_sizes(g, s, zs, error), "split stage sizes");
        c.gdn.resize(zs.gdn, 13); c.ple.resize(zs.ple, 13);
        c.tails.resize(zs.tail, 13); c.dead.resize(zs.dead, 13); c.block_pos.resize(zs.block_pos, 13);
        return c;
    };
    image.live = part(stage0, 9);
    image.live.stage_parts.push_back(part(stage1, 9));
    ConversationCheckpoint five = part(stage0, 5);
    five.stage_parts.push_back(part(stage1, 5));
    image.checkpoints.push_back(std::move(five));
    image.kv.push_back(first.image(g, true));    // stage 0's owned layer
    image.kv.push_back(second.image(g, true));   // stage 1's owned layer
    image.kv.push_back(draft.image(g, false));   // the drafter lives on stage 1; the flat order pins it
    check(conversation_snapshot_validate(image, stages, g, draft.st, error), "two-stage image validates without CUDA");
    size_t estimate = 0;
    check(conversation_snapshot_bytes({image.live.ids,image.live.imgs,image.checkpoints,true},stages,g,draft.st,estimate,error),
          "split capture estimate works without CUDA");
    check(estimate == image.bytes(), "estimate covers both stages' parts and their directories");
    auto unchanged = [&] {
        auto pristine = [](const auto& bytes) { return std::all_of(bytes.begin(),bytes.end(),[](uint8_t b){return b==0xa5;}); };
        if (!pristine(gdn0) || !pristine(hist0) || !pristine(gdn1) || !pristine(hist1)) return false;
        for (const auto* p : {&first,&second,&draft}) for (const auto& bytes : p->data) if (!pristine(bytes)) return false;
        return stage0.ple_prev[0] == -1 && stage0.ple_prev[1] == -1 &&
               stage1.ple_prev[0] == -1 && stage1.ple_prev[1] == -1;
    };
    auto bad = image; bad.live.stage_parts[0].layer_lo = 0;
    check(conversation_snapshot_restore(bad, stages, g, draft.st, error) == ConversationRestore::invalid,
          "stage part from another carve rejected before writes");
    check(unchanged(), "invalid split restore did not touch any stage");
    bad = image; bad.live.stage_parts.pop_back();
    check(conversation_snapshot_restore(bad, stages, g, draft.st, error) == ConversationRestore::invalid,
          "missing stage part rejected before writes");
    check(unchanged(), "short part list did not touch any stage");
#if defined(CONVERSATION_TEST_TRANSFERS)
    // Real device numbers here: a -1 device cannot show which card a copy was made on, and the point is that
    // each stage is drained on its own card before anything there is read or written.
    stages = {{&stage0, 0}, {&stage1, 1}};
    auto on_stage1 = [&](const void* p) {
        auto inside = [&](const std::vector<uint8_t>& v) {
            return p >= (const void*) v.data() && p < (const void*) (v.data() + v.size());
        };
        if (inside(gdn1) || inside(hist1)) return true;
        for (const auto& bytes : second.data) if (inside(bytes)) return true;
        return false;
    };
    reset_counters();
    check(conversation_snapshot_restore(image, stages, g, draft.st, error) == ConversationRestore::restored,
          "injected host transfer backend restores both stages");
    check(gdn0 == image.live.gdn && hist0 == image.live.ple &&
          gdn1 == image.live.stage_parts[0].gdn && hist1 == image.live.stage_parts[0].ple,
          "each stage's running state landed in its own buffers");
    check(first.data[5] == image.live.tails && first.data[6] == image.live.dead &&
          first.data[7] == image.live.block_pos, "stage 0's indexer payload landed");
    check(second.data[5] == image.live.stage_parts[0].tails &&
          second.data[6] == image.live.stage_parts[0].dead &&
          second.data[7] == image.live.stage_parts[0].block_pos, "stage 1's indexer payload landed");
    check(stage0.ple_prev[0] == 8 && stage0.ple_prev[1] == 9 &&
          stage1.ple_prev[0] == 8 && stage1.ple_prev[1] == 9, "both stages got the PLE window");
    bool after_drain = true, stage1_written = false;
    for (const CopyEvent& e : copy_order) {
        if (e.drains < 1) after_drain = false;
        if (!on_stage1(e.dst)) continue;
        stage1_written = true;
        if (e.device != 1) after_drain = false;   // the copy was made on another card's scope
    }
    check(after_drain && stage1_written,
          "stage 1's buffers were written on its own device after that device was drained");
    reset_counters();
    SavedConversation captured;
    check(conversation_snapshot_save(captured, {image.live.ids, image.live.imgs, image.checkpoints, true},
                                     stages, g, draft.st, error), "split capture through the host backend");
    bool stage1_read = false, read_after_drain = true;
    for (const CopyEvent& e : copy_order) {
        if (e.drains < 1) read_after_drain = false;
        if (!on_stage1(e.src)) continue;
        stage1_read = true;
        if (e.device != 1) read_after_drain = false;
    }
    check(read_after_drain && stage1_read,
          "stage 1's state was read on its own device after that device was drained");
#endif
}
}

int main() {
    for (int format : {0,1,2}) for (int experts : {256,512})
        for (bool zero_qsa : {false,true}) for (bool ple : {false,true}) fixture(format,experts,zero_qsa,ple);
    for (int format : {0,1,2}) split_fixture(format);
    std::printf("conversation_validation_test: %d host-only checks passed\n", checks);
}
