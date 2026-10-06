// The hisshi solver: transposition table management and Solver::Solve
// (parallel search, verification, answer).

#include "hisshi_dfpn.hpp"

#include "answer.hpp"

namespace hisshi {

using namespace detail;

// ---------------------------------------------------------------------------
// Transposition table

void Solver::Resize(std::size_t mb) {
  const std::size_t bytes = std::max<std::size_t>(mb, 1) * 1024 * 1024;
  std::size_t clusters = bytes / sizeof(Cluster);
  if (clusters < 1024) clusters = 1024;
  if (clusters == cluster_count_ && !table_.empty()) return;
  table_.clear();
  table_.shrink_to_fit();
  table_.resize(clusters);
  cluster_count_ = clusters;
  Clear();
}

void Solver::Clear() {
  if (!table_.empty()) std::memset(static_cast<void*>(table_.data()), 0, table_.size() * sizeof(Cluster));
  for (std::size_t i = 0; i < kRepTableSize; ++i) rep_table_[i].store(0, std::memory_order_relaxed);
  dirty_ = false;
}

// After a proof is found only proofs matter: drops everything else, so that
// re-proofs during verification and answer building find free slots, and
// from then on proofs are replaced only by proofs (see Store).
void Solver::KeepOnlyProofs() {
  for (Cluster& c : table_)
    for (int j = 0; j < kClusterSize; ++j)
      if (c.key[j] != 0 && c.d[j].pn != 0) c.key[j] = 0;
  proofs_only_ = true;
}

int Solver::Hashfull() const {
  if (table_.empty()) return 0;
  int used = 0;
  const std::size_t sample = std::min<std::size_t>(table_.size(), 1000);
  for (std::size_t i = 0; i < sample; ++i) {
    // The search threads write the table meanwhile: the cluster's lock (the
    // same as SearchImpl::ClusterLock's).
    Lock* l = shared_ ? &locks_[i & (kLocks - 1)] : nullptr;
    if (l)
      while (l->v.exchange(1, std::memory_order_acquire))
        while (l->v.load(std::memory_order_relaxed)) _mm_pause();
    for (int j = 0; j < kClusterSize; ++j) used += table_[i].key[j] != 0;
    if (l) l->v.store(0, std::memory_order_release);
  }
  return static_cast<int>(used * 1000 / (sample * kClusterSize));
}

namespace {

// ---------------------------------------------------------------------------
// Parallel search

// Search parameters of helper thread t: each varies the child order (see
// SearchImpl::Tie) and some parameters, for diversity.
Options HelperOptions(const Options& opt, int t, int cache_slots) {
  static const int kLazyDn[] = {64, 32, 128, 48, 96, 80, 40, 160};
  static const int kStageDn[] = {16, 12, 24, 16, 20, 14, 16, 32};
  static const int kCost[] = {2, 3, 2, 2, 3, 2};
  static const int kCheckDn[] = {16, 8, 32, 16, 64, 12, 24};
  static const int kEps[] = {0, 50, 100, 25, 200};
  static const int kDeep[] = {16, 8, 24, 12, 32, 0, 20, 10, 16};
  Options h = opt;
  h.lazy_dn = kLazyDn[t % 8];
  h.and_stage_dn = kStageDn[t % 8];
  h.non_check_cost = kCost[t % 6];
  h.check_stage_dn = kCheckDn[t % 7];
  h.eps_percent = kEps[t % 5];
  if (opt.deep_pn > 0) h.deep_pn = kDeep[t % 9];
  h.cache_slots = cache_slots;
  return h;
}

// Children cache slots of the main thread and of each helper: together
// about a quarter of the hash size (a quarter of it for the main thread, the
// rest shared by the helpers), at most the configured numbers.
void CacheSlots(std::size_t hash_bytes, const Options& opt, int nthreads, int& main_slots, int& helper_slots) {
  const std::size_t cache_bytes = hash_bytes / 4;
  auto slots = [](std::size_t bytes, int cap) {
    return static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(cap),
                                                  std::max<std::size_t>(64, bytes / sizeof(SearchImpl::CacheSlot))));
  };
  main_slots = slots(nthreads > 1 ? cache_bytes / 4 : cache_bytes, opt.cache_slots);
  helper_slots = nthreads > 1 ? slots(cache_bytes * 3 / 4 / static_cast<std::size_t>(nthreads - 1),
                                      std::min(opt.cache_slots, opt.helper_cache_slots))
                              : 0;
}

// What the verification and the answer building need besides the main
// search: the root, the options and the threads.
struct SolveContext {
  Solver& solver;
  Position& root;
  const std::string root_sfen;
  const Options& opt;
  const Limits& limits;
  bool (*should_stop)();
  int nthreads;
};

// ---------------------------------------------------------------------------
// Verification

// Not hisshi: accepted only after the independent check with every legal
// attack (a disproof resting on a repetition or the depth limit of the path
// is not a result).
void VerifyDisproof(SearchImpl& s, Position& root, Result& res) {
  const std::uint64_t before = s.nodes;
  const auto t0 = s.ElapsedMs();
  DisproofVerifier v(s);
  bool cyc = false;
  s.PushPath(root.key());
  const bool ok = v.Refuted(0, kModeHisshi, cyc);
  s.PopPath();
  res.verified = ok;
  std::ostringstream os;
  os << "visited=" << v.visited << " searches=" << v.searches << " research_nodes=" << (s.nodes - before)
     << " time_ms=" << (s.ElapsedMs() - t0);
  if (!ok) os << " error=" << v.error;
  res.verify_info = os.str();
  if (!ok) res.status = Status::kUnknown;
}

// Splits the proof into subtrees (verified on the way down to their roots),
// deepening until there are enough tasks for the threads. False when the
// proof cannot be split (the sequential verification then decides).
bool CollectVerifyTasks(SearchImpl& s, Position& root, int nthreads, std::vector<VerifyTask>& tasks,
                        std::uint64_t& visited) {
  for (int d = 2; d <= 20; d += 2) {
    tasks.clear();
    Verifier c(s);
    c.own_log2 = 18;  // only the top of the proof is walked
    c.collect_depth = d;
    c.tasks = &tasks;
    s.PushPath(root.key());
    const bool ok = c.Verify(0, kModeHisshi);
    s.PopPath();
    if (!ok) return false;
    if (tasks.size() >= 1024u * static_cast<std::size_t>(nthreads) || d >= 20 || tasks.empty()) {
      visited += c.visited;
      return true;
    }
  }
  return false;
}

// Verifies the tasks on all threads, recording the verified positions in
// `done`. True when every task passed.
bool VerifyTasksInParallel(const SolveContext& ctx, Color attacker, const std::vector<VerifyTask>& tasks,
                           AtomicHandSet& done, std::uint64_t& visited) {
  std::atomic<std::size_t> next{0};
  std::atomic<bool> failed{false};
  std::atomic<std::uint64_t> total_visited{0};
  ctx.solver.SetShared(true);
  std::vector<std::thread> workers;
  for (int t = 0; t < ctx.nthreads; ++t) {
    workers.emplace_back([&]() {
      Options wopt = ctx.opt;
      wopt.cache_slots = 256;
      wopt.threads = 1;
      Limits wlim = ctx.limits;
      wlim.pv_interval_ms = 0;
      // One position, searcher and verifier per thread, reused for every task.
      Position wp;
      StateInfo root_st;
      wp.set(ctx.root_sfen, &root_st, ctx.root.this_thread());
      SearchImpl w(ctx.solver, wp, wopt, wlim, ctx.should_stop, 0, nullptr, attacker);
      Verifier v(w);
      v.shared = &done;
      std::vector<StateInfo> mst(1024);
      for (std::size_t i; !failed && (i = next.fetch_add(1)) < tasks.size();) {
        const VerifyTask& task = tasks[i];
        if (task.moves.size() >= mst.size()) { failed = true; break; }
        for (std::size_t j = 0; j < task.moves.size(); ++j) {
          if (task.moves[j] == MOVE_NULL) wp.do_null_move(mst[j]);
          else wp.do_move(task.moves[j], mst[j]);
        }
        v.on_path = task.ancestors;
        v.raw_path = task.ancestors_raw;
        v.error.clear();
        w.PushPath(wp.key());
        const bool ok = v.Verify(static_cast<int>(task.moves.size()), task.mode);
        w.PopPath();
        for (std::size_t j = task.moves.size(); j-- > 0;) {
          if (task.moves[j] == MOVE_NULL) wp.undo_null_move();
          else wp.undo_move(task.moves[j]);
        }
        if (!ok) failed = true;
      }
      total_visited += v.visited;
    });
  }
  for (auto& th : workers) th.join();
  ctx.solver.SetShared(false);
  visited += total_visited;
  return !failed;
}

// Independent verification of the proof (all defences, including the
// virtual pass after every non-check): in parallel when the proof is large
// enough, else (or after a failure) sequentially, reusing what the parallel
// verification has checked. Returns the verified positions when the proof
// passed (the answer's line is checked against them).
std::unique_ptr<AtomicHandSet> VerifyProof(const SolveContext& ctx, SearchImpl& s, Result& res) {
  const std::uint64_t before = s.nodes;
  const auto t0 = s.ElapsedMs();
  std::uint64_t visited = 0;
  std::string error;
  bool parallel_ok = false;
  std::unique_ptr<AtomicHandSet> done;
  // Verified-position memo: about an eighth of the hash size (8-byte slots).
  int memo_log2 = 18;
  while (memo_log2 < 25 && (std::size_t(8) << (memo_log2 + 1)) <= ctx.solver.TableBytes() / 8) ++memo_log2;
  if (ctx.nthreads > 1) {
    std::vector<VerifyTask> tasks;
    if (CollectVerifyTasks(s, ctx.root, ctx.nthreads, tasks, visited)) {
      if (tasks.empty()) {
        parallel_ok = true;  // the whole proof was checked while collecting
      } else if (tasks.size() >= 16u * static_cast<std::size_t>(ctx.nthreads)) {  // else not worth the threads
        done = std::make_unique<AtomicHandSet>(memo_log2);
        parallel_ok = VerifyTasksInParallel(ctx, s.atk, tasks, *done, visited);
      }
    }
  }
  bool ok = parallel_ok;
  if (!parallel_ok) {
    Verifier v(s);
    if (!done) done = std::make_unique<AtomicHandSet>(std::min(v.own_log2, memo_log2));
    v.shared = done.get();
    s.PushPath(ctx.root.key());
    ok = v.Verify(0, kModeHisshi);
    s.PopPath();
    visited = v.visited;
    error = v.error;
  }
  res.verified = ok;
  std::ostringstream os;
  os << "visited=" << visited << " research_nodes=" << (s.nodes - before) << " time_ms=" << (s.ElapsedMs() - t0)
     << (parallel_ok ? " parallel" : "");
  if (!ok) os << " error=" << error;
  res.verify_info = os.str();
  if (!ok) done.reset();
  return done;
}

// ---------------------------------------------------------------------------
// Answer

// The displayed answer (futile interpositions left out), built after the
// verification; the shorter-answer attempts use all threads.
void BuildAnswer(const SolveContext& ctx, SearchImpl& s, AtomicHandSet* verified, Result& res) {
  const auto t0 = s.ElapsedMs();
  s.FlushNodes();
  const std::uint64_t n0 = ctx.solver.TotalNodes();
  PvBuilder builder(ctx.solver, ctx.opt, s.atk);
  builder.verified = verified;
  std::unique_ptr<AnswerPool> pool;
  if (ctx.nthreads > 1) {
    pool = std::make_unique<AnswerPool>(ctx.solver, ctx.root_sfen, ctx.root.this_thread(), ctx.opt, ctx.limits,
                                        ctx.should_stop, ctx.nthreads, s.atk);
    builder.pool = pool.get();
  }
  res.pv = builder.Pv(s);
  pool.reset();  // (its searchers add their nodes)
  s.FlushNodes();
  res.verify_info += std::string(" answer=") + (builder.exact ? "longest" : "greedy") +
                     " answer_ms=" + std::to_string(s.ElapsedMs() - t0) +
                     " answer_nodes=" + std::to_string(ctx.solver.TotalNodes() - n0);
}

}  // namespace

// ---------------------------------------------------------------------------
// Solve

Result Solver::Solve(Position& root, const Limits& given_limits, const Options& opt, bool (*should_stop)()) {
  // Every searcher of this Solve counts the time limit from now.
  Limits limits = given_limits;
  if (limits.origin == std::chrono::steady_clock::time_point{}) limits.origin = std::chrono::steady_clock::now();
  InitTables();
  if (table_.empty()) Resize(256);
  // A large table takes a noticeable time to clear: only after a search.
  if (dirty_) Clear();
  dirty_ = true;
  proofs_only_ = false;
  Result res;
  if (root.king_square(~root.side_to_move()) == SQ_NB) {
    res.status = Status::kDisproven;
    res.pn = kInf;
    res.dn = 0;
    return res;
  }
  const int nthreads = std::max(1, opt.threads);
  const SolveContext ctx{*this, root, root.sfen(), opt, limits, should_stop, nthreads};
  int main_slots = 0, helper_slots = 0;
  CacheSlots(TableBytes(), opt, nthreads, main_slots, helper_slots);

  // Helper threads search the same root with a shared TT, until any thread
  // has decided it.
  std::atomic<bool> done{false};
  std::atomic<bool> helper_rep_disproof{false};  // a helper's root: a disproof resting on a repetition
  std::vector<std::thread> helpers;
  total_nodes_ = 0;
  shared_ = nthreads > 1;
  for (int t = 1; t < nthreads; ++t) {
    helpers.emplace_back([&, t]() {
      Position hp;
      StateInfo hsi;
      hp.set(ctx.root_sfen, &hsi, root.this_thread());
      Limits hlim = limits;
      hlim.pv_interval_ms = 0;
      SearchImpl h(*this, hp, HelperOptions(opt, t, helper_slots), hlim, should_stop, t, &done);
      const NodeResult hr = h.Run(kModeHisshi);
      if (hr.dn == 0 && hr.rep) helper_rep_disproof = true;
      if (hr.pn == 0 || hr.dn == 0) done = true;
    });
  }
  // A single search has no helpers that vary the parameters: it uses the
  // 1+epsilon trick against switching between near-equal alternatives
  // (the helpers' main means of escaping such seesaws).
  Options main_opt = opt;
  if (nthreads == 1 && main_opt.eps_percent == 0) main_opt.eps_percent = opt.single_thread_eps;
  main_opt.cache_slots = main_slots;
  SearchImpl s(*this, root, main_opt, limits, should_stop, 0, nthreads > 1 ? &done : nullptr);
  s.report = true;
  s.root_sfen = ctx.root_sfen;
  NodeResult r = s.Run(kModeHisshi);
  done = true;
  for (auto& th : helpers) th.join();
  shared_ = false;
  if (nthreads > 1) {
    // A helper may have decided the root: read the shared TT.
    s.shared_stop = nullptr;
    const Probe rp = s.Lookup(BoardKey(root), root.hand_of(s.atk), kModeHisshi);
    if (rp.pn == 0 || (rp.dn == 0 && !rp.rep)) {
      r.pn = rp.pn;
      r.dn = rp.dn;
      r.rep = false;
    } else if (r.pn != 0 && r.dn != 0 && helper_rep_disproof) {
      // A helper disproved the root through a repetition (not in the TT for
      // the root's own path lookup): the independent check decides.
      r.pn = kInf;
      r.dn = 0;
      r.rep = true;
    }
  }
  res.pn = r.pn;
  res.dn = r.dn;
  s.FlushNodes();  // (the helpers added theirs when they ended)
  res.nodes = TotalNodes();
  res.elapsed_ms = s.ElapsedMs();
  if (r.pn == 0) {
    res.status = Status::kProven;
  } else if (r.dn == 0) {
    // A disproof resting on a repetition with the root's path: the
    // independent check (its path starts at the root) decides.
    res.status = Status::kDisproven;
    VerifyDisproof(s, root, res);
    if (r.rep) res.verify_info += " rep=1";
  }
  if (res.status != Status::kProven) return res;

  s.report = false;  // verification and answer searches print no root progress
  if (limits.pv_interval_ms > 0)
    sync_cout << "info string proof found (" << s.ElapsedMs() << " ms), verifying" << sync_endl;
  if (Hashfull() > 300) KeepOnlyProofs();  // a full scan: only when the table is in use
  const std::unique_ptr<AtomicHandSet> verified = VerifyProof(ctx, s, res);
  // A verification cut off by a limit (time, nodes, stop) is not a failed one.
  const bool cut_off = !res.verified && s.LimitReached();
  if (cut_off) res.verify_info += " stopped=limit";
  if (limits.pv_interval_ms > 0)
    sync_cout << "info string proof "
              << (res.verified ? "verified" : cut_off ? "verification stopped by the limit" : "NOT verified") << " ("
              << s.ElapsedMs() << " ms)" << (res.verified ? ", building the answer" : "") << sync_endl;
  // An unverified proof is not reported (no answer is shown for it).
  if (res.verified) BuildAnswer(ctx, s, verified.get(), res);

  s.FlushNodes();  // (the verification and answer threads added theirs)
  res.nodes = TotalNodes();
  res.elapsed_ms = s.ElapsedMs();
  return res;
}

}  // namespace hisshi
