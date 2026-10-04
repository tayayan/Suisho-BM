// Internal header of the hisshi solver (included by solver.cpp only):
// the displayed answer (longest resistance, shortest attack, futile
// interpositions left out), built after the proof has been verified.

#ifndef HISSHI_ANSWER_HPP_
#define HISSHI_ANSWER_HPP_

#include "verify.hpp"

namespace hisshi {
namespace detail {

// Worker threads for the attempts at shorter answers of the answer building
// (is `move` a hisshi? see PvBuilder::TryShorterHere). Each worker has its own
// position and searcher; they share the TT (locked while the pool lives).
// A batch of tasks is run while the calling thread waits.
struct AnswerPool {
  struct Task {
    std::string sfen;  // position before the move
    Move move;
    std::uint64_t budget, threat_budget;
    bool result = false;
    std::uint64_t nodes = 0;
  };

  Solver& solver;
  std::vector<std::thread> threads;
  std::mutex mu;
  std::condition_variable wake, finished_cv;
  std::vector<Task>* batch = nullptr;
  std::atomic<std::size_t> next{0}, finished{0};
  std::uint64_t generation = 0;
  int active = 0;  // workers inside a batch (guarded by mu)
  bool quit = false;
  bool prev_shared;

  AnswerPool(Solver& s, const std::string& root_sfen, Thread* th, const Options& opt, const Limits& lim,
             bool (*should_stop)(), int n, Color atk)
      : solver(s), prev_shared(s.Shared()) {
    solver.SetShared(true);
    for (int t = 0; t < n; ++t) {
      threads.emplace_back([this, root_sfen, th, opt, lim, should_stop, t, atk]() {
        Options wopt = opt;
        wopt.cache_slots = 256;
        wopt.threads = 1;
        Limits wlim = lim;
        wlim.time_ms = 0;
        wlim.nodes = 0;
        wlim.pv_interval_ms = 0;
        Position wp;
        StateInfo root_st, st1, st2;
        wp.set(root_sfen, &root_st, th);
        SearchImpl w(solver, wp, wopt, wlim, should_stop, 100 + t, nullptr, atk);
        std::uint64_t seen = 0;
        for (;;) {
          std::vector<Task>* current;
          {
            std::unique_lock<std::mutex> lk(mu);
            wake.wait(lk, [&] { return quit || generation != seen; });
            if (quit) return;
            seen = generation;
            current = batch;
            if (!current) continue;  // that batch is over already
            ++active;
          }
          std::vector<Task>& tasks = *current;
          for (std::size_t i; (i = next.fetch_add(1)) < tasks.size();) {
            Task& task = tasks[i];
            const std::uint64_t n0 = w.nodes;
            wp.set(task.sfen, &root_st, th);
            const bool check = wp.gives_check(task.move);
            wp.do_move(task.move, st1);
            bool threat = check;  // a quiet attack must be a tsumero (the pass is mated)
            if (!check) {
              wp.do_null_move(st2);
              threat = w.SubSearch(kModeMate, task.threat_budget).pn == 0;
              wp.undo_null_move();
            }
            task.result = threat && w.SubSearch(kModeHisshi, task.budget).pn == 0;
            task.nodes = w.nodes - n0;
            finished.fetch_add(1);
          }
          {
            std::lock_guard<std::mutex> lk(mu);
            --active;
            finished_cv.notify_all();
          }
        }
      });
    }
  }
  ~AnswerPool() {
    {
      std::lock_guard<std::mutex> lk(mu);
      quit = true;
    }
    wake.notify_all();
    for (auto& t : threads) t.join();
    solver.SetShared(prev_shared);
  }
  void Run(std::vector<Task>& tasks) {
    if (tasks.empty()) return;
    std::unique_lock<std::mutex> lk(mu);
    batch = &tasks;
    next = 0;
    finished = 0;
    ++generation;
    wake.notify_all();
    finished_cv.wait(lk, [&] { return finished.load() >= tasks.size() && active == 0; });
    batch = nullptr;
  }
};

// Builds the displayed answer. The proof itself always covers every defence;
// only the choice of the displayed defences follows the problem conventions:
// the defender plays the longest resistance, the answer ends when hisshi is
// reached, and futile interpositions (無駄合い, paper 3.4) are ignored.
//
// Futile interposition (Kakinoki's method extended to hisshi, paper 3.4): as
// in the paper, only drops onto a square covered by the slider the attacker
// touched last are examined. The slider takes the dropped piece; the drop is
// futile when, not counting the drop and the capture, hisshi is reached no
// later than through the other defences. Lengths are plies until hisshi is
// reached (a mated reply counts 0), computed exactly by depth-limited
// searches over proven moves. Unlike Kakinoki's tsume procedure the captured
// piece is not handed back to the defender: with it handed back, the paper's
// Fig. 4 (▽9一飛 ▲9一同飛成) would not be futile. Inside these length
// searches an interposition counts as futile only when the capture brings
// hisshi back at once.
struct PvBuilder {
  Solver& solver;
  const Options& opt;
  Color atk;
  PvBuilder(Solver& s, const Options& o, Color attacker) : solver(s), opt(o), atk(attacker) {}
  static constexpr std::uint64_t kMateProbe = 10000;  // "is this reply simply mated?"
  static constexpr int kMaxDepth = 9;                 // nested judgements (a line has at most 8 squares)


  AnswerPool* pool = nullptr;  // worker threads for the shorter-answer attempts (several search threads)
  AtomicHandSet* verified = nullptr;  // positions checked by the proof verification (reused by VerifyLine)
  std::unordered_map<Key, bool> mated_memo;
  std::unordered_set<Key> not_known_mated;  // neither in the TT nor a one-ply mate

  // Is the attacker-to-move position after reply m mated by checks alone?
  bool ReplyMated(SearchImpl& h, Move m, StateInfo& st) {
    Position& p = h.pos;
    p.do_move(m, st);
    const Key k = p.key();
    bool mated;
    const auto it = mated_memo.find(k);
    if (it != mated_memo.end()) {
      mated = it->second;
    } else {
      Probe pm = h.Lookup(BoardKey(p), p.hand_of(atk), kModeMate);
      if (pm.pn != 0 && !p.in_check() && Mate::mate_1ply(p) != MOVE_NONE) pm.pn = 0;
      if (pm.pn != 0 && pm.dn != 0) {
        h.SubSearch(kModeMate, kMateProbe);
        pm = h.Lookup(BoardKey(p), p.hand_of(atk), kModeMate);
      }
      mated = pm.pn == 0;
      mated_memo[k] = mated;
    }
    p.undo_move(m);
    return mated;
  }

  bool IsInterposition(const Position& p, Move r, Square slider) const {
    if (!is_drop(r) || slider == SQ_NB) return false;
    const Piece pc = p.piece_on(slider);
    if (pc == NO_PIECE || color_of(pc) != atk || !IsSlider(type_of(pc))) return false;
    return static_cast<bool>(effects_from(pc, slider, p.pieces()) & Bitboard(to_sq(r)));
  }

  // Replies of the defender-to-move position, longest resistance first
  // (cached per position).
  std::unordered_map<Key, std::vector<Move>> replies_cache;
  const std::vector<Move>& Replies(SearchImpl& h) {
    const auto it = replies_cache.find(h.pos.key());
    if (it != replies_cache.end()) return it->second;
    return replies_cache[h.pos.key()] = RepliesUncached(h);
  }

  std::vector<Move> RepliesUncached(SearchImpl& h) {
    Position& p = h.pos;
    StateInfo st;
    std::vector<std::pair<int, Move>> replies;
    for (const auto& em : MoveList<LEGAL_ALL>(p)) {
      p.do_move(em.move, st);
      const Probe ph = h.Lookup(BoardKey(p), p.hand_of(atk), kModeHisshi);
      p.undo_move(em.move);
      replies.emplace_back(ph.pn == 0 ? ph.len : 0, em.move);
    }
    std::stable_sort(replies.begin(), replies.end(),
                     [](const auto& x, const auto& y) { return x.first > y.first; });
    std::vector<Move> out;
    for (const auto& rm : replies) out.push_back(rm.second);
    return out;
  }

  // Is the attacker-to-move position after reply m already known as mated
  // (in the TT, or by a one-ply mate)?
  bool KnownMated(SearchImpl& h, Move m, StateInfo& st) {
    Position& p = h.pos;
    p.do_move(m, st);
    const auto it = mated_memo.find(p.key());
    bool mated;
    if (it != mated_memo.end()) {
      mated = it->second;
    } else if (not_known_mated.count(p.key())) {
      mated = false;
    } else {
      mated = h.Lookup(BoardKey(p), p.hand_of(atk), kModeMate).pn == 0 ||
              (!p.in_check() && Mate::mate_1ply(p) != MOVE_NONE);
      if (mated) mated_memo[p.key()] = true;
      else not_known_mated.insert(p.key());
    }
    p.undo_move(m);
    return mated;
  }

  // The first real defence at the defender-to-move position, or MOVE_NONE
  // when hisshi has been reached, with interpositions counted as futile only
  // when their capture brings hisshi back at once (used inside the length
  // searches). Cheap refutations first: only existence matters.
  Move FirstRealDefence(SearchImpl& h, Square slider, int depth) {
    StateInfo st;
    std::vector<Move> open, inter;
    for (const Move m : Replies(h))
      if (!KnownMated(h, m, st)) (IsInterposition(h.pos, m, slider) ? inter : open).push_back(m);
    for (const Move m : open)
      if (!ReplyMated(h, m, st)) return m;
    for (const Move m : inter) {
      if (ReplyMated(h, m, st)) continue;
      if (depth < kMaxDepth && FutileAtOnce(h, m, slider, depth)) continue;
      return m;
    }
    return MOVE_NONE;
  }

  std::unordered_map<Key, bool> at_once_memo;

  bool FutileAtOnce(SearchImpl& h, Move r, Square slider, int depth) {
    Position& p = h.pos;
    StateInfo s1, s2;
    bool futile = false;
    p.do_move(r, s1);
    for (const Move cap : Captures(p, slider, to_sq(r))) {
      p.do_move(cap, s2);
      // The slider now stands on the interposition square.
      const Key k = Mix(p.key(), to_sq(r));
      const auto it = at_once_memo.find(k);
      if (it != at_once_memo.end()) {
        futile = it->second;
      } else {
        futile = FirstRealDefence(h, to_sq(r), depth + 1) == MOVE_NONE;
        at_once_memo[k] = futile;
      }
      p.undo_move(cap);
      if (futile) break;
    }
    p.undo_move(r);
    return futile;
  }

  static Key Mix(Key k, std::uint64_t a) { return k ^ ((a + 1) * 0x9E3779B97F4A7C15ULL); }

  // The slider's legal captures on sq (with and without promotion).
  std::vector<Move> Captures(const Position& p, Square slider, Square sq) const {
    std::vector<Move> out;
    for (const bool promote : {true, false}) {
      const Move cap = p.to_move(promote ? make_move_promote16(slider, sq) : make_move16(slider, sq));
      if (cap != MOVE_NONE && p.pseudo_legal_s<true>(cap) && p.legal(cap)) out.push_back(cap);
    }
    return out;
  }

  // ---- exact lengths (plies until hisshi is reached) ------------------------
  static constexpr int kMaxLength = 12;               // futility compares lengths up to this
  static constexpr int kMaxAnswer = 255;              // longest answer searched for
  static constexpr std::uint64_t kBudget = 3000000;   // length-search visits per answer
  static constexpr std::uint64_t kReprove = 5000000;       // re-proof of an evicted position
  static constexpr std::uint64_t kReproveFutile = 200000;  // proof of a position of a futility test
  int futility_depth = 0;  // inside a futility test (positions outside the proof)
  std::uint64_t ReproveBudget() const { return futility_depth ? kReproveFutile : kReprove; }
  std::uint64_t visits = 0;
  bool exact = false;  // the answer came from the exact length search
  // Known bounds of a length: "not within lo" and "within hi".
  struct Bounds {
    int lo = -1;
    int hi = 1 << 30;
  };
  std::unordered_map<Key, Bounds> bounds;

  // Attacker moves with a proven result (shortest proof first) at the current
  // position. When the position's proof has been evicted from the TT (the
  // table can be full on large problems), it is proven again first.
  std::unordered_map<Key, std::vector<std::pair<int, Move>>> proven_cache;
  const std::vector<std::pair<int, Move>>& ProvenMoves(SearchImpl& h) {
    const auto it = proven_cache.find(h.pos.key());
    if (it != proven_cache.end()) return it->second;
    return proven_cache[h.pos.key()] = ProvenMovesUncached(h);
  }

  std::vector<std::pair<int, Move>> ProvenMovesUncached(SearchImpl& h) {
    Position& p = h.pos;
    std::vector<std::pair<int, Move>> moves;
    for (int attempt = 0; attempt < 2 && moves.empty(); ++attempt) {
      if (attempt == 1) {
        const Probe self = h.Lookup(BoardKey(p), p.hand_of(atk), kModeHisshi);
        if (self.dn == 0) break;
        if (self.pn != 0) {
          // Usually a plain mate: try the cheaper mate search first.
          if (h.SubSearch(kModeMate, kMateProbe).pn != 0) h.SubSearch(kModeHisshi, ReproveBudget());
        } else {
          // Proven, but the children's entries were evicted: prove again the
          // child reached by the stored best move.
          const Move b = p.to_move(Move16(self.best));
          if (b == MOVE_NONE || !p.pseudo_legal_s<true>(b) || !p.legal(b)) break;
          StateInfo st;
          p.do_move(b, st);
          h.SubSearch(kModeHisshi, ReproveBudget());
          p.undo_move(b);
        }
      }
      if (!p.in_check()) {
        const Move mate = Mate::mate_1ply(p);
        if (mate != MOVE_NONE) moves.emplace_back(0, mate);
      }
      for (const auto& em : MoveList<LEGAL_ALL>(p)) {
        Child c{};
        c.move = em.move;
        const Probe pr = h.ProbeChild(c, kModeHisshi);
        if (pr.pn == 0) moves.emplace_back(pr.len, em.move);
      }
    }
    std::stable_sort(moves.begin(), moves.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
    return moves;
  }

  // Attacker to move: can hisshi be reached within k plies? Only moves proven
  // in the TT (and a one-ply mate) are tried: a winning move is proven there.
  bool OrWithin(SearchImpl& h, Square slider, int k) {
    if (k <= 0) return false;
    Position& p = h.pos;
    Bounds& bd = bounds[Mix(Mix(p.key(), slider), 1)];
    if (k >= bd.hi) return true;
    if (k <= bd.lo || visits >= kBudget) return false;
    ++visits;
    const std::vector<std::pair<int, Move>> moves = ProvenMoves(h);
    bool ok = false;
    StateInfo st;
    for (const auto& lm : moves) {
      const Move m = lm.second;
      const Square next = IsSlider(type_of(p.moved_piece_after(m))) ? to_sq(m) : slider;
      p.do_move(m, st);
      ok = AndWithin(h, next, k - 1);
      p.undo_move(m);
      if (ok) break;
    }
    Bounds& b2 = bounds[Mix(Mix(p.key(), slider), 1)];
    if (ok) b2.hi = std::min(b2.hi, k);
    else if (visits < kBudget) b2.lo = std::max(b2.lo, k);
    return ok;
  }

  // Defender to move: is hisshi reached within k plies against every real
  // defence? A mated reply needs nothing more; a futile interposition is not
  // a defence.
  bool AndWithin(SearchImpl& h, Square slider, int k) {
    if (k < 0) return false;
    Position& p = h.pos;
    Bounds& bd = bounds[Mix(Mix(p.key(), slider), 2)];
    if (k >= bd.hi) return true;
    if (k <= bd.lo || visits >= kBudget) return false;
    ++visits;
    bool ok = true;
    StateInfo st;
    for (const Move m : Replies(h)) {
      if (KnownMated(h, m, st)) continue;
      bool answered = false;
      if (k > 0) {
        p.do_move(m, st);
        answered = OrWithin(h, slider, k - 1);
        p.undo_move(m);
      }
      if (answered) continue;
      if (ReplyMated(h, m, st)) continue;
      if (IsInterposition(p, m, slider) && FutileAtOnce(h, m, slider, 1)) continue;
      ok = false;
      break;
    }
    Bounds& b2 = bounds[Mix(Mix(p.key(), slider), 2)];
    if (ok) b2.hi = std::min(b2.hi, k);
    else if (visits < kBudget) b2.lo = std::max(b2.lo, k);
    return ok;
  }

  // Is interposition r at the current (defender-to-move) node futile?
  bool Futile(SearchImpl& h, Move r, Square slider) {
    ++futility_depth;
    const bool f = FutileImpl(h, r, slider);
    --futility_depth;
    return f;
  }

  bool FutileImpl(SearchImpl& h, Move r, Square slider) {
    Position& p = h.pos;
    StateInfo s1, s2, s3;
    // Length after the capture, not counting the drop and the capture.
    int after = -1;
    p.do_move(r, s1);
    for (const Move cap : Captures(p, slider, to_sq(r))) {
      p.do_move(cap, s2);
      for (int k = 0; k <= kMaxLength && after < 0; k += 2)
        if (AndWithin(h, to_sq(r), k)) after = k;
      p.undo_move(cap);
      if (after >= 0) break;
    }
    p.undo_move(r);
    if (after < 0) return false;  // not shown to win quickly after the capture
    if (after == 0) return true;  // hisshi at once
    // Futile when another (non-interposition) real defence lasts at least as
    // long: it needs at least `after - 1` more plies from its position.
    for (const Move o : Replies(h)) {
      if (IsInterposition(p, o, slider) || ReplyMated(h, o, s3)) continue;
      p.do_move(o, s3);
      const bool shorter = OrWithin(h, slider, after - 2);
      p.undo_move(o);
      if (!shorter) return true;
    }
    return false;
  }

  // The answer: the attacker reaches hisshi as soon as possible, the defender
  // resists as long as possible, futile interpositions excluded (all among
  // the proven moves). Falls back to a greedy line when the length search
  // does not finish.
  // The search proves one winning line and leaves other attacks unproven,
  // so the shortest answer among proven moves can be longer than the true
  // shortest (e.g. a check proven first, a one-move hisshi never tried).
  // After building the answer, the unproven attacks along it are tried with
  // a small budget, from the end of the line back to the root (near hisshi a
  // shorter answer is both likelier and cheaper to prove); when one is proven
  // the exact length search is repeated, while the budget lasts.
  static constexpr int kShortenRounds = 16;
  static constexpr std::uint64_t kShortenBudget = 1500000;  // nodes for all attempts of one answer
  static constexpr std::uint64_t kShortenChild = 10000;     // one attack
  static constexpr std::uint64_t kShortenThreat = 2000;     // "is it a tsumero?" pre-check
  std::uint64_t shorten_nodes = 0;
  std::unordered_set<Key> shorten_tried;  // (position, attack) already tried in an earlier round

  std::vector<Move> Pv(SearchImpl& h) {
    std::vector<Move> pv, first;
    int total = -1;
    bool first_exact = false;
    for (int round = 0;; ++round) {
      pv = PvOnce(h, total);
      if (round == 0) { first = pv; first_exact = exact; }
      if (!exact || total <= 1 || round == kShortenRounds || shorten_nodes >= kShortenBudget) break;
      if (!TryShorterAlong(h, pv, total)) break;
      // New proofs can only shorten lengths: "within k" stays true, "not
      // within k" may not. Mate probes are facts (a failed probe stays a
      // conservative "not known").
      for (auto& [key, b] : bounds) b.lo = -1;
      proven_cache.clear();
      visits = 0;
    }
    if (pv != first && !VerifyLine(h, pv)) {
      // A proof found here failed the independent check: keep the answer
      // built from the verified proof.
      exact = first_exact;
      return first;
    }
    return pv;
  }

  // Independent check of the answer's attacks: the position after each
  // attacker move of `pv` is a verified hisshi (the root proof itself was
  // verified before the answer was built).
  // One verifier for the whole line, the position nearest to hisshi first
  // (the earlier positions' proofs contain the later ones), which also reuses
  // the positions checked by the proof verification.
  bool VerifyLine(SearchImpl& h, const std::vector<Move>& pv) {
    if (pv.empty()) return true;
    Position& p = h.pos;
    std::vector<StateInfo> st(pv.size() + 1);
    Verifier v(h);
    v.own_log2 = 24;
    v.shared = verified;  // nullptr: own set
    // Attacker moves are pv[0], pv[2], ...: positions after an odd number of moves.
    for (std::size_t end = pv.size() % 2 ? pv.size() : pv.size() - 1;; end -= 2) {
      for (std::size_t i = 0; i < end; ++i) p.do_move(pv[i], st[i]);
      v.on_path.clear();
      v.raw_path.clear();
      v.error.clear();
      h.PushPath(p.key());
      const bool ok = v.Verify(0, kModeHisshi);
      h.PopPath();
      for (std::size_t i = end; i > 0; --i) p.undo_move(pv[i - 1]);
      if (!ok) return false;
      if (end < 2) break;
    }
    return true;
  }

  // Tries to prove the unproven attacks at the attacker nodes of `pv`
  // (where a shorter answer is possible), the node nearest to hisshi first.
  // True when one was proven.
  bool TryShorterAlong(SearchImpl& h, const std::vector<Move>& pv, int total) {
    Position& p = h.pos;
    std::vector<StateInfo> st(pv.size() + 1);
    // Attacker nodes with at least 3 plies left (a shorter answer needs 2 fewer).
    std::vector<std::size_t> nodes;
    for (std::size_t ply = 0; ply < pv.size(); ply += 2)
      if (total - static_cast<int>(ply) >= 3) nodes.push_back(ply);
    for (auto it = nodes.rbegin(); it != nodes.rend() && shorten_nodes < kShortenBudget; ++it) {
      const std::size_t ply = *it;
      for (std::size_t i = 0; i < ply; ++i) p.do_move(pv[i], st[i]);
      const bool found = TryShorterHere(h);
      for (std::size_t i = ply; i > 0; --i) p.undo_move(pv[i - 1]);
      if (found) return true;  // rebuild from the improved answer first
    }
    return false;
  }

  bool TryShorterHere(SearchImpl& h) {
    Position& p = h.pos;
    if (pool) {
      std::vector<AnswerPool::Task> tasks;
      const std::string sfen = p.sfen();
      for (const auto& em : MoveList<LEGAL_ALL>(p)) {
        const Move m = em.move;
        if (SearchImpl::UselessNonPromotion(p, m)) continue;
        Child c{};
        c.move = m;
        if (h.ProbeChild(c, kModeHisshi).pn == 0) continue;  // already proven
        if (!shorten_tried.insert(Mix(p.key(), m)).second) continue;
        tasks.push_back(AnswerPool::Task{sfen, m, kShortenChild, kShortenThreat});
      }
      pool->Run(tasks);
      bool any = false;
      for (const auto& t : tasks) {
        shorten_nodes += t.nodes;
        if (t.result) any = true;
      }
      return any;
    }
    bool found = false;
    StateInfo s1, s2;
    for (const auto& em : MoveList<LEGAL_ALL>(p)) {
      if (shorten_nodes >= kShortenBudget) break;
      const Move m = em.move;
      if (SearchImpl::UselessNonPromotion(p, m)) continue;
      Child c{};
      c.move = m;
      if (h.ProbeChild(c, kModeHisshi).pn == 0) continue;  // already proven
      if (!shorten_tried.insert(Mix(p.key(), m)).second) continue;
      const bool check = p.gives_check(m);
      p.do_move(m, s1);
      const std::uint64_t n0 = h.nodes;
      bool threat = check;
      if (!check) {
        // A quiet attack must be a tsumero: the pass is mated.
        p.do_null_move(s2);
        threat = h.SubSearch(kModeMate, kShortenThreat).pn == 0;
        p.undo_null_move();
      }
      if (threat && h.SubSearch(kModeHisshi, kShortenChild).pn == 0) {
        found = true;
      }
      shorten_nodes += h.nodes - n0;
      p.undo_move(m);
    }
    return found;
  }

  std::vector<Move> PvOnce(SearchImpl& h, int& total) {
    Position& p = h.pos;
    total = -1;
    exact = false;
    for (int k = 1; k <= kMaxAnswer && visits < kBudget; k += 2)
      if (OrWithin(h, SQ_NB, k)) { total = k; break; }
    if (total < 0) return GreedyPv(h);
    exact = true;

    std::vector<StateInfo> st(kMaxAnswer + 2);
    std::vector<Move> pv;
    Square slider = SQ_NB;
    int k = total;  // plies left until hisshi
    int ply = 0;
    while (k > 0) {
      if (p.side_to_move() == atk) {
        // A move reaching hisshi within k plies (the shortest first).
        Move best = MOVE_NONE;
        for (const auto& lm : ProvenMoves(h)) {
          const Move m = lm.second;
          const Square next = IsSlider(type_of(p.moved_piece_after(m))) ? to_sq(m) : slider;
          p.do_move(m, st[ply]);
          const bool ok = AndWithin(h, next, k - 1);
          p.undo_move(m);
          if (ok) { best = m; slider = next; break; }
        }
        if (best == MOVE_NONE) break;  // should not happen
        pv.push_back(best);
        p.do_move(best, st[ply++]);
        --k;
        continue;
      }
      // Defender: a real defence that lasts exactly k - 1 more plies, i.e.
      // one that is not answered within k - 3 plies (the longest resistance).
      if (AndWithin(h, slider, 0)) break;  // hisshi reached
      Move chosen = MOVE_NONE;
      for (const Move m : Replies(h)) {
        if (KnownMated(h, m, st[ply])) continue;
        p.do_move(m, st[ply]);
        const bool shorter = OrWithin(h, slider, k - 3);
        p.undo_move(m);
        if (shorter) continue;
        if (ReplyMated(h, m, st[ply])) continue;
        if (IsInterposition(p, m, slider) && FutileAtOnce(h, m, slider, 1)) continue;
        if (IsInterposition(p, m, slider) && Futile(h, m, slider)) continue;
        chosen = m;
        break;
      }
      if (chosen == MOVE_NONE) {
        // The replies that last k - 1 plies were all left out as futile
        // interpositions (the length search counts them): the longest of the
        // remaining real defences, with its own length.
        int best_len = -1;
        for (const Move m : Replies(h)) {
          if (KnownMated(h, m, st[ply]) || ReplyMated(h, m, st[ply])) continue;
          if (IsInterposition(p, m, slider) && FutileAtOnce(h, m, slider, 1)) continue;
          if (IsInterposition(p, m, slider) && Futile(h, m, slider)) continue;
          p.do_move(m, st[ply]);
          int len = -1;
          for (int j = 1; j <= k - 3; j += 2)
            if (OrWithin(h, slider, j)) { len = j; break; }
          p.undo_move(m);
          if (len > best_len) { best_len = len; chosen = m; }
        }
        if (chosen == MOVE_NONE || best_len < 0) break;
        k = best_len + 1;
      }
      pv.push_back(chosen);
      p.do_move(chosen, st[ply++]);
      --k;
    }
    while (ply > 0) p.undo_move(pv[--ply]);
    return pv;
  }

  // Greedy answer (fallback): shortest proof for the attacker, longest proof
  // for the defender, futile interpositions excluded.
  std::vector<Move> GreedyPv(SearchImpl& h) {
    Position& p = h.pos;
    std::vector<StateInfo> st(1024);
    std::vector<Move> pv;
    std::unordered_set<Key> seen;
    Square slider = SQ_NB;  // the slider the attacker touched last
    int ply = 0;
    while (ply < 1000) {
      if (seen.count(p.key())) break;
      seen.insert(p.key());
      if (p.side_to_move() == atk) {
        Move best = MOVE_NONE;
        if (!p.in_check()) best = Mate::mate_1ply(p);
        if (best == MOVE_NONE) {
          int bl = 0x7fffffff;
          for (const auto& em : MoveList<LEGAL_ALL>(p)) {
            Child c{};
            c.move = em.move;
            const Probe pr = h.ProbeChild(c, kModeHisshi);
            if (pr.pn == 0 && pr.len < bl) { bl = pr.len; best = em.move; }
          }
        }
        if (best == MOVE_NONE) break;
        if (IsSlider(type_of(p.moved_piece_after(best)))) slider = to_sq(best);
        pv.push_back(best);
        p.do_move(best, st[ply++]);
        continue;
      }
      // Defender: the longest real defence; none left means hisshi.
      Move chosen = MOVE_NONE;
      for (const Move m : Replies(h)) {
        if (ReplyMated(h, m, st[ply])) continue;
        if (IsInterposition(p, m, slider) && Futile(h, m, slider)) continue;
        chosen = m;
        break;
      }
      if (chosen == MOVE_NONE) break;
      pv.push_back(chosen);
      p.do_move(chosen, st[ply++]);
    }
    while (ply > 0) p.undo_move(pv[--ply]);
    return pv;
  }
};

}  // namespace detail
}  // namespace hisshi

#endif  // HISSHI_ANSWER_HPP_
