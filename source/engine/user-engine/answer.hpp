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
        Limits wlim = lim;  // (the limits of the Solve hold here too)
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
// hisshi back at once, or when it belongs to a chain judged futile (see
// ChainFutile). Futile interpositions are told apart only with
// Options::futile; without it every interposition is a defence.
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

  // Every futility rule starts here: without Options::futile no
  // interposition is told apart from the other defences.
  bool IsInterposition(const Position& p, Move r, Square slider) const {
    if (!opt.futile || !is_drop(r) || slider == SQ_NB) return false;
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
  static constexpr int kMaxLengthChain = 40;          // the same in the chain judgements
  static constexpr int kMaxAnswer = 255;              // longest answer searched for
  static constexpr std::uint64_t kBudget = 3000000;   // length-search visits per answer
  static constexpr std::uint64_t kReprove = 5000000;       // re-proof of an evicted position
  static constexpr std::uint64_t kReproveFutile = 200000;  // proof of a position of a futility test
  int futility_depth = 0;  // inside a futility test (positions outside the proof)
  std::uint64_t ReproveBudget() const { return futility_depth ? kReproveFutile : kReprove; }
  std::uint64_t visits = 0;
  bool exact = false;  // the answer came from the exact length search
  bool limit_reached = false;  // a limit of the Solve: the answer is finished with what is known
  // The length searches' budget, also spent when a limit of the Solve is
  // reached (checked now and then).
  bool OutOfBudget(SearchImpl& h) {
    if ((visits & 1023) == 0 && !limit_reached && h.LimitReached()) limit_reached = true;
    if (limit_reached) visits = kBudget;
    return visits >= kBudget;
  }
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
  std::unordered_set<Key> reprove_tried;  // (position, budget) re-proofs tried inside futility tests
  const std::vector<std::pair<int, Move>>& ProvenMoves(SearchImpl& h) {
    // Inside futility tests evicted proofs are re-proved with a smaller
    // budget (or not at all): those lists are kept apart.
    const Key key = futility_depth ? Mix(h.pos.key(), 29) : h.pos.key();
    const auto it = proven_cache.find(key);
    if (it != proven_cache.end()) return it->second;
    return proven_cache[key] = ProvenMovesUncached(h);
  }

  std::vector<std::pair<int, Move>> ProvenMovesUncached(SearchImpl& h) {
    Position& p = h.pos;
    std::vector<std::pair<int, Move>> moves;
    for (int attempt = 0; attempt < 2 && moves.empty(); ++attempt) {
      if (attempt == 1) {
        const Probe self = h.Lookup(BoardKey(p), p.hand_of(atk), kModeHisshi);
        if (self.dn == 0) break;
        // Inside futility tests (positions outside the proof, many of them) a
        // re-proof is tried once per position for the whole answer.
        if (futility_depth && !reprove_tried.insert(Mix(p.key(), ReproveBudget())).second) break;
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
    Bounds& bd = bounds[Mix(Mix(p.key(), slider), chain_eval ? 11 : 1)];
    if (k >= bd.hi) return true;
    if (k <= bd.lo || OutOfBudget(h)) return false;
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
    Bounds& b2 = bounds[Mix(Mix(p.key(), slider), chain_eval ? 11 : 1)];
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
    Bounds& bd = bounds[Mix(Mix(p.key(), slider), chain_eval ? 12 : 2)];
    if (k >= bd.hi) return true;
    if (k <= bd.lo || OutOfBudget(h)) return false;
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
      if (IsInterposition(p, m, slider) &&
          (FutileAtOnce(h, m, slider, 1) || (!chain_eval && ChainDrop(p, m, slider) && ChainFutile(h, m, slider))))
        continue;
      ok = false;
      break;
    }
    Bounds& b2 = bounds[Mix(Mix(p.key(), slider), chain_eval ? 12 : 2)];
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

  // ---- chains of interpositions ---------------------------------------------
  // A drop of a chain: between the slider and the defender's king, reached by
  // the slider and capturable by it.
  bool ChainDrop(Position& p, Move r, Square slider) {
    if (!IsInterposition(p, r, slider)) return false;
    const Square ksq = p.king_square(~atk);
    if (ksq == SQ_NB) return false;
    if (!(between_bb(slider, ksq) & Bitboard(to_sq(r)))) return false;
    StateInfo st;
    p.do_move(r, st);
    const bool capturable = !Captures(p, slider, to_sq(r)).empty();
    p.undo_move(r);
    return capturable;
  }

  // Defender to move: is hisshi reached within k plies against every reply
  // except the drops of the chain on the slider's line (the real defences)?
  bool RealWithin(SearchImpl& h, Square slider, int k) {
    Position& p = h.pos;
    StateInfo st;
    for (const Move m : Replies(h)) {
      if (ChainDrop(p, m, slider)) continue;
      if (KnownMated(h, m, st)) continue;
      // In chain judgements most replies of the (new) positions are simply
      // mated: the mate probe first, before a length search that may have to
      // re-prove positions.
      if (ReplyMated(h, m, st)) continue;
      if (k > 0) {
        p.do_move(m, st);
        const bool ok = OrWithin(h, slider, k - 1);
        p.undo_move(m);
        if (ok) continue;
      }
      if (IsInterposition(p, m, slider) && FutileAtOnce(h, m, slider, 1)) continue;
      return false;
    }
    return true;
  }

  // An interposition of a chain is futile when, after its capture, hisshi is
  // reached no later than through the real defences of the position without
  // it: the drop and the capture only add their two plies. The chain is
  // judged from its end: a later interposition judged futile does not count
  // (its two plies are taken off), a later one that is not futile counts
  // with its length. The lengths compared are made shortest first by
  // bounded-length searches (a search of its own, in a table of its own); a
  // proof found is copied into the main table and the answer is rebuilt with
  // it in the next round.
  static constexpr std::size_t kShortTableMb = 128;
  static constexpr std::uint64_t kShortCall = 3000000;    // nodes of one bounded search (proofs need ~1M)
  static constexpr std::uint64_t kShortBudget = 40000000;  // nodes of all bounded searches of one answer
  // Nodes of the bounded searches of one chain judgement (with the later
  // interpositions it judges): each judgement gets its share, so that the
  // first ones do not leave nothing to the others.
  static constexpr std::uint64_t kJudgeBudget = 6000000;
  int chain_depth = 0;           // nested chain judgements
  std::uint64_t judge_left = 0;  // bounded-search nodes left to the current judgement
  std::uint64_t last_bs_nodes = 0;
  std::unique_ptr<Solver> short_solver;
  std::unique_ptr<SearchImpl> short_search;
  std::uint64_t short_nodes = 0;
  int chain_eval = 0;          // inside a chain judgement (no other chain judged from its lengths)
  bool chain_restart = false;  // a shorter proof was copied: rebuild the answer
  std::unordered_map<Key, bool> chain_memo;     // per round
  std::unordered_map<Key, int> real_memo;       // per round
  std::unordered_map<Key, int> shortest_memo;   // kept across rounds
  std::unordered_map<Key, bool> bound_memo;     // kept across rounds

  SearchImpl& ShortSearcher(SearchImpl& h) {
    if (!short_search) {
      short_solver = std::make_unique<Solver>();
      short_solver->Resize(kShortTableMb);
      Options so = opt;
      so.cache_slots = 256;
      so.threads = 1;
      so.eps_percent = std::max(so.eps_percent, 50);  // without it the bounded searches rarely decide
      Limits sl = h.lim;  // the time limit of the Solve (its own table: no node total)
      sl.nodes = 0;
      sl.pv_interval_ms = 0;
      short_search = std::make_unique<SearchImpl>(*short_solver, h.pos, so, sl, nullptr, 0, nullptr, atk);
    }
    return *short_search;
  }

  // Attacker to move: is hisshi reached within d plies? One bounded search
  // (a proof found is copied into the main table).
  bool BoundedSearch(SearchImpl& h, int d, std::uint64_t cap = kShortCall) {
    last_bs_nodes = 0;
    std::uint64_t budget = std::min(cap, kShortBudget - std::min(short_nodes, kShortBudget));
    if (chain_depth) budget = std::min(budget, judge_left);
    if (d < 1 || budget == 0) return false;
    SearchImpl& s2 = ShortSearcher(h);
    short_solver->Clear();
    const std::uint64_t n0 = s2.nodes;
    const bool ok = s2.ShortSearch(d, budget);
    last_bs_nodes = s2.nodes - n0;
    short_nodes += last_bs_nodes;
    if (chain_depth) judge_left -= std::min(judge_left, last_bs_nodes);
    if (!ok) return false;
    CopyProof(s2, h, d);
    chain_restart = true;
    return true;
  }

  // Copies the proof of the current position from the bounded search's table
  // into the main table: the best move at attacker nodes, every reply at
  // defender nodes, `depth` plies deep (with the whole attacker's hand).
  void CopyProof(SearchImpl& from, SearchImpl& to, int depth) {
    Position& p = to.pos;
    const Key board = BoardKey(p);
    const Hand hand = p.hand_of(atk);
    const Probe pr = from.Lookup(board, hand, kModeHisshi);
    if (pr.pn != 0) return;
    const bool or_node = p.side_to_move() == atk;
    const Move best = or_node ? p.to_move(Move16(pr.best)) : MOVE_NONE;
    to.Store(board, hand, kModeHisshi, 0, kInf, pr.len, best, false, 1000);
    if (depth <= 0) return;
    StateInfo st;
    if (or_node) {
      if (best == MOVE_NONE || !p.pseudo_legal_s<true>(best) || !p.legal(best)) return;
      p.do_move(best, st);
      CopyProof(from, to, depth - 1);
      p.undo_move(best);
      return;
    }
    for (const auto& em : MoveList<LEGAL_ALL>(p)) {
      p.do_move(em.move, st);
      CopyProof(from, to, depth - 1);
      p.undo_move(em.move);
    }
  }

  // Attacker to move, hisshi known within `upper` plies: bounded searches at
  // upper - 2, upper - 4, ... (the longer limits first) until one fails.
  void Shortest(SearchImpl& h, int upper) {
    const Key key = Mix(h.pos.key(), 13);
    if (shortest_memo.count(key)) return;
    int best = upper;
    // A shorter limit rarely needs much more than the last success: a
    // failing step (mostly running out of its budget) costs less so.
    std::uint64_t cap = kShortCall;
    for (int d = upper - 2; d >= 1 && BoundedSearch(h, d, cap); d -= 2) {
      best = d;
      cap = std::min(kShortCall, std::max<std::uint64_t>(4 * last_bs_nodes, 300000));
    }
    shortest_memo[key] = best;
  }

  // Attacker to move: hisshi within d plies by a bounded search, tried once
  // per position and d.
  bool WithinByBoundedSearch(SearchImpl& h, int d) {
    const Key key = Mix(Mix(h.pos.key(), static_cast<std::uint64_t>(d)), 23);
    const auto it = bound_memo.find(key);
    if (it != bound_memo.end()) return it->second;
    return bound_memo[key] = BoundedSearch(h, d);
  }

  // Defender to move: the length of the real defences, -1 when not within
  // kMaxLengthChain. The longest real defence is made shortest first.
  int RealLen(SearchImpl& h, Square slider) {
    const Key key = Mix(Mix(h.pos.key(), slider), 17);
    const auto it = real_memo.find(key);
    if (it != real_memo.end()) return it->second;
    Position& p = h.pos;
    int len = -1;
    for (int k = 0; k <= kMaxLengthChain && visits < kBudget; k += 2)
      if (RealWithin(h, slider, k)) { len = k; break; }
    if (len > 0) {
      StateInfo st;
      for (const Move m : Replies(h)) {
        if (ChainDrop(p, m, slider) || KnownMated(h, m, st)) continue;
        p.do_move(m, st);
        const bool longest = !OrWithin(h, slider, len - 3);
        if (longest) Shortest(h, len - 1);
        p.undo_move(m);
        if (longest) break;
      }
    }
    return real_memo[key] = len;
  }

  // Interposition r of a chain at the current (defender-to-move) node: futile
  // when, after its capture, hisshi is reached within `real` plies, the
  // length of the real defences without r.
  bool ChainFutile(SearchImpl& h, Move r, Square slider) {
    Position& p = h.pos;
    // Shared by every piece dropped on the square and every distribution of
    // the hands (an approximation: per piece and hand the positions of a
    // chain grow like a tree).
    const Key key = Mix(Mix(BoardKey(p) ^ (p.side_to_move() == BLACK ? 0 : 0x5bd1e995ULL), to_sq(r)), slider);
    const auto it = chain_memo.find(key);
    if (it != chain_memo.end()) return it->second;
    if (chain_depth++ == 0) judge_left = kJudgeBudget;
    ++chain_eval;
    ++futility_depth;  // positions outside the proof: the smaller re-proof budget
    // Length-search visits of its own (the answer's length search keeps its own).
    const std::uint64_t saved_visits = visits;
    visits = 0;
    bool futile = false;
    const int real = RealLen(h, slider);
    if (real >= 0) {
      StateInfo s1, s2, s3;
      p.do_move(r, s1);
      for (const Move cap : Captures(p, slider, to_sq(r))) {
        p.do_move(cap, s2);
        const Square next = to_sq(r);
        bool within = RealWithin(h, next, real);
        if (!within) {
          // The longest real defence after the capture may have a shorter
          // answer never proven: one bounded search at the needed length.
          StateInfo s4;
          for (const Move m : Replies(h)) {
            if (ChainDrop(p, m, next) || KnownMated(h, m, s4)) continue;
            p.do_move(m, s4);
            const bool longer = real < 1 || !OrWithin(h, next, real - 1);
            if (longer) WithinByBoundedSearch(h, real - 1);
            p.undo_move(m);
            if (longer) break;
          }
        }
        // The chain goes on: a later interposition counts unless futile.
        for (const Move r2 : Replies(h)) {
          if (!within) break;
          if (!ChainDrop(p, r2, next)) continue;
          --chain_eval;
          const bool f2 = ChainFutile(h, r2, next);
          ++chain_eval;
          if (f2) continue;
          p.do_move(r2, s3);
          within = real >= 1 && OrWithin(h, next, real - 1);
          p.undo_move(r2);
        }
        p.undo_move(cap);
        if (within) { futile = true; break; }
      }
      p.undo_move(r);
    }
    visits = saved_visits;
    --futility_depth;
    --chain_eval;
    --chain_depth;
    return chain_memo[key] = futile;
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
    bool last_exact = false;
    for (int round = 0;; ++round) {
      std::vector<Move> next = PvOnce(h, total);
      // An answer ends with the attacker's move: a line cut after a defence
      // (lengths that disagree with the judgements) is not exact.
      if (exact && next.size() % 2 == 0) exact = false;
      if (round > 0 && !exact && last_exact) {
        // The length search failed in a later round: keep the last exact
        // answer rather than a greedy line.
        exact = true;
        break;
      }
      pv = std::move(next);
      last_exact = exact;
      if (round == 0) { first = pv; first_exact = exact; }
      bool more = exact && total > 1 && round < kShortenRounds && shorten_nodes < kShortenBudget &&
                  !limit_reached && !h.LimitReached();
      if (chain_restart && exact && round < kShortenRounds) {
        chain_restart = false;  // rebuild with the shorter proofs copied by the chain judgements
        more = true;
      } else if (more) {
        more = TryShorterAlong(h, pv, total);
      }
      if (!more) break;
      // New proofs can only shorten lengths: without futility "within k"
      // stays true, "not within k" may not. With futility the judgements
      // change too: everything is computed again. Mate probes are facts (a
      // failed probe stays a conservative "not known").
      if (opt.futile) bounds.clear();
      else for (auto& [key, b] : bounds) b.lo = -1;
      proven_cache.clear();
      chain_memo.clear();
      real_memo.clear();
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
        if (ChainDrop(p, m, slider) && ChainFutile(h, m, slider)) continue;
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

  // ---- the mate after a final check -----------------------------------------
  // An answer whose last attack is a check ends where every reply is mated:
  // the mate itself is shown then, up to the checkmate (the defender's longest
  // resistance, the attacker's shortest mate, futile interpositions left out
  // with Options::futile). Mate lengths come from the mate-mode proofs (re-
  // proven when evicted); the line is left as it was when one is missing.
  static constexpr int kMaxMateTail = 255;               // plies of the shown mate
  static constexpr std::uint64_t kMateTailProbe = 1000000;  // re-proof of an evicted mate position

  // Plies to the checkmate at the current position (attacker to move: his
  // shortest mate; defender to move: his longest resistance, 0 when mated),
  // -1 when no mate is known.
  int MateLen(SearchImpl& h) {
    Position& p = h.pos;
    if (p.side_to_move() != atk && MoveList<LEGAL_ALL>(p).size() == 0) return p.in_check() ? 0 : -1;
    if (p.side_to_move() == atk && !p.in_check() && Mate::mate_1ply(p) != MOVE_NONE) return 1;
    Probe pr = h.Lookup(BoardKey(p), p.hand_of(atk), kModeMate);
    if (pr.pn != 0 && pr.dn != 0) {
      h.SubSearch(kModeMate, kMateTailProbe);
      pr = h.Lookup(BoardKey(p), p.hand_of(atk), kModeMate);
    }
    return pr.pn == 0 ? pr.len : -1;
  }

  // The legal move of the piece on `from` to `to` (a capture of an
  // interposition: the non-promotion when both are legal), MOVE_NONE if none.
  static Move MoveFromTo(const Position& p, Square from, Square to) {
    Move found = MOVE_NONE;
    for (const auto& em : MoveList<LEGAL_ALL>(p))
      if (!is_drop(em.move) && from_sq(em.move) == from && to_sq(em.move) == to) {
        if (found == MOVE_NONE || is_promote(found)) found = em.move;
      }
    return found;
  }

  // Appends the mate after `pv` when its last move is a check. Returns the
  // number of plies appended.
  int AppendMateLine(SearchImpl& h, std::vector<Move>& pv) {
    Position& p = h.pos;
    std::vector<StateInfo> st(pv.size() + kMaxMateTail + 2);
    std::size_t ply = 0;
    for (; ply < pv.size(); ++ply) p.do_move(pv[ply], st[ply]);
    const std::size_t base = pv.size();
    bool ok = !pv.empty() && p.side_to_move() != atk && p.in_check();
    Square slider = SQ_NB;  // the checking slider (futile interpositions)
    if (ok && IsSlider(type_of(p.piece_on(to_sq(pv.back()))))) slider = to_sq(pv.back());
    bool mated = false;
    while (ok && pv.size() - base < static_cast<std::size_t>(kMaxMateTail)) {
      Move chosen = MOVE_NONE;
      if (p.side_to_move() != atk) {
        if (MoveList<LEGAL_ALL>(p).size() == 0) {
          mated = true;
          break;
        }
        // The longest resistance; an interposition is futile when the
        // slider's capture mates no later than the best other defence.
        int best = -1, best_other = -1;
        std::vector<std::pair<int, Move>> inters;
        for (const auto& em : MoveList<LEGAL_ALL>(p)) {
          p.do_move(em.move, st[ply]);
          const int len = MateLen(h);
          p.undo_move(em.move);
          if (len < 0) { ok = false; break; }
          if (IsInterposition(p, em.move, slider)) {
            inters.emplace_back(len, em.move);
            continue;
          }
          if (len > best_other) best_other = len;
          if (len > best) { best = len; chosen = em.move; }
        }
        if (!ok) break;
        for (const auto& [len, m] : inters) {
          if (best_other >= 0) {
            StateInfo s1, s2;
            p.do_move(m, s1);
            const Move cap = MoveFromTo(p, slider, to_sq(m));
            int after = -1;
            if (cap != MOVE_NONE) {
              p.do_move(cap, s2);
              after = MateLen(h);
              p.undo_move(cap);
            }
            p.undo_move(m);
            if (after >= 0 && after + 1 <= best_other) continue;  // futile
          }
          if (len > best) { best = len; chosen = m; }
        }
      } else {
        // The shortest mate (a check whose position is mated soonest).
        int best = 0x7fffffff;
        for (const auto& em : MoveList<LEGAL_ALL>(p)) {
          if (!p.gives_check(em.move)) continue;
          p.do_move(em.move, st[ply]);
          const int len = MateLen(h);
          p.undo_move(em.move);
          if (len >= 0 && len < best) { best = len; chosen = em.move; }
        }
        if (chosen != MOVE_NONE && IsSlider(type_of(p.moved_piece_after(chosen)))) slider = to_sq(chosen);
        else if (chosen != MOVE_NONE) slider = SQ_NB;
      }
      if (chosen == MOVE_NONE) { ok = false; break; }
      pv.push_back(chosen);
      p.do_move(chosen, st[ply++]);
    }
    while (ply > 0) p.undo_move(pv[--ply]);
    if (!ok || !mated) {
      pv.resize(base);  // no complete mate: the answer as it was
      return 0;
    }
    return static_cast<int>(pv.size() - base);
  }

  // Greedy answer (fallback): shortest proof for the attacker, longest proof
  // for the defender, futile interpositions excluded.
  std::vector<Move> GreedyPv(SearchImpl& h) {
    // After a limit of the Solve the probes still run (with their own small
    // budgets): without them the line could stop short of hisshi.
    struct IgnoreLimits {
      SearchImpl& s;
      explicit IgnoreLimits(SearchImpl& x) : s(x) { s.ignore_limits = true; }
      ~IgnoreLimits() { s.ignore_limits = false; }
    } ignore_limits(h);
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
        if (ChainDrop(p, m, slider) && ChainFutile(h, m, slider)) continue;
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
