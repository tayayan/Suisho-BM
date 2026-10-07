// Internal header of the hisshi solver (included by solver.cpp only):
// the independent verification of a proof and of a disproof.

#ifndef HISSHI_VERIFY_HPP_
#define HISSHI_VERIFY_HPP_

#include "search.hpp"
#include "tree_refute.hpp"

namespace hisshi {
namespace detail {

// Open-addressing set of non-zero 64-bit keys.
class KeySet {
 public:
  bool Contains(Key k) const {
    if (t_.empty()) return false;
    k = k ? k : 1;
    for (std::size_t i = Index(k);; i = (i + 1) & mask_) {
      if (t_[i] == k) return true;
      if (t_[i] == 0) return false;
    }
  }
  void Insert(Key k) {
    if ((count_ + 1) * 2 > t_.size()) Grow();
    k = k ? k : 1;
    for (std::size_t i = Index(k);; i = (i + 1) & mask_) {
      if (t_[i] == k) return;
      if (t_[i] == 0) { t_[i] = k; ++count_; return; }
    }
  }

 private:
  std::size_t Index(Key k) const { return static_cast<std::size_t>((k * 0x9E3779B97F4A7C15ULL) >> 20) & mask_; }
  void Grow() {
    std::vector<Key> old;
    old.swap(t_);
    t_.assign(old.empty() ? (1u << 16) : old.size() * 2, 0);
    mask_ = t_.size() - 1;
    count_ = 0;
    for (Key k : old)
      if (k) Insert(k);
  }
  std::vector<Key> t_;
  std::size_t mask_ = 0, count_ = 0;
};

// Lock-free set of verified positions shared by the verification threads:
// board key (with the mode salt) and attacker's hand. Like the TT it uses
// hand dominance: a position verified with a hand is verified with any
// superior hand (the defender then has fewer pieces; the board is the same).
// A slot packs 43 key bits and the 21-bit hand. When the probe window is full
// an older slot is overwritten: a lost entry only repeats work.
class AtomicHandSet {
 public:
  explicit AtomicHandSet(int log2) : size_(std::size_t(1) << log2), t_(new std::atomic<std::uint64_t>[size_]) {
    for (std::size_t i = 0; i < size_; ++i) t_[i].store(0, std::memory_order_relaxed);
  }
  // A position with this board and an equal or inferior hand is recorded.
  bool Contains(Key board, Hand hand) const {
    const std::uint64_t kb = KeyBits(board);
    for (std::size_t n = 0, i = Index(board); n < kProbe; ++n, i = (i + 1) & (size_ - 1)) {
      const std::uint64_t cur = t_[i].load(std::memory_order_acquire);
      if (cur == 0) return false;
      if ((cur >> kHandBits) == kb && hand_is_equal_or_superior(hand, Unpack(cur))) return true;
    }
    return false;
  }
  void Insert(Key board, Hand hand) {
    const std::uint64_t v = (KeyBits(board) << kHandBits) | Pack(hand);
    for (std::size_t n = 0, i = Index(board); n < kProbe; ++n, i = (i + 1) & (size_ - 1)) {
      std::uint64_t cur = t_[i].load(std::memory_order_acquire);
      if (cur == v) return;
      if (cur == 0) {
        if (t_[i].compare_exchange_strong(cur, v, std::memory_order_acq_rel)) return;
        if (cur == v) return;
      }
    }
    // Slots never become empty again, so Contains() may still stop at an
    // empty slot after this overwrite.
    t_[(Index(board) + static_cast<std::size_t>(v & (kProbe - 1))) & (size_ - 1)].store(v, std::memory_order_release);
  }

 private:
  static constexpr std::size_t kProbe = 16;
  static constexpr int kHandBits = 21;
  // 43 key bits, never 0 (0 marks an empty slot).
  static std::uint64_t KeyBits(Key k) {
    const std::uint64_t b = static_cast<std::uint64_t>(k) >> kHandBits;
    return b ? b : 1;
  }
  // Hand counts P(5 bits) L N S(3) B R(2) G(3) -> 21 bits.
  static std::uint64_t Pack(Hand h) {
    const std::uint32_t x = static_cast<std::uint32_t>(h);
    return (x & 31) | ((x >> 8 & 7) << 5) | ((x >> 12 & 7) << 8) | ((x >> 16 & 7) << 11) |
           ((x >> 20 & 3) << 14) | ((x >> 24 & 3) << 16) | ((x >> 28 & 7) << 18);
  }
  static Hand Unpack(std::uint64_t v) {
    const std::uint32_t x = static_cast<std::uint32_t>(v);
    return static_cast<Hand>((x & 31) | ((x >> 5 & 7) << 8) | ((x >> 8 & 7) << 12) | ((x >> 11 & 7) << 16) |
                             ((x >> 14 & 3) << 20) | ((x >> 16 & 3) << 24) | ((x >> 18 & 7) << 28));
  }
  std::size_t Index(Key k) const { return static_cast<std::size_t>((k * 0x9E3779B97F4A7C15ULL) >> 20) & (size_ - 1); }
  std::size_t size_;
  std::unique_ptr<std::atomic<std::uint64_t>[]> t_;
};

// A subtree of the proof to verify on another thread: the moves from the
// root (MOVE_NULL = the pass), its mode, and the positions above it.
struct VerifyTask {
  std::vector<Move> moves;
  std::uint8_t mode;
  std::vector<Key> ancestors;      // with the mode salt (cycle check)
  std::vector<Key> ancestors_raw;  // position keys (re-proof avoiding the path)
};

struct Verifier {
  SearchImpl& s;
  Position& pos;
  std::unique_ptr<AtomicHandSet> own;  // verified positions (sequential verification)
  int own_log2 = 22;                   // its size (2^n slots of 8 bytes, allocated when first used)
  AtomicHandSet* shared = nullptr;     // shared by the threads of the parallel verification
  std::vector<Key> on_path;    // positions on the current verification path
  std::vector<Key> raw_path;   // the same, position keys without the mode salt
  bool cycle_seen = false;     // a stored proof led back to a position on the path
  std::uint64_t visited = 0;
  std::string error;
  std::vector<StateInfo> st;
  // Task collection: subtrees at this depth are recorded instead of verified.
  int collect_depth = -1;
  std::vector<Move> path;
  std::vector<VerifyTask>* tasks = nullptr;

  static constexpr std::uint64_t kMemoMinVisits = 3;  // smaller subtrees are not recorded as verified

  explicit Verifier(SearchImpl& si) : s(si), pos(si.pos) { st.resize(4096); }

  AtomicHandSet& Done() {
    if (shared) return *shared;
    if (!own) own = std::make_unique<AtomicHandSet>(own_log2);
    return *own;
  }
  // Verified positions are recorded by board (with the mode salt) and attacker's hand.
  bool IsDone(Key board, Hand hand) { return Done().Contains(board, hand); }
  void MarkDone(Key board, Hand hand) { Done().Insert(board, hand); }

  static Key Salt(Key k, std::uint8_t mode) { return mode ? k ^ 0x9e3779b97f4a7c15ULL : k; }

  bool ProveHere(std::uint8_t mode) {
    // Ensure the current node is proven in the TT, searching if needed.
    const Probe p = s.Lookup(BoardKey(pos), pos.hand_of(s.atk), mode);
    if (p.pn == 0) return true;
    return s.SubSearch(mode, 5000000).pn == 0;
  }

  bool Verify(int depth, std::uint8_t mode) {
    if (depth >= 4000) { error = "too deep"; return false; }
    const Key k = Salt(pos.key(), mode);
    const Key board = Salt(BoardKey(pos), mode);
    const Hand hand = pos.hand_of(s.atk);
    if (IsDone(board, hand)) return true;
    if (std::find(on_path.begin(), on_path.end(), k) != on_path.end()) {
      error = "cycle in proof";
      cycle_seen = true;
      return false;
    }
    if (depth == collect_depth) {
      tasks->push_back(VerifyTask{path, mode, on_path, raw_path});
      MarkDone(board, hand);  // one task per position
      return true;
    }
    const std::uint64_t visited_before = visited++;
    on_path.push_back(k);
    raw_path.push_back(s.PathEntry(pos.key(), mode));
    bool ok = true;
    if (pos.side_to_move() == s.atk) {
      ok = VerifyOr(depth, mode);
    } else {
      ok = VerifyAnd(depth, mode);
    }
    on_path.pop_back();
    raw_path.pop_back();
    // Only positions whose check took some work are recorded: most positions
    // of a proof are leaves (mated positions, one-ply mates), and recording
    // them would push the costly positions out of the shared set of the
    // parallel verification.
    if (ok && visited - visited_before >= kMemoMinVisits) MarkDone(board, hand);
    return ok;
  }

  bool VerifyOr(int depth, std::uint8_t mode) {
    // Prefer an immediate mate (most attacker nodes of a proof; it needs no
    // TT entry, which may have been evicted).
    if (!pos.in_check()) {
      const Move mate = Mate::mate_1ply(pos);
      if (mate != MOVE_NONE && TryOrMove(depth, mode, mate)) return true;
    }
    if (s.Lookup(BoardKey(pos), pos.hand_of(s.atk), mode).pn != 0) {
      // Not in the TT any more (evicted): an attack whose result was already
      // verified through another path is enough, otherwise the node would be
      // proven again on every path reaching it.
      for (const auto& em : MoveList<LEGAL_ALL>(pos)) {
        const Move m = em.move;
        if (mode == kModeMate && !pos.gives_check(m)) continue;
        Child c{};
        c.move = m;
        Key cb, cf;
        Hand ch;
        std::uint8_t cm;
        s.ChildKey(c, mode, cb, ch, cm, cf);
        if (IsDone(Salt(cb, cm), ch)) return true;
      }
    }
    if (!ProveHere(mode)) { error = "OR node not proven"; return false; }
    // The move stored with the proof.
    {
      const Probe p = s.Lookup(BoardKey(pos), pos.hand_of(s.atk), mode);
      const Move best = p.pn == 0 ? pos.to_move(Move16(p.best)) : MOVE_NONE;
      if (best != MOVE_NONE && pos.pseudo_legal_s<true>(best) && pos.legal(best) &&
          (mode != kModeMate || pos.gives_check(best)) && TryOrMove(depth, mode, best))
        return true;
    }
    // The stored proof led back to a position on the path: repair it here
    // before the other attacks are tried (each could run into the loop too).
    if (cycle_seen && ReproveAndRetry(depth, mode)) return true;
    // Otherwise any attack proven in the TT, the shortest first.
    std::vector<std::pair<int, Move>> cands;
    for (const auto& em : MoveList<LEGAL_ALL>(pos)) {
      const Move m = em.move;
      if (mode == kModeMate && !pos.gives_check(m)) continue;
      Child c{};
      c.move = m;
      const Probe p = s.ProbeChild(c, mode);
      if (p.pn == 0) cands.emplace_back(p.len, m);
    }
    std::sort(cands.begin(), cands.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& [len, m] : cands) {
      if (TryOrMove(depth, mode, m)) return true;
    }
    // Fall back to the stored best move after re-proving.
    const Probe p = s.Lookup(BoardKey(pos), pos.hand_of(s.atk), mode);
    const Move best = pos.to_move(Move16(p.best));
    if (best != MOVE_NONE && pos.pseudo_legal_s<true>(best) && pos.legal(best) &&
        (mode != kModeMate || pos.gives_check(best)))
      if (TryOrMove(depth, mode, best)) return true;
    if (cycle_seen && ReproveAndRetry(depth, mode)) return true;
    if (error.empty()) error = "no verified attack";
    return false;
  }

  // Stored proofs made at different times may form a loop once entries are
  // replaced: prove the node again without the path and check that proof.
  bool ReproveAndRetry(int depth, std::uint8_t mode) {
    if (!ReproveAvoidingPath(mode)) return false;
    const Probe q = s.Lookup(BoardKey(pos), pos.hand_of(s.atk), mode);
    const Move b = pos.to_move(Move16(q.best));
    return q.pn == 0 && b != MOVE_NONE && pos.pseudo_legal_s<true>(b) && pos.legal(b) &&
           (mode != kModeMate || pos.gives_check(b)) && TryOrMove(depth, mode, b);
  }

  // Drops the proofs of the current node and proves it again with the
  // positions of the verification path treated as repetitions.
  bool ReproveAvoidingPath(std::uint8_t mode) {
    cycle_seen = false;
    s.EraseProofs(BoardKey(pos), pos.hand_of(s.atk), mode);
    for (std::size_t i = 0; i + 1 < raw_path.size(); ++i) s.PushPath(raw_path[i]);
    const NodeResult r = s.SubSearch(mode, 5000000);
    for (std::size_t i = 0; i + 1 < raw_path.size(); ++i) s.PopPath();
    return r.pn == 0;
  }

  bool TryOrMove(int depth, std::uint8_t mode, Move m) {
    pos.do_move(m, st[depth]);
    path.push_back(m);
    const bool ok = Verify(depth + 1, mode);
    path.pop_back();
    pos.undo_move(m);
    if (!ok) return false;
    error.clear();
    return true;
  }

  bool VerifyAnd(int depth, std::uint8_t mode) {
    const bool check = pos.in_check();
    if (mode == kModeMate && !check) { error = "mate-mode node without check"; return false; }
    if (!check) {
      // The pass must be mated: the attack was a tsumero.
      pos.do_null_move(st[depth]);
      path.push_back(MOVE_NULL);
      const bool ok = Verify(depth + 1, kModeMate);
      path.pop_back();
      pos.undo_null_move();
      if (!ok) { if (error.empty()) error = "pass not mated"; return false; }
    }
    for (const auto& em : MoveList<LEGAL_ALL>(pos)) {
      const Move m = em.move;
      pos.do_move(m, st[depth]);
      path.push_back(m);
      const bool ok = Verify(depth + 1, mode);
      path.pop_back();
      pos.undo_move(m);
      if (!ok) return false;
    }
    return true;
  }
};

// Independent check of a disproof ("not hisshi"). Unlike the search, the
// attacker may play every legal move here (checks only below a pass), so a
// verified disproof does not depend on the candidate moves of paper 3.1. A
// repetition counts as a failure of the attacker (a draw, or a loss when he
// keeps checking). Each defender node needs one refuting reply (the pass
// counts after a non-check); replies disproven in the TT are tried first,
// otherwise the node is searched again.
struct DisproofVerifier {
  SearchImpl& s;
  Position& pos;
  KeySet done;              // refuted positions (key ^ mode salt), not depending on the path
  std::vector<Key> on_path;
  std::uint64_t visited = 0, searches = 0;
  std::string error;
  std::vector<StateInfo> st;
  static constexpr std::uint64_t kBudget = 2000000;  // node budget of one re-search
  static constexpr int kTries = 4;                   // refuting replies tried per defender node and round
  static constexpr int kRounds = 3;                  // re-searches of a defender node (budget x8 each)
  static constexpr std::size_t kTreeNodes = std::size_t(1) << 22;  // nodes of one search tree (TreeRefuter)
  static constexpr std::uint64_t kTreeTotal = std::uint64_t(1) << 26;  // nodes of all the trees of one check
  std::uint64_t trees = 0, trees_disproven = 0, tree_nodes = 0, last_tree_nodes = 0;
  bool last_tree_full = false;

  explicit DisproofVerifier(SearchImpl& si) : s(si), pos(si.pos) { st.resize(4096); }

  static Key Salt(Key k, std::uint8_t mode) { return mode ? k ^ 0x9e3779b97f4a7c15ULL : k; }

  static constexpr int kNoDep = 0x7fffffff;  // a result resting on no repetition

  // True when the attacker cannot force hisshi (mate below a pass) here.
  // `dep` is lowered to the shallowest path index of a repetition the result
  // rests on. A result resting only on repetitions with this node or below
  // holds on every path to it (such a cycle is one on every path), so it is
  // recorded as refuted.
  bool Refuted(int depth, std::uint8_t mode, int& dep) {
    const Key k = Salt(pos.key(), mode);
    if (done.Contains(k)) return true;
    const auto it = std::find(on_path.begin(), on_path.end(), k);
    if (it != on_path.end()) {
      dep = std::min(dep, static_cast<int>(it - on_path.begin()));
      return true;
    }
    if (depth >= 4000) { error = "too deep"; return false; }
    ++visited;
    const int index = static_cast<int>(on_path.size());
    on_path.push_back(k);
    int my_dep = kNoDep;
    const bool ok = pos.side_to_move() == s.atk ? OrRefuted(depth, mode, my_dep) : AndRefuted(depth, mode, my_dep);
    on_path.pop_back();
    if (ok) {
      if (my_dep >= index) done.Insert(k);
      else dep = std::min(dep, my_dep);
    }
    return ok;
  }

  // Attacker to move: every legal move (every check below a pass) must fail.
  bool OrRefuted(int depth, std::uint8_t mode, int& dep) {
    for (const auto& em : MoveList<LEGAL_ALL>(pos)) {
      const Move m = em.move;
      if (mode == kModeMate && !pos.gives_check(m)) continue;
      pos.do_move(m, st[depth]);
      s.PushPath(s.PathEntry(pos.key(), mode));
      const bool ok = Refuted(depth + 1, mode, dep);
      s.PopPath();
      pos.undo_move(m);
      if (!ok) {
        if (error.empty()) error = "attack " + to_usi_string(m) + " not refuted";
        return false;
      }
    }
    return true;
  }

  // Probe of a defender's reply (MOVE_NULL = the pass).
  // The reply's TT result for this path (the searcher's path is the
  // verification path, so disproofs resting on a repetition with it count).
  Probe ProbeReply(Move r, std::uint8_t mode) {
    Child c{};
    c.move = r;
    c.pass = r == MOVE_NULL;
    Key board, full;
    Hand hand;
    std::uint8_t cmode;
    s.ChildKey(c, mode, board, hand, cmode, full);
    // Disproven through a repetition with this path (also when the TT holds
    // a proof of the position, whose lines then run through the path).
    if (s.RepAt(s.ChildPathKey(full))) {
      Probe p;
      p.pn = kInf;
      p.dn = 0;
      p.found = true;
      p.rep = true;
      return p;
    }
    return s.Lookup(board, hand, cmode, s.ChildPathKey(full));
  }

  bool ReplyOnPath(Move r, std::uint8_t mode) {
    Child c{};
    c.move = r;
    c.pass = r == MOVE_NULL;
    Key board, full;
    Hand hand;
    std::uint8_t cmode;
    s.ChildKey(c, mode, board, hand, cmode, full);
    return s.InPath(full);
  }

  bool TryReply(int depth, std::uint8_t mode, Move r, int& dep) {
    int c = kNoDep;
    bool ok;
    if (r == MOVE_NULL) {
      pos.do_null_move(st[depth]);
      s.PushPath(s.PathEntry(pos.key(), kModeMate));
      ok = Refuted(depth + 1, kModeMate, c);
      s.PopPath();
      pos.undo_null_move();
    } else {
      pos.do_move(r, st[depth]);
      s.PushPath(s.PathEntry(pos.key(), mode));
      ok = Refuted(depth + 1, mode, c);
      s.PopPath();
      pos.undo_move(r);
    }
    if (ok) dep = std::min(dep, c);
    return ok;
  }

  // Defender to move: one reply must refute.
  bool AndRefuted(int depth, std::uint8_t mode, int& dep) {
    const bool check = pos.in_check();
    std::vector<Move> replies;
    if (!check) replies.push_back(MOVE_NULL);
    for (const auto& em : MoveList<LEGAL_ALL>(pos)) replies.push_back(em.move);
    if (check && replies.empty()) { error = "mated"; return false; }
    std::vector<Move> tried;
    std::uint64_t budget = kBudget;
    for (int round = 0; round < kRounds; ++round) {
      // Replies disproven in the TT: path-independent ones first.
      std::vector<std::pair<int, Move>> cands;
      for (Move r : replies) {
        if (std::find(tried.begin(), tried.end(), r) != tried.end()) continue;
        const Probe p = ProbeReply(r, mode);
        // A reply back to a position of the path is a repetition (the search
        // does not store those): the verification sees it on its own path.
        if (p.dn == 0) cands.emplace_back(p.rep ? 1 : 0, r);
        else if (ReplyOnPath(r, mode)) cands.emplace_back(1, r);
      }
      std::stable_sort(cands.begin(), cands.end(),
                       [](const auto& a, const auto& b) { return a.first < b.first; });
      for (const auto& [rep, r] : cands) {
        if (static_cast<int>(tried.size()) >= kTries * (round + 1)) break;
        tried.push_back(r);
        std::string saved = error;
        if (TryReply(depth, mode, r, dep)) { error.clear(); return true; }
        if (saved.empty() && !error.empty()) saved = error;
        error = saved;
      }
      if (round == kRounds - 1 || s.LimitReached()) break;
      // Search this node again to find (another) disproof.
      ++searches;
      // (SubSearch puts this node on the path itself: off the path meanwhile,
      // so that its path keys are the verification path's)
      const Key self = s.path.back();
      s.PopPath();
      s.rederive_root = true;
      s.rederive_ply = static_cast<int>(s.path.size()) + 1;  // (SubSearch's root ply)
      const NodeResult nr = s.SubSearch(mode, budget);
      budget *= 8;  // a hard node: a larger search next time
      s.rederive_root = false;
      s.PushPath(self);
      if (nr.pn == 0) break;  // (proven: no refuting reply)
    }
    // The TT shows no refuting reply (disproofs resting on repetitions hold
    // only for the paths they were found on): a small search tree without
    // transpositions from this node decides, its repetitions judged on its
    // own paths below the verification path.
    if (tree_nodes < kTreeTotal) {
      const std::vector<Key> above(s.path.begin(), s.path.end() - 1);
      TreeRefuter tree(pos, s.atk, mode, above, &s, kTreeNodes, [this]() { return s.LimitReached(); });
      const auto tr = tree.Run();
      ++trees;
      tree_nodes += tr.nodes;
      last_tree_nodes = tr.nodes;
      last_tree_full = tr.full;
      if (tr.disproven) {
        ++trees_disproven;
        if (tr.rep_above) dep = std::min(dep, tr.rep_index);  // (resting on the path above)
        error.clear();
        return true;
      }
    }
    if (error.empty()) error = "no refuting reply at " + pos.sfen() + " (tree " + std::to_string(last_tree_nodes) + (last_tree_full ? " full)" : ")");
    return false;
  }
};

}  // namespace detail
}  // namespace hisshi

#endif  // HISSHI_VERIFY_HPP_
