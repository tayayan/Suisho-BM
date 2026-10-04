#include "hisshi_dfpn.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <x86intrin.h>

#include "../../bitboard.h"
#include "../../mate/mate.h"
#include "../../misc.h"
#include "../../usi.h"

namespace hisshi {
namespace {

constexpr std::uint8_t kModeHisshi = 0;
constexpr std::uint8_t kModeMate = 1;

inline PnDn Add(PnDn a, PnDn b) {
  const PnDn s = a + b;  // both <= kInf, no overflow in 32 bits
  return s >= kInf ? kInf : s;
}

// Squares around the defending king (paper, Fig. 2). zone12: squares where
// an attacking effect counts as an attack; zone3: squares two ranks in front
// of the king that only a knight effect counts for.
Bitboard g_zone12[SQ_NB_PLUS1][COLOR_NB];
Bitboard g_zone3[SQ_NB_PLUS1][COLOR_NB];
bool g_tables_ready = false;

void InitTables() {
  if (g_tables_ready) return;
  for (Square k = SQ_ZERO; k < SQ_NB; ++k) {
    for (Color def : COLOR) {
      // "forward" for the defender = towards the attacker's camp.
      const int fwd = def == WHITE ? 1 : -1;
      const int kf = file_of(k), kr = rank_of(k);
      Bitboard z12 = ZERO_BB, z3 = ZERO_BB;
      auto add = [&](Bitboard& bb, int f, int r) {
        if (f < 0 || f > 8 || r < 0 || r > 8) return;
        bb |= Bitboard(Square(f * 9 + r));
      };
      for (int df = -1; df <= 1; ++df) add(z12, kf + df, kr - fwd);
      for (int df = -2; df <= 2; ++df)
        if (df != 0) add(z12, kf + df, kr);
      for (int df = -2; df <= 2; ++df) add(z12, kf + df, kr + fwd);
      for (int df = -1; df <= 1; ++df) add(z3, kf + df, kr + 2 * fwd);
      g_zone12[k][def] = z12;
      g_zone3[k][def] = z3;
    }
  }
  for (Color def : COLOR) {
    g_zone12[SQ_NB][def] = ZERO_BB;
    g_zone3[SQ_NB][def] = ZERO_BB;
  }
  g_tables_ready = true;
}

inline bool IsSlider(PieceType t) {
  return t == LANCE || t == BISHOP || t == ROOK || t == HORSE || t == DRAGON;
}

inline int Sign(int x) { return (x > 0) - (x < 0); }

inline Key BoardKey(const Position& p) { return p.state()->board_key(); }

// ---- proof pieces (証明駒, paper 3.3) --------------------------------------
constexpr PieceType kHandTypes[7] = {PAWN, LANCE, KNIGHT, SILVER, BISHOP, ROOK, GOLD};

inline Hand HandUnion(Hand a, Hand b) {
  Hand r = static_cast<Hand>(0);
  for (PieceType p : kHandTypes) add_hand(r, p, std::max(hand_count(a, p), hand_count(b, p)));
  return r;
}

// Proof pieces of a defender-to-move node from the union of its children's.
// Pieces of a kind the defender does not hold are never handed over: that
// could create new drops for the defender.
inline Hand AndProofHand(Hand uni, Hand atk_hand, Hand def_hand) {
  Hand r = static_cast<Hand>(0);
  for (PieceType p : kHandTypes)
    add_hand(r, p, hand_exists(def_hand, p) ? hand_count(uni, p) : hand_count(atk_hand, p));
  return r;
}

// Proof pieces of an attacker-to-move node proven by move m whose child
// needs child_ph.
inline Hand OrProofHand(Hand child_ph, Move m, const Position& pos) {
  Hand r = child_ph;
  if (is_drop(m)) {
    add_hand(r, move_dropped_piece(m));
  } else {
    const Piece cap = pos.piece_on(to_sq(m));
    if (cap != NO_PIECE) {
      const PieceType pt = raw_type_of(cap);
      if (hand_exists(r, pt)) sub_hand(r, pt);
    }
  }
  return r;
}

struct Child {
  Move move;
  Hand ph;  // proof pieces of the child when proven
  PnDn pn;
  PnDn dn;
  std::uint16_t group;
  std::uint16_t len;
  std::uint8_t cost;
  std::uint8_t pass;
  std::uint8_t rep;
  std::uint8_t check;
  std::uint8_t stage;
  std::uint8_t sim;  // proof simulation already tried
};

// A list of children in caller-provided storage: a per-ply buffer, or a
// children-cache slot that is then updated in place.
struct ChildList {
  Child* p = nullptr;
  int n = 0;
  int cap = 0;
  Child& operator[](std::size_t i) { return p[i]; }
  const Child& operator[](std::size_t i) const { return p[i]; }
  std::size_t size() const { return static_cast<std::size_t>(n); }
  void clear() { n = 0; }
  void push_back(const Child& c) {
    if (n < cap) p[n++] = c;
  }
  Child* begin() { return p; }
  Child* end() { return p + n; }
  const Child* begin() const { return p; }
  const Child* end() const { return p + n; }
};

constexpr int kMaxChildren = 640;  // > max. number of legal moves (593) + pass

struct StageInfo {
  int lazy = 0;  // 1: stage 1 of an OR node, 2: stages 1-2 of a defender node not generated yet
  Move last = MOVE_NONE;  // attacker's last move (grouping of lazily generated defences)
  int nstages = 0;
  int begin[4] = {0, 0, 0, 0};  // begin[s] .. begin[s+1]
  PnDn est_pn[3] = {0, 0, 0};
  PnDn est_dn[3] = {0, 0, 0};
};

struct ChildKeyInfo {
  Key board;
  Key full;
  Hand hand;
  std::uint8_t mode;
};

struct NodeResult {
  PnDn pn, dn;
  std::uint16_t len;
  bool rep;
  Hand ph;  // proof pieces when proven
};

struct Probe {
  PnDn pn = 1, dn = 1;
  std::uint16_t len = 0;
  std::uint16_t best = 0;
  bool rep = false;
  bool found = false;
  Hand ph = static_cast<Hand>(0);  // proof pieces when proven
};

}  // namespace

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
  for (std::size_t i = 0; i < sample; ++i)
    for (int j = 0; j < kClusterSize; ++j) used += table_[i].key[j] != 0;
  return static_cast<int>(used * 1000 / (sample * kClusterSize));
}

// ---------------------------------------------------------------------------
// Search

struct SearchImpl {
  Solver& solver;
  Position& pos;
  const Options opt;
  const Limits lim;
  bool (*should_stop)();

  Color atk;
  Color def;
  Key side_key = 0;
  std::uint64_t nodes = 0;
  std::uint64_t next_check = 0;
  std::uint64_t next_report_ms = 0;
  std::chrono::steady_clock::time_point start;
  bool stop = false;
  std::uint64_t node_cap = 0;
  PnDn root_pn = 1, root_dn = 1;

  // Path for repetition detection.
  std::vector<Key> path;
  std::vector<std::uint16_t> path_count;

  std::vector<std::vector<Child>> child_buf;
  std::vector<std::vector<PnDn>> gmax_buf;
  std::vector<Move> killers;
  std::vector<std::vector<ChildKeyInfo>> key_buf;
  std::vector<std::vector<Key>> full_buf;
  std::vector<std::vector<int>> live_buf;

  static constexpr int kCacheMinChildren = 12;
  static constexpr int kCacheMaxChildren = kMaxChildren;
  struct CacheSlot {
    Key board;
    std::uint32_t hand;
    std::uint8_t mode;
    std::uint8_t valid;
    std::uint8_t in_use;  // a node on the current path works on this slot
    std::uint8_t active;
    std::uint16_t n;
    std::uint16_t groups;
    std::uint16_t active_end;
    StageInfo si;
    Child ch[kCacheMaxChildren];
    Key full[kCacheMaxChildren];
  };
  std::vector<CacheSlot> cache;
  std::vector<StateInfo> states;

  // Parallel search: thread 0 is the main thread; helpers share the TT and
  // stop when any thread has decided the root.
  int thread_id = 0;
  std::atomic<bool>* shared_stop = nullptr;

  SearchImpl(Solver& s, Position& p, const Options& o, const Limits& l, bool (*stop_fn)(),
             int tid = 0, std::atomic<bool>* sstop = nullptr, Color attacker = COLOR_NB)
      : solver(s), pos(p), opt(o), lim(l), should_stop(stop_fn), thread_id(tid), shared_stop(sstop) {
    cache.resize(opt.cache_slots);
    atk = attacker == COLOR_NB ? pos.side_to_move() : attacker;
    def = ~atk;
    {
      StateInfo si;
      const Key before = BoardKey(pos);
      pos.do_null_move(si);
      side_key = BoardKey(pos) ^ before;
      pos.undo_null_move();
    }
    path.reserve(opt.max_ply + 8);
    path_count.assign(1 << 16, 0);
    child_buf.resize(opt.max_ply + 8);
    gmax_buf.resize(opt.max_ply + 8);
    killers.assign(opt.max_ply + 8, MOVE_NONE);
    key_buf.resize(opt.max_ply + 8);
    full_buf.resize(opt.max_ply + 8);
    live_buf.resize(opt.max_ply + 8);
    // child_buf / full_buf of a ply are allocated when the ply is first
    // reached (most searches stay far below max_ply).
    states.resize(opt.max_ply + 8);
    start = std::chrono::steady_clock::now();
  }

  std::uint64_t ElapsedMs() const {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count());
  }

  // ---- TT ----------------------------------------------------------------
  static constexpr std::uint64_t kFinalWeight = 1000;
  static Key NonZero(Key board) { return board ? board : 1; }
  // The key bits kept in an entry (with the cluster index they identify the board).
  static std::uint32_t TagHi(Key board) {
    const auto t = static_cast<std::uint32_t>(static_cast<std::uint64_t>(board) >> 32);
    return t ? t : 1;
  }
  static std::uint16_t TagLo(Key board) { return static_cast<std::uint16_t>(static_cast<std::uint64_t>(board) >> 16); }

  Cluster* GetCluster(Key board) {
    return &solver.table_[static_cast<std::uint64_t>(board) % solver.cluster_count_];
  }

  // Holds the lock stripe of a cluster while helper threads are running.
  struct ClusterLock {
    Solver::Lock* l = nullptr;
    ClusterLock(Solver& s, Key board) {
      if (!s.shared_) return;
      l = &s.locks_[(static_cast<std::uint64_t>(board) % s.cluster_count_) & (Solver::kLocks - 1)];
      while (l->v.exchange(1, std::memory_order_acquire))
        while (l->v.load(std::memory_order_relaxed)) _mm_pause();
    }
    ~ClusterLock() {
      if (l) l->v.store(0, std::memory_order_release);
    }
  };

  Probe Lookup(Key board, Hand hand, std::uint8_t mode) {
    board = NonZero(board);
    Probe r;
    Cluster* c = GetCluster(board);
    ClusterLock lock(solver, board);
    const std::uint32_t hi = TagHi(board);
    const std::uint16_t lo = TagLo(board);
    PnDn pn_lb = 1, dn_lb = 1;
    int disproof = -1;  // a proof takes precedence (disproofs may depend on the path)
    for (int i = 0; i < kClusterSize; ++i) {
      if (c->key[i] != hi || c->d[i].tag != lo) continue;
      const EntryData& e = c->d[i];
      const Hand eh = static_cast<Hand>(e.hand);
      const bool we_sup = hand_is_equal_or_superior(hand, eh);  // our hand >= entry hand
      const bool we_inf = hand_is_equal_or_superior(eh, hand);  // entry hand >= our hand
      if (e.pn == 0 && we_sup && (e.Mode() == mode || (mode == kModeHisshi && e.Mode() == kModeMate))) {
        r.pn = 0; r.dn = kInf; r.len = e.Len(); r.best = e.best; r.found = true; r.rep = false;
        r.ph = eh;
        return r;
      }
      if (e.dn == 0 && we_inf && (e.Mode() == mode || (mode == kModeMate && e.Mode() == kModeHisshi))) {
        if (disproof < 0 || (c->d[disproof].Rep() && !e.Rep())) disproof = i;
        continue;
      }
      if (e.Mode() == mode && eh == hand) {
        r.found = true;
        r.pn = FromTT(e.pn); r.dn = FromTT(e.dn); r.len = e.Len(); r.best = e.best;
        continue;
      }
      // Superiority gives lower bounds on the unknown values.
      if (we_inf && (e.Mode() == mode || (e.Mode() == kModeHisshi && mode == kModeMate)))
        pn_lb = std::max(pn_lb, FromTT(e.pn));
      if (we_sup && (e.Mode() == mode || (e.Mode() == kModeMate && mode == kModeHisshi)))
        dn_lb = std::max(dn_lb, FromTT(e.dn));
    }
    if (disproof >= 0) {
      const EntryData& e = c->d[disproof];
      r.pn = kInf; r.dn = 0; r.len = e.Len(); r.best = e.best; r.found = true; r.rep = e.Rep();
      return r;
    }
    if (r.found) {
      r.pn = std::max(r.pn, pn_lb);
      r.dn = std::max(r.dn, dn_lb);
    } else {
      r.pn = pn_lb;
      r.dn = dn_lb;
    }
    return r;
  }

  void Store(Key board, Hand hand, std::uint8_t mode, PnDn pn, PnDn dn, std::uint16_t len,
             Move best, bool rep, std::uint64_t amount) {
    board = NonZero(board);
    Cluster* c = GetCluster(board);
    ClusterLock lock(solver, board);
    const std::uint32_t hi = TagHi(board);
    const std::uint16_t lo = TagLo(board);
    auto same_board = [&](int i) { return c->key[i] == hi && c->d[i].tag == lo; };
    int target = -1;
    for (int i = 0; i < kClusterSize; ++i) {
      if (same_board(i) && c->d[i].Mode() == mode && c->d[i].hand == hand) {
        target = i;
        break;
      }
    }
    if (target >= 0) {
      // A final result is never replaced by an estimate (another path or
      // thread may still be working on the node), and a proof is never
      // replaced by a disproof, which may depend on the path.
      const EntryData& t = c->d[target];
      if (t.pn == 0 && pn != 0) return;
      if (t.dn == 0 && !t.Rep() && pn != 0 && dn != 0) return;
    }
    if (dn == 0) {
      // Proven for a hand we dominate: the disproof must be path-dependent.
      for (int i = 0; i < kClusterSize; ++i) {
        if (!same_board(i) || c->d[i].pn != 0) continue;
        const EntryData& e = c->d[i];
        if (hand_is_equal_or_superior(hand, static_cast<Hand>(e.hand)) &&
            (e.Mode() == mode || (mode == kModeHisshi && e.Mode() == kModeMate)))
          return;
      }
    }
    if (pn == 0 || dn == 0) {
      // Remove entries made redundant by this result (a disproof never
      // removes a proof).
      for (int i = 0; i < kClusterSize; ++i) {
        if (i == target || !same_board(i) || c->d[i].Mode() != mode) continue;
        const Hand eh = static_cast<Hand>(c->d[i].hand);
        if (pn == 0 && hand_is_equal_or_superior(eh, hand)) c->key[i] = 0;
        else if (dn == 0 && c->d[i].pn != 0 && hand_is_equal_or_superior(hand, eh)) c->key[i] = 0;
      }
    }
    const std::uint16_t amt = EncodeAmount(amount);
    if (target < 0) {
      for (int i = 0; i < kClusterSize; ++i)
        if (c->key[i] == 0) { target = i; break; }
    }
    if (target < 0) {
      // Replace the least valuable entry: the one that would cost the least
      // search to recompute. As in KomoringHeights, a final result (proof or
      // disproof) counts as kFinalWeight times its search effort instead
      // of being kept unconditionally (a table full of small proofs pushes
      // the working estimates out). After the search only proofs are kept.
      std::uint64_t worst = ~0ULL;
      for (int i = 0; i < kClusterSize; ++i) {
        const EntryData& e = c->d[i];
        std::uint64_t v = DecodeAmount(e.amount);
        if (e.pn == 0 || e.dn == 0) v = v * kFinalWeight;
        if (e.pn == 0 && solver.proofs_only_) v += 1ULL << 40;
        if (v < worst) { worst = v; target = i; }
      }
      c->d[target].amount = 0;
    }
    EntryData& e = c->d[target];
    if (same_board(target) && e.Mode() == mode && e.hand == hand)
      e.amount = std::max(e.amount, amt);
    else
      e.amount = amt;
    c->key[target] = hi;
    e.tag = lo;
    e.hand = static_cast<std::uint32_t>(hand);
    e.pn = ToTT(pn);
    e.dn = ToTT(dn);
    e.lmr = static_cast<std::uint16_t>(std::min<int>(len, 0x3fff) | ((mode & 1) << 14) | ((rep ? 1 : 0) << 15));
    e.best = Move16(best).to_u16();
  }

  // Removes the proofs that cover this position (verification repairs).
  void EraseProofs(Key board, Hand hand, std::uint8_t mode) {
    board = NonZero(board);
    Cluster* c = GetCluster(board);
    ClusterLock lock(solver, board);
    const std::uint32_t hi = TagHi(board);
    const std::uint16_t lo = TagLo(board);
    for (int i = 0; i < kClusterSize; ++i) {
      if (c->key[i] != hi || c->d[i].tag != lo || c->d[i].pn != 0) continue;
      const EntryData& e = c->d[i];
      if ((e.Mode() == mode || e.Mode() == kModeMate) && hand_is_equal_or_superior(hand, static_cast<Hand>(e.hand)))
        c->key[i] = 0;
    }
  }

  // ---- path --------------------------------------------------------------
  bool InPath(Key k) const {
    if (!path_count[k & 0xffff]) return false;
    for (Key p : path)
      if (p == k) return true;
    return false;
  }
  void PushPath(Key k) {
    path.push_back(k);
    ++path_count[k & 0xffff];
  }
  void PopPath() {
    --path_count[path.back() & 0xffff];
    path.pop_back();
  }

  // ---- move generation ----------------------------------------------------
  static bool UselessNonPromotion(const Position& p, Move m) {
    if (is_drop(m) || is_promote(m)) return false;
    const Piece pc = p.moved_piece_before(m);
    const PieceType t = type_of(pc);
    if (t != PAWN && t != BISHOP && t != ROOK) return false;
    const Color us = color_of(pc);
    const Square from = from_sq(m), to = to_sq(m);
    auto in_enemy = [&](Square s) {
      const int r = rank_of(s);
      return us == BLACK ? r <= RANK_3 : r >= RANK_7;
    };
    return in_enemy(from) || in_enemy(to);
  }

  bool IsCandidate(Move m) const {
    const Square ksq = pos.king_square(def);
    const Square to = to_sq(m);
    const Piece after = pos.moved_piece_after(m);
    const PieceType t = type_of(after);
    Bitboard occ = pos.pieces();
    if (!is_drop(m)) occ ^= Bitboard(from_sq(m));
    occ |= Bitboard(to);
    const Bitboard eff = effects_from(after, to, occ);
    if (eff & g_zone12[ksq][def]) return true;
    if (t == KNIGHT && (eff & g_zone3[ksq][def])) return true;
    if (!is_drop(m)) {
      const Piece cap = pos.piece_on(to);
      if (cap != NO_PIECE && type_of(cap) != PAWN) return true;
    }
    const PieceType raw = is_drop(m) ? move_dropped_piece(m) : type_of(pos.moved_piece_before(m));
    if ((raw == PAWN || raw == LANCE) && pos.attackers_to(def, to).pop_count() >= 2) return true;
    const int f = file_of(to), r = rank_of(to), kf = file_of(ksq), kr = rank_of(ksq);
    if (t == ROOK || t == DRAGON) {
      if (std::abs(f - kf) <= 1 || std::abs(r - kr) <= 1) return true;
    } else if (t == BISHOP || t == HORSE) {
      if (std::abs((f + r) - (kf + kr)) <= 1 || std::abs((f - r) - (kf - kr)) <= 1) return true;
    } else if (t == LANCE) {
      const bool toward = atk == BLACK ? r > kr : r < kr;
      if (std::abs(f - kf) <= 1 && toward) return true;
    }
    return false;
  }

  // Group id for the dn aggregation of an OR node (paper 3.1: attacks from
  // the same direction / captures of the same piece are represented by the
  // maximum).
  int OrGroupKey(Move m, int index) const {
    const Square to = to_sq(m);
    if (!is_drop(m) && pos.piece_on(to) != NO_PIECE) return 16 + to;  // capture of that piece
    const Square ksq = pos.king_square(def);
    const Piece after = pos.moved_piece_after(m);
    if (ksq != SQ_NB && IsSlider(type_of(after)) && (effects_from(after, to, ZERO_BB) & Bitboard(ksq))) {
      const int df = Sign(file_of(ksq) - file_of(to)), dr = Sign(rank_of(ksq) - rank_of(to));
      return (df + 1) * 3 + (dr + 1);  // 0..8
    }
    return 1000 + index;
  }

  // Group id for the pn aggregation of an AND node.
  enum : int { kGEscape = 110, kGCounter = 111, kGBlock = 112, kGOther = 113 };

  int AndGroupKey(Move m, int index, Move last) const {
    const Square ksq = pos.king_square(def);
    const Square to = to_sq(m);
    if (pos.in_check()) {
      if (!is_drop(m) && from_sq(m) == ksq) return 1000 + index;
      if (!is_drop(m) && pos.piece_on(to) != NO_PIECE) return 1000 + index;
      // Interpositions: one group per square (pieces on the same square alike).
      return 120 + static_cast<int>(to);
    }
    if (!is_drop(m) && from_sq(m) == ksq) return 1000 + index;
    const Bitboard around = kingEffect(ksq);
    if (!is_drop(m)) {
      const Piece cap = pos.piece_on(to);
      if (cap != NO_PIECE) {
        const PieceType ct = raw_type_of(cap);
        const bool major = !(ct == PAWN || ct == LANCE || ct == KNIGHT);
        if (major || (around & Bitboard(to)) || (is_ok(last) && to == to_sq(last))) return 16 + to;
        return kGOther;
      }
      if (around & Bitboard(from_sq(m))) return kGEscape;
      const PieceType mt = type_of(pos.moved_piece_before(m));
      if (mt == BISHOP || mt == ROOK || mt == HORSE || mt == DRAGON) return kGOther;
    }
    const Piece after = pos.moved_piece_after(m);
    Bitboard occ = pos.pieces();
    if (!is_drop(m)) occ ^= Bitboard(from_sq(m));
    occ |= Bitboard(to);
    if (effects_from(after, to, occ) & (around | g_zone3[ksq][def])) return kGCounter;
    if (is_ok(last)) {
      const Square lto = to_sq(last);
      const Piece lp = pos.piece_on(lto);
      if (lp != NO_PIECE && color_of(lp) == atk && IsSlider(type_of(lp)) &&
          (effects_from(lp, lto, pos.pieces()) & Bitboard(to)))
        return kGBlock;
    }
    return kGOther;
  }

  // Generates the non-checking attacks of a lazily expanded OR node
  // (appended as stage 1).
  void ExpandLazy(ChildList& ch, int& groups) {
    static thread_local int key_to_group[256];
    std::fill(std::begin(key_to_group), std::end(key_to_group), -1);
    const std::size_t n0 = ch.size();
    for (const auto& em : MoveList<NON_EVASIONS_ALL>(pos)) {
      const Move m = em.move;
      if (UselessNonPromotion(pos, m)) continue;
      if (!opt.full_width && !IsCandidate(m)) continue;
      if (pos.gives_check(m) || !pos.legal(m)) continue;
      bool dup = false;
      for (std::size_t i = 0; i < n0; ++i)
        if (ch[i].move == m) { dup = true; break; }
      if (dup) continue;
      Child c{};
      c.move = m;
      c.pn = 1;
      c.dn = 1;
      c.cost = static_cast<std::uint8_t>(opt.non_check_cost);
      c.stage = 1;
      const int key = OrGroupKey(m, static_cast<int>(ch.size()));
      if (key >= 1000) {
        c.group = static_cast<std::uint16_t>(groups++);
      } else {
        int& g = key_to_group[key];
        if (g < 0) g = groups++;
        c.group = static_cast<std::uint16_t>(g);
      }
      ch.push_back(c);
    }
  }

  // pn estimate of a defender node's relevant replies before generating them:
  // the king's apparently safe squares plus one group for the other defences.
  PnDn EstimateRelevantDefences() const {
    const Square ksq = pos.king_square(def);
    Bitboard to = kingEffect(ksq) & ~pos.pieces(def);
    PnDn n = 1;
    while (to) {
      const Square sq = to.pop();
      if (!pos.effected_to(atk, sq, ksq)) ++n;
    }
    return n;
  }

  // Generates the real replies of a lazily expanded defender node (after the
  // pass): stage 1 = relevant defences, stage 2 = the others.
  void ExpandLazyAnd(ChildList& ch, int& groups, StageInfo& si) {
    static thread_local int key_to_group[256];
    std::fill(std::begin(key_to_group), std::end(key_to_group), -1);
    const std::size_t n0 = ch.size();
    for (const auto& em : MoveList<NON_EVASIONS_ALL>(pos)) {
      const Move m = em.move;
      if (!pos.legal(m)) continue;
      const int key = AndGroupKey(m, static_cast<int>(ch.size()), si.last);
      Child c{};
      c.move = m;
      c.pn = 1;
      c.dn = 1;
      c.stage = static_cast<std::uint8_t>(key == kGOther ? 2 : 1);
      if (key >= 1000) {
        c.group = static_cast<std::uint16_t>(groups++);
      } else {
        int& g = key_to_group[key];
        if (g < 0) g = groups++;
        c.group = static_cast<std::uint16_t>(g);
      }
      ch.push_back(c);
    }
    std::stable_sort(ch.begin() + n0, ch.end(),
                     [](const Child& x, const Child& y) { return x.stage < y.stage; });
    int b2 = static_cast<int>(ch.size());
    for (std::size_t i = n0; i < ch.size(); ++i)
      if (ch[i].stage == 2) { b2 = static_cast<int>(i); break; }
    si.begin[2] = b2;
    si.begin[3] = static_cast<int>(ch.size());
    // Exact pn contributions of the two stages now that they are known.
    static thread_local std::vector<std::uint8_t> seen;
    seen.assign(groups, 0);
    si.est_pn[1] = si.est_pn[2] = 0;
    for (std::size_t i = n0; i < ch.size(); ++i)
      if (!seen[ch[i].group]) { seen[ch[i].group] = 1; si.est_pn[ch[i].stage] += 1; }
    si.lazy = 0;
  }

  // Fill children (sorted by stage) and per-stage information. Children of a
  // later stage are only looked up / searched when needed (delayed
  // expansion, paper 3.1: the defender's moves are generated in category
  // order and eventually all of them).

  int Expand(ChildList& ch, bool or_node, std::uint8_t mode, Move last, int ply,
             StageInfo& si) {
    ch.clear();
    static thread_local int key_to_group[256];
    std::fill(std::begin(key_to_group), std::end(key_to_group), -1);
    int groups = 0;
    auto group_of = [&](int key) -> int {
      if (key >= 1000) return groups++;
      int& g = key_to_group[key];
      if (g < 0) g = groups++;
      return g;
    };

    auto push = [&](Move m, bool pass, bool check, int cost, int gkey, int stage) {
      Child c{};
      c.move = m;
      c.pn = 1;
      c.dn = 1;
      c.cost = static_cast<std::uint8_t>(cost);
      c.pass = pass;
      c.check = check;
      c.stage = static_cast<std::uint8_t>(stage);
      c.group = static_cast<std::uint16_t>(group_of(gkey));
      ch.push_back(c);
    };
    const Move killer = killers[ply];

    if (or_node) {
      if (mode == kModeMate && pos.in_check()) {
        for (const auto& em : MoveList<EVASIONS_ALL>(pos)) {
          const Move m = em.move;
          if (!pos.gives_check(m) || !pos.legal(m)) continue;
          push(m, false, true, 0, OrGroupKey(m, static_cast<int>(ch.size())), 0);
        }
      } else if (mode == kModeMate) {
        for (const auto& em : MoveList<CHECKS_ALL>(pos)) {
          const Move m = em.move;
          if (!pos.legal(m)) continue;
          push(m, false, true, 0, OrGroupKey(m, static_cast<int>(ch.size())), 0);
        }
      } else if (pos.in_check()) {
        for (const auto& em : MoveList<EVASIONS_ALL>(pos)) {
          const Move m = em.move;
          if (!pos.legal(m)) continue;
          const bool chk = pos.gives_check(m);
          push(m, false, chk, chk ? 0 : opt.non_check_cost, OrGroupKey(m, static_cast<int>(ch.size())), 0);
        }
      } else {
        // Checks now; the other attacks are generated when first needed.
        for (const auto& em : MoveList<CHECKS_ALL>(pos)) {
          const Move m = em.move;
          if (!pos.legal(m)) continue;
          push(m, false, true, 0, OrGroupKey(m, static_cast<int>(ch.size())), 0);
        }
        if (killer != MOVE_NONE && pos.pseudo_legal_s<true>(killer) && pos.legal(killer) &&
            !pos.gives_check(killer))
          push(killer, false, false, opt.non_check_cost, OrGroupKey(killer, static_cast<int>(ch.size())), 0);
        for (std::size_t i = 0; i < ch.size(); ++i)
          if (ch[i].move == killer) {
            std::rotate(ch.begin(), ch.begin() + i, ch.begin() + i + 1);
            break;
          }
        si = StageInfo{};
        si.nstages = 2;
        si.begin[0] = 0;
        si.begin[1] = si.begin[2] = static_cast<int>(ch.size());
        si.est_pn[0] = 1;
        si.est_pn[1] = static_cast<PnDn>(opt.lazy_pn + opt.non_check_cost);
        si.est_dn[1] = static_cast<PnDn>(opt.lazy_dn);
        si.lazy = 1;
        return groups;
      }
      // Killer (move that proved a sibling) first.
      if (killer != MOVE_NONE) {
        for (std::size_t i = 0; i < ch.size(); ++i)
          if (ch[i].move == killer) {
            std::rotate(ch.begin(), ch.begin() + i, ch.begin() + i + 1);
            break;
          }
      }
    } else {
      if (pos.in_check()) {
        // Interpositions after the king moves and captures; only one dropped
        // piece per square at first (the others follow once the active ones
        // are proven, reusing their proofs).
        std::uint32_t drop_seen = 0;  // per square index bit (at most 7 squares between)
        Square drop_sq[8];
        int ndrop_sq = 0;
        for (const auto& em : MoveList<EVASIONS_ALL>(pos)) {
          const Move m = em.move;
          if (!pos.legal(m)) continue;
          const int g = AndGroupKey(m, static_cast<int>(ch.size()), last);
          int stage = (g >= 120 && g < 120 + SQ_NB) ? 1 : 0;
          if (stage == 1 && is_drop(m)) {
            int k = 0;
            while (k < ndrop_sq && drop_sq[k] != to_sq(m)) ++k;
            if (k == ndrop_sq && ndrop_sq < 8) drop_sq[ndrop_sq++] = to_sq(m);
            if (k < 8 && (drop_seen >> k & 1)) stage = 2;
            else if (k < 8) drop_seen |= 1u << k;
          }
          push(m, false, false, 0, g, stage);
        }
      } else {
        // The pass now; the real replies are generated when first needed.
        push(MOVE_NULL, true, false, 0, 1000 + 0, 0);
        si = StageInfo{};
        si.nstages = 3;
        si.begin[0] = 0;
        si.begin[1] = si.begin[2] = si.begin[3] = 1;
        si.est_pn[1] = EstimateRelevantDefences();
        si.est_pn[2] = 1;
        si.est_dn[1] = si.est_dn[2] = static_cast<PnDn>(opt.and_stage_dn);
        si.lazy = 2;
        si.last = last;
        return groups;
      }
    }

    // Order by stage and compute the stage boundaries / estimates.
    std::stable_sort(ch.begin(), ch.end(),
                     [](const Child& x, const Child& y) { return x.stage < y.stage; });
    si = StageInfo{};
    int last_stage = -1;
    static thread_local std::vector<std::uint8_t> seen_group;
    seen_group.assign(groups, 0);
    for (int i = 0; i < static_cast<int>(ch.size()); ++i) {
      const int st = ch[i].stage;
      while (last_stage < st) {
        ++last_stage;
        si.begin[last_stage] = i;
        si.est_pn[last_stage] = or_node ? kInf : 0;
        si.est_dn[last_stage] = or_node ? 0
                                : static_cast<PnDn>(pos.in_check() ? opt.check_stage_dn : opt.and_stage_dn);
      }
      const Child& c = ch[i];
      if (or_node) {
        si.est_pn[st] = std::min<PnDn>(si.est_pn[st], 1 + c.cost);
        if (!seen_group[c.group]) { seen_group[c.group] = 1; si.est_dn[st] += 1; }
      } else {
        if (!seen_group[c.group]) { seen_group[c.group] = 1; si.est_pn[st] += 1; }
      }
    }
    si.nstages = last_stage + 1;
    si.begin[si.nstages] = static_cast<int>(ch.size());
    return groups;
  }

  // Child's key/hand/mode for lookup.
  void ChildKey(const Child& c, std::uint8_t mode, Key& board, Hand& hand, std::uint8_t& cmode,
                Key& full) {
    if (c.pass) {
      board = BoardKey(pos) ^ side_key;
      hand = pos.hand_of(atk);
      cmode = kModeMate;
      full = pos.key() ^ side_key;
      return;
    }
    board = pos.board_key_after(c.move);
    full = pos.key_after(c.move);
    hand = pos.hand_of(atk);
    cmode = mode;
    if (pos.side_to_move() == atk) {
      if (is_drop(c.move)) {
        sub_hand(hand, move_dropped_piece(c.move));
      } else {
        const Piece cap = pos.piece_on(to_sq(c.move));
        if (cap != NO_PIECE) add_hand(hand, raw_type_of(cap));
      }
    }
  }

  void CheckLimits() {
    next_check = nodes + 4096;
    solver.total_nodes_.fetch_add(nodes - nodes_reported, std::memory_order_relaxed);
    nodes_reported = nodes;
    const auto ms = ElapsedMs();
    if ((node_cap && nodes >= node_cap) || (lim.nodes && nodes >= lim.nodes) || (lim.time_ms && static_cast<std::int64_t>(ms) >= lim.time_ms) ||
        (should_stop && should_stop()) || (shared_stop && shared_stop->load(std::memory_order_relaxed)))
      stop = true;
    if (report && lim.pv_interval_ms > 0 && ms >= next_report_ms) {
      next_report_ms = ms + lim.pv_interval_ms;
      PrintProgress(ms);
    }
  }

  // ---- USI progress output (cf. KomoringHeights' UsiInfo / PrintIfNeeded) --
  // One standard info line (read by GUIs such as ShogiGUI) and one info
  // string line; the pv takes the rest of its line, so pn/dn go to the
  // string line.
  std::uint64_t nodes_reported = 0;
  PnDn live_root_pn = 1, live_root_dn = 1;
  Move live_root_best = MOVE_NONE;
  bool report = false;       // the main thread of the main search prints progress
  int seldepth = 0;          // deepest ply since the last output
  std::string root_sfen;     // to replay the current best line

  // Evaluation from the root's pn/dn (KomoringHeights' Score::Unknown:
  // -600 log(pn/dn), a winning-rate style value).
  static std::string ScoreString(PnDn pn, PnDn dn) {
    if (pn == 0) return "mate +";
    if (dn == 0) return "mate -";
    double v = -600.0 * std::log(static_cast<double>(pn) / static_cast<double>(std::max<PnDn>(dn, 1)));
    v = std::max(-30000.0, std::min(30000.0, v));
    return "cp " + std::to_string(static_cast<int>(v));
  }

  // The current best line from the root, following the best moves stored
  // in the TT (up to a pass, which USI cannot show).
  std::string BestLine(std::uint8_t mode, int& len) const {
    len = 0;
    if (root_sfen.empty()) return "";
    Position p;
    std::vector<StateInfo> st(65);
    p.set(root_sfen, &st[64], pos.this_thread());
    std::string out;
    std::vector<Key> seen;
    for (int i = 0; i < 64; ++i) {
      if (std::find(seen.begin(), seen.end(), p.key()) != seen.end()) break;
      seen.push_back(p.key());
      const Probe pr = const_cast<SearchImpl*>(this)->Lookup(BoardKey(p), p.hand_of(atk), mode);
      Move m = MOVE_NONE;
      if (i == 0 && live_root_best != MOVE_NONE && pr.pn != 0) {
        m = p.to_move(Move16(live_root_best));
      } else {
        if (!pr.found && pr.pn != 0 && pr.dn != 0) break;
        m = p.to_move(Move16(pr.best));
      }
      if (m == MOVE_NONE || m == MOVE_NULL || !p.pseudo_legal_s<true>(m) || !p.legal(m)) break;
      out += (out.empty() ? "" : " ") + to_usi_string(m);
      p.do_move(m, st[i]);
      ++len;
      if (pr.pn == 0 && pr.len <= 1 && p.side_to_move() != atk && MoveList<LEGAL_ALL>(p).size() == 0) break;  // mated
    }
    return out;
  }

  void PrintProgress(std::uint64_t ms) {
    const std::uint64_t total = solver.total_nodes_.load(std::memory_order_relaxed);
    const std::uint64_t nps = ms ? total * 1000 / ms : 0;
    // Live values of the root (the search loop updates root_pn/dn only between iterations).
    PnDn pn = live_root_pn, dn = live_root_dn;
    if (!root_sfen.empty()) {
      // Decided by another thread already?
      Position p;
      StateInfo si;
      p.set(root_sfen, &si, pos.this_thread());
      const Probe pr = Lookup(BoardKey(p), p.hand_of(atk), kModeHisshi);
      if (pr.pn == 0 || (pr.dn == 0 && !pr.rep)) { pn = pr.pn; dn = pr.dn; }
    }
    int len = 0;
    const std::string line = BestLine(kModeHisshi, len);
    std::ostringstream os;
    os << "info depth " << path.size() << " seldepth " << std::max<int>(seldepth, static_cast<int>(path.size()))
       << " time " << ms << " nodes " << total << " nps " << nps << " hashfull " << solver.Hashfull()
       << " score " << ScoreString(pn, dn);
    if (!line.empty()) os << " pv " << line;
    sync_cout << os.str() << sync_endl;
    sync_cout << "info string pn=" << (pn >= kInf ? std::string("inf") : std::to_string(pn))
              << " dn=" << (dn >= kInf ? std::string("inf") : std::to_string(dn))
              << " threads=" << std::max(1, opt.threads) << sync_endl;
    seldepth = 0;
  }

  // Order among children with equal values: generation order for the main
  // thread, a per-thread pseudo-random order for helpers (search diversity).
  std::uint64_t Tie(const ChildList& ch, int i) const {
    if (i < 0) return ~0ULL;
    if (thread_id == 0) return static_cast<std::uint64_t>(i);
    return (static_cast<std::uint64_t>(ch[i].move) * 0x9E3779B97F4A7C15ULL +
            static_cast<std::uint64_t>(thread_id) * 0x632BE59BD9B4E019ULL) >> 16;
  }

  // 1+epsilon trick (Pawlewicz & Lew): let the best child run a little past
  // the second best before returning, which reduces switching between them.
  // Below a pass (mate mode) at least kMateEps: without it the search
  // alternates between checks of nearly equal proof numbers and rarely
  // completes a disproof (no-mate positions stayed undecided for tens of
  // millions of nodes; with it they are disproved in thousands).
  static constexpr int kMateEps = 50;
  PnDn Epsilon(PnDn second, std::uint8_t mode) const {
    const int eps = mode == kModeMate ? std::max(opt.eps_percent, kMateEps) : opt.eps_percent;
    if (eps <= 0 || second >= kInf) return second;
    return Add(second, (second * static_cast<PnDn>(eps) + 99) / 100);
  }

  // ---- df-pn+ ---------------------------------------------------------------
  NodeResult Search(int ply, std::uint8_t mode, PnDn thpn, PnDn thdn, Move last) {
    ++nodes;
    if (ply > seldepth) seldepth = ply;
    if (nodes >= next_check) CheckLimits();
    const bool or_node = pos.side_to_move() == atk;
    const Key board = BoardKey(pos);
    const Hand hand = pos.hand_of(atk);
    const std::uint64_t nodes_at_start = nodes;

    if (ply >= opt.max_ply) {
      return NodeResult{kInf, 0, 0, true, hand};
    }

    bool self_found = false;
    {
      // The node's own TT entry: return at once when it is already decided or
      // above the thresholds (e.g. values changed through a transposition).
      const Probe self = Lookup(board, hand, mode);
      if (self.pn == 0 || self.dn == 0 || self.pn >= thpn || self.dn >= thdn) {
        return NodeResult{self.pn, self.dn, self.len, self.rep, self.pn == 0 ? self.ph : hand};
      }
      self_found = self.found;
    }

    // A node with its own TT entry was expanded before; its one-ply mate
    // check already failed then.
    if (or_node && !self_found && !pos.in_check()) {
      const Move mate = Mate::mate_1ply(pos);
      if (mate != MOVE_NONE) {
        // The mated child needs no pieces except those that keep the defender
        // from getting new kinds of drops.
        Hand after = hand;
        if (is_drop(mate)) {
          sub_hand(after, move_dropped_piece(mate));
        } else if (pos.piece_on(to_sq(mate)) != NO_PIECE) {
          add_hand(after, raw_type_of(pos.piece_on(to_sq(mate))));
        }
        const Hand child_ph = AndProofHand(static_cast<Hand>(0), after, pos.hand_of(def));
        const Hand ph = OrProofHand(child_ph, mate, pos);
        // Not stored: found again at once (most proofs are such leaves).
        return NodeResult{0, kInf, 1, false, ph};
      }
    }

    ChildList ch;
    Key* fulls = nullptr;
    StageInfo si;
    int groups = 0;

    // Children cache: reuse the expanded children of a recently visited node.
    CacheSlot* slot = nullptr;
    bool cached = false;
    if (!cache.empty()) {
      const std::uint64_t h = static_cast<std::uint64_t>(board) ^
                              (static_cast<std::uint64_t>(hand) * 0x9E3779B97F4A7C15ULL) ^ mode;
      slot = &cache[(h >> 7) % cache.size()];
      if (slot->in_use) slot = nullptr;  // used by an ancestor
      cached = slot != nullptr && slot->valid && slot->board == board &&
               slot->hand == static_cast<std::uint32_t>(hand) && slot->mode == mode;
    }
    int active = 0;      // number of active stages
    int active_end = 0;  // children [0, active_end) are active
    if (cached) {
      si = slot->si;
      groups = slot->groups;
      active = slot->active;
      active_end = slot->active_end;
      // Work on the slot's children in place.
      slot->in_use = 1;
      ch.p = slot->ch;
      ch.n = slot->n;
      ch.cap = kCacheMaxChildren;
      fulls = slot->full;
      // Refresh path-dependent information.
      for (int i = 0; i < active_end; ++i) {
        Child& c = ch[i];
        if (InPath(fulls[i])) {
          c.pn = kInf; c.dn = 0; c.rep = 1;
          continue;
        }
        if (!c.rep) continue;
        ChildKeyInfo k;
        ChildKey(c, mode, k.board, k.hand, k.mode, k.full);
        const Probe p = Lookup(k.board, k.hand, k.mode);
        c.pn = p.pn; c.dn = p.dn; c.len = p.len; c.rep = p.rep; c.ph = p.ph;
      }
    } else {
      if (child_buf[ply].empty()) {
        child_buf[ply].resize(kMaxChildren);
        full_buf[ply].resize(kMaxChildren);
      }
      ch.p = child_buf[ply].data();
      ch.n = 0;
      ch.cap = kMaxChildren;
      fulls = full_buf[ply].data();
      groups = Expand(ch, or_node, mode, last, ply, si);
    }
    int n = static_cast<int>(ch.size());

    if (n == 0 && !si.lazy) {
      if (or_node) {
        Store(board, hand, mode, kInf, 0, 0, MOVE_NONE, false, 1);
        return NodeResult{kInf, 0, 0, false, hand};
      }
      const Hand ph = AndProofHand(static_cast<Hand>(0), hand, pos.hand_of(def));
      Store(board, ph, mode, 0, kInf, 0, MOVE_NONE, false, 1);
      return NodeResult{0, kInf, 0, false, ph};
    }

    std::vector<PnDn>& gmax = gmax_buf[ply];
    gmax.assign(groups, 0);
    // Children that can still change the node's values. Final children (a
    // proven reply of a defender node, a refuted attack of an attacker node)
    // are dropped from the scan.
    std::vector<int>& live = live_buf[ply];
    live.clear();

    auto activate = [&]() {
      if (si.lazy && active == 1) {
        if (si.lazy == 2) {
          ExpandLazyAnd(ch, groups, si);
        } else {
          ExpandLazy(ch, groups);
          si.begin[2] = static_cast<int>(ch.size());
          si.lazy = 0;
        }
        n = static_cast<int>(ch.size());
        gmax.resize(groups);
      }
      const int b = si.begin[active], e = si.begin[active + 1];
      // Compute the child keys and prefetch their clusters first so that the
      // (mostly missing) lookups overlap.
      std::vector<ChildKeyInfo>& keys = key_buf[ply];
      keys.resize(e - b);
      for (int i = b; i < e; ++i) {
        ChildKeyInfo& k = keys[i - b];
        ChildKey(ch[i], mode, k.board, k.hand, k.mode, k.full);
        fulls[i] = k.full;
        _mm_prefetch(reinterpret_cast<const char*>(GetCluster(NonZero(k.board))), _MM_HINT_T0);
      }
      for (int i = b; i < e; ++i) {
        Child& c = ch[i];
        const ChildKeyInfo& k = keys[i - b];
        if (InPath(k.full)) {
          c.pn = kInf; c.dn = 0; c.rep = 1;
          continue;
        }
        const Probe p = Lookup(k.board, k.hand, k.mode);
        c.pn = p.pn;
        c.dn = p.dn;
        if (!p.found && opt.deep_pn > 0) {
          // Unknown positions look harder the deeper they are (deep df-pn):
          // keeps the search from diving endlessly along lines whose pn stays 1.
          c.pn = 1 + static_cast<PnDn>(ply + 1) / static_cast<PnDn>(opt.deep_pn);
        }
        c.len = p.len;
        c.rep = p.rep;
        c.ph = p.ph;
      }
      for (int i = b; i < e; ++i) live.push_back(i);
      ++active;
      active_end = e;
    };
    if (cached) {
      for (int i = 0; i < active_end; ++i) live.push_back(i);
    } else {
      activate();
    }

    PnDn pn = 1, dn = 1;
    int best = 0;
    for (;;) {
      std::fill(gmax.begin(), gmax.end(), 0);
      PnDn inactive_pn = 0, inactive_dn = 0;
      for (int st = active; st < si.nstages; ++st) {
        inactive_pn = Add(inactive_pn, si.est_pn[st]);
        inactive_dn = Add(inactive_dn, si.est_dn[st]);
      }
      if (or_node) {
        PnDn best_v = kInf + 1, second_v = kInf + 1;
        best = -1;
        for (std::size_t k = 0; k < live.size();) {
          const int i = live[k];
          const Child& c = ch[i];
          if (c.dn == 0) {  // refuted: contributes nothing any more
            live[k] = live.back();
            live.pop_back();
            continue;
          }
          ++k;
          const PnDn v = c.pn == 0 ? 0 : Add(c.pn, c.cost);
          if (v < best_v || (v == best_v && Tie(ch, i) < Tie(ch, best))) { second_v = best_v; best_v = v; best = i; }
          else if (v < second_v) second_v = v;
          gmax[c.group] = std::max(gmax[c.group], c.dn);
        }
        const PnDn virt = active < si.nstages ? si.est_pn[active] : kInf + 1;
        pn = std::min(best_v, virt);
        if (pn > kInf) pn = kInf;
        dn = inactive_dn;
        for (int g = 0; g < groups; ++g) dn = Add(dn, gmax[g]);
        if (pn == 0) dn = kInf;
        if (ply == 0) {
          // The root never returns before the end (its thresholds are
          // infinite): keep its values for the progress output.
          live_root_pn = pn;
          live_root_dn = dn;
          live_root_best = best >= 0 && !ch[best].pass ? ch[best].move : MOVE_NONE;
        }
        if (pn >= thpn || dn >= thdn || stop) break;
        if (virt < best_v) { activate(); continue; }
        second_v = std::min(second_v, virt);
        Child& c = ch[best];
        const PnDn sec = second_v > kInf ? kInf : second_v;
        PnDn cthpn = std::min(thpn, Add(Epsilon(sec, mode), 1));
        cthpn = cthpn >= kInf ? kInf : cthpn - c.cost;
        PnDn others = inactive_dn;
        for (int g = 0; g < groups; ++g)
          if (g != c.group) others = Add(others, gmax[g]);
        const PnDn cthdn = thdn >= kInf ? kInf : thdn - others;
        SearchChild(ply, mode, c, cthpn, cthdn);
        if (c.pn == 0) killers[ply] = c.move;
      } else {
        PnDn best_v = kInf + 1, second_v = kInf + 1;
        best = -1;
        for (std::size_t k = 0; k < live.size();) {
          const int i = live[k];
          const Child& c = ch[i];
          if (c.pn == 0) {  // proven reply: contributes nothing any more
            live[k] = live.back();
            live.pop_back();
            continue;
          }
          ++k;
          const PnDn v = c.dn;
          if (v < best_v || (v == best_v && Tie(ch, i) < Tie(ch, best))) { second_v = best_v; best_v = v; best = i; }
          else if (v < second_v) second_v = v;
          gmax[c.group] = std::max(gmax[c.group], c.pn);
        }
        PnDn active_pn = 0;
        for (int g = 0; g < groups; ++g) active_pn = Add(active_pn, gmax[g]);
        if (active_pn == 0 && active < si.nstages) { activate(); continue; }
        // Not yet activated replies act as one virtual child with a small dn
        // estimate: when it is the easiest to disprove, activate them.
        const PnDn virt = active < si.nstages ? si.est_dn[active] : kInf + 1;
        dn = std::min(best_v, virt);
        if (dn > kInf) dn = kInf;
        pn = Add(active_pn, inactive_pn);
        if (dn == 0) pn = kInf;
        if (pn >= thpn || dn >= thdn || stop) break;
        if (virt < best_v) { activate(); continue; }
        second_v = std::min(second_v, virt);
        Child& c = ch[best];
        if (!c.pass && !c.sim) {
          c.sim = 1;
          // Replay a proven sibling's proof (the pass first) on this reply.
          int sidx = -1;
          for (int i = 0; i < active_end; ++i) {
            if (ch[i].pn != 0) continue;
            if (ch[i].pass) { sidx = i; break; }
            if (sidx < 0) sidx = i;
          }
          bool sim_ok = false;
          if (sidx >= 0) {
            const bool by_pass = ch[sidx].pass != 0;
            sim_ok = TrySimulate(c, by_pass ? MOVE_NULL : ch[sidx].move, mode, by_pass ? kModeMate : mode);
            if (sim_ok) continue;
          }
        }
        const PnDn sec = second_v > kInf ? kInf : second_v;
        const PnDn cthdn = std::min(thdn, Add(Epsilon(sec, mode), 1));
        PnDn others = inactive_pn;
        for (int g = 0; g < groups; ++g)
          if (g != c.group) others = Add(others, gmax[g]);
        const PnDn cthpn = thpn >= kInf ? kInf : thpn - others;
        SearchChild(ply, mode, c, cthpn, cthdn);
      }
    }
    if (slot != nullptr) {
      if (pn == 0 || dn == 0 || n < kCacheMinChildren || n > kCacheMaxChildren) {
        if (cached) slot->valid = 0;  // leave another node's entry alone
      } else {
        if (!cached) {
          std::copy(ch.begin(), ch.end(), slot->ch);
          std::copy(fulls, fulls + n, slot->full);
          ch.p = slot->ch;
          fulls = slot->full;
        }
        slot->valid = 1;
        slot->board = board;
        slot->hand = static_cast<std::uint32_t>(hand);
        slot->mode = mode;
        slot->n = static_cast<std::uint16_t>(n);
        slot->groups = static_cast<std::uint16_t>(groups);
        slot->active = static_cast<std::uint8_t>(active);
        slot->active_end = static_cast<std::uint16_t>(active_end);
        slot->si = si;
      }
      slot->in_use = 0;
    }
    ch.n = active_end;

    // Result, best move, proof length, repetition dependence.
    std::uint16_t len = 0;
    Move best_move = MOVE_NONE;
    bool rep = false;
    Hand store_hand = hand;
    if (pn == 0) {
      if (or_node) {
        int bl = 0x7fffffff;
        Hand child_ph = hand;
        for (const auto& c : ch)
          if (c.pn == 0 && c.len < bl) { bl = c.len; best_move = c.move; child_ph = c.ph; }
        len = static_cast<std::uint16_t>(std::min(65535, bl + 1));
        store_hand = OrProofHand(child_ph, best_move, pos);
      } else {
        int ml = -1;
        Hand uni = static_cast<Hand>(0);
        for (const auto& c : ch) {
          uni = HandUnion(uni, c.ph);
          if (c.pass) continue;
          if (static_cast<int>(c.len) > ml) { ml = c.len; best_move = c.move; }
        }
        len = static_cast<std::uint16_t>(std::min(65535, ml + 1));
        store_hand = AndProofHand(uni, hand, pos.hand_of(def));
      }
    } else if (dn == 0) {
      if (or_node) {
        for (const auto& c : ch) rep |= c.rep != 0;
      } else {
        rep = true;
        for (const auto& c : ch)
          if (c.dn == 0 && !c.rep) { rep = false; best_move = c.move; break; }
        if (rep)
          for (const auto& c : ch)
            if (c.dn == 0) { best_move = c.move; break; }
      }
    } else if (best >= 0) {
      best_move = ch[best].move;
    }
    Store(board, store_hand, mode, pn, dn, len, best_move, rep, nodes - nodes_at_start + 1);
    return NodeResult{pn, dn, len, rep, store_hand};
  }

  void SearchChild(int ply, std::uint8_t mode, Child& c, PnDn thpn, PnDn thdn) {
    StateInfo& st = states[ply];
    std::uint8_t cmode = mode;
    const Move m = c.move;
    if (c.pass) {
      pos.do_null_move(st);
      cmode = kModeMate;
    } else {
      pos.do_move(m, st);
    }
    PushPath(pos.key());
    const NodeResult r = Search(ply + 1, cmode, thpn, thdn, c.pass ? MOVE_NONE : m);
    PopPath();
    if (c.pass) pos.undo_null_move();
    else pos.undo_move(m);
    c.pn = r.pn;
    c.dn = r.dn;
    c.len = r.len;
    c.rep = r.rep;
    c.ph = r.ph;
  }

  // ---- proof simulation --------------------------------------------------
  // A defender's reply is often answered by the same mate that proved a
  // sibling (typically the virtual pass: the threat's mate still works). We
  // replay the sibling's proof on the reply: attacker moves are taken from the
  // TT of the sibling ("shadow") line, every defender move is generated in the
  // real position. Only checking sequences are replayed, and every node of the
  // real line is checked on the real board, so a success is a real proof.
  static constexpr int kSimMaxDepth = 256;
  const std::uint64_t kSimBudget = static_cast<std::uint64_t>(opt.sim_budget);
  Position shadow;
  std::vector<StateInfo> sim_states = std::vector<StateInfo>(kSimMaxDepth + 4);
  std::vector<StateInfo> shadow_states = std::vector<StateInfo>(kSimMaxDepth + 4);
  std::uint64_t sim_budget = 0;
  std::uint64_t sim_nodes = 0;  // nodes replayed (effort of the stored proofs)

  // A proven position for the replay: a mate proof keeps the replayed line a
  // pure mate (stored in mate mode); a hisshi proof only proves hisshi.
  bool SimLeaf(std::uint8_t mode, std::uint16_t& len, bool& pure) {
    const Key board = BoardKey(pos);
    const Hand hand = pos.hand_of(atk);
    const Probe pm = Lookup(board, hand, kModeMate);
    if (pm.pn == 0) { len = pm.len; return true; }
    if (mode != kModeMate) {
      const Probe p = Lookup(board, hand, mode);
      if (p.pn == 0) { len = p.len; pure = false; return true; }
    }
    return false;
  }

  bool SimOr(int d, std::uint8_t mode, std::uint8_t smode, std::uint16_t& len, bool& pure) {
    if (sim_budget == 0 || d >= kSimMaxDepth) return false;
    const std::uint64_t sim_nodes0 = sim_nodes;
    --sim_budget;
    ++sim_nodes;
    ++nodes;
    const Key board = BoardKey(pos);
    const Hand hand = pos.hand_of(atk);
    if (SimLeaf(mode, len, pure)) return true;
    if (!pos.in_check()) {
      const Move mate = Mate::mate_1ply(pos);
      if (mate != MOVE_NONE) {
        len = 1;
        return true;
      }
    }
    const Probe sp = Lookup(BoardKey(shadow), shadow.hand_of(atk), smode);
    if (sp.pn != 0 || sp.best == 0) return false;
    const Move sm = shadow.to_move(Move16(sp.best));
    if (sm == MOVE_NONE || !shadow.pseudo_legal_s<true>(sm) || !shadow.legal(sm)) return false;
    const Move m = pos.to_move(Move16(sp.best));
    if (m == MOVE_NONE || !pos.pseudo_legal_s<true>(m) || !pos.legal(m) || !pos.gives_check(m)) return false;
    pos.do_move(m, sim_states[d]);
    shadow.do_move(sm, shadow_states[d]);
    std::uint16_t clen = 0;
    bool cpure = true;
    const bool ok = SimAnd(d + 1, mode, smode, clen, cpure);
    shadow.undo_move(sm);
    pos.undo_move(m);
    if (!ok) return false;
    pure = pure && cpure;
    len = static_cast<std::uint16_t>(std::min(65535, clen + 1));
    // The replayed subtree's size is its effort (the entry stands for a whole proof).
    Store(board, hand, cpure ? kModeMate : mode, 0, kInf, len, m, false, sim_nodes - sim_nodes0 + 1);
    return true;
  }

  bool SimAnd(int d, std::uint8_t mode, std::uint8_t smode, std::uint16_t& len, bool& pure) {
    if (sim_budget == 0 || d >= kSimMaxDepth) return false;
    const std::uint64_t sim_nodes0 = sim_nodes;
    --sim_budget;
    ++sim_nodes;
    ++nodes;
    const Key board = BoardKey(pos);
    const Hand hand = pos.hand_of(atk);
    int maxlen = -1;
    bool all_pure = true;
    // Evasions the shadow cannot mirror must already be proven: check them
    // first so that a failing replay stops before any deeper work.
    std::pair<Move, Move> mapped_moves[kMaxChildren];
    int nmapped = 0;
    for (const auto& em : MoveList<EVASIONS_ALL>(pos)) {
      const Move e = em.move;
      if (!pos.legal(e)) continue;
      const Move se = shadow.to_move(Move16(e));
      if (se != MOVE_NONE && shadow.pseudo_legal_s<true>(se) && shadow.legal(se)) {
        mapped_moves[nmapped++] = {e, se};
        continue;
      }
      std::uint16_t clen = 0;
      pos.do_move(e, sim_states[d]);
      const bool ok = SimLeaf(mode, clen, all_pure);
      pos.undo_move(e);
      if (!ok) return false;
      maxlen = std::max<int>(maxlen, clen);
    }
    for (int i = 0; i < nmapped; ++i) {
      const auto [e, se] = mapped_moves[i];
      std::uint16_t clen = 0;
      pos.do_move(e, sim_states[d]);
      shadow.do_move(se, shadow_states[d]);
      const bool ok = SimOr(d + 1, mode, smode, clen, all_pure);
      shadow.undo_move(se);
      pos.undo_move(e);
      if (!ok) return false;
      maxlen = std::max<int>(maxlen, clen);
    }
    pure = pure && all_pure;
    len = static_cast<std::uint16_t>(std::min(65535, maxlen + 1));
    Store(board, hand, all_pure ? kModeMate : mode, 0, kInf, len, MOVE_NONE, false, sim_nodes - sim_nodes0 + 1);
    return true;
  }

  // Try to prove child c of the current (defender-to-move) node by replaying
  // the proof of the sibling reached by shadow_move (MOVE_NULL = the pass).
  StateInfo shadow_root_st, shadow_move_st;
  Key shadow_base = 0;  // the node and sibling the shadow is currently set up for

  bool TrySimulate(Child& c, Move shadow_move, std::uint8_t mode, std::uint8_t smode) {
    // Cheap pre-check: the shadow's first attack must be a legal check after
    // this reply.
    {
      const Key sboard = shadow_move == MOVE_NULL ? (BoardKey(pos) ^ side_key) : pos.board_key_after(shadow_move);
      const Probe sp = Lookup(sboard, pos.hand_of(atk), smode);
      if (sp.pn != 0 || sp.best == 0) return false;
      StateInfo sr;
      pos.do_move(c.move, sr);
      const Move m = pos.to_move(Move16(sp.best));
      const bool first_ok = m != MOVE_NONE && pos.pseudo_legal_s<true>(m) && pos.legal(m) && pos.gives_check(m);
      pos.undo_move(c.move);
      if (!first_ok) return false;
    }
    const Key base = pos.key() ^ (static_cast<Key>(shadow_move) * 0x9E3779B97F4A7C15ULL);
    if (base != shadow_base) {
      shadow.set(pos.sfen(), &shadow_root_st, pos.this_thread());
      if (shadow_move == MOVE_NULL) shadow.do_null_move(shadow_move_st);
      else shadow.do_move(shadow_move, shadow_move_st);
      shadow_base = base;
    }
    StateInfo sr;
    pos.do_move(c.move, sr);
    sim_budget = kSimBudget;
    std::uint16_t len = 0;
    bool pure = true;
    const bool ok = SimOr(0, mode, smode, len, pure);
    pos.undo_move(c.move);
    if (ok) {
      c.pn = 0;
      c.dn = kInf;
      c.len = len;
      c.rep = 0;
      c.ph = pos.hand_of(atk);
    }
    return ok;
  }

  // Root search loop.
  NodeResult Run(std::uint8_t mode) {
    PushPath(pos.key());
    NodeResult r{1, 1, 0, false};
    while (!stop) {
      r = Search(0, mode, kInf, kInf, MOVE_NONE);
      root_pn = r.pn;
      root_dn = r.dn;
      if (r.pn == 0 || r.dn == 0) break;
      if (r.pn >= kInf || r.dn >= kInf) break;  // saturated: no progress possible
    }
    PopPath();
    return r;
  }

  // ---- helpers for PV / verification ------------------------------------
  Probe ProbeChild(const Child& c, std::uint8_t mode) {
    Key cb, cf;
    Hand chand;
    std::uint8_t cm;
    ChildKey(c, mode, cb, chand, cm, cf);
    return Lookup(cb, chand, cm);
  }

  // Search the current position with a node budget (used after the main
  // search to fill gaps). Returns the proof status.
  NodeResult SubSearch(std::uint8_t mode, std::uint64_t budget) {
    NodeResult r{1, 1, 0, false};
    const bool saved_stop = stop;
    stop = false;
    node_cap = nodes + budget;
    next_check = nodes;  // re-evaluate limits immediately
    PushPath(pos.key());
    while (!stop) {
      r = Search(static_cast<int>(path.size()), mode, kInf, kInf, MOVE_NONE);
      if (r.pn == 0 || r.dn == 0) break;
    }
    PopPath();
    node_cap = 0;
    stop = saved_stop;
    return r;
  }
};

// ---------------------------------------------------------------------------
// PV extraction and proof verification

namespace {

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
    raw_path.push_back(pos.key());
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
  static constexpr int kTries = 4;                   // refuting replies tried per defender node

  explicit DisproofVerifier(SearchImpl& si) : s(si), pos(si.pos) { st.resize(4096); }

  static Key Salt(Key k, std::uint8_t mode) { return mode ? k ^ 0x9e3779b97f4a7c15ULL : k; }

  // True when the attacker cannot force hisshi (mate below a pass) here.
  // `cyc` is set when the result relies on a repetition with the path.
  bool Refuted(int depth, std::uint8_t mode, bool& cyc) {
    const Key k = Salt(pos.key(), mode);
    if (done.Contains(k)) return true;
    if (std::find(on_path.begin(), on_path.end(), k) != on_path.end()) { cyc = true; return true; }
    if (depth >= 4000) { error = "too deep"; return false; }
    ++visited;
    on_path.push_back(k);
    bool my_cyc = false;
    const bool ok = pos.side_to_move() == s.atk ? OrRefuted(depth, mode, my_cyc) : AndRefuted(depth, mode, my_cyc);
    on_path.pop_back();
    if (ok && !my_cyc) done.Insert(k);
    cyc |= my_cyc;
    return ok;
  }

  // Attacker to move: every legal move (every check below a pass) must fail.
  bool OrRefuted(int depth, std::uint8_t mode, bool& cyc) {
    for (const auto& em : MoveList<LEGAL_ALL>(pos)) {
      const Move m = em.move;
      if (mode == kModeMate && !pos.gives_check(m)) continue;
      pos.do_move(m, st[depth]);
      const bool ok = Refuted(depth + 1, mode, cyc);
      pos.undo_move(m);
      if (!ok) {
        if (error.empty()) error = "attack " + to_usi_string(m) + " not refuted";
        return false;
      }
    }
    return true;
  }

  // Probe of a defender's reply (MOVE_NULL = the pass).
  Probe ProbeReply(Move r, std::uint8_t mode) {
    if (r == MOVE_NULL) return s.Lookup(BoardKey(pos) ^ s.side_key, pos.hand_of(s.atk), kModeMate);
    Child c{};
    c.move = r;
    return s.ProbeChild(c, mode);
  }

  bool TryReply(int depth, std::uint8_t mode, Move r, bool& cyc) {
    bool c = false;
    bool ok;
    if (r == MOVE_NULL) {
      pos.do_null_move(st[depth]);
      ok = Refuted(depth + 1, kModeMate, c);
      pos.undo_null_move();
    } else {
      pos.do_move(r, st[depth]);
      ok = Refuted(depth + 1, mode, c);
      pos.undo_move(r);
    }
    if (ok) cyc |= c;
    return ok;
  }

  // Defender to move: one reply must refute.
  bool AndRefuted(int depth, std::uint8_t mode, bool& cyc) {
    const bool check = pos.in_check();
    std::vector<Move> replies;
    if (!check) replies.push_back(MOVE_NULL);
    for (const auto& em : MoveList<LEGAL_ALL>(pos)) replies.push_back(em.move);
    if (check && replies.empty()) { error = "mated"; return false; }
    std::vector<Move> tried;
    for (int round = 0; round < 2; ++round) {
      // Replies disproven in the TT: path-independent ones first.
      std::vector<std::pair<int, Move>> cands;
      for (Move r : replies) {
        if (std::find(tried.begin(), tried.end(), r) != tried.end()) continue;
        const Probe p = ProbeReply(r, mode);
        if (p.dn == 0) cands.emplace_back(p.rep ? 1 : 0, r);
      }
      std::stable_sort(cands.begin(), cands.end(),
                       [](const auto& a, const auto& b) { return a.first < b.first; });
      for (const auto& [rep, r] : cands) {
        if (static_cast<int>(tried.size()) >= kTries * 2) break;
        tried.push_back(r);
        std::string saved = error;
        if (TryReply(depth, mode, r, cyc)) { error.clear(); return true; }
        if (saved.empty() && !error.empty()) saved = error;
        error = saved;
      }
      if (round == 1) break;
      // Search this node again to find (another) disproof.
      ++searches;
      const NodeResult nr = s.SubSearch(mode, kBudget);
      if (nr.dn != 0) break;
    }
    if (error.empty()) error = "no refuting reply";
    return false;
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

struct PvBuilder {
  Solver& solver;
  const Options& opt;
  Color atk;
  static constexpr std::uint64_t kMateProbe = 10000;  // "is this reply simply mated?"
  static constexpr int kMaxDepth = 9;                 // nested judgements (a line has at most 8 squares)


  AnswerPool* pool = nullptr;  // worker threads for the shorter-answer attempts (several search threads)
  AtomicHandSet* verified = nullptr;  // positions checked by the proof verification (reused by VerifyLine)
  std::unordered_map<Key, bool> mated_memo;
  std::unordered_set<Key> not_known_mated;  // neither in the TT nor a one-ply mate
  std::uint64_t probe_calls = 0, probe_nodes = 0, probe_mated = 0, reprove_calls = 0, reprove_nodes = 0;

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
        const std::uint64_t n0 = h.nodes;
        h.SubSearch(kModeMate, kMateProbe);
        pm = h.Lookup(BoardKey(p), p.hand_of(atk), kModeMate);
        ++probe_calls;
        probe_nodes += h.nodes - n0;
        if (pm.pn == 0) ++probe_mated;
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
        const std::uint64_t n0 = h.nodes;
        ++reprove_calls;
        if (self.pn != 0) {
          // Usually a plain mate: try the cheaper mate search first.
          if (h.SubSearch(kModeMate, kMateProbe).pn != 0) h.SubSearch(kModeHisshi, ReproveBudget());
          reprove_nodes += h.nodes - n0;
        } else {
          // Proven, but the children's entries were evicted: prove again the
          // child reached by the stored best move.
          const Move b = p.to_move(Move16(self.best));
          if (b == MOVE_NONE || !p.pseudo_legal_s<true>(b) || !p.legal(b)) break;
          StateInfo st;
          p.do_move(b, st);
          h.SubSearch(kModeHisshi, ReproveBudget());
          reprove_nodes += h.nodes - n0;
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
  std::uint64_t shorten_nodes = 0, shorten_found = 0;
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
        if (t.result) { any = true; ++shorten_found; }
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
        ++shorten_found;
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

}  // namespace

Result Solver::Solve(Position& root, const Limits& limits, const Options& opt,
                     bool (*should_stop)()) {
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
  // Helper threads search the same root with a shared TT; each uses a
  // different child order and search parameters (cost, staging estimates,
  // 1+epsilon) for diversity.
  const int nthreads = std::max(1, opt.threads);
  std::atomic<bool> done{false};
  std::vector<std::thread> helpers;
  std::vector<std::uint64_t> helper_nodes(nthreads, 0);
  const std::string root_sfen = root.sfen();
  shared_ = nthreads > 1;
  // Children caches take about a quarter of the hash size (a quarter of it
  // for the main thread, the rest shared by the helpers), at most the
  // configured numbers of slots.
  const std::size_t hash_bytes = table_.size() * sizeof(Cluster);
  const std::size_t slot_bytes = sizeof(SearchImpl::CacheSlot);
  const std::size_t cache_bytes = hash_bytes / 4;
  auto slots = [&](std::size_t bytes, int cap) {
    return static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(cap), std::max<std::size_t>(64, bytes / slot_bytes)));
  };
  const int main_slots = slots(nthreads > 1 ? cache_bytes / 4 : cache_bytes, opt.cache_slots);
  const int helper_slots = nthreads > 1 ? slots(cache_bytes * 3 / 4 / static_cast<std::size_t>(nthreads - 1),
                                                std::min(opt.cache_slots, opt.helper_cache_slots)) : 0;
  for (int t = 1; t < nthreads; ++t) {
    helpers.emplace_back([&, t]() {
      Position hp;
      StateInfo hsi;
      hp.set(root_sfen, &hsi, root.this_thread());
      static const int kLazyDn[] = {64, 32, 128, 48, 96, 80, 40, 160};
      static const int kStageDn[] = {16, 12, 24, 16, 20, 14, 16, 32};
      Options hopt = opt;
      static const int kCost[] = {2, 3, 2, 2, 3, 2};
      static const int kCheckDn[] = {16, 8, 32, 16, 64, 12, 24};
      static const int kEps[] = {0, 50, 100, 25, 200};
      hopt.lazy_dn = kLazyDn[t % 8];
      hopt.and_stage_dn = kStageDn[t % 8];
      hopt.non_check_cost = kCost[t % 6];
      hopt.check_stage_dn = kCheckDn[t % 7];
      hopt.eps_percent = kEps[t % 5];
      if (opt.deep_pn > 0) {
        static const int kDeep[] = {16, 8, 24, 12, 32, 0, 20, 10, 16};
        hopt.deep_pn = kDeep[t % 9];
      }
      hopt.cache_slots = helper_slots;
      Limits hlim = limits;
      hlim.pv_interval_ms = 0;
      SearchImpl h(*this, hp, hopt, hlim, should_stop, t, &done);
      const NodeResult hr = h.Run(kModeHisshi);
      helper_nodes[t] = h.nodes;
      if (hr.pn == 0 || hr.dn == 0) done = true;
    });
  }
  // A single search has no helpers that vary the parameters: it uses the
  // 1+epsilon trick against switching between near-equal alternatives
  // (the helpers' main means of escaping such seesaws).
  Options main_opt = opt;
  if (nthreads == 1 && main_opt.eps_percent == 0) main_opt.eps_percent = opt.single_thread_eps;
  main_opt.cache_slots = main_slots;
  total_nodes_ = 0;
  SearchImpl s(*this, root, main_opt, limits, should_stop, 0, nthreads > 1 ? &done : nullptr);
  s.report = true;
  s.root_sfen = root_sfen;
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
    }
  }
  res.pn = r.pn;
  res.dn = r.dn;
  res.nodes = s.nodes;
  for (std::uint64_t hn : helper_nodes) res.nodes += hn;
  res.elapsed_ms = s.ElapsedMs();
  if (r.pn == 0) {
    res.status = Status::kProven;
  } else if (r.dn == 0 && !r.rep) {
    // Not hisshi: accepted only after the independent check with every
    // legal attack (a disproof resting on a repetition or the depth limit
    // of the path is not a result).
    res.status = Status::kDisproven;
    const std::uint64_t before = s.nodes;
    const auto t0 = s.ElapsedMs();
    DisproofVerifier v(s);
    bool cyc = false;
    s.PushPath(root.key());
    const bool ok = v.Refuted(0, kModeHisshi, cyc);
    s.PopPath();
    res.verified = ok;
    res.verify_nodes = v.visited;
    std::ostringstream os;
    os << "visited=" << v.visited << " searches=" << v.searches << " research_nodes=" << (s.nodes - before)
       << " time_ms=" << (s.ElapsedMs() - t0);
    if (!ok) os << " error=" << v.error;
    res.verify_info = os.str();
    if (!ok) res.status = Status::kUnknown;
  }
  if (res.status != Status::kProven) return res;

  // The displayed answer (futile interpositions excluded), built after the
  // verification.
  std::string answer_info;
  std::unique_ptr<AtomicHandSet> verified_set;  // positions checked by the verification below
  auto build_answer = [&]() {
    const auto t0 = s.ElapsedMs();
    const std::uint64_t n0 = s.nodes;
    PvBuilder builder{*this, opt, s.atk};
    builder.verified = verified_set.get();
    std::unique_ptr<AnswerPool> pool;
    if (nthreads > 1) {
      pool = std::make_unique<AnswerPool>(*this, root_sfen, root.this_thread(), opt, limits, should_stop, nthreads, s.atk);
      builder.pool = pool.get();
    }
    res.pv = builder.Pv(s);
    pool.reset();
    answer_info = std::string(" answer=") + (builder.exact ? "longest" : "greedy") +
                  " answer_ms=" + std::to_string(s.ElapsedMs() - t0) +
                  " answer_nodes=" + std::to_string(s.nodes - n0);
  };
  s.report = false;  // verification and answer searches print no root progress
  if (limits.pv_interval_ms > 0)
    sync_cout << "info string proof found (" << s.ElapsedMs() << " ms), verifying" << sync_endl;
  if (Hashfull() > 300) KeepOnlyProofs();  // a full scan: only when the table is in use
  // Independent verification of the proof (all defences, including the
  // virtual pass after every non-check).
  {
    const std::uint64_t before = s.nodes;
    const auto t0 = s.ElapsedMs();
    bool ok = false;
    std::uint64_t visited = 0;
    std::string error;
    bool parallel_ok = false;
    // Positions verified by the parallel verification; a sequential fallback
    // reuses them and only checks what is left.
    std::unique_ptr<AtomicHandSet> shared_done;
    // Verified-position memo: about an eighth of the hash size (8-byte slots).
    int memo_log2 = 18;
    while (memo_log2 < 25 && (std::size_t(8) << (memo_log2 + 1)) <= hash_bytes / 8) ++memo_log2;
    if (nthreads > 1) {
      // Split the proof into subtrees and verify them on all threads; any
      // failure falls back to the sequential verification below.
      std::vector<VerifyTask> tasks;
      bool collected = false;
      for (int d = 2; d <= 20 && !collected; d += 2) {
        tasks.clear();
        Verifier c(s);
        c.own_log2 = 18;  // only the top of the proof is walked
        c.collect_depth = d;
        c.tasks = &tasks;
        s.PushPath(root.key());
        const bool cok = c.Verify(0, kModeHisshi);
        s.PopPath();
        if (!cok) break;
        collected = tasks.size() >= 1024u * static_cast<std::size_t>(nthreads) || d >= 20;
        if (collected || tasks.empty()) {
          visited += c.visited;
          collected = true;
        }
      }
      if (collected && tasks.empty()) {
        parallel_ok = true;  // the whole proof was checked while collecting
      } else if (collected && tasks.size() < 16u * static_cast<std::size_t>(nthreads)) {
        // Too small to be worth the threads: verify sequentially below.
      } else if (collected) {
        shared_done = std::make_unique<AtomicHandSet>(memo_log2);
        std::atomic<std::size_t> next{0};
        std::atomic<bool> failed{false};
        std::atomic<std::uint64_t> total_visited{0};
        shared_ = true;
        std::vector<std::thread> workers;
        for (int t = 0; t < nthreads; ++t) {
          workers.emplace_back([&]() {
            Options wopt = opt;
            wopt.cache_slots = 256;
            wopt.threads = 1;
            Limits wlim = limits;
            wlim.pv_interval_ms = 0;
            // One position, searcher and verifier per thread, reused for every task.
            Position wp;
            StateInfo root_st;
            wp.set(root_sfen, &root_st, root.this_thread());
            SearchImpl w(*this, wp, wopt, wlim, should_stop, 0, nullptr, s.atk);
            Verifier v(w);
            v.shared = shared_done.get();
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
              const bool vok = v.Verify(static_cast<int>(task.moves.size()), task.mode);
              w.PopPath();
              for (std::size_t j = task.moves.size(); j-- > 0;) {
                if (task.moves[j] == MOVE_NULL) wp.undo_null_move();
                else wp.undo_move(task.moves[j]);
              }
              if (!vok) failed = true;
            }
            total_visited += v.visited;
          });
        }
        for (auto& th : workers) th.join();
        shared_ = false;
        visited += total_visited;
        parallel_ok = !failed;
      }
    }
    if (parallel_ok) {
      ok = true;
    } else {
      Verifier v(s);
      if (!shared_done) shared_done = std::make_unique<AtomicHandSet>(std::min(v.own_log2, memo_log2));
      v.shared = shared_done.get();
      s.PushPath(root.key());
      ok = v.Verify(0, kModeHisshi);
      s.PopPath();
      visited = v.visited;
      error = v.error;
    }
    res.verified = ok;
    res.verify_nodes = visited;
    if (ok) verified_set = std::move(shared_done);  // kept for checking the answer's line
    std::ostringstream os;
    os << "visited=" << visited << " research_nodes=" << (s.nodes - before)
       << " time_ms=" << (s.ElapsedMs() - t0) << (parallel_ok ? " parallel" : "");
    if (!ok) os << " error=" << error;
    res.verify_info = os.str();
  }

  if (limits.pv_interval_ms > 0)
    sync_cout << "info string proof " << (res.verified ? "verified" : "NOT verified") << " (" << s.ElapsedMs()
              << " ms), building the answer" << sync_endl;
  build_answer();
  res.verify_info += answer_info;

  res.nodes = s.nodes;
  for (std::uint64_t hn : helper_nodes) res.nodes += hn;
  res.elapsed_ms = s.ElapsedMs();
  return res;
}

}  // namespace hisshi
