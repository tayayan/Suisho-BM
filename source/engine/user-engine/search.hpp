// Internal header of the hisshi solver (included by solver.cpp only):
// the df-pn+ search with the virtual pass, its move generation, the access
// to the transposition table and the proof replay.

#ifndef HISSHI_SEARCH_HPP_
#define HISSHI_SEARCH_HPP_

#include "hisshi_dfpn.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
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
namespace detail {

constexpr std::uint8_t kModeHisshi = 0;
constexpr std::uint8_t kModeMate = 1;

inline PnDn Add(PnDn a, PnDn b) {
  const PnDn s = a + b;  // both <= kInf, no overflow in 32 bits
  return s >= kInf ? kInf : s;
}

// Squares around the defending king (paper, Fig. 2). zone12: squares where
// an attacking effect counts as an attack; zone3: squares two ranks in front
// of the king that only a knight effect counts for.
inline Bitboard g_zone12[SQ_NB_PLUS1][COLOR_NB];
inline Bitboard g_zone3[SQ_NB_PLUS1][COLOR_NB];
inline bool g_tables_ready = false;

inline void InitTables() {
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
    // Live values of the root (kept by the root's OR loop, see Search).
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

  // The working state of one node: its children (in a per-ply buffer or a
  // children-cache slot), the activated stages and the children that can
  // still change the node's values.
  struct NodeWork {
    ChildList ch;
    Key* fulls = nullptr;     // full keys of the children (repetition checks)
    StageInfo si;
    int groups = 0;
    int active = 0;           // number of active stages
    int active_end = 0;       // children [0, active_end) are active
    int n = 0;                // children generated so far
    std::vector<PnDn>* gmax;  // per group: the maximum of the summed number
    std::vector<int>* live;   // undecided active children
    CacheSlot* slot = nullptr;
    bool cached = false;      // the children came from the cache
  };

  // An attacker node mated in one: the proof without a TT entry (found again
  // at once; most proofs are such leaves). The mated child needs no pieces
  // except those that keep the defender from getting new kinds of drops.
  bool MateInOne(Hand hand, NodeResult& r) {
    const Move mate = Mate::mate_1ply(pos);
    if (mate == MOVE_NONE) return false;
    Hand after = hand;
    if (is_drop(mate)) {
      sub_hand(after, move_dropped_piece(mate));
    } else if (pos.piece_on(to_sq(mate)) != NO_PIECE) {
      add_hand(after, raw_type_of(pos.piece_on(to_sq(mate))));
    }
    const Hand child_ph = AndProofHand(static_cast<Hand>(0), after, pos.hand_of(def));
    r = NodeResult{0, kInf, 1, false, OrProofHand(child_ph, mate, pos)};
    return true;
  }

  // The children of the node: those of a recently visited node from the
  // children cache (path-dependent values refreshed), else generated.
  void OpenNode(NodeWork& w, Key board, Hand hand, std::uint8_t mode, bool or_node, Move last, int ply) {
    if (!cache.empty()) {
      const std::uint64_t h = static_cast<std::uint64_t>(board) ^
                              (static_cast<std::uint64_t>(hand) * 0x9E3779B97F4A7C15ULL) ^ mode;
      w.slot = &cache[(h >> 7) % cache.size()];
      if (w.slot->in_use) w.slot = nullptr;  // used by an ancestor
      w.cached = w.slot != nullptr && w.slot->valid && w.slot->board == board &&
                 w.slot->hand == static_cast<std::uint32_t>(hand) && w.slot->mode == mode;
    }
    if (w.cached) {
      CacheSlot& slot = *w.slot;
      w.si = slot.si;
      w.groups = slot.groups;
      w.active = slot.active;
      w.active_end = slot.active_end;
      // Work on the slot's children in place.
      slot.in_use = 1;
      w.ch.p = slot.ch;
      w.ch.n = slot.n;
      w.ch.cap = kCacheMaxChildren;
      w.fulls = slot.full;
      for (int i = 0; i < w.active_end; ++i) {
        Child& c = w.ch[i];
        if (InPath(w.fulls[i])) {
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
      w.ch.p = child_buf[ply].data();
      w.ch.n = 0;
      w.ch.cap = kMaxChildren;
      w.fulls = full_buf[ply].data();
      w.groups = Expand(w.ch, or_node, mode, last, ply, w.si);
    }
    w.n = static_cast<int>(w.ch.size());
  }

  // Activates the next stage of children: generates the lazily expanded
  // moves when their stage comes, then looks the stage's children up.
  void Activate(NodeWork& w, std::uint8_t mode, int ply) {
    if (w.si.lazy && w.active == 1) {
      if (w.si.lazy == 2) {
        ExpandLazyAnd(w.ch, w.groups, w.si);
      } else {
        ExpandLazy(w.ch, w.groups);
        w.si.begin[2] = static_cast<int>(w.ch.size());
        w.si.lazy = 0;
      }
      w.n = static_cast<int>(w.ch.size());
      w.gmax->resize(w.groups);
    }
    const int b = w.si.begin[w.active], e = w.si.begin[w.active + 1];
    // Compute the child keys and prefetch their clusters first so that the
    // (mostly missing) lookups overlap.
    std::vector<ChildKeyInfo>& keys = key_buf[ply];
    keys.resize(e - b);
    for (int i = b; i < e; ++i) {
      ChildKeyInfo& k = keys[i - b];
      ChildKey(w.ch[i], mode, k.board, k.hand, k.mode, k.full);
      w.fulls[i] = k.full;
      _mm_prefetch(reinterpret_cast<const char*>(GetCluster(NonZero(k.board))), _MM_HINT_T0);
    }
    for (int i = b; i < e; ++i) {
      Child& c = w.ch[i];
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
    for (int i = b; i < e; ++i) w.live->push_back(i);
    ++w.active;
    w.active_end = e;
  }

  // Before searching reply c of a defender node: replay a proven sibling's
  // proof (the pass first) on it, once. True when that proved c.
  bool TryReplay(NodeWork& w, Child& c, std::uint8_t mode) {
    if (c.pass || c.sim) return false;
    c.sim = 1;
    int sidx = -1;
    for (int i = 0; i < w.active_end; ++i) {
      if (w.ch[i].pn != 0) continue;
      if (w.ch[i].pass) { sidx = i; break; }
      if (sidx < 0) sidx = i;
    }
    if (sidx < 0) return false;
    const bool by_pass = w.ch[sidx].pass != 0;
    return TrySimulate(c, by_pass ? MOVE_NULL : w.ch[sidx].move, mode, by_pass ? kModeMate : mode);
  }

  // Leaves the node: keeps its children in the cache slot for the next
  // visit (unless decided, or too few or too many to be worth it).
  void CloseNode(NodeWork& w, Key board, Hand hand, std::uint8_t mode, PnDn pn, PnDn dn) {
    if (w.slot != nullptr) {
      CacheSlot& slot = *w.slot;
      if (pn == 0 || dn == 0 || w.n < kCacheMinChildren || w.n > kCacheMaxChildren) {
        if (w.cached) slot.valid = 0;  // leave another node's entry alone
      } else {
        if (!w.cached) {
          std::copy(w.ch.begin(), w.ch.end(), slot.ch);
          std::copy(w.fulls, w.fulls + w.n, slot.full);
          w.ch.p = slot.ch;
          w.fulls = slot.full;
        }
        slot.valid = 1;
        slot.board = board;
        slot.hand = static_cast<std::uint32_t>(hand);
        slot.mode = mode;
        slot.n = static_cast<std::uint16_t>(w.n);
        slot.groups = static_cast<std::uint16_t>(w.groups);
        slot.active = static_cast<std::uint8_t>(w.active);
        slot.active_end = static_cast<std::uint16_t>(w.active_end);
        slot.si = w.si;
      }
      slot.in_use = 0;
    }
    w.ch.n = w.active_end;
  }

  // The node's result from its active children: the best move, the proof
  // length and proof pieces of a proof, the path dependence of a disproof;
  // stored in the TT with the search effort spent.
  NodeResult Conclude(const NodeWork& w, Key board, Hand hand, std::uint8_t mode, bool or_node, PnDn pn, PnDn dn,
                      int best, std::uint64_t effort) {
    const ChildList& ch = w.ch;
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
    Store(board, store_hand, mode, pn, dn, len, best_move, rep, effort);
    return NodeResult{pn, dn, len, rep, store_hand};
  }

  NodeResult Search(int ply, std::uint8_t mode, PnDn thpn, PnDn thdn, Move last) {
    ++nodes;
    if (ply > seldepth) seldepth = ply;
    if (nodes >= next_check) CheckLimits();
    const bool or_node = pos.side_to_move() == atk;
    const Key board = BoardKey(pos);
    const Hand hand = pos.hand_of(atk);
    const std::uint64_t nodes_at_start = nodes;

    if (ply >= opt.max_ply) return NodeResult{kInf, 0, 0, true, hand};

    // The node's own TT entry: return at once when it is already decided or
    // above the thresholds (e.g. values changed through a transposition).
    const Probe self = Lookup(board, hand, mode);
    if (self.pn == 0 || self.dn == 0 || self.pn >= thpn || self.dn >= thdn)
      return NodeResult{self.pn, self.dn, self.len, self.rep, self.pn == 0 ? self.ph : hand};

    // A node with its own TT entry was expanded before; its one-ply mate
    // check already failed then.
    NodeResult mated;
    if (or_node && !self.found && !pos.in_check() && MateInOne(hand, mated)) return mated;

    NodeWork w;
    w.gmax = &gmax_buf[ply];
    w.live = &live_buf[ply];
    OpenNode(w, board, hand, mode, or_node, last, ply);
    if (w.n == 0 && !w.si.lazy) {
      if (or_node) {
        Store(board, hand, mode, kInf, 0, 0, MOVE_NONE, false, 1);
        return NodeResult{kInf, 0, 0, false, hand};
      }
      const Hand ph = AndProofHand(static_cast<Hand>(0), hand, pos.hand_of(def));
      Store(board, ph, mode, 0, kInf, 0, MOVE_NONE, false, 1);
      return NodeResult{0, kInf, 0, false, ph};
    }

    std::vector<PnDn>& gmax = *w.gmax;
    std::vector<int>& live = *w.live;
    ChildList& ch = w.ch;
    StageInfo& si = w.si;
    gmax.assign(w.groups, 0);
    // Final children (a proven reply of a defender node, a refuted attack of
    // an attacker node) are dropped from `live`.
    live.clear();
    if (w.cached) {
      for (int i = 0; i < w.active_end; ++i) live.push_back(i);
    } else {
      Activate(w, mode, ply);
    }

    PnDn pn = 1, dn = 1;
    int best = 0;
    for (;;) {
      std::fill(gmax.begin(), gmax.end(), 0);
      // Stages not yet activated count with their estimates.
      PnDn inactive_pn = 0, inactive_dn = 0;
      for (int st = w.active; st < si.nstages; ++st) {
        inactive_pn = Add(inactive_pn, si.est_pn[st]);
        inactive_dn = Add(inactive_dn, si.est_dn[st]);
      }
      PnDn best_v = kInf + 1, second_v = kInf + 1;
      best = -1;
      if (or_node) {
        // pn: the smallest child (with its cost); dn: the sum over groups of
        // the groups' maxima.
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
        const PnDn virt = w.active < si.nstages ? si.est_pn[w.active] : kInf + 1;
        pn = std::min(best_v, virt);
        if (pn > kInf) pn = kInf;
        dn = inactive_dn;
        for (int g = 0; g < w.groups; ++g) dn = Add(dn, gmax[g]);
        if (pn == 0) dn = kInf;
        if (ply == 0) {
          // The root never returns before the end (its thresholds are
          // infinite): keep its values for the progress output.
          live_root_pn = pn;
          live_root_dn = dn;
          live_root_best = best >= 0 && !ch[best].pass ? ch[best].move : MOVE_NONE;
        }
        if (pn >= thpn || dn >= thdn || stop) break;
        if (virt < best_v) { Activate(w, mode, ply); continue; }
        second_v = std::min(second_v, virt);
        Child& c = ch[best];
        const PnDn sec = second_v > kInf ? kInf : second_v;
        PnDn cthpn = std::min(thpn, Add(Epsilon(sec, mode), 1));
        cthpn = cthpn >= kInf ? kInf : cthpn - c.cost;
        PnDn others = inactive_dn;
        for (int g = 0; g < w.groups; ++g)
          if (g != c.group) others = Add(others, gmax[g]);
        const PnDn cthdn = thdn >= kInf ? kInf : thdn - others;
        SearchChild(ply, mode, c, cthpn, cthdn);
        if (c.pn == 0) killers[ply] = c.move;
      } else {
        // dn: the smallest child; pn: the sum over groups of the groups'
        // maxima.
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
        for (int g = 0; g < w.groups; ++g) active_pn = Add(active_pn, gmax[g]);
        if (active_pn == 0 && w.active < si.nstages) { Activate(w, mode, ply); continue; }
        // Not yet activated replies act as one virtual child with a small dn
        // estimate: when it is the easiest to disprove, activate them.
        const PnDn virt = w.active < si.nstages ? si.est_dn[w.active] : kInf + 1;
        dn = std::min(best_v, virt);
        if (dn > kInf) dn = kInf;
        pn = Add(active_pn, inactive_pn);
        if (dn == 0) pn = kInf;
        if (pn >= thpn || dn >= thdn || stop) break;
        if (virt < best_v) { Activate(w, mode, ply); continue; }
        second_v = std::min(second_v, virt);
        Child& c = ch[best];
        if (TryReplay(w, c, mode)) continue;
        const PnDn sec = second_v > kInf ? kInf : second_v;
        const PnDn cthdn = std::min(thdn, Add(Epsilon(sec, mode), 1));
        PnDn others = inactive_pn;
        for (int g = 0; g < w.groups; ++g)
          if (g != c.group) others = Add(others, gmax[g]);
        const PnDn cthpn = thpn >= kInf ? kInf : thpn - others;
        SearchChild(ply, mode, c, cthpn, cthdn);
      }
    }
    CloseNode(w, board, hand, mode, pn, dn);
    return Conclude(w, board, hand, mode, or_node, pn, dn, best, nodes - nodes_at_start + 1);
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
    NodeResult r{1, 1, 0, false, static_cast<Hand>(0)};
    while (!stop) {
      r = Search(0, mode, kInf, kInf, MOVE_NONE);
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
    NodeResult r{1, 1, 0, false, static_cast<Hand>(0)};
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

}  // namespace detail
}  // namespace hisshi

#endif  // HISSHI_SEARCH_HPP_
