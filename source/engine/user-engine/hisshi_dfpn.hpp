#ifndef HISSHI_DFPN_HPP_
#define HISSHI_DFPN_HPP_

// Hisshi (必至) solver based on
//   長井歩「難解な必至問題を解くアルゴリズムとその実装」(GPW 2011).
//
// A single df-pn+ search. The defender may "pass" after a non-checking
// attack; below a pass only checks are generated (mate mode), so the pass
// child proves that the preceding attack was a tsumero (詰めろ). There is no
// separate mate engine: a flag in the search path switches move generation.

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../../position.h"

namespace hisshi {

// Proof/disproof numbers are 64-bit in the search (sums over many children
// grow quickly); the TT keeps them in 32 bits, clamped below infinity.
using PnDn = std::uint64_t;
constexpr PnDn kInf = 1ULL << 60;
constexpr std::uint32_t kTTInf = 0xffffffffU;
inline std::uint32_t ToTT(PnDn v) {
  return v >= kInf ? kTTInf : static_cast<std::uint32_t>(v < kTTInf - 1 ? v : kTTInf - 1);
}
inline PnDn FromTT(std::uint32_t v) { return v == kTTInf ? kInf : v; }

// Search parameters. Helper threads vary some of them (see Solver::Solve).
struct Options {
  int max_ply = 400;
  bool full_width = false;   // attacker considers every legal move (else the candidates of paper 3.1)
  int non_check_cost = 2;    // cost of a non-checking attack in pn (df-pn+, paper: 2)
  int lazy_dn = 64;          // dn estimate of the not yet generated non-checking attacks
  int lazy_pn = 1;           // pn estimate (h) of the not yet generated attacks, before their cost
  int sim_budget = 4096;     // node limit of one proof replay
  int check_stage_dn = 16;   // dn estimate of not yet activated interpositions
  int and_stage_dn = 16;     // dn estimate of not yet activated defences (the pass gets a head start)
  int cache_slots = 4096;    // children cache of the main thread (at most)
  int helper_cache_slots = 1024;  // children cache of each helper thread (at most)
  int eps_percent = 0;       // 1+epsilon trick for child thresholds (percent)
  int single_thread_eps = 50;  // eps_percent of a search with one thread (no helpers to vary it)
  int deep_pn = 16;          // unknown children start with pn 1 + depth / deep_pn (0: off)
  int threads = 1;           // search threads sharing the TT
  // Futile interpositions (無駄合い) left out of the displayed answer: 0 none
  // (every interposition is a defence), 1 the futility rules with the chains
  // of interpositions judged from their end.
  int futile = 0;
};

enum class Status { kUnknown, kProven, kDisproven };

struct Result {
  Status status = Status::kUnknown;
  PnDn pn = 1, dn = 1;
  std::uint64_t nodes = 0;
  std::uint64_t elapsed_ms = 0;
  std::vector<Move> pv;
  bool verified = false;
  std::string verify_info;
};

struct Limits {
  std::uint64_t nodes = 0;       // 0 = unlimited
  std::int64_t time_ms = 0;      // 0 = unlimited
  int pv_interval_ms = 1000;
};

// Transposition table. An entry takes 24 bytes: a cluster keeps the upper 32
// bits of the 8 board keys together (one cache line, read first) and 16 more
// key bits with each entry's data (48 bits checked besides the cluster
// index); the attacker's hand is stored for superiority lookups.
struct EntryData {
  std::uint32_t hand;    // attacker's hand
  std::uint32_t pn;      // ToTT()/FromTT()
  std::uint32_t dn;
  std::uint16_t best;    // Move16
  std::uint16_t lmr;     // proof length (14 bits) | mode << 14 | rep << 15
  std::uint16_t amount;  // search effort, EncodeAmount() (replacement priority)
  std::uint16_t tag;     // bits 16..31 of the board key
  std::uint16_t Len() const { return lmr & 0x3fff; }
  std::uint8_t Mode() const { return (lmr >> 14) & 1; }  // 0 = hisshi, 1 = mate (below a pass)
  bool Rep() const { return (lmr >> 15) != 0; }           // disproof depends on path repetition
};
static_assert(sizeof(EntryData) == 20, "entry size");

// Search effort in 16 bits: 12-bit mantissa, 4-bit exponent (monotonic).
inline std::uint16_t EncodeAmount(std::uint64_t a) {
  int e = 0;
  while (a >= 4096 && e < 15) { a >>= 1; ++e; }
  if (a >= 4096) a = 4095;
  return static_cast<std::uint16_t>((e << 12) | a);
}
inline std::uint64_t DecodeAmount(std::uint16_t c) { return static_cast<std::uint64_t>(c & 4095) << (c >> 12); }

constexpr int kClusterSize = 8;
struct alignas(64) Cluster {
  std::uint32_t key[kClusterSize];  // bits 32..63 of the board key (never 0), 0 = empty
  EntryData d[kClusterSize];
};
static_assert(sizeof(Cluster) == 192, "cluster size");

namespace detail {
struct SearchImpl;
}

class Solver {
 public:
  void Resize(std::size_t mb);
  void Clear();
  Result Solve(Position& root, const Limits& limits, const Options& opt,
               bool (*should_stop)());
  int Hashfull() const;
  std::size_t TableBytes() const { return table_.size() * sizeof(Cluster); }
  void KeepOnlyProofs();
  // TT locking while several threads use the table.
  bool Shared() const { return shared_; }
  void SetShared(bool v) { shared_ = v; }

 private:
  friend struct detail::SearchImpl;
  std::vector<Cluster> table_;
  std::size_t cluster_count_ = 0;

  // Striped spin locks guarding the clusters while several threads search.
  struct alignas(64) Lock {
    std::atomic<std::uint32_t> v{0};
  };
  static constexpr std::size_t kLocks = 1 << 14;
  std::unique_ptr<Lock[]> locks_{new Lock[kLocks]};
  bool shared_ = false;  // true while helper threads run
  std::atomic<std::uint64_t> total_nodes_{0};  // nodes of all search threads (progress output)
  bool dirty_ = false;   // the table holds entries of a search
  bool proofs_only_ = false;  // after the search: proofs are kept (verification, answer)
};

}  // namespace hisshi

#endif  // HISSHI_DFPN_HPP_
