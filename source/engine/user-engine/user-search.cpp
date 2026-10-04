#include <cmath>
#include <cstdlib>
#include <sstream>

#include "../../misc.h"
#include "../../search.h"
#include "../../thread.h"
#include "../../usi.h"

#include "hisshi_dfpn.hpp"

#if defined(USER_ENGINE)

namespace {
hisshi::Solver g_solver;

bool ShouldStop() { return Threads.stop.load(); }

std::string ToString(hisshi::PnDn v) {
  return v >= hisshi::kInf ? std::string("inf") : std::to_string(v);
}
}  // namespace

// "user" USI command (unused).
void user_test(Position& /*pos*/, std::istringstream& /*is*/) {}

// Only options a user may want to change. Search parameters are fixed at
// their tuned values (hisshi::Options defaults).
void USI::extra_option(USI::OptionsMap& o) {
  // Attacker considers every legal move instead of the paper's candidates
  // (slower; a "nomate" answer then covers all attacks).
  o["HisshiFullWidth"] << USI::Option(false);
  // Progress output interval in ms (0 = none).
  o["PvInterval"] << USI::Option(1000, 0, 1000000);
  // Futile interpositions (無駄合い) in the answer: 0 every interposition is
  // shown as a defence, 1 futile interpositions are left out.
  o["HisshiFutile"] << USI::Option(0, 0, 1);
}

void Search::init() {}

void Search::clear() { g_solver.Resize(static_cast<std::size_t>(static_cast<int>(Options["USI_Hash"]))); }

void MainThread::search() {
  const bool is_mate_search = Search::Limits.mate != 0;

  hisshi::Limits lim;
  lim.nodes = Search::Limits.nodes;
  if (is_mate_search) {
    lim.time_ms = Search::Limits.mate >= INT32_MAX ? 0 : Search::Limits.mate;
  } else if (Search::Limits.movetime > 0) {
    lim.time_ms = Search::Limits.movetime;
  }
  lim.pv_interval_ms = static_cast<int>(Options["PvInterval"]);

  hisshi::Options opt;
  opt.full_width = static_cast<bool>(Options["HisshiFullWidth"]);
  opt.futile = static_cast<int>(Options["HisshiFutile"]);
  opt.threads = static_cast<int>(Options["Threads"]);

  const auto res = g_solver.Solve(rootPos, lim, opt, &ShouldStop);

  const std::uint64_t nps = res.elapsed_ms ? res.nodes * 1000 / res.elapsed_ms : res.nodes;
  std::ostringstream pv;
  for (Move m : res.pv) pv << to_usi_string(m) << ' ';
  std::string pv_str = pv.str();
  if (!pv_str.empty()) pv_str.pop_back();

  // A standard info line for GUIs (score: "mate +N" with the answer's length when
  // proven, "mate -" when disproven, else an evaluation from pn/dn) ...
  {
    std::ostringstream os;
    const int len = static_cast<int>(res.pv.size());
    os << "info depth " << len << " seldepth " << len << " time " << res.elapsed_ms << " nodes " << res.nodes
       << " nps " << nps << " hashfull " << g_solver.Hashfull() << " score ";
    if (res.status == hisshi::Status::kProven && res.verified) os << "mate +" << len;
    else if (res.status == hisshi::Status::kDisproven) os << "mate -";
    else {
      double v = -600.0 * std::log(static_cast<double>(std::max<hisshi::PnDn>(res.pn, 1)) /
                                   static_cast<double>(std::max<hisshi::PnDn>(res.dn, 1)));
      os << "cp " << static_cast<int>(std::max(-30000.0, std::min(30000.0, v)));
    }
    if (res.status == hisshi::Status::kProven && res.verified && !pv_str.empty()) os << " pv " << pv_str;
    sync_cout << os.str() << sync_endl;
  }
  // ... and the summary line read by the scripts.
  sync_cout << "info time " << res.elapsed_ms << " nodes " << res.nodes << " nps " << nps
            << " hashfull " << g_solver.Hashfull() << " string pn=" << ToString(res.pn)
            << " dn=" << ToString(res.dn) << sync_endl;
  if (res.status != hisshi::Status::kProven && !res.verify_info.empty())
    sync_cout << "info string disproof_verified=" << (res.verified ? 1 : 0) << " " << res.verify_info << sync_endl;
  if (res.status == hisshi::Status::kProven) {
    sync_cout << "info string proof_verified=" << (res.verified ? 1 : 0) << " " << res.verify_info
              << sync_endl;
  }

  if (is_mate_search) {
    if (res.status == hisshi::Status::kProven && res.verified)
      sync_cout << "checkmate " << pv_str << sync_endl;
    else if (res.status == hisshi::Status::kDisproven)
      sync_cout << "checkmate nomate" << sync_endl;
    else
      sync_cout << "checkmate timeout" << sync_endl;
    return;
  }

  while (!Threads.stop && Search::Limits.infinite) Tools::sleep(1);
  if (res.status == hisshi::Status::kProven && !res.pv.empty())
    sync_cout << "bestmove " << to_usi_string(res.pv.front()) << sync_endl;
  else
    sync_cout << "bestmove resign" << sync_endl;
}

void Thread::search() {}

#endif  // USER_ENGINE
