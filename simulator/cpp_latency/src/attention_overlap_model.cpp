#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Cycle = std::uint64_t;

// 当前 M=256、9 个 head 的 block-13 self-attention RTL/MIG 阶段计数。
constexpr Cycle kReferenceSelf = 1131363;
constexpr Cycle kReferenceBlock = 4442603;
constexpr Cycle kReferenceTransaction = 4663834;
constexpr int kHeads = 9;
constexpr int kQueryTiles = 8;  // 256 query / 32 rows。
constexpr int kKeyTiles = 8;    // 256 keys / 32 keys。
constexpr int kGroups = kHeads * kQueryTiles;
constexpr int kTiles = kGroups * kKeyTiles;

// 每 head 的 V 装载和每个 query tile 的标量输出，先保持 RTL 现状。
constexpr Cycle kVLoadPerHead = 256 * 64;
constexpr Cycle kOutputPerGroup = 32 * 64;
constexpr Cycle kFixed = kHeads * kVLoadPerHead + kGroups * kOutputPerGroup;

// 每个 32-query x 32-key tile 的结构性周期：
// Q/K 各 64 拍、QK clear+feed、QK 逐行 drain、两个 PV 半列块和合并。
// QK WAIT_ACC、状态切换、在线 softmax 与最终归一化留在未分解项中。
constexpr Cycle kPrepare = 128;
constexpr Cycle kQkArray = 1 + 64 + 32 + 32 + 3;  // clear + (K + 2*32 + 3)
constexpr Cycle kScoreDrain = 32;
constexpr Cycle kPvArray = 2 * (1 + (3 * 32 + 3) + 3); // 每半块 clear/feed/drain
constexpr Cycle kMerge = 2 * 32;
constexpr Cycle kKnownPerTile = kPrepare + kQkArray + kScoreDrain + kPvArray + kMerge;
constexpr Cycle kResidual = kReferenceSelf - kFixed - kTiles * kKnownPerTile;

enum class Kind { Prepare, Qk, Drain, Softmax, Pv, Merge };
enum class Resource { Prepare, Array, Drain, Softmax, Merge, Count };

struct Task {
  Kind kind;
  int tile;
  Resource resource;
  Cycle duration;
  std::vector<int> deps;
  int state = 0;  // 0=未启动，1=执行中，2=完成。
  Cycle start = 0;
  Cycle finish = 0;
};

const char* label(Kind kind) {
  switch (kind) {
    case Kind::Prepare: return "qk_prepare";
    case Kind::Qk: return "qk_array";
    case Kind::Drain: return "score_drain";
    case Kind::Softmax: return "online_softmax";
    case Kind::Pv: return "pv_array_snapshot";
    case Kind::Merge: return "pv_merge";
  }
  return "unknown";
}

const char* resource_label(Resource resource) {
  switch (resource) {
    case Resource::Prepare: return "qk_prepare";
    case Resource::Array: return "shared_array";
    case Resource::Drain: return "score_write";
    case Resource::Softmax: return "softmax";
    case Resource::Merge: return "pv_merge";
    case Resource::Count: break;
  }
  return "unknown";
}

int priority(Kind kind) {
  // ARRAY 同时有 PV 和 QK 请求时优先清理 PV，避免双缓冲过早被占满。
  return kind == Kind::Pv ? 0 : kind == Kind::Qk ? 1 : 2;
}

struct GroupResult {
  Cycle makespan = 0;
  Cycle array_busy = 0;
  std::vector<Task> tasks;
};

GroupResult schedule_group(const std::array<Cycle, kKeyTiles>& softmax_cycles) {
  std::vector<Task> tasks;
  std::array<std::array<int, 6>, kKeyTiles> ids{};
  const auto add = [&tasks](Kind kind, int tile, Resource resource, Cycle duration,
                            std::vector<int> deps) {
    const int id = static_cast<int>(tasks.size());
    tasks.push_back({kind, tile, resource, duration, std::move(deps)});
    return id;
  };
  for (int i = 0; i < kKeyTiles; ++i) {
    auto& id = ids[i];
    // 前一个 QK 的输入读取结束后，才能改写同一组 Q/K 输入 bank。
    id[0] = add(Kind::Prepare, i, Resource::Prepare, kPrepare,
                i ? std::vector<int>{ids[i-1][1]} : std::vector<int>{});
    // 现有 QK 的 acc_snap 在 drain 完成前不能被下一 QK 覆盖。
    std::vector<int> qk_deps{id[0]};
    if (i) qk_deps.push_back(ids[i-1][2]);
    id[1] = add(Kind::Qk, i, Resource::Array, kQkArray + 1, qk_deps);

    // score 双缓冲：同一个槽只有在 i-2 的 softmax 读完后才能覆盖。
    std::vector<int> drain_deps{id[1]};
    if (i >= 2) drain_deps.push_back(ids[i-2][3]);
    id[2] = add(Kind::Drain, i, Resource::Drain, kScoreDrain, drain_deps);

    // online m/l 状态要按 key tile 顺序更新；raw_exp/alpha 双缓冲
    // 直到 i-2 的 PV 合并完成后才能复用。
    std::vector<int> sm_deps{id[2]};
    if (i) sm_deps.push_back(ids[i-1][3]);
    if (i >= 2) sm_deps.push_back(ids[i-2][5]);
    id[3] = add(Kind::Softmax, i, Resource::Softmax, softmax_cycles[i], sm_deps);

    // 每个 PV tile 分成两个 32 列 feature 块。假设各有一个 32x32 INT32
    // 快照，阵列完成两块后合并；上一 tile 合并结束后才能复用这两个快照。
    std::vector<int> pv_deps{id[3]};
    if (i) pv_deps.push_back(ids[i-1][5]);
    id[4] = add(Kind::Pv, i, Resource::Array, kPvArray + 2, pv_deps);
    id[5] = add(Kind::Merge, i, Resource::Merge, kMerge, {id[4]});
  }

  std::array<int, static_cast<int>(Resource::Count)> owner{};
  owner.fill(-1);
  Cycle now = 0;
  std::size_t completed = 0;
  while (completed < tasks.size()) {
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      auto& t = tasks[i];
      if (t.state == 1 && t.finish == now) {
        t.state = 2;
        owner[static_cast<int>(t.resource)] = -1;
        ++completed;
      }
    }
    // 同一个时间点可启动多个不同资源，但 ARRAY 只能选一个任务。
    bool started = true;
    while (started) {
      started = false;
      int best = -1;
      for (std::size_t i = 0; i < tasks.size(); ++i) {
        const auto& t = tasks[i];
        if (t.state || owner[static_cast<int>(t.resource)] != -1) continue;
        const bool ready = std::all_of(t.deps.begin(), t.deps.end(),
            [&tasks](int dep) { return tasks[dep].state == 2; });
        if (!ready) continue;
        if (best < 0 || priority(t.kind) < priority(tasks[best].kind) ||
            (priority(t.kind) == priority(tasks[best].kind) && t.tile < tasks[best].tile))
          best = static_cast<int>(i);
      }
      if (best >= 0) {
        auto& t = tasks[best];
        t.state = 1;
        t.start = now;
        t.finish = now + t.duration;
        owner[static_cast<int>(t.resource)] = best;
        started = true;
      }
    }
    if (completed == tasks.size()) break;
    Cycle next = std::numeric_limits<Cycle>::max();
    for (const auto& t : tasks) if (t.state == 1) next = std::min(next, t.finish);
    if (next == std::numeric_limits<Cycle>::max())
      throw std::runtime_error("attention dependency graph deadlocked");
    now = next;
  }

  GroupResult result;
  result.makespan = now;
  for (const auto& t : tasks) {
    for (int dep : t.deps)
      if (tasks[dep].finish > t.start) throw std::logic_error("dependency violation");
    if (t.resource == Resource::Array) result.array_busy += t.duration;
  }
  for (std::size_t i = 0; i < tasks.size(); ++i)
    for (std::size_t j = i + 1; j < tasks.size(); ++j)
      if (tasks[i].resource == tasks[j].resource &&
          tasks[i].start < tasks[j].finish && tasks[j].start < tasks[i].finish)
        throw std::logic_error("resource overlap");
  result.tasks = std::move(tasks);
  return result;
}

struct Scenario {
  unsigned softmax_percent;
  Cycle overlappable_residual = 0;
  Cycle residual_serial = 0;
  Cycle modeled_array_busy = 0;
  Cycle optimized_self = 0;
  GroupResult first_group;
};

Scenario evaluate(unsigned percent) {
  if (!percent || percent > 100) throw std::invalid_argument("percent must be in 1..100");
  Scenario result{percent};
  result.optimized_self = kFixed;
  // 以整数周期分配残差，保证所有 tile 的残差总和恰好等于阶段参考值。
  for (int group = 0; group < kGroups; ++group) {
    std::array<Cycle, kKeyTiles> soft{};
    for (int tile = 0; tile < kKeyTiles; ++tile) {
      const int i = group * kKeyTiles + tile;
      const Cycle residual = kResidual / kTiles + (i < static_cast<int>(kResidual % kTiles));
      soft[tile] = residual * percent / 100;
      result.overlappable_residual += soft[tile];
      result.residual_serial += residual - soft[tile];
    }
    auto schedule = schedule_group(soft);
    if (!group) result.first_group = schedule;
    result.modeled_array_busy += schedule.array_busy;
    result.optimized_self += schedule.makespan;
  }
  result.optimized_self += result.residual_serial;
  return result;
}

void emit_trace(const std::string& path, const GroupResult& group) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write trace file");
  out << "tile,task,resource,start_ui,finish_ui\n";
  auto ordered = group.tasks;
  std::stable_sort(ordered.begin(), ordered.end(), [](const Task& a, const Task& b) {
    return a.start < b.start || (a.start == b.start && a.finish < b.finish);
  });
  for (const auto& t : ordered)
    out << t.tile << ',' << label(t.kind) << ',' << resource_label(t.resource)
        << ',' << t.start << ',' << t.finish << '\n';
}

void verify() {
  if (kFixed + kTiles * kKnownPerTile + kResidual != kReferenceSelf)
    throw std::logic_error("baseline arithmetic mismatch");
  std::array<Cycle, kKeyTiles> cost{};
  cost.fill(100);
  const auto group = schedule_group(cost);
  if (group.tasks.size() != 48 || group.array_busy != 8 * (kQkArray + kPvArray + 3))
    throw std::logic_error("task count or shared-array accounting mismatch");
  const auto scenario = evaluate(50);
  if (scenario.overlappable_residual + scenario.residual_serial != kResidual)
    throw std::logic_error("residual accounting mismatch");
  if (scenario.optimized_self < kFixed + scenario.modeled_array_busy + scenario.residual_serial)
    throw std::logic_error("predicted self time below modeled resource lower bound");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::string report_path, trace_path;
    bool self_test = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--self-test") self_test = true;
      else if ((arg == "--report" || arg == "--trace") && i + 1 < argc) {
        if (arg == "--report") report_path = argv[++i];
        else trace_path = argv[++i];
      } else throw std::invalid_argument("usage: attention_overlap_model [--self-test] [--report FILE] [--trace FILE]");
    }
    verify();
    if (self_test) { std::cout << "attention schedule invariants PASS\n"; return 0; }
    const std::array<unsigned, 4> splits{25, 50, 75, 90};
    std::vector<Scenario> scenarios;
    for (const auto split : splits) scenarios.push_back(evaluate(split));
    if (!trace_path.empty()) emit_trace(trace_path, scenarios[1].first_group);

    std::ofstream file;
    if (!report_path.empty()) {
      file.open(report_path);
      if (!file) throw std::runtime_error("cannot write report file");
    }
    std::ostream& out = report_path.empty() ? std::cout : file;
    out << "{\n  \"model\":\"single_array_online_attention_sensitivity\",\n"
        << "  \"prediction_status\":\"illustrative_not_calibrated\",\n"
        << "  \"ui_period_ps\":3332,\n"
        << "  \"reference_self_ui\":" << kReferenceSelf << ",\n"
        << "  \"reference_block_ui\":" << kReferenceBlock << ",\n"
        << "  \"reference_transaction_ui\":" << kReferenceTransaction << ",\n"
        << "  \"fixed_v_and_output_ui\":" << kFixed << ",\n"
        << "  \"tile_count\":" << kTiles << ",\n"
        << "  \"known_baseline_cycles_per_tile\":" << kKnownPerTile << ",\n"
        << "  \"unmeasured_residual_ui\":" << kResidual << ",\n"
        << "  \"scenarios\":[\n";
    for (std::size_t i = 0; i < scenarios.size(); ++i) {
      const auto& s = scenarios[i];
      const auto saved = static_cast<std::int64_t>(kReferenceSelf) - static_cast<std::int64_t>(s.optimized_self);
      const auto block = static_cast<std::int64_t>(kReferenceBlock) - saved;
      const auto transaction = static_cast<std::int64_t>(kReferenceTransaction) - saved;
      out << "    {\"softmax_share_percent\":" << s.softmax_percent
          << ",\"softmax_cycles\":" << s.overlappable_residual
          << ",\"remaining_serial_cycles\":" << s.residual_serial
          << ",\"array_reserved_ui\":" << s.modeled_array_busy
          << ",\"self_ui\":" << s.optimized_self
          << ",\"self_reduction_percent\":" << std::fixed << std::setprecision(3)
          << (100.0 * saved / kReferenceSelf)
          << ",\"block_ui\":" << block
          << ",\"transaction_ui\":" << transaction << '}';
      out << (i + 1 == scenarios.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    if (file.is_open()) std::cout << "report=" << report_path << " trace=" << trace_path << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "attention_overlap_model: " << error.what() << '\n';
    return 2;
  }
}
