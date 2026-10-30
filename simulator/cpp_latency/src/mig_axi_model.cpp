#include "mig_axi_model.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace supra2::latency {
namespace {

constexpr std::uint64_t kAxiPageBytes = 4096;

}  // namespace

MigAxiModel::MigAxiModel(MigAxiConfig config)
    : config_(config), ddr_(config.ddr),
      bank_rows_(config.ddr.bank_groups * config.ddr.banks_per_group) {
  if (!config_.ui_period_ps || !config_.ddr_ck_per_ui ||
      !config_.axi_data_width_bits || config_.axi_data_width_bits % 8 ||
      !config_.memory_data_width_bits || config_.memory_data_width_bits % 8 ||
      !config_.max_outstanding_requests ||
      config_.axi_data_width_bits != 512 ||
      config_.memory_data_width_bits != 64 ||
      config_.ddr.bank_groups != 2 || config_.ddr.banks_per_group != 4 ||
      config_.ddr.row_bits != 15 || config_.ddr.column_bits != 10 ||
      config_.ddr.tck_ps * config_.ddr_ck_per_ui != config_.ui_period_ps) {
    throw std::invalid_argument("invalid MIG/AXI model configuration");
  }
  // 调度器会反复尝试尚未满足时序的命令，以便寻找其他可运行 bank。
  // 这些尝试并没有真的出现在 DDR 引脚上，因此不计入器件违例统计。
  ddr_.set_rejection_counting(false);
}

bool MigAxiModel::has_capacity() const {
  // 未完成请求总数包括：等待读写地址时隙的请求，以及 MIG 已接收但未响应的请求。
  // 容量满时 submit() 返回 false，相当于上游看到反压。
  return active_requests_.size() + read_address_queue_.size() +
             write_address_queue_.size() <
         config_.max_outstanding_requests;
}

bool MigAxiModel::submit(const AxiRequest& request) {
  // 这个版本只接受整宽 AXI512 beat，所以地址必须按 64 字节对齐。
  // AXI 规范还要求 burst 不跨 4 KiB 边界；这里先检查这两个输入条件。
  if (!request.beats) throw std::invalid_argument("AXI burst must contain beats");
  const auto beat_bytes = config_.axi_data_width_bits / 8;
  if (request.address % beat_bytes)
    throw std::invalid_argument("v1 model requires AXI512 beat alignment");
  const auto bytes = static_cast<std::uint64_t>(request.beats) * beat_bytes;
  if ((request.address & (kAxiPageBytes - 1)) + bytes > kAxiPageBytes)
    throw std::invalid_argument("AXI burst crosses a 4-KiB boundary");
  if (!has_capacity()) return false;

  AxiRequest queued = request;
  // 请求编号为 0 时由模型分配一个唯一编号，方便把多个 DDR beat 归回原 burst。
  if (!queued.id) {
    while (used_ids_.count(next_id_)) ++next_id_;
    queued.id = next_id_++;
  }
  if (!used_ids_.insert(queued.id).second)
    throw std::invalid_argument("duplicate AXI request ID");
  auto& queue = queued.direction == AxiDirection::Read ? read_address_queue_
                                                        : write_address_queue_;
  queue.push_back({queued, ui_cycle_});
  ++stats_.accepted_requests;
  return true;
}

DdrAddress MigAxiModel::map_axi_address(const MigAxiConfig& config,
                                        std::uint64_t axi_byte_address) {
  const auto payload_bytes = config.memory_data_width_bits / 8;
  // AXI 地址以字节计数。64-bit DDR payload 每个地址单位是 8 字节，
  // 因而把 AXI 字节地址除以 8，等价于忽略 AXI 地址最低 3 位。
  const std::uint64_t app = axi_byte_address / payload_bytes;
  DdrAddress out;
  // 依据 PG150 的 DDR4 4Gb x16 ROW_COLUMN_BANK 表拆地址：
  // app[3] -> BG0，app[5:4] -> BA，app[12:6] 与 app[2:0] -> Column，
  // app[27:13] -> Row。四颗 x16 芯片并联成同一个 x64 rank，所以只译码一次。
  out.bank_group = static_cast<std::uint8_t>((app >> 3) & 0x1);
  out.bank = static_cast<std::uint8_t>((app >> 4) & 0x3);
  const std::uint32_t col_low = static_cast<std::uint32_t>(app & 0x7);
  const std::uint32_t col_high = static_cast<std::uint32_t>((app >> 6) & 0x7f);
  out.column = static_cast<std::uint16_t>((col_high << 3) | col_low);
  out.row = static_cast<std::uint32_t>((app >> 13) & 0x7fff);
  return out;
}

std::size_t MigAxiModel::bank_index(const DdrAddress& address) const {
  // 把二维 (BG, BA) 坐标转换成 bank_rows_ 的一维数组下标。
  return static_cast<std::size_t>(address.bank_group) *
             config_.ddr.banks_per_group +
         address.bank;
}

void MigAxiModel::grant_axi_address() {
  // TDM 只选择本 UI 周期可以被 MIG 接收的 AR 或 AW 地址请求。
  // 它不表示 DDR 芯片上的 READ/WRITE 命令必须交替发出。
  if (read_address_queue_.empty() && write_address_queue_.empty()) {
    // 固定 TDM 时隙在空闲周期也继续轮转，不能因暂无请求而冻结相位。
    if (config_.tdm_read_write_arbitration) read_tdm_slot_ = !read_tdm_slot_;
    return;
  }
  QueuedRequest queued;
  bool granted = false;
  if (!config_.tdm_read_write_arbitration) {
    if (!read_address_queue_.empty()) {
      queued = read_address_queue_.front();
      read_address_queue_.pop_front();
      granted = true;
    } else {
      queued = write_address_queue_.front();
      write_address_queue_.pop_front();
      granted = true;
    }
  } else {
    // PG150 TDM：读、写地址通道固定交替占用时隙；空时隙不会借给另一通道。
    auto& selected = read_tdm_slot_ ? read_address_queue_ : write_address_queue_;
    if (!selected.empty()) {
      queued = selected.front();
      selected.pop_front();
      granted = true;
    } else {
      ++stats_.tdm_slot_wait_ui;
    }
    read_tdm_slot_ = !read_tdm_slot_;
  }
  if (!granted) return;

  // 获得 AXI 地址时隙后，把一个 burst 登记为活动请求，并拆成逐 beat 的 DDR 请求。
  RequestState state;
  state.request = queued.request;
  state.submitted_ui = queued.submitted_ui;
  state.accepted_ui = ui_cycle_;
  state.beats_not_issued = queued.request.beats;
  active_requests_.push_back(state);
  if (queued.request.direction == AxiDirection::Read)
    read_beat_index_.emplace(queued.request.id,
        std::vector<std::size_t>(queued.request.beats, read_beat_events_.max_size()));
  stats_.queue_wait_ui += ui_cycle_ - queued.submitted_ui;
  for (std::uint32_t i = 0; i < queued.request.beats; ++i)
    make_beat_ready(active_requests_.back(), i);
}

void MigAxiModel::make_beat_ready(RequestState& state, std::uint32_t beat_index) {
  // AXI512 每拍 64 字节。逐拍计算地址，并在这里转换成 DDR 的 BG/BA/Row/Column。
  const auto beat_bytes = config_.axi_data_width_bits / 8;
  Beat beat;
  beat.request_id = state.request.id;
  beat.beat_index = beat_index;
  beat.direction = state.request.direction;
  beat.address = state.request.address + static_cast<std::uint64_t>(beat_index) * beat_bytes;
  beat.decoded = map_axi_address(config_, beat.address);
  pending_beats_.push_back(beat);
}

void MigAxiModel::issue_one_ddr_command(std::uint64_t ck) {
  if (pending_beats_.empty()) return;
  // 在待处理 beat 中轮转查找。某个 bank 因 tRCD/tCCD 等待时，继续找其他 bank，
  // 让独立 bank 有机会使用命令总线；每个 DDR CK 最多接受一条命令。
  for (std::size_t offset = 0; offset < pending_beats_.size(); ++offset) {
    const auto pos = (round_robin_ + offset) % pending_beats_.size();
    auto& beat = pending_beats_[pos];
    const auto bi = bank_index(beat.decoded);
    auto& bank = bank_rows_[bi];
    Ddr4Command command;
    command.ck = ck;
    command.bank_group = beat.decoded.bank_group;
    command.bank = beat.decoded.bank;
    if (!bank.open_row) {
      // bank 当前关闭：先 ACT，把目标行装入该 bank 的行缓冲。
      command.kind = Ddr4CommandKind::Act;
      command.row = beat.decoded.row;
    } else if (*bank.open_row != beat.decoded.row) {
      // bank 开着另一行：先 PRE 关行，后续 CK 再尝试 ACT 目标行。
      command.kind = Ddr4CommandKind::Pre;
    } else {
      // 目标行已开：行命中，可直接发列命令。
      command.kind = beat.direction == AxiDirection::Read
                         ? Ddr4CommandKind::Read
                         : Ddr4CommandKind::Write;
      command.column = beat.decoded.column;
    }

    // Ddr4Device 检查该 CK 是否满足 DDR 器件时序；不满足则本 CK 不发这条命令。
    const auto result = ddr_.issue(command);
    if (!result.accepted) continue;
    if (command.kind == Ddr4CommandKind::Act) {
      bank.open_row = beat.decoded.row;
      ++stats_.row_misses;
    } else if (command.kind == Ddr4CommandKind::Pre) {
      bank.open_row.reset();
      ++stats_.row_conflicts;
    } else if (command.kind == Ddr4CommandKind::Read ||
               command.kind == Ddr4CommandKind::Write) {
      // 列命令被接受后，该 beat 已从命令队列发出；等待其 DQ 数据窗口结束。
      ++stats_.row_hits;
      beat.data_end_ck = result.data_end_ck.value_or(
          ck + config_.ddr.burst_length / 2);
      auto request_it = std::find_if(active_requests_.begin(), active_requests_.end(),
          [&beat](const RequestState& s) { return s.request.id == beat.request_id; });
      if (request_it != active_requests_.end()) {
        request_it->last_data_end_ck = std::max(request_it->last_data_end_ck,
                                                beat.data_end_ck);
        if (request_it->beats_not_issued) --request_it->beats_not_issued;
      }
      if (command.kind == Ddr4CommandKind::Read) {
        stats_.read_bytes += config_.axi_data_width_bits / 8;
        // 数据窗口结束后，还要经过配置的 MIG 读返回延迟。
        const auto ratio = config_.ddr_ck_per_ui;
        const auto ready_ui = (beat.data_end_ck + ratio - 1) / ratio +
                              config_.read_response_latency_ui;
        read_beat_events_.push_back(
            {beat.request_id, beat.beat_index, beat.address, ready_ui});
        read_beat_index_.at(beat.request_id)[beat.beat_index] =
            read_beat_events_.size() - 1;
      } else
        stats_.write_bytes += config_.axi_data_width_bits / 8;
      pending_beats_.erase(pending_beats_.begin() + static_cast<std::ptrdiff_t>(pos));
      round_robin_ = pending_beats_.empty() ? 0 : pos % pending_beats_.size();
      return;  // 当前 CK 已使用；共享 DDR 命令/地址总线不能再发第二条命令。
    }
    round_robin_ = (pos + 1) % pending_beats_.size();
    return;
  }
}

void MigAxiModel::retire_responses() {
  // 只有 burst 中所有 beat 都已发出、最后一段 DQ 数据已结束、响应附加延迟已到，
  // 才把 AXI 请求记为完成并释放 outstanding 容量。
  for (auto it = active_requests_.begin(); it != active_requests_.end();) {
    if (it->beats_not_issued != 0 ||
        it->last_data_end_ck > ui_cycle_ * config_.ddr_ck_per_ui)
      { ++it; continue; }
    const auto response_delay = it->request.direction == AxiDirection::Read
                                    ? config_.read_response_latency_ui
                                    : config_.write_response_latency_ui;
    const auto ratio = config_.ddr_ck_per_ui;
    const auto earliest = (it->last_data_end_ck + ratio - 1) / ratio +
                          response_delay;
    if (ui_cycle_ < earliest) { ++stats_.response_wait_ui; ++it; continue; }
    completions_.push_back({it->request.id, it->request.direction,
                            it->submitted_ui, ui_cycle_});
    completed_ids_.insert(it->request.id);
    ++stats_.completed_requests;
    it = active_requests_.erase(it);
  }
}

const AxiBeatEvent* MigAxiModel::find_read_beat(
    std::uint64_t id, std::uint32_t index) const {
  const auto found = read_beat_index_.find(id);
  if (found == read_beat_index_.end() || index >= found->second.size()) return nullptr;
  const auto position = found->second[index];
  if (position == read_beat_events_.max_size()) return nullptr;
  return &read_beat_events_[position];
}

bool MigAxiModel::is_complete(std::uint64_t id) const {
  return completed_ids_.count(id) != 0;
}

void MigAxiModel::step_ui() {
  // 一个 UI 周期的顺序：先给 AR/AW 地址时隙，再推进对应的 4 个 DDR CK，最后检查响应。
  grant_axi_address();
  const auto first_ck = ui_cycle_ * config_.ddr_ck_per_ui;
  for (std::uint32_t sub = 0; sub < config_.ddr_ck_per_ui; ++sub)
    issue_one_ddr_command(first_ck + sub);
  ++ui_cycle_;
  retire_responses();
}

void MigAxiModel::run_until_idle(std::uint64_t max_ui_cycles) {
  // 离线跑完当前所有请求；max_ui_cycles 是防止错误状态导致无限循环的保护上限。
  std::uint64_t elapsed = 0;
  while (!idle() && elapsed < max_ui_cycles) {
    step_ui();
    ++elapsed;
  }
  if (!idle()) throw std::runtime_error("MIG model did not become idle before guard");
}

bool MigAxiModel::idle() const {
  // 四类队列/状态都清空，才代表 AXI 请求全部返回。
  return read_address_queue_.empty() && write_address_queue_.empty() &&
         pending_beats_.empty() && active_requests_.empty();
}

std::string MigAxiModel::summary() const {
  std::ostringstream out;
  out << "ui_cycles=" << ui_cycle_ << " accepted=" << stats_.accepted_requests
      << " completed=" << stats_.completed_requests
      << " read_bytes=" << stats_.read_bytes
      << " write_bytes=" << stats_.write_bytes
      << " row_hits=" << stats_.row_hits
      << " row_misses=" << stats_.row_misses
      << " row_conflicts=" << stats_.row_conflicts
      << " queue_wait_ui=" << stats_.queue_wait_ui
      << " tdm_empty_slots=" << stats_.tdm_slot_wait_ui;
  return out.str();
}

}  // namespace supra2::latency
