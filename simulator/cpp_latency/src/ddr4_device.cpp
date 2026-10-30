#include "ddr4_device.h"

#include <algorithm>
#include <stdexcept>

namespace supra2::latency {

Ddr4Device::Ddr4Device(Ddr4Config config) : config_(config) {
  // 构造时检查几何参数。vector 下标由 BG 和 BA 合成；
  // 每个元素只保存时序与状态，不分配实际的 4-Gb 数据单元数组。
  if (!config_.tck_ps || !config_.bank_groups || !config_.banks_per_group ||
      !config_.row_bits || config_.row_bits > 17 || !config_.column_bits ||
      config_.column_bits > 10 || !config_.dq_bits ||
      (config_.burst_length != 8 && config_.burst_length != 4)) {
    throw std::invalid_argument("invalid DDR4 geometry or clock");
  }
  banks_.resize(config_.bank_groups * config_.banks_per_group);
  last_write_end_by_bg_.resize(config_.bank_groups);
}

std::uint64_t Ddr4Device::cycles(std::uint64_t ps) const {
  // 若时序约束在两个时钟沿之间结束，只能在后一个沿发命令，
  // 因此将 ps 向上取整为 CK，不能直接截断。
  return (ps + config_.tck_ps - 1) / config_.tck_ps;
}

std::size_t Ddr4Device::index(std::uint8_t bg, std::uint8_t bank) const {
  // 将二维地址 (BG, BA) 映射到 banks_ 的一维下标。
  return static_cast<std::size_t>(bg) * config_.banks_per_group + bank;
}

Ddr4Command Ddr4Device::decode_pins(const Ddr4Pins& pins) {
  // 板级 x16 wrapper 在 ACT_n=1 时把 ADR[16:14] 送到 RAS_n/CAS_n/WE_n；
  // 发 ACT 时这些引脚则参与表示 row 地址。
  Ddr4Command out;  // 解码结果；缺省为 NOP，未选中或 CKE 关闭时直接返回。
  out.ck = pins.ck;
  out.bank_group = pins.bank_group;
  out.bank = pins.bank;
  if (pins.cs_n || !pins.cke) return out;
  if (!pins.act_n) {
    // row 位宽是否合法留给掌握几何配置的 issue() 检查。
    out.kind = Ddr4CommandKind::Act;
    out.row = pins.address;
    return out;
  }
  const bool ras_n = (pins.address >> 16) & 1; // RAS_n：高/低电平参与命令真值表。
  const bool cas_n = (pins.address >> 15) & 1; // CAS_n：高/低电平参与命令真值表。
  const bool we_n = (pins.address >> 14) & 1;  // WE_n：高/低电平参与命令真值表。
  // 对列命令，A10 表示自动预充电；对 PRE，A10 表示 PREA。
  // A12=1 选择板级 BL8 模式；BC4 可以解码，但当前模型会拒绝。
  if (ras_n && !cas_n) {
    out.kind = we_n ? Ddr4CommandKind::Read : Ddr4CommandKind::Write;
    out.column = static_cast<std::uint16_t>(pins.address & 0x3ff);
    out.auto_precharge = (pins.address >> 10) & 1;
    out.burst_chop = !((pins.address >> 12) & 1);
  } else if (!ras_n && cas_n && !we_n) {
    out.kind = ((pins.address >> 10) & 1) ? Ddr4CommandKind::PreAll
                                            : Ddr4CommandKind::Pre;
  } else if (!ras_n && !cas_n && we_n) {
    out.kind = Ddr4CommandKind::Refresh;
  } else if (ras_n && cas_n && we_n) {
    out.kind = Ddr4CommandKind::Nop;
  } else {
    out.kind = Ddr4CommandKind::Unsupported;
  }
  return out;
}

void Ddr4Device::apply_auto_precharges(std::uint64_t ck) {
  // 无需遍历每个空闲 CK：跳到下一条命令时，结算所有已满足
  // tRAS/tRTP/tWR 和 tRP 恢复条件的隐式 PRE。
  for (auto& bank : banks_) {
    // auto_precharge_end_ck 是该 bank 的自动 PRE 完成时间。
    if (bank.auto_precharge_end_ck && ck >= *bank.auto_precharge_end_ck) {
      bank.row.reset();
      bank.auto_precharge_end_ck.reset();
      ++stats_.precharges;
    }
  }
}

Ddr4IssueResult Ddr4Device::reject(const Ddr4Command& command,
                                  std::uint64_t earliest,
                                  const std::string& reason) {
  // 被拒绝的命令本身不提交状态变化；但随时间流逝，
  // 之前安排的自动预充电可能已完成，这是另一项状态事件。
  if (count_rejections_) ++stats_.rejected_commands;
  // earliest 不能早于命令本身的 CK；reason 解释是哪条规则拒绝了命令。
  return {false, std::max(command.ck, earliest), reason, {}, {}};
}

std::uint64_t Ddr4Device::precharge_ready(const BankState& bank) const {
  // ACT 通过 tRAS 给出最早 PRE 时间；READ/WRITE 又可能通过
  // tRTP 或写数据结束时间+tWR 将其推后，PRE 必须满足最大值。
  return bank.next_pre_ck;
}

std::uint64_t Ddr4Device::act_ready(const Ddr4Command& command,
                                   const BankState& bank) const {
  // 单 bank 的 tRP/tRC 和整个 rank 的 tRFC 是独立门限。
  std::uint64_t ready = std::max(bank.next_act_ck, refresh_end_ck_); // bank 与刷新恢复的较晚者。
  if (last_act_) {
    // 同一 bank group 内两次 ACT 使用较长的 tRRD_L；
    // 跨 bank group 使用较短的 tRRD_S，与 bank 当前是否开行无关。
    ready = std::max(ready, last_act_->ck +
        (last_act_->bg == command.bank_group ? cycles(config_.trrd_l_ps)
                                             : cycles(config_.trrd_s_ps)));
  }
  if (recent_acts_.size() >= 4) // 至少四条 ACT 仍在窗口中，才需要施加第五次 ACT 限制。
    // 滚动 tFAW 窗口内已有四次 ACT 时，第五次必须等待最早一次离开窗口。
    ready = std::max(ready, recent_acts_.front() + cycles(config_.tfaw_ps));
  return ready;
}

std::optional<std::uint32_t> Ddr4Device::open_row(std::uint8_t bg,
                                                  std::uint8_t bank,
                                                  std::uint64_t at_ck) const {
  if (bg >= config_.bank_groups || bank >= config_.banks_per_group) return {};
  const auto& state = banks_[index(bg, bank)]; // 指定 BG/BA 对应的 bank 状态。
  // 即使尚未在 at_ck 调用 issue()，只读查询也能判断隐式 PRE 是否已结束。
  if (state.auto_precharge_end_ck && at_ck >= *state.auto_precharge_end_ck)
    return {};
  return state.row;
}
// ACT：DRAM 阵列的一行 → bank 行缓冲
// READ：行缓冲指定列 → 等 CL → DQ 总线
// WRITE：DQ 总线 → 等 CWL → 行缓冲指定列/DRAM 阵列
// PRE：关闭行缓冲中的当前行
// REF：占用 rank 执行刷新
Ddr4IssueResult Ddr4Device::issue(const Ddr4Command& command) {
  // 事件推进顺序：结算延迟完成的自动 PRE，检查 rank 共用门限，
  // 再检查当前命令涉及的 bank 门限，最后提交计时器和统计量。
  // last_command_ck_ 只记录已接受命令；MIG 轨迹必须按 CK 非递减输入。
  if (last_command_ck_ && command.ck < *last_command_ck_)//检查命令 CK 是否早于上一条已接受命令。轨迹必须按时间顺序输入
    return reject(command, *last_command_ck_, "trace is not chronological");
  apply_auto_precharges(command.ck);//把已经到期的自动预充电结算掉。它模拟时间经过后，bank 的行缓冲关闭。
  if (command.kind == Ddr4CommandKind::Nop) return {true, command.ck, {}, {}, {}};//NOP 不访问阵列，直接返回。
  if (command.kind == Ddr4CommandKind::Unsupported)//拒绝当前模型不支持的命令，例如初始化期间的 MRS/ZQ。
    return reject(command, command.ck, "unsupported command (MRS/ZQ/init)");
  if (last_command_ck_ && command.ck == *last_command_ck_)
    return reject(command, command.ck + 1, "one command per rank per CK");
  if (command.kind != Ddr4CommandKind::PreAll &&
      command.kind != Ddr4CommandKind::Refresh &&
      (command.bank_group >= config_.bank_groups ||
       command.bank >= config_.banks_per_group))
    return reject(command, command.ck, "bank address out of range");
  if (command.ck < refresh_end_ck_)
    return reject(command, refresh_end_ck_, "tRFC refresh recovery");

  BankState* bank = nullptr; // PREA/REF 操作范围不是单 bank，因此不绑定单个 BankState。
  if (command.kind != Ddr4CommandKind::PreAll &&
      command.kind != Ddr4CommandKind::Refresh)
    bank = &banks_[index(command.bank_group, command.bank)];

  if (command.kind == Ddr4CommandKind::Act) {//ACT 指定一个 bank 和 row
    // ACT 把目标 row 装入行缓冲；同一 bank 再次 ACT 前，
    // 必须已显式或自动 PRE，且所有激活间隔约束均已满足。
    if (command.row >= (1u << config_.row_bits))//检查row地址
      return reject(command, command.ck, "row address out of range");
    if (bank->row)//banK当前没有打开其他row 
      return reject(command, bank->auto_precharge_end_ck.value_or(command.ck),
                    "bank has an open row");
    const auto ready = act_ready(command, *bank); // 合并 bank、tRRD、tFAW、tRFC 限制后的最早 CK。tRP、tRC、tRRD、tFAW 和 tRFC 都已满足。
    if (command.ck < ready) return reject(command, ready, "tRP/tRC/tRRD/tFAW");
    if (bank->previous_row && *bank->previous_row != command.row)
      ++stats_.row_switches;
    bank->row = command.row;//做更新，模型把bank->row更新为新行，并设置3个最早允许时间
    bank->previous_row = command.row;
    bank->next_col_ck = command.ck + cycles(config_.trcd_ps);//最早可以 READ 或 WRITE 的周期，主要由 tRCD 决定。
    bank->next_pre_ck = command.ck + cycles(config_.tras_ps);//最早可以 PRE 的周期，取决于 tRAS、tRTP、tWR。
    bank->next_act_ck = command.ck + cycles(config_.trc_ps);//bank 最早可以再次 ACT 的周期，受 tRC、tRP 等限制。
    bank->columns_since_act = 0;
    last_act_ = LastCommand{command.ck, command.bank_group, command.kind};
    // 先移除 tFAW 滚动窗口之外的旧 ACT，再记录本次 ACT。
    while (!recent_acts_.empty() &&
           command.ck >= recent_acts_.front() + cycles(config_.tfaw_ps))
      recent_acts_.pop_front();
    recent_acts_.push_back(command.ck);
    ++stats_.acts;
  } else if (command.kind == Ddr4CommandKind::Read ||
             command.kind == Ddr4CommandKind::Write) {
    // 列命令要求目标 bank 已开行；对同一开行再次访问列时
    // 无需重复 ACT，统计为一次 row 复用。
    if (!bank->row) return reject(command, command.ck, "no open row");//目标bank是否开行
    if (bank->auto_precharge_end_ck)//是否有自动预充电未完成
      return reject(command, *bank->auto_precharge_end_ck,
                    "auto-precharge pending");
    if (command.column >= (1u << config_.column_bits))//列地址和Burst模式是否受支持
      return reject(command, command.ck, "column address out of range");
    if (command.burst_chop)
      return reject(command, command.ck, "BC4 not configured in this model");
    std::uint64_t ready = bank->next_col_ck; // 本 bank 的 tRCD 限制先作为候选下界。
    if (last_column_)
      // tCCD 约束作用于整个 rank；同一 bank group 内采用较长间隔。
      ready = std::max(ready, last_column_->ck +
          (last_column_->bg == command.bank_group ? config_.tccd_l_ck
                                                   : config_.tccd_s_ck));
    const auto latency = command.kind == Ddr4CommandKind::Read
                             ? config_.read_latency_ck : config_.write_latency_ck;
    const auto burst_ck = config_.burst_length / 2; // DDR 每 CK 两次传输，BL8=4 CK。
    if (command.kind == Ddr4CommandKind::Write && last_read_ck_)
      ready = std::max(ready, *last_read_ck_ + config_.trtw_ck);
    if (command.kind == Ddr4CommandKind::Read && config_.write_latency_ck) {
      // 写后读从最后写数据结束计时；同 BG 记录不能被其他 BG 的写命令覆盖。
      if (last_write_data_end_ck_)
        ready = std::max(ready, *last_write_data_end_ck_ + config_.twtr_s_ck);
      const auto& same_bg_end = last_write_end_by_bg_[command.bank_group];
      if (same_bg_end) ready = std::max(ready, *same_bg_end + config_.twtr_l_ck);
    }
    const auto data_begin = command.ck + latency;   // 命令发出后 CL/CWL 个 CK 开始传数据。
    const auto data_end = data_begin + burst_ck;    // DQ 窗口的结束边界（不包含该边界）。
    // BL8 每个 CK 的两个边沿各传一次，总共占用 DQ 四个 CK。
    // 若 CL/CWL 未知，则暂不检查对应命令类型的 DQ 窗口冲突。
    if (latency && data_begin < dq_busy_until_ck_)
      ready = std::max(ready, dq_busy_until_ck_ - latency);
    if (command.ck < ready)
      return reject(command, ready, "tRCD/tCCD/tRTW/tWTR/DQ overlap");
    if (bank->columns_since_act++ > 0) ++stats_.row_reuses;
    last_column_ = LastCommand{command.ck, command.bank_group, command.kind};
    if (command.kind == Ddr4CommandKind::Read) {
      last_read_ck_ = command.ck;
      ++stats_.reads;
      bank->next_pre_ck = std::max(bank->next_pre_ck,
                                   command.ck + cycles(config_.trtp_ps));
    } else {
      last_write_data_end_ck_ = data_end;
      last_write_end_by_bg_[command.bank_group] = data_end;
      ++stats_.writes;
      // CWL 未知时，data_end 暂按零 CWL 估算。
      // 在 MIG 模式寄存器确定 CWL 前，不能把写后 PRE 检查视为完整验证。
      bank->next_pre_ck = std::max(bank->next_pre_ck,
                                   data_end + cycles(config_.twr_ps));
    }
    if (latency) dq_busy_until_ck_ = data_end;
    if (command.auto_precharge) {
      // 隐式 PRE 要等待最小开行时间和读写恢复时间；
      // 随后的 tRP 决定 bank 何时可以重新 ACT。
      const auto pre_start = bank->next_pre_ck;
      bank->auto_precharge_end_ck = pre_start + cycles(config_.trp_ps);
      bank->next_act_ck = std::max(bank->next_act_ck,
                                   *bank->auto_precharge_end_ck);
    }
    last_command_ck_ = command.ck;
    return {true, command.ck, {},
            latency ? std::optional<std::uint64_t>(data_begin) : std::nullopt,
            latency ? std::optional<std::uint64_t>(data_end) : std::nullopt};
  } else if (command.kind == Ddr4CommandKind::Pre ||
             command.kind == Ddr4CommandKind::PreAll) {
    // PREA 按原子操作处理：先检查全部 bank，全部通过后才修改状态，
    // 避免某个 bank 违例时，其余 bank 已被部分预充电。
    auto begin = command.kind == Ddr4CommandKind::Pre ? index(command.bank_group, command.bank) : 0;
    auto end = command.kind == Ddr4CommandKind::Pre ? begin + 1 : banks_.size();
    // PRE 的半开遍历范围只有目标 bank；PREA 的范围覆盖 banks_ 全部元素。
    for (auto i = begin; i < end; ++i) {
      if (banks_[i].auto_precharge_end_ck)
        return reject(command, *banks_[i].auto_precharge_end_ck,
                      "auto-precharge pending");
      if (banks_[i].row && command.ck < precharge_ready(banks_[i]))
        return reject(command, precharge_ready(banks_[i]), "tRAS/tRTP/tWR");
    }
    for (auto i = begin; i < end; ++i) {
      if (!banks_[i].row) continue;
      banks_[i].row.reset();
      banks_[i].next_act_ck = std::max(banks_[i].next_act_ck,
                                       command.ck + cycles(config_.trp_ps));
      ++stats_.precharges;
    }
  } else if (command.kind == Ddr4CommandKind::Refresh) {
    // REF 要求所有 bank 已关闭并完成恢复；发出后整个 rank 在 tRFC 内不可操作。
    // 当前模型只检查显式 REF，不会按 tREFI 自动插入刷新命令。
    std::uint64_t ready = command.ck;
    for (const auto& state : banks_) {
      if (state.row) return reject(command, state.auto_precharge_end_ck.value_or(command.ck),
                                   "refresh requires all banks precharged");
      ready = std::max(ready, state.next_act_ck);
    }
    if (command.ck < ready) return reject(command, ready, "precharge recovery before REF");
    refresh_end_ck_ = command.ck + cycles(config_.trfc_ps);
    ++stats_.refreshes;
  }
  last_command_ck_ = command.ck;
  return {true, command.ck, {}, {}, {}};
}

}  // namespace supra2::latency
