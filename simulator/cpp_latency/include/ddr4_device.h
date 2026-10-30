#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace supra2::latency {

// 输入是 MIG 到 DDR4 器件引脚边界上的命令流：
//   DMA/AXI 请求 -> MIG 队列、地址译码和调度 -> Ddr4Command。
// 本类只建模最后一级，不预测 MIG 如何调度 AXI 请求。
enum class Ddr4CommandKind {
  Nop,        // 空操作；不访问存储阵列。
  Act,        // 激活一行，把该行载入 bank 的行缓冲。
  Read,       // 从已打开的行读取一个 burst。
  Write,      // 向已打开的行写入一个 burst。
  Pre,        // 关闭指定 bank 当前打开的行。
  PreAll,     // 关闭 rank 内所有 bank 的打开行。
  Refresh,    // 刷新整个 rank，期间所有 bank 都不能接收普通访问。
  Unsupported // 当前简化模型不处理的命令，例如 MRS、ZQ。
};

// 一条命令对应 DDR CK 的一个上升沿。row 和 column 是器件侧物理坐标，
// 不是 AXI 字节地址。
struct Ddr4Command {
  std::uint64_t ck = 0;              // 这条命令所在的 DDR 时钟周期编号。
  Ddr4CommandKind kind = Ddr4CommandKind::Nop; // 命令类型。
  std::uint8_t bank_group = 0;       // 目标 bank group 编号（BG）。
  std::uint8_t bank = 0;             // 目标 bank 编号（BA）。
  std::uint32_t row = 0;             // ACT 命令携带的行号。
  std::uint16_t column = 0;          // READ/WRITE 命令携带的列号。
  bool auto_precharge = false;       // READ/WRITE 后是否自动关闭当前行（A10）。
  bool burst_chop = false;           // 是否请求 BC4；当前模型只接受 BL8。
};

struct Ddr4Pins {
  std::uint64_t ck = 0;          // 采样到这组引脚值时的 DDR CK 编号。
  bool act_n = true;             // ACT_n 低表示 ACT；高表示由 RAS/CAS/WE 解码命令。
  std::uint32_t address = 0;     // MIG ADR[16:0]；ACT_n=1 时 A16/A15/A14 是 RAS/CAS/WE。
  std::uint8_t bank = 0;          // BA 引脚：bank 地址。
  std::uint8_t bank_group = 0;    // BG 引脚：bank group 地址。
  bool cs_n = false;             // CS_n 高时 rank 未被选中，本周期视作 NOP。
  bool cke = true;                // CKE 低时器件不接收普通命令，本模型视作 NOP。
};

struct Ddr4Config {
  // 板上四颗 x16 器件并联成一个 64-bit rank，共用命令引脚；
  // 因此一条命令同时作用于四颗芯片，只需维护一份逻辑命令状态。
  std::uint64_t tck_ps = 833;            // 一个 DDR CK 的周期长度，单位 ps。
  std::uint32_t bank_groups = 2;         // rank 中的 bank group 数量。
  std::uint32_t banks_per_group = 4;     // 每个 bank group 中的 bank 数量。
  std::uint32_t row_bits = 15;           // 行地址位数；行总数为 2^row_bits。
  std::uint32_t column_bits = 10;        // 列地址位数；列总数为 2^column_bits。
  std::uint32_t dq_bits = 64;            // rank 的 DQ 数据总线位宽（四颗 x16 并联）。
  std::uint32_t burst_length = 8;        // 每条 READ/WRITE 命令的数据传输数量（BL8）。
  // 以下板级 XCI 的整数 CK 参数按 tCK=833 ps 编码为 ps。
  // cycles() 向上取整；使用 ps 保存参数，也便于替换为其他频率下的器件时序值。
  std::uint64_t trcd_ps = 17 * 833;      // ACT 到 READ/WRITE 的最小间隔。
  std::uint64_t trp_ps = 17 * 833;       // PRE 后 bank 再次 ACT 前的恢复时间。
  std::uint64_t tras_ps = 39 * 833;      // ACT 后一行必须保持打开的最短时间。
  std::uint64_t trc_ps = 44500;          // 同一 bank 两次 ACT 的最短间隔；TS_833 向上取整为 54 CK。
  std::uint64_t trtp_ps = 10 * 833;      // READ 后到 PRE 的最小间隔。
  std::uint64_t twr_ps = 19 * 833;       // 写数据结束到 PRE 的恢复时间。
  std::uint64_t trrd_s_ps = 7 * 833;     // 不同 bank group 之间 ACT 的最小间隔。
  std::uint64_t trrd_l_ps = 8 * 833;     // 同一 bank group 之间 ACT 的最小间隔。
  std::uint64_t tfaw_ps = 37 * 833;      // 滚动窗口内最多允许四次 ACT 的约束窗口。
  std::uint64_t trfc_ps = 313 * 833;     // REF 后 rank 必须等待的恢复时间。
  std::uint64_t trefi_ps = 9363 * 833;   // 两次刷新之间的目标间隔；当前只记录参数，不自动插 REF。
  std::uint32_t tccd_s_ck = 4;           // 不同 bank group 的列命令间隔。
  std::uint32_t tccd_l_ck = 6;           // 同一 bank group 的列命令间隔。
  std::uint32_t trtw_ck = 14;           // READ 到 WRITE 命令间隔；生成的 MIG RTL 参数。
  std::uint32_t twtr_s_ck = 4;          // 写数据结束到跨 BG READ 的间隔；XCI 参数。
  std::uint32_t twtr_l_ck = 10;         // 写数据结束到同 BG READ 的间隔；XCI 参数。
  // 来源：板级 XCI PARAM_VALUE.C0.DDR4_CasLatency / CasWriteLatency。
  // 这两项分别是 18 / 12；同文件的 MODELPARAM DDR4_CL/CWL=0 不是实际 CAS 延时。
  // 显式设置 0 仍可关闭对应 DQ 窗口检查；命令/DQS 轨迹用于后续独立核对。
  std::uint32_t read_latency_ck = 18;    // READ 命令到读数据开始的 CK 数（CL）。
  std::uint32_t write_latency_ck = 12;   // WRITE 命令到写数据开始的 CK 数（CWL）。
};

struct Ddr4IssueResult {
  bool accepted = false;                 // true 表示命令通过当前已实现的检查并生效。
  // 对时序违例，给出当前已建模约束允许的最早 CK。
  // 对状态违例（例如没有开行就 READ），仍须先发其他命令修复状态。
  std::uint64_t earliest_ck = 0;         // 时序限制导致拒绝时，模型给出的最早合法 CK。
  std::string reason;                    // 拒绝原因；接受时为空。
  std::optional<std::uint64_t> data_begin_ck; // CL/CWL 已配置时，DQ burst 的开始 CK。
  std::optional<std::uint64_t> data_end_ck;   // CL/CWL 已配置时，DQ burst 的结束 CK。
};

struct Ddr4Stats {
  std::uint64_t acts = 0;                 // 已接受的 ACT 数。
  std::uint64_t reads = 0;                // 已接受的 READ burst 数。
  std::uint64_t writes = 0;               // 已接受的 WRITE burst 数。
  std::uint64_t precharges = 0;           // 已完成的 PRE/PREA/自动 PRE 数。
  std::uint64_t refreshes = 0;            // 已接受的 REF 数。
  std::uint64_t row_reuses = 0;           // 同一行打开期间，第二次及后续列访问数。
  std::uint64_t row_switches = 0;         // 同一 bank 前后两次 ACT 使用不同 row 的次数。
  std::uint64_t rejected_commands = 0;     // 被模型规则拒绝的命令数。
};

// 一个逻辑 64-bit rank 的离散事件命令级时序检查器。
// 每次 issue() 直接跳到命令所在 CK，不逐周期模拟空闲时间。
// 只处理初始化完成后的命令，不模拟 PHY 训练和存储数据内容。
class Ddr4Device {
 public:
  explicit Ddr4Device(Ddr4Config config = {});
  // 解码板级 wrapper 的 ACT_n/ADR[16:0]/BA/BG 引脚。
  static Ddr4Command decode_pins(const Ddr4Pins& pins);
  // 先检查命令是否合法，再提交状态变化和统计计数。
  Ddr4IssueResult issue(const Ddr4Command& command);
  // Controller schedulers may probe timing availability while choosing a
  // command. Disable rejection counting for those expected scheduling probes.
  void set_rejection_counting(bool enabled) { count_rejections_ = enabled; }
  const Ddr4Config& config() const { return config_; }
  const Ddr4Stats& stats() const { return stats_; }
  // 查询指定 CK 的 bank 开行状态，不推进模型时间。
  std::optional<std::uint32_t> open_row(std::uint8_t bg, std::uint8_t bank,
                                        std::uint64_t at_ck) const;

 private:
  struct BankState {
    // ACT 打开一行；PRE、PREA 或完成的自动预充电将其关闭。
    std::optional<std::uint32_t> row;
    // 仅用于统计同一 bank 两次 ACT 之间是否切换了 row。
    std::optional<std::uint32_t> previous_row;
    // 三类独立的最早发命令时间；相关门限必须同时满足。
    std::uint64_t next_act_ck = 0;        // 此 bank 最早可再次 ACT 的周期。
    std::uint64_t next_col_ck = 0;        // 此 bank 最早可 READ/WRITE 的周期（tRCD）。
    std::uint64_t next_pre_ck = 0;        // 此 bank 最早可 PRE 的周期（tRAS/tRTP/tWR 最大值）。
    // 自动预充电完成恢复前，bank 仍不可再次 ACT。
    std::optional<std::uint64_t> auto_precharge_end_ck;
    std::uint32_t columns_since_act = 0;  // 当前 ACT 后已经执行的 READ/WRITE 命令数。
  };
  struct LastCommand {
    std::uint64_t ck = 0;                 // 上一条相关命令的周期。
    std::uint8_t bg = 0;                  // 上一条命令所属的 bank group。
    Ddr4CommandKind kind = Ddr4CommandKind::Nop; // 上一条命令类型。
  };

  std::uint64_t cycles(std::uint64_t ps) const;
  std::size_t index(std::uint8_t bg, std::uint8_t bank) const;
  void apply_auto_precharges(std::uint64_t ck);
  Ddr4IssueResult reject(const Ddr4Command& command, std::uint64_t earliest,
                         const std::string& reason);
  std::uint64_t precharge_ready(const BankState& bank) const;
  std::uint64_t act_ready(const Ddr4Command& command, const BankState& bank) const;

  Ddr4Config config_;                     // 几何结构、频率和时序参数。
  Ddr4Stats stats_;                       // 当前模型累计的统计结果。
  bool count_rejections_ = true;          // 是否统计被拒命令（调度探测可关闭）。
  std::vector<BankState> banks_;          // 所有 bank 的状态，按 BG/BA 合并索引。
  std::optional<std::uint64_t> last_command_ck_; // 上一条已接受命令的 CK，用于保证轨迹有序。
  std::optional<LastCommand> last_act_;   // 最近一次 ACT，用于计算 tRRD_S/L。
  std::optional<LastCommand> last_column_; // 最近一次 READ/WRITE，用于计算 tCCD_S/L。
  std::optional<std::uint64_t> last_read_ck_;
  std::optional<std::uint64_t> last_write_data_end_ck_;
  std::vector<std::optional<std::uint64_t>> last_write_end_by_bg_;
  // 保存近期 ACT 时间戳，用于检查四次激活窗口 tFAW。
  std::deque<std::uint64_t> recent_acts_;
  // 整个 rank 共用的门限：REF 恢复期，以及 CL/CWL 已知时的 DQ 总线占用。
  std::uint64_t refresh_end_ck_ = 0;      // tRFC 结束前，rank 最早可再次接收命令的 CK。
  std::uint64_t dq_busy_until_ck_ = 0;    // 已知 CL/CWL 时，rank 的 DQ 总线空闲 CK。
};

}  // namespace supra2::latency
