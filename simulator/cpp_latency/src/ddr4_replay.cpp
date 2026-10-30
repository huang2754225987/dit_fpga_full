#include "ddr4_device.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using supra2::latency::Ddr4Device;
using supra2::latency::Ddr4Pins;
using supra2::latency::Ddr4Config;

namespace {
// 板级 testbench 每个 DDR CK 采样一次命令引脚。
// 将 CSV 解析与 Ddr4Device 分开，其他来源也可直接提交解码后的命令，
// 不必依赖 CSV 格式或某一种仿真器。
std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> fields; // 当前 CSV 行按逗号切分后的各列。
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(field);
  return fields;
}
std::uint64_t number(const std::string& text, int base = 10) {
  std::size_t used = 0; // stoull 实际解析的字符数，用于发现尾部非法字符。
  const auto value = std::stoull(text, &used, base); // base=16 用于 ADR 十六进制列。
  if (used != text.size()) throw std::invalid_argument("invalid number: " + text);
  return value;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Ddr4Config config; // 当前回放采用的 DDR 几何和时序；可由命令行覆盖 CL/CWL。
    std::string path;  // 输入的 DDR 引脚 CSV 文件路径。
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        std::cout << "Usage: ddr4_replay --trace pin.csv [--read-latency-ck N] "
                     "[--write-latency-ck N]\n"
                     "CSV: ck,act_n,adr,ba,bg,cs_n,cke (adr is hexadecimal)\n";
        return 0;
      }
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
      if (arg == "--trace") path = argv[++i];
      else if (arg == "--read-latency-ck")
        config.read_latency_ck = static_cast<std::uint32_t>(number(argv[++i]));
      else if (arg == "--write-latency-ck")
        config.write_latency_ck = static_cast<std::uint32_t>(number(argv[++i]));
      else throw std::invalid_argument("unknown option: " + arg);
    }
    if (path.empty()) throw std::invalid_argument("--trace is required");
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open trace: " + path);
    Ddr4Device device(config); // 持有 bank 状态并检查逐条命令。
    std::string line;          // 文件中当前读取的原始文本行。
    std::size_t line_no = 0;   // 用于在报错中定位 CSV 行号。
    while (std::getline(input, line)) {
      ++line_no;
      if (line.empty() || line[0] == '#' || line.rfind("ck,", 0) == 0) continue;
      const auto fields = split(line); // 列顺序：ck,act_n,adr,ba,bg,cs_n,cke。
      if (fields.size() != 7) throw std::runtime_error("expected 7 CSV fields at line " + std::to_string(line_no));
      Ddr4Pins pins; // 将 CSV 的 DDR 引脚采样值装入有语义的结构体。
      pins.ck = number(fields[0]);
      pins.act_n = number(fields[1]) != 0;
      pins.address = static_cast<std::uint32_t>(number(fields[2], 16));
      pins.bank = static_cast<std::uint8_t>(number(fields[3]));
      pins.bank_group = static_cast<std::uint8_t>(number(fields[4]));
      pins.cs_n = number(fields[5]) != 0;
      pins.cke = number(fields[6]) != 0;
      // 回放 MIG 实际发出的命令序列；这里不根据 AXI 地址自行生成调度，
      // 只检查已经到达 DDR4 器件引脚的命令。
      const auto command = Ddr4Device::decode_pins(pins); // 引脚电平 -> ACT/RD/WR/PRE 等命令。
      const auto result = device.issue(command); // 执行时序检查，成功时更新模型状态。
      if (!result.accepted)
        std::cout << "line=" << line_no << " ck=" << pins.ck << " REJECT "
                  << result.reason << " earliest_ck=" << result.earliest_ck << '\n';
    }
    const auto& s = device.stats(); // 取回放结束后的累计命令统计。
    // 一条被接受的 BL8 命令在 64-bit rank 上传送八次数据。
    // 这里统计的是 DDR 引脚流量，不是 AXI 或 DMA 事务数量。
    const auto bursts = s.reads + s.writes; // 读 burst 与写 burst 总数。
    const auto bytes = bursts * config.burst_length * config.dq_bits / 8; // DQ 上的理论字节数。
    std::cout << "ACT=" << s.acts << " RD=" << s.reads << " WR=" << s.writes
              << " PRE=" << s.precharges << " REF=" << s.refreshes
              << " row_reuses=" << s.row_reuses << " row_switches=" << s.row_switches
              << " device_bytes=" << bytes << " rejected=" << s.rejected_commands
              << '\n';
    return s.rejected_commands ? 2 : 0;
  } catch (const std::exception& e) {
    std::cerr << "ddr4_replay: " << e.what() << '\n';
    return 1;
  }
}
