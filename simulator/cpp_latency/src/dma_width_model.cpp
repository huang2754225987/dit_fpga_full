#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using Cycle = std::uint64_t;

// 本实验仅覆盖 block-13 输入组件 0..9 和结果输出，不包含中途的权重 DMA。
constexpr std::array<int, 10> kInputBytes = {294912, 36864, 1152, 1152, 1152,
                                             1152, 1152, 1152, 1152, 1152};
constexpr int kOutputBeats = 294912 / 64;
constexpr Cycle kInputReference = 178086;
constexpr Cycle kOutputReference = 43145;
constexpr Cycle kBlockReference = 4442603;

struct Result {
  Cycle cycles = 0;
  Cycle accepted = 0;
  Cycle completed = 0;
  Cycle backpressure = 0;
  int max_fifo = 0;
};

// 接收 FIFO 按 512-bit beat 计数，正在拆包的 beat 单独占一个寄存器。
// 在同一周期，先从 FIFO 启动拆包，再接收新 beat；新接收数据最早下周期使用。
Result input_component(int beats, int lanes, int depth, int arrival_period) {
  if (beats < 0 || lanes <= 0 || 32 % lanes || depth < 1 || arrival_period < 1)
    throw std::invalid_argument("invalid input configuration");
  Result r;
  int fifo = 0, remaining = 0;
  while (r.completed < static_cast<Cycle>(beats)) {
    if (remaining == 0 && fifo > 0) { --fifo; remaining = 32 / lanes; }
    if (remaining > 0 && --remaining == 0) ++r.completed;
    if (r.accepted < static_cast<Cycle>(beats) && r.cycles % arrival_period == 0) {
      if (fifo < depth) { ++fifo; ++r.accepted; }
      else ++r.backpressure;
    }
    r.max_fifo = std::max(r.max_fifo, fifo);
    ++r.cycles;
  }
  return r;
}

// 假设结果 RAM 可每周期提供 read_bits 位；拼包完成后的 beat 进入有限 FIFO。
// 写通道每 write_period 周期最多接收一个 beat，FIFO 满则暂停片上读取。
// 假设 RAM 读延迟为一个周期并充分流水，最后一个分段读出后下一周期入 FIFO。
Result output_stream(int beats, int read_bits, int depth, int write_period) {
  if (beats < 0 || read_bits <= 0 || 512 % read_bits || depth < 1 || write_period < 1)
    throw std::invalid_argument("invalid output configuration");
  Result r;
  int fifo = 0, parts = 0;
  bool pending = false;
  while (r.completed < static_cast<Cycle>(beats)) {
    if (fifo > 0 && r.cycles % write_period == 0) { --fifo; ++r.completed; }
    if (pending && fifo < depth) { ++fifo; pending = false; ++r.accepted; }
    if (pending) ++r.backpressure;
    else if (r.accepted < static_cast<Cycle>(beats)) {
      if (++parts == 512 / read_bits) { parts = 0; pending = true; }
    }
    r.max_fifo = std::max(r.max_fifo, fifo);
    ++r.cycles;
  }
  return r;
}

void self_test() {
  auto a = input_component(2, 8, 1, 1);
  if (a.cycles != 9 || a.accepted != 2 || a.completed != 2) throw std::runtime_error("input timing");
  auto b = input_component(12, 8, 1, 1);
  if (b.backpressure == 0 || b.completed != 12 || b.max_fifo > 1) throw std::runtime_error("input backpressure");
  auto c = output_stream(2, 256, 1, 1);
  if (c.cycles != 6 || c.accepted != 2 || c.completed != 2) throw std::runtime_error("output timing");
  auto d = output_stream(20, 512, 1, 5);
  if (d.backpressure == 0 || d.completed != 20 || d.max_fifo > 1) throw std::runtime_error("output backpressure");
  if (input_component(0, 4, 2, 1).cycles != 0 || output_stream(0, 128, 2, 1).cycles != 0)
    throw std::runtime_error("empty transfer");
}

int main(int argc, char** argv) try {
  if (argc == 2 && std::string(argv[1]) == "--self-test") { self_test(); return 0; }
  if (argc != 2) { std::cerr << "usage: dma_width_model <report.csv> | --self-test\n"; return 1; }
  std::ofstream out(argv[1]);
  if (!out) throw std::runtime_error("cannot open report");
  out << "input_lanes,output_read_bits,fifo_depth,axi_period,input_cycles,output_cycles,transaction_cycles,input_backpressure,output_backpressure,input_fifo_peak,output_fifo_peak\n";
  for (int period : {1, 4}) for (int lanes : {4, 8}) for (int bits : {128, 256, 512}) {
    Result input;
    for (int bytes : kInputBytes) {
      Result part = input_component(bytes / 64, lanes, 4, period);
      input.cycles += part.cycles; input.backpressure += part.backpressure;
      input.max_fifo = std::max(input.max_fifo, part.max_fifo);
    }
    auto output = output_stream(kOutputBeats, bits, 4, period);
    out << lanes << ',' << bits << ",4," << period << ',' << input.cycles << ','
        << output.cycles << ',' << input.cycles + kBlockReference + output.cycles << ','
        << input.backpressure << ',' << output.backpressure << ',' << input.max_fifo
        << ',' << output.max_fifo << '\n';
  }
  std::cout << "Reference input=" << kInputReference << " block=" << kBlockReference
            << " output=" << kOutputReference << " cycles\n";
  std::cout << "Report: " << argv[1] << " (idealized, not calibrated)\n";
  return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
