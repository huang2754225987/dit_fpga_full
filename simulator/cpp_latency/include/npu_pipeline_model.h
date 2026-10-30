#pragma once

#include "mig_axi_model.h"
#include "task_graph.h"

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace supra2::latency {

enum class PipelineKind { DmaRead, DmaWrite, Compute };

struct BufferSlice {
  std::string buffer;          // 片上缓冲区名称，例如 WEIGHT 或 ACTIVATION。
  std::string tile;            // 缓冲区中的 tile 标识。
  std::uint64_t bytes = 0;     // 本任务需要或产生的数据量。
  bool release_after_read = true; // 计算读取后是否释放；权重重用可设为 false。
};

struct PipelineOperation {
  std::string id;                        // 操作在依赖图中的唯一编号。
  PipelineKind kind = PipelineKind::Compute;
  std::vector<std::string> dependencies; // 完成这些操作后，当前操作才可启动。
  std::uint64_t ddr_address = 0;         // DMA 使用的 AXI 字节地址。
  std::uint64_t bytes = 0;               // DMA 总传输字节数，须为 64 的倍数。
  std::uint64_t local_offset = 0;        // DMA 对应的片上缓冲区字节偏移。
  std::vector<BufferSlice> inputs;       // 计算输入；写 DMA 使用第一个输入。
  BufferSlice output;                    // 计算输出；读 DMA 使用此目标。
  std::string resource;                  // 计算资源，例如 ARRAY、LN、SOFTMAX。
  Cycle compute_cycles = 0;             // 纯计算周期，不能包含被本模型重算的 DMA 等待。
};

struct BufferConfig {
  std::string name;
  std::uint64_t capacity_bytes = 0;
  std::uint32_t bank_count = 1;
  std::uint32_t line_bytes = 64;
  std::uint32_t write_ports_per_bank = 1;
};

struct ComputeResourceConfig {
  std::string name;
  std::uint32_t instances = 1; // 同一资源最多并行运行多少个任务。
};

class OnChipTileBuffer {
 public:
  explicit OnChipTileBuffer(BufferConfig config);
  void begin_ui();
  bool can_store(std::uint64_t bytes) const;
  bool accept_dma_beat(const std::string& tile, std::uint64_t offset,
                       std::uint64_t bytes, bool use_reservation = false);
  bool has(const std::string& tile, std::uint64_t bytes) const;
  bool can_read(const BufferSlice& input) const;
  void begin_read(const BufferSlice& input);
  void end_read(const BufferSlice& input);
  void consume(const std::string& tile, std::uint64_t bytes);
  void reserve(std::uint64_t bytes);
  void publish(const std::string& tile, std::uint64_t bytes);
  std::uint64_t free_bytes() const;
  std::uint64_t capacity_bytes() const { return config_.capacity_bytes; }

 private:
  BufferConfig config_;
  std::uint64_t occupied_bytes_ = 0;
  std::uint64_t reserved_bytes_ = 0;
  std::unordered_map<std::string, std::uint64_t> tile_bytes_;
  std::unordered_map<std::string, std::uint32_t> tile_readers_;
  std::unordered_map<std::string, bool> consuming_reader_;
  std::vector<std::uint32_t> bank_writes_this_ui_;
};

struct PipelineTiming {
  std::string id;
  Cycle start_ui = 0;
  Cycle finish_ui = 0;
};

struct DmaStats {
  std::uint64_t bursts = 0;
  std::uint64_t read_beats_accepted = 0;
  std::uint64_t write_beats_staged = 0;
  std::uint64_t axi_backpressure_ui = 0;
  std::uint64_t buffer_backpressure_ui = 0;
  std::uint64_t memory_wait_ui = 0;
};

struct PipelineResult {
  Cycle makespan_ui = 0;
  std::vector<PipelineTiming> operations;
  MigAxiStats mig_stats{};
  DmaStats dma_stats{};
  std::unordered_map<std::string, Cycle> resource_busy_ui;
};

// 组合单通道 DMA、片上 tile 缓冲区、计算资源和 MIG/DDR 事件时间线。
class NpuPipelineModel {
 public:
  NpuPipelineModel(MigAxiConfig mig_config,
                   std::vector<BufferConfig> buffers,
                   std::vector<ComputeResourceConfig> resources);
  PipelineResult run(const std::vector<PipelineOperation>& operations,
                     Cycle max_ui_cycles = 100000000);

 private:
  struct ActiveDma {
    const PipelineOperation* op = nullptr;
    Cycle started_ui = 0;
    std::uint64_t bytes_finished = 0;
    std::uint64_t burst_id = 0;
    std::uint32_t burst_beats = 0;
    std::uint32_t beats_delivered = 0;
    bool burst_active = false;
  };
  struct ActiveCompute {
    const PipelineOperation* op = nullptr;
    Cycle finish_ui = 0;
  };

  void progress_dma(Cycle now, std::vector<PipelineTiming>& finished);
  void progress_compute(Cycle now, std::vector<PipelineTiming>& finished);
  bool start_compute(const PipelineOperation& op, Cycle now);
  bool start_dma(const PipelineOperation& op, Cycle now);
  OnChipTileBuffer& buffer(const std::string& name);

  MigAxiModel mig_;
  std::uint64_t beat_bytes_ = 64;
  std::uint64_t next_axi_id_ = 1;
  std::unordered_map<std::string, OnChipTileBuffer> buffers_;
  std::unordered_map<std::string, std::uint32_t> resource_capacity_;
  std::unordered_map<std::string, std::uint32_t> resource_busy_;
  ActiveDma dma_{};
  std::vector<ActiveCompute> computing_;
  DmaStats dma_stats_{};
  std::unordered_map<std::string, Cycle> resource_busy_ui_;
  bool used_ = false;
};

}  // namespace supra2::latency
