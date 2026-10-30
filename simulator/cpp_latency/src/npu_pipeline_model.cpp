#include "npu_pipeline_model.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>

namespace supra2::latency {

OnChipTileBuffer::OnChipTileBuffer(BufferConfig config)
    : config_(std::move(config)), bank_writes_this_ui_(config_.bank_count, 0) {
  if (config_.name.empty() || !config_.capacity_bytes || !config_.bank_count ||
      config_.line_bytes < 64 || config_.line_bytes % 64 ||
      !config_.write_ports_per_bank)
    throw std::invalid_argument("invalid on-chip buffer configuration");
}

void OnChipTileBuffer::begin_ui() {
  // 新 UI 周期开始时，片上每个 bank 的写端口重新可用。
  std::fill(bank_writes_this_ui_.begin(), bank_writes_this_ui_.end(), 0);
}

std::uint64_t OnChipTileBuffer::free_bytes() const {
  return config_.capacity_bytes - occupied_bytes_ - reserved_bytes_;
}

bool OnChipTileBuffer::can_store(std::uint64_t bytes) const {
  return bytes <= free_bytes();
}

bool OnChipTileBuffer::accept_dma_beat(const std::string& tile,
                                       std::uint64_t offset,
                                       std::uint64_t bytes,
                                       bool use_reservation) {
  // DMA 读数据进入片上缓冲区，受到容量和目标 bank 写端口共同限制。
  const auto bank = (offset / config_.line_bytes) % config_.bank_count;
  const auto readers = tile_readers_.find(tile);
  const bool capacity_ready = use_reservation ? reserved_bytes_ >= bytes : can_store(bytes);
  if ((readers != tile_readers_.end() && readers->second) || !capacity_ready ||
      bank_writes_this_ui_[bank] >= config_.write_ports_per_bank)
    return false;
  ++bank_writes_this_ui_[bank];
  if (use_reservation) reserved_bytes_ -= bytes;
  tile_bytes_[tile] += bytes;
  occupied_bytes_ += bytes;
  return true;
}

bool OnChipTileBuffer::has(const std::string& tile, std::uint64_t bytes) const {
  const auto it = tile_bytes_.find(tile);
  const auto readers = tile_readers_.find(tile);
  return it != tile_bytes_.end() && it->second >= bytes &&
         (readers == tile_readers_.end() || readers->second == 0);
}

bool OnChipTileBuffer::can_read(const BufferSlice& input) const {
  const auto data = tile_bytes_.find(input.tile);
  if (data == tile_bytes_.end() || data->second < input.bytes) return false;
  const auto readers = tile_readers_.find(input.tile);
  const auto consuming = consuming_reader_.find(input.tile);
  if (input.release_after_read)
    return readers == tile_readers_.end() || readers->second == 0;
  return consuming == consuming_reader_.end() || !consuming->second;
}

void OnChipTileBuffer::begin_read(const BufferSlice& input) {
  if (!can_read(input)) throw std::logic_error("tile is unavailable or in use");
  ++tile_readers_[input.tile];
  if (input.release_after_read) consuming_reader_[input.tile] = true;
}

void OnChipTileBuffer::end_read(const BufferSlice& input) {
  auto& readers = tile_readers_.at(input.tile);
  if (!readers) throw std::logic_error("tile read lease underflow");
  --readers;
  if (input.release_after_read) {
    consuming_reader_.erase(input.tile);
    consume(input.tile, input.bytes);
  }
  if (!readers) tile_readers_.erase(input.tile);
}

void OnChipTileBuffer::consume(const std::string& tile, std::uint64_t bytes) {
  // DMA 写回或计算读租约结束后释放数据；计算运行期间不能提前复用该空间。
  if (!has(tile, bytes)) throw std::logic_error("on-chip tile data unavailable");
  auto& available = tile_bytes_.at(tile);
  available -= bytes;
  occupied_bytes_ -= bytes;
  if (!available) tile_bytes_.erase(tile);
}

void OnChipTileBuffer::reserve(std::uint64_t bytes) {
  // 计算开始时预留输出空间，防止 DMA 在计算期间填满该缓冲区。
  if (!can_store(bytes)) throw std::logic_error("on-chip output buffer full");
  reserved_bytes_ += bytes;
}

void OnChipTileBuffer::publish(const std::string& tile, std::uint64_t bytes) {
  // 计算结束时，把预留容量转成可供后续任务使用的有效 tile 数据。
  if (reserved_bytes_ < bytes) throw std::logic_error("output was not reserved");
  reserved_bytes_ -= bytes;
  occupied_bytes_ += bytes;
  tile_bytes_[tile] += bytes;
}

NpuPipelineModel::NpuPipelineModel(
    MigAxiConfig mig_config, std::vector<BufferConfig> buffers,
    std::vector<ComputeResourceConfig> resources)
    : mig_(mig_config), beat_bytes_(mig_config.axi_data_width_bits / 8) {
  if (beat_bytes_ != 64)
    throw std::invalid_argument("pipeline v1 requires AXI512 / 64-byte beats");
  for (auto& config : buffers) {
    const auto name = config.name;
    if (!buffers_.emplace(name, OnChipTileBuffer(std::move(config))).second)
      throw std::invalid_argument("duplicate on-chip buffer: " + name);
  }
  for (const auto& resource : resources) {
    if (resource.name.empty() || !resource.instances ||
        !resource_capacity_.emplace(resource.name, resource.instances).second)
      throw std::invalid_argument("invalid or duplicate compute resource");
  }
}

OnChipTileBuffer& NpuPipelineModel::buffer(const std::string& name) {
  const auto it = buffers_.find(name);
  if (it == buffers_.end()) throw std::invalid_argument("unknown buffer: " + name);
  return it->second;
}

bool NpuPipelineModel::start_dma(const PipelineOperation& op, Cycle now) {
  // 当前 RTL 读 DMA 只有一个 outstanding burst；这里也只运行一个 DMA 描述符。
  if (dma_.op) return false;
  if (op.kind == PipelineKind::DmaRead) {
    if (op.output.buffer.empty() || op.output.tile.empty() ||
        !buffer(op.output.buffer).can_store(op.bytes)) return false;
    buffer(op.output.buffer).reserve(op.bytes);
  } else {
    if (op.inputs.size() != 1 ||
        !buffer(op.inputs.front().buffer).has(op.inputs.front().tile, op.bytes))
      return false;
  }
  dma_.op = &op;
  dma_.started_ui = now;
  return true;
}

void NpuPipelineModel::progress_dma(
    Cycle now, std::vector<PipelineTiming>& finished) {
  if (!dma_.op) return;
  const auto& op = *dma_.op;

  if (!dma_.burst_active && dma_.bytes_finished < op.bytes) {
    // 64-beat 上限和 4-KiB 页边界来自 supra2_axi_read_dma 的拆分规则。
    const auto address = op.ddr_address + dma_.bytes_finished;
    const auto page_bytes = 4096 - (address & 4095);
    const auto bytes = std::min<std::uint64_t>(
        {op.bytes - dma_.bytes_finished, page_bytes, 64 * beat_bytes_});
    const auto beats = static_cast<std::uint32_t>(bytes / beat_bytes_);
    const bool read = op.kind == PipelineKind::DmaRead;
    if (!read && !buffer(op.inputs.front().buffer).has(op.inputs.front().tile, bytes))
      return;
    const std::uint64_t id = next_axi_id_;
    if (!mig_.submit({id, read ? AxiDirection::Read : AxiDirection::Write,
                      address, beats})) {
      ++dma_stats_.axi_backpressure_ui;
      return;
    }
    ++next_axi_id_;
    ++dma_stats_.bursts;
    dma_.burst_id = id;
    dma_.burst_beats = beats;
    dma_.beats_delivered = 0;
    dma_.burst_active = true;
    // 写 DMA 当前按整 burst 预取片上数据，再由 MIG 模型处理 AW/DDR 写完成。
    if (!read) {
      buffer(op.inputs.front().buffer).consume(op.inputs.front().tile, bytes);
      dma_stats_.write_beats_staged += beats;
    }
    return;
  }

  if (dma_.burst_active) {
    if (op.kind == PipelineKind::DmaRead) {
      // 按 AXI burst 顺序、最多每 UI 周期一个 512-bit beat，写入片上 tile。
      if (dma_.beats_delivered < dma_.burst_beats) {
        const auto* beat = mig_.find_read_beat(dma_.burst_id, dma_.beats_delivered);
        if (beat && beat->ready_ui <= now) {
          if (buffer(op.output.buffer).accept_dma_beat(
                  op.output.tile, op.local_offset + dma_.bytes_finished,
                  beat_bytes_, true)) {
            ++dma_.beats_delivered;
            ++dma_stats_.read_beats_accepted;
            dma_.bytes_finished += beat_bytes_;
          } else {
            ++dma_stats_.buffer_backpressure_ui;
          }
        } else {
          ++dma_stats_.memory_wait_ui;
        }
      }
      if (dma_.beats_delivered == dma_.burst_beats &&
          mig_.is_complete(dma_.burst_id)) dma_.burst_active = false;
    } else if (mig_.is_complete(dma_.burst_id)) {
      dma_.bytes_finished += static_cast<std::uint64_t>(dma_.burst_beats) * beat_bytes_;
      dma_.burst_active = false;
    } else {
      ++dma_stats_.memory_wait_ui;
    }
  }

  if (!dma_.burst_active && dma_.bytes_finished == op.bytes) {
    finished.push_back({op.id, dma_.started_ui, now});
    dma_ = {};
  }
}

bool NpuPipelineModel::start_compute(const PipelineOperation& op, Cycle now) {
  const auto capacity_it = resource_capacity_.find(op.resource);
  if (capacity_it == resource_capacity_.end())
    throw std::invalid_argument("unknown compute resource: " + op.resource);
  if (resource_busy_[op.resource] >= capacity_it->second) return false;
  // 一个输入 tile 在该计算任务结束前保持驻留。只读权重可被多个任务共享。
  using TileKey = std::pair<std::string, std::string>;
  std::map<TileKey, bool> seen;
  for (const auto& input : op.inputs) {
    if (input.buffer.empty() || input.tile.empty() || !input.bytes)
      throw std::invalid_argument("compute input needs buffer, tile, and bytes");
    const TileKey key{input.buffer, input.tile};
    if (!seen.emplace(key, true).second)
      throw std::invalid_argument("combine duplicate input tile references into one slice");
    if (!buffer(input.buffer).can_read(input)) return false;
  }

  if (op.output.bytes) {
    auto& destination = buffer(op.output.buffer);
    if (op.output.bytes > destination.free_bytes()) return false;
  }

  // 输入 tile 在任务开始时被占有；任务执行期间资源保持 busy。
  // compute_cycles 应包含该 RTL 单元内部的 feed/流水线时间，不能再含 DMA 等待。
  for (const auto& input : op.inputs) buffer(input.buffer).begin_read(input);
  if (op.output.bytes) buffer(op.output.buffer).reserve(op.output.bytes);
  ++resource_busy_[op.resource];
  resource_busy_ui_[op.resource] += op.compute_cycles;
  computing_.push_back({&op, now + op.compute_cycles});
  return true;
}

void NpuPipelineModel::progress_compute(
    Cycle now, std::vector<PipelineTiming>& finished) {
  for (auto it = computing_.begin(); it != computing_.end();) {
    if (it->finish_ui > now) { ++it; continue; }
    const auto& op = *it->op;
    for (const auto& input : op.inputs) buffer(input.buffer).end_read(input);
    if (op.output.bytes)
      buffer(op.output.buffer).publish(op.output.tile, op.output.bytes);
    --resource_busy_.at(op.resource);
    finished.push_back({op.id, it->finish_ui - op.compute_cycles, now});
    it = computing_.erase(it);
  }
}

PipelineResult NpuPipelineModel::run(
    const std::vector<PipelineOperation>& operations, Cycle max_ui_cycles) {
  if (used_) throw std::logic_error("NpuPipelineModel is one-shot");
  used_ = true;
  std::unordered_map<std::string, unsigned> state; // 0=待启动，1=运行中，2=完成。
  for (const auto& op : operations) {
    if (op.id.empty() || !state.emplace(op.id, 0).second)
      throw std::invalid_argument("pipeline operations need unique nonempty IDs");
    if (op.kind == PipelineKind::Compute) {
      if (!op.compute_cycles || op.resource.empty())
        throw std::invalid_argument("compute operation needs resource and cycles");
      if (op.output.bytes && op.output.tile.empty())
        throw std::invalid_argument("compute output needs a tile name");
      if (op.output.bytes &&
          op.output.bytes > buffer(op.output.buffer).capacity_bytes())
        throw std::invalid_argument("compute output exceeds buffer capacity");
    } else if (!op.bytes || op.bytes % beat_bytes_ ||
               op.ddr_address % beat_bytes_ ||
               op.local_offset % beat_bytes_ ||
               op.ddr_address >= (1ULL << 32) ||
               op.bytes > (1ULL << 32) - op.ddr_address) {
      throw std::invalid_argument("DMA operation needs aligned 32-bit address and bytes");
    } else if (op.kind == PipelineKind::DmaRead) {
      if (op.output.tile.empty() || op.output.bytes != op.bytes ||
          op.bytes > buffer(op.output.buffer).capacity_bytes())
        throw std::invalid_argument("DMA read needs matching output tile capacity");
    } else {
      if (op.inputs.size() != 1 || op.inputs.front().tile.empty() ||
          op.inputs.front().bytes < op.bytes)
        throw std::invalid_argument("DMA write needs one source tile with enough bytes");
      buffer(op.inputs.front().buffer);
    }
  }
  for (const auto& op : operations)
    for (const auto& dep : op.dependencies)
      if (!state.count(dep)) throw std::invalid_argument("missing dependency: " + dep);

  PipelineResult result;
  while (result.operations.size() < operations.size() &&
         mig_.ui_cycle() < max_ui_cycles) {
    const auto now = mig_.ui_cycle();
    for (auto& entry : buffers_) entry.second.begin_ui();
    std::vector<PipelineTiming> finished;
    progress_dma(now, finished);
    progress_compute(now, finished);
    for (const auto& item : finished) {
      state.at(item.id) = 2;
      result.operations.push_back(item);
      result.makespan_ui = std::max(result.makespan_ui, item.finish_ui);
    }

    // 依赖满足、数据已在片上且对应硬件空闲时，启动可运行任务。
    bool started_any = false;
    for (const auto& op : operations) {
      if (state.at(op.id) != 0) continue;
      const bool ready = std::all_of(op.dependencies.begin(), op.dependencies.end(),
          [&state](const std::string& dep) { return state.at(dep) == 2; });
      if (!ready) continue;
      const bool started = op.kind == PipelineKind::Compute
                               ? start_compute(op, now) : start_dma(op, now);
      if (started) { state.at(op.id) = 1; started_any = true; }
    }
    if (!started_any && finished.empty() && !dma_.op && computing_.empty() &&
        mig_.idle() && result.operations.size() != operations.size())
      throw std::runtime_error("pipeline cannot progress: dependency or data unavailable");
    mig_.step_ui();
  }
  if (result.operations.size() != operations.size())
    throw std::runtime_error("pipeline did not finish: dependency or resource deadlock");
  result.mig_stats = mig_.stats();
  result.dma_stats = dma_stats_;
  result.resource_busy_ui = resource_busy_ui_;
  std::sort(result.operations.begin(), result.operations.end(),
            [](const PipelineTiming& a, const PipelineTiming& b) {
              return a.start_ui < b.start_ui ||
                     (a.start_ui == b.start_ui && a.id < b.id);
            });
  return result;
}

}  // namespace supra2::latency
