#pragma once

#include "ddr4_device.h"
#include "model_config.h"

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace supra2::latency {

enum class AxiDirection { Read, Write };

struct AxiRequest {
  std::uint64_t id = 0;
  AxiDirection direction = AxiDirection::Read;
  std::uint64_t address = 0;  // AXI byte address.
  std::uint32_t beats = 1;    // Number of full 512-bit AXI beats.
};

struct DdrAddress {
  std::uint32_t row = 0;
  std::uint16_t column = 0;
  std::uint8_t bank_group = 0;
  std::uint8_t bank = 0;
};

struct MigAxiConfig {
  std::uint64_t ui_period_ps = 3332;
  std::uint32_t ddr_ck_per_ui = 4; // 一个 MIG UI 周期内有 4 个 DDR CK。
  std::uint32_t axi_data_width_bits = 512;
  std::uint32_t memory_data_width_bits = 64;
  std::uint32_t max_outstanding_requests = 16;
  std::uint32_t read_response_latency_ui = 2;
  std::uint32_t write_response_latency_ui = 2;
  // 实际 PARAM_VALUE.C0.DDR4_AxiArbitrationScheme=RD_PRI_REG。
  // false 使用简化读优先；方向保持、QoS、饥饿计数仍待逐项复现。true 用于 TDM 对比。
  bool tdm_read_write_arbitration = false;
  Ddr4Config ddr{};
};

struct MigAxiStats {
  std::uint64_t accepted_requests = 0;
  std::uint64_t completed_requests = 0;
  std::uint64_t tdm_slot_wait_ui = 0;
  std::uint64_t queue_wait_ui = 0;
  std::uint64_t row_hits = 0;
  std::uint64_t row_misses = 0;
  std::uint64_t row_conflicts = 0;
  std::uint64_t read_bytes = 0;
  std::uint64_t write_bytes = 0;
  std::uint64_t response_wait_ui = 0;
};

struct AxiCompletion {
  std::uint64_t id = 0;
  AxiDirection direction = AxiDirection::Read;
  std::uint64_t submitted_ui = 0;
  std::uint64_t completed_ui = 0;
};

// DDR read beat availability. The DMA still decides when to accept this beat.
struct AxiBeatEvent {
  std::uint64_t request_id = 0;
  std::uint32_t beat_index = 0;
  std::uint64_t address = 0;
  std::uint64_t ready_ui = 0;
};

// Request-level MIG/AXI timing model for the board's AXI512-to-DDR64 path.
// It models the address shim, TDM address grants, a bounded request queue,
// bank row state, DDR command issue, DQ occupancy, and AXI response delay.
class MigAxiModel {
 public:
  explicit MigAxiModel(MigAxiConfig config = {});

  // Submit an AXI burst at the current UI cycle. False means backpressure:
  // the bounded outstanding-request capacity is full.
  bool submit(const AxiRequest& request);

  // Advance exactly one MIG UI clock (four DDR CK by default).
  void step_ui();

  // Advance until all submitted requests have returned, subject to a guard.
  void run_until_idle(std::uint64_t max_ui_cycles = 100000000);

  static DdrAddress map_axi_address(const MigAxiConfig& config,
                                    std::uint64_t axi_byte_address);

  std::uint64_t ui_cycle() const { return ui_cycle_; }
  const MigAxiStats& stats() const { return stats_; }
  const std::vector<AxiCompletion>& completions() const { return completions_; }
  const std::vector<AxiBeatEvent>& read_beat_events() const { return read_beat_events_; }
  const AxiBeatEvent* find_read_beat(std::uint64_t id,
                                     std::uint32_t index) const;
  bool is_complete(std::uint64_t id) const;
  const Ddr4Device& ddr_device() const { return ddr_; }
  bool idle() const;
  std::string summary() const;

 private:
  struct Beat {
    std::uint64_t request_id = 0;
    std::uint32_t beat_index = 0;
    AxiDirection direction = AxiDirection::Read;
    std::uint64_t address = 0;
    DdrAddress decoded{};
    std::uint64_t data_end_ck = 0;
  };
  struct RequestState {
    AxiRequest request{};
    std::uint64_t submitted_ui = 0;
    std::uint64_t accepted_ui = 0;
    std::uint32_t beats_not_issued = 0;
    std::uint64_t last_data_end_ck = 0;
  };
  struct BankRow {
    std::optional<std::uint32_t> open_row;
  };
  struct QueuedRequest {
    AxiRequest request{};
    std::uint64_t submitted_ui = 0;
  };

  bool has_capacity() const;
  void grant_axi_address();
  void make_beat_ready(RequestState& state, std::uint32_t beat_index);
  void issue_one_ddr_command(std::uint64_t ck);
  void retire_responses();
  std::size_t bank_index(const DdrAddress& address) const;

  MigAxiConfig config_;
  Ddr4Device ddr_;
  std::uint64_t ui_cycle_ = 0;
  std::uint64_t next_id_ = 1;
  std::deque<QueuedRequest> read_address_queue_;
  std::deque<QueuedRequest> write_address_queue_;
  std::deque<Beat> pending_beats_;
  std::deque<RequestState> active_requests_;
  std::vector<BankRow> bank_rows_;
  std::vector<AxiCompletion> completions_;
  std::vector<AxiBeatEvent> read_beat_events_;
  std::unordered_map<std::uint64_t, std::vector<std::size_t>> read_beat_index_;
  std::unordered_set<std::uint64_t> used_ids_;
  std::unordered_set<std::uint64_t> completed_ids_;
  MigAxiStats stats_;
  bool read_tdm_slot_ = true;
  std::size_t round_robin_ = 0;
};

}  // namespace supra2::latency
