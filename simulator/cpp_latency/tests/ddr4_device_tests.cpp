#include "ddr4_device.h"

#include <cassert>

using namespace supra2::latency;

namespace {
Ddr4Command command(std::uint64_t ck, Ddr4CommandKind kind,
                    std::uint8_t bank = 0, std::uint8_t bg = 0) {
  Ddr4Command c;
  c.ck = ck;
  c.kind = kind;
  c.bank = bank;
  c.bank_group = bg;
  return c;
}
}  // namespace

int main() {
  Ddr4Device device;
  assert(!device.issue(command(0, Ddr4CommandKind::Read)).accepted);
  assert(device.issue(command(0, Ddr4CommandKind::Act)).accepted);
  assert(!device.issue(command(16, Ddr4CommandKind::Read)).accepted);
  assert(device.issue(command(17, Ddr4CommandKind::Read)).accepted);
  assert(!device.issue(command(22, Ddr4CommandKind::Read)).accepted);
  assert(device.issue(command(23, Ddr4CommandKind::Read)).accepted);
  assert(!device.issue(command(38, Ddr4CommandKind::Pre)).accepted);
  assert(device.issue(command(39, Ddr4CommandKind::Pre)).accepted);
  assert(!device.issue(command(55, Ddr4CommandKind::Act)).accepted);
  auto next = command(56, Ddr4CommandKind::Act);
  next.row = 1;
  assert(device.issue(next).accepted);
  assert(device.open_row(0, 0, 56) == 1);
  assert(device.stats().row_switches == 1);

  Ddr4Device four_act;
  for (unsigned i = 0; i < 4; ++i)
    assert(four_act.issue(command(i * 8, Ddr4CommandKind::Act, i)).accepted);
  assert(!four_act.issue(command(31, Ddr4CommandKind::Act, 0, 1)).accepted);
  assert(four_act.issue(command(37, Ddr4CommandKind::Act, 0, 1)).accepted);

  Ddr4Device ap;
  assert(ap.issue(command(0, Ddr4CommandKind::Act)).accepted);
  auto rd = command(16, Ddr4CommandKind::Read);
  rd.ck = 17;
  rd.auto_precharge = true;
  assert(ap.issue(rd).accepted);
  assert(!ap.issue(command(23, Ddr4CommandKind::Read)).accepted);
  assert(!ap.open_row(0, 0, 56));
  assert(ap.issue(command(56, Ddr4CommandKind::Act)).accepted);

  Ddr4Pins pins;
  pins.ck = 16;
  pins.address = 0x15000;
  const auto decoded = Ddr4Device::decode_pins(pins);
  assert(decoded.kind == Ddr4CommandKind::Read);
  assert(!decoded.auto_precharge && !decoded.burst_chop);
  pins.act_n = false;
  pins.address = 0x1234;
  assert(Ddr4Device::decode_pins(pins).row == 0x1234);

  Ddr4Config timed_config;
  timed_config.read_latency_ck = 17;
  Ddr4Device timed(timed_config);
  assert(timed.issue(command(0, Ddr4CommandKind::Act)).accepted);
  const auto result = timed.issue(command(17, Ddr4CommandKind::Read));
  assert(result.accepted && result.data_begin_ck == 34 && result.data_end_ck == 38);
  return 0;
}
