# MIG AXI to DDR4 request model

## Modeled path

The model adds the controller layer between AXI traffic and the existing DDR4
pin command checker:

```text
AXI512 request -> read/write address arbitration -> bounded request admission
 -> AXI byte address to app_addr -> BG/BA/row/column decode
 -> bank row state and command selection -> DDR4 timing checker
 -> DQ burst completion -> AXI response completion
```

`MigAxiModel` is request-level timing estimation. `Ddr4Device` remains the
device-side timing checker. The controller does not model MIG PHY training,
calibration, actual memory contents, or electrical behavior.

## Board parameters and address decode

Defaults reflect `build/zcu106_pl/ip_axi/ddr4_pl_axi_8/ddr4_pl_axi.xci`:

| Parameter | Default | Source |
| --- | ---: | --- |
| AXI data width | 512 bits | Board MIG XCI |
| DDR payload width | 64 bits | Board MIG XCI |
| UI period | 3332 ps | Board MIG XCI (300 MHz nominal) |
| UI clocks per DDR CK | 4 | Board MIG XCI `nCK_PER_CLK` |
| DDR burst length | BL8 | DDR4/MIG configuration |
| Address map | ROW_COLUMN_BANK | Board MIG XCI |
| AR/AW arbitration | RD_PRI_REG (currently approximated by read priority) | XCI `PARAM_VALUE.C0.DDR4_AxiArbitrationScheme` and generated RTL |
| CL / CWL | 18 / 12 CK | XCI `CasLatency` / `CasWriteLatency` |
| Bank geometry | 2 BG x 4 banks, 15 row bits, 10 column bits | Four x16 MT40A256M16GE-075E devices |

For a 64-bit DDR payload, the model converts an AXI byte address to the flat
payload address with `app_addr = axi_byte_address >> 3`. It then applies the
PG150 4-Gb x16 `ROW_COLUMN_BANK` mapping: BG0 from `app_addr[3]`, BA from
`app_addr[5:4]`, column from `{app_addr[12:6], app_addr[2:0]}`, and row from
`app_addr[27:13]`. Address mapping is centralized in
`MigAxiModel::map_axi_address()` so it can be checked against the MIG pin
trace. The model accepts full 512-bit beats only; narrow AXI beats are not
implemented despite the IP interface advertising narrow-burst support.

## Per-cycle behavior

1. `submit()` places an AXI burst into the read-address or write-address
   queue. It returns `false` when the configured outstanding-request capacity
   is full, representing backpressure to the caller. Each beat must be 64-byte
   aligned, full width, and the burst cannot cross a 4-KiB boundary.
2. `step_ui()` grants at most one queued address in each UI cycle. The board
   uses `RD_PRI_REG`; the model's default is a read-priority approximation
   without the generated arbiter's hold/wait/starvation state. Optional TDM
   mode alternates read/write slots even while idle. The XCI BUSIF metadata's
   TDM entry is not the instantiated arbitration parameter.
3. A granted AXI beat becomes one 64-byte DDR BL8 request. The scheduler
   examines pending beats in rotating order. It issues ACT for a closed bank,
   PRE for a row conflict, or READ/WRITE for an open row hit. At most one
   command is accepted per DDR CK. Four CK command opportunities are modeled
   per UI step.
4. `Ddr4Device::issue()` checks the selected command against the configured
   device timing state. A command that is too early remains pending; the
   scheduler can issue a command for another bank during that CK instead.
5. A request completes after all its BL8 data windows end and the configured
   AXI response delay elapses. `AxiCompletion` records submit and completion
   UI cycles.

## Build and run

```powershell
cmake -S simulator/cpp_latency -B build/cpp_latency
cmake --build build/cpp_latency --config Release
build/cpp_latency/Release/mig_axi_model.exe --requests simulator/cpp_latency/examples/mig_axi_requests.csv
```

The executable prints total UI cycles and each request's latency. The class
API is in `include/mig_axi_model.h`; the implementation is in
`src/mig_axi_model.cpp`.

`read_beat_events()` and `find_read_beat(id, index)` expose the earliest UI
cycle at which each read beat is available to a DMA consumer. These are
data-ready events, not copies of payload bytes. `is_complete(id)` reports
the request-level completion. The integrated DMA model accepts read beats
in AXI burst order and can delay its own completion when its tile buffer
cannot receive a beat.

## Interpretation and known limits

This first version captures the architectural path and DDR bank timing, but
it is not a cycle-exact clone of the MIG RTL. In particular, MIG's dynamic
bank-machine allocation, separate row/column arbiters, write-data channel
arrival, write buffering, read/write batching, exact queue depths, response
reordering, refresh insertion, and all DQ turnaround rules are simplified or
omitted. CL=18 and CWL=12 come from the XCI user parameters, not the zero-valued
MODELPARAM CL/CWL entries. Generated RTL also confirms the mode-register
configuration and tRTW=14 CK. A command/DQS trace should independently verify
the configured timing and the added UI response delay.

The model's `Ddr4Device` currently checks a subset of DDR4 timings. A passing
request run is a software-model result, not validation of MIG RTL, a complete
DDR4 protocol proof, or an FPGA measurement. Compare its command sequence and
phase cycle counts with the existing block-13 Questa/MIG simulation before
using it for latency claims.

## References

- AMD, [PG150 AXI Addressing](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/AXI-Addressing)
- AMD, [PG150 DDR4 Address Map](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/Address-Map)
- AMD, [PG150 TDM Arbitration](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/Time-Division-Multiplexing-TDM)
- AMD, [PG150 Bank Machines](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/Bank-Machines)
- AMD, [PG150 Command Reordering](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/Reordering)
