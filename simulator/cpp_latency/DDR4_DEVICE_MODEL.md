# DDR4 device command/timing model

## Boundary

`Ddr4Device` models one logical 64-bit DDR4 rank formed by four parallel x16
MT40A256M16GE-075E devices. The four chips share ACT_n, address, BA, BG,
CS_n, and CK, so the command checker runs once for the rank. This is a
post-initialization command-level timing model. It is separate from the MIG
controller and from `axi_ddr_model.h` (which estimates AXI transfer time).

The exact path in the current design is:

```text
NPU DMA / host AXI512 -> MIG AXI port -> MIG command scheduler
    -> DDR4 pins (ACT_n, ADR, BA, BG, CS_n, CK) -> four x16 DDR4 chips
```

The C++ model consumes commands at the **DDR4 pin boundary**. It cannot infer
MIG's command schedule merely from an AXI transaction. A future AXI-to-DDR
predictor must include a controller scheduler, address mapping, queue limits,
refresh policy, and read/write batching; it should then feed `Ddr4Device`.

## Geometry and address interpretation

Board MIG XCI: one bank-group bit, two bank bits, 15 row bits, 10 column bits,
64 DQ bits, 833 ps DDR clock, 512-bit AXI port. The four device wrappers tie
the second BG pin low. The default model therefore has 2 bank groups x 4
banks/group, 32,768 rows/bank, 1,024 columns/row, and BL8. One BL8 column
command transfers 8 x 64 / 8 = 64 bytes on DQ over 4 CK. This equals one
full AXI512 beat in byte count, but it does not imply one AXI beat becomes
exactly one DDR command after MIG buffering or write combining.

For ACT_n=0, ADR[14:0] is a row address. For ACT_n=1, ADR[16:14] carries
RAS_n/CAS_n/WE_n: `101` READ, `100` WRITE, `010` PRE, `001` REF, `111` NOP.
For READ/WRITE, A10 requests auto-precharge and A12 selects BL8 when high.
The model rejects burst-chop (BC4). PRE with A10 high precharges all banks.
Column is ADR[9:0]. The MIG AXI-address to row/bank/column transformation is
**not** implemented here; pin replay uses the already decoded addresses.

## State and timing

Each bank stores its open row, previous row, next ACT/RD-WR/PRE issue cycles,
pending auto-precharge completion, and column count since ACT. Global state
stores the last ACT, last column command, four-ACT window, refresh recovery,
and optional DQ data-window occupancy. `issue()` advances directly to each
command's CK rather than ticking through idle cycles.

| Command | State transition / acceptance rules |
| --- | --- |
| ACT | Bank closed; tRP, tRC, tRRD_S/L, tFAW, tRFC elapsed. Opens row and starts tRCD/tRAS/tRC timers. |
| READ | Row open; tRCD and tCCD_S/L elapsed. Counts row reuse and read bytes. PRE must later respect tRTP. |
| WRITE | Same row/tCCD checks. PRE must respect write data end plus tWR. |
| PRE/PREA | Open banks satisfy tRAS, tRTP, and tWR; close banks and start tRP. |
| REF | All banks closed and recovered; blocks commands for tRFC. |
| READ/WRITE with A10 | Starts implicit PRE after recovery; bank becomes available after tRP. |

The board XCI provides integer CK values used as defaults: tRCD=17,
tRP=17, tRAS=39, tRTP=10, tWR=19, tRRD_S=7, tRRD_L=8, tFAW=37,
tRFC=313, tREFI=9363. The C++ configuration stores ps and rounds up to
CK; its default ps values encode these XCI integers at tCK=833 ps. tRC=54 CK
comes from the device model's TS_833 44,500 ps value. tCCD_S/L defaults are
4/6 CK. The XCI controller values and the device timing-task values serve
different purposes; do not silently replace one with the other.

`read_latency_ck` and `write_latency_ck` default to 18/12 CK, taken from XCI
`PARAM_VALUE.C0.DDR4_CasLatency` / `CasWriteLatency`. The separate zero-valued
MODELPARAM CL/CWL entries are not physical zero latency. READ/WRITE results
expose DQ begin/end CK and reject overlapping DQ windows. The model also
enforces tRTW=14 CK from generated MIG RTL and write-data-end to READ guards
tWTR_S/L=4/10 CK, including a same-BG write history. Explicit zero CL/CWL can
still disable the corresponding DQ checks. It does not model rank switching,
ODT, DQS edges, data contents,
masking, periodic-refresh debt, or initialization/MRS/ZQ commands. Therefore
an accepted command sequence is a **partial timing check**, not proof of
complete DDR4 protocol compliance. REF is checked when present but not
automatically inserted at tREFI.

### Worked command timeline

The example CSV makes the timer updates visible. With the default 833 ps CK:

1. `ACT bank0,row1` at CK 0 opens the row. `next_col_ck=17` from tRCD,
   `next_pre_ck=39` from tRAS, and `next_act_ck=54` from tRC.
2. `READ col0` at CK 17 is the first legal column command. A second READ at
   CK 23 satisfies same-bank-group tCCD_L=6. The row remains open, so this
   second read needs no ACT and increments `row_reuses`.
3. `PRE bank0` at CK 39 satisfies tRAS and the READ-to-PRE guard. It closes
   the row and moves `next_act_ck` to max(54, 39+17)=56.
4. `ACT bank0,row2` at CK 56 is the first legal reopen. The change from row1
   to row2 increments `row_switches`.
5. `READ` at CK 73 returns data at CK 91..95 with CL=18. `WRITE` at CK 87
   satisfies tRTW=14 and sends data at CK 99..103 with CWL=12. PRE at CK 122
   satisfies write data end plus tWR=19; REF at CK 139 then satisfies tRP=17.
   These are configured-timing model calculations, not a measured DQS trace.

This illustrates the model's central rule: the next legal command time is the
maximum of every applicable bank-level and rank-level timer. A resource can
be free in one dimension (for example, the row is already open) while another
timer (such as tCCD or DQ occupancy) still blocks the command.

## API and replay

- `decode_pins(Ddr4Pins)`: convert a sampled command bus into ACT/RD/WR/PRE/
  PREA/REF/NOP or Unsupported.
- `issue(Ddr4Command)`: update state on acceptance, or return rejection reason
  and earliest legal CK for modeled timing rules. A rejected command does not
  commit its own state transition; an already pending auto-PRE can still
  complete as replay time advances.
- `open_row(bg, bank, at_ck)`: inspect row state, including pending auto-PRE.
- `stats()`: command counts, per-command bus bytes, row reuse/switch counts,
  and rejected commands. These are device counters, not DMA or AXI counters.

The testbench supports `+DDR_COMMAND_TRACE=<output.csv>` to record MIG pin
commands after calibration. Use the existing block-13 simulation invocation
with that plusarg only when a new trace is needed; the C++ build does not run
Vivado or the vendor model. It records one logical rank's shared command pins,
not four duplicate chip commands. Example replay:

```powershell
build/cpp_latency/ddr4_replay.exe --trace simulator/cpp_latency/examples/ddr4_pin_trace_example.csv
build/cpp_latency/ddr4_replay.exe --trace build/ddr_commands.csv
```

The example trace has ACT, same-row READs, PRE, another ACT, READ, WRITE,
PRE, and REF. The second command shows the 17-CK ACT-to-READ delay. A return
code of 2 means at least one command violated a modeled rule. The CLI reports
the first legal CK where a timing constraint can be resolved; state errors
such as an absent row require another command and have no numerical remedy.

## Relation to GPGPU-Sim

GPGPU-Sim's `dram_t` and `bank_t` inspired the command/bank separation and
per-bank state machine. `dram_sched.cc` also demonstrates how the controller
chooses requests, but controller scheduling belongs in a separate MIG
predictor. This implementation is original code; no GPGPU-Sim source was
copied. Reference source: `gpgpu-sim_distribution-dev/src/gpgpu-sim/dram.h`,
`dram.cc`, `dram_sched.cc`, `addrdec.cc`. The vendor DDR4 RTL remains the
functional and protocol reference for later comparison.

## Validation level

The small C++ test covers ACT-to-column, tCCD, PRE, tRP, tFAW, auto-PRE,
pin decoding, and DQ interval calculation. Replaying a real MIG pin trace
will check a larger command sequence. This is **software model validation**;
it is neither a new RTL/MIG simulation nor FPGA board measurement. Total
block latency still requires DMA, MIG/AXI queueing, compute, and scheduler
timelines; do not add command time to an already measured block interval.
