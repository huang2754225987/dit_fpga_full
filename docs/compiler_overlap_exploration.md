# Compiler overlap exploration

The existing `compile_block13_tile_program.py` is a static schedule prototype. Its `Task` graph, token dependencies and exclusive-resource scheduler are useful starting points, but `GEMM_TILE` currently includes preprocessing, the connected sequencer executes one command at a time, and the hardware has one weight store. Its 1,280-byte abstract weight banks represent 1-KiB tiles plus parameters, whereas the `DMA_WEIGHT` tasks describe whole-stage loads. They must not be treated as capacity for whole matrices.

Compiler changes for a future parallel executor:

1. Lower GEMM into activation preparation, weight load, full-K compute and output tasks with explicit source, destination, byte range, shape, metadata and buffer generation. Keep the full-K reduction and apply epilogue only after its last partial sum. A K-split variant requires explicit partial-sum lifetime and merging.
2. Allocate buffers according to actual capacity and ports. Choose either complete stage-weight buffers with sufficient space or individually described weight tiles with matching compute consumers. Track scale, bias, token scale and ownership as well as data. Insert RAW, WAR and WAW dependencies, including all consumers before buffer reuse; resource mutual exclusion alone does not preserve lifetime.
3. Schedule DMA, preparation and array tasks independently, using a consistent finite-buffer model, realistic port conflicts and calibrated durations. Separate assumed and measured inputs. Compare a serial schedule and an overlap schedule with identical task granularity and costs.
4. Define a versioned descriptor/ISA and runtime contract for multiple in-flight commands, dependency tokens, finite dispatch windows, multiple completions, end-of-program draining, buffer ownership and token lifetime. Current low eight bits encode dependency count; the JSON encoding label calling them flags is misleading. Do not emit new words as executable until decoder and scheduler support them.
5. Validate dependencies, buffer capacities and versions, port conflicts, issue-window progress, token limits and end draining. Check numerical output and temporal ordering in the matching integrated RTL, then measure synthesis/timing. Do not treat a static schedule replay as datapath validation.

`compiler/supra2/explore_instruction_overlap.py` supplies a deliberately small non-executable two-stage graph. It models separate DMA/prep/array resources, stage weights in W0/W1, and two activation buffers with read-before-overwrite dependencies. The stages' hypothetical durations are arbitrary; the base and prefetch variants both overlap activation prep and differ only in when stage 1 weight loading may occur. Running `python compiler/supra2/explore_instruction_overlap.py --output <report.json>` checks the graph and emits event times. The hypothetical result 1,100 versus 1,040 cycles illustrates scheduling only; it is not a block-13 prediction, an ISA implementation or RTL verification. Actual stage weight capacity and hardware feasibility are outside this toy graph.

## Block-13 per-stage capacity and frontend exploration

`compiler/supra2/explore_block13_gemm_pipeline.py` independently analyzes each of the seven actual non-programmed GEMM shapes. It checks capacity for a whole INT8 stage matrix plus FP32 scale/bias and for a 32-row activation tile plus FP32 token scales. A task graph holds each activation slot through its GEMM consumer and records exclusive DMA, frontend and array intervals. The serial comparison waits for weight load and processes prep and GEMM row by row; it is a synthetic comparison, not the current RTL's all-frontend-first baseline. The overlap case permits the weight load and prep to start together and allows later prep to overlap compute. Each stage is isolated and assumes input available at cycle zero: this does not implement or estimate cross-stage prefetch, inter-stage dependencies or a full block.

The default weight capacity is 1,280 bytes per buffer, so every whole-stage case is rejected. For a hypothetical two-buffer whole-stage design, MLP0 needs 1,345,536 bytes per weight buffer and MLP2 needs 73,856 bytes per activation buffer. The test configuration provides these maxima *per buffer*; two of each imply 2,691,072 + 147,712 logical payload bytes before padding, routing, other buffers and FPGA memory mapping. It is an assumption, not evidence those resources fit.

Example runs:

```
python -m unittest discover -s compiler/supra2 -p test_block13_gemm_pipeline.py -v
python compiler/supra2/explore_block13_gemm_pipeline.py --output build/cpp_latency_msys2/block13_capacity_check.json
python compiler/supra2/explore_block13_gemm_pipeline.py --weight-buffer-bytes 1345536 --output build/cpp_latency_msys2/block13_gemm_exploration.json
```

Frontend time defaults to a hypothetical 64 cycles per row for K=576, with assumed linear K scaling. GEMM time uses full-K feed, one assumed wait cycle and a serial 32-row drain; weight durations are existing stage wait counters used only as illustrative proxies. The comparisons omit DDR scheduling, memory-port contention, controller costs and downstream computation. They are not calibrated RTL latency or whole-block speedups. This path emits no instruction binary and does not alter the existing compiler's executable output.
