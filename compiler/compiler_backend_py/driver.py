"""Driver — NPU compiler backend pipeline.

Runs the 10-pass compile pipeline plus the codegen pass, producing
``ucode.bin``.

Usage:
    python driver.py -i graph.json [-o ucode.bin] [--precision int8|fp16]
                      [--core 1] [--dsp 1024]

Pipeline (C++ order):
    dataflow -> lower_dfg -> loop_partition -> loop_memory_coalesce
    -> loop_opt -> loop_unroll -> loop_allocator -> ada
    -> loop_scheduler -> loop_eventgen
    + codegen: MachineCodeGen
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# ensure this package dir is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bir.module import Module
from bir.arch import Arch
import isa
from isa import Precision
from passes.dataflow import DataFlowPass
from passes.machine_codegen import MachineCodeGen


# Pass registry — each is (name, factory). Passes that are not yet
# implemented are registered as no-op stubs so the pipeline still runs
# end-to-end and produces ucode.bin.
def _make_passes(arch: Arch):
    from passes.loop_partition import LoopPartitionPass
    from passes.loop_coalesce import LoopCoalescePass
    from passes.loop_opt import LoopOptPass
    from passes.loop_unroll import LoopUnrollPass
    from passes.loop_allocator import LoopAllocatorPass
    from passes.ada import ADAPass
    from passes.loop_scheduler import LoopSchedulerPass
    from passes.loop_eventgen import LoopEventGenPass
    from passes.lower_dfg import LowerDFGPass

    return [
        ("dataflow",            DataFlowPass()),
        ("lower_dfg",           LowerDFGPass()),
        ("loop_partition",      LoopPartitionPass(arch.core_num)),
        ("loop_memory_coalesce", LoopCoalescePass()),
        ("loop_opt",            LoopOptPass()),
        ("loop_unroll",         LoopUnrollPass()),
        ("loop_allocator",      LoopAllocatorPass()),
        ("ada",                 ADAPass()),
        ("loop_scheduler",      LoopSchedulerPass()),
        ("loop_eventgen",       LoopEventGenPass()),
    ]


def run_pipeline(module: Module, arch: Arch, do_codegen: bool = True,
                  out_bin: str = "ucode.bin", out_txt: str = "ucode.txt",
                  precision: Precision = Precision.INT8,
                  max_layers: int = 0, mode: str = "decode"):
    passes = _make_passes(arch)
    print("=" * 60)
    print(f"NPU Compiler Backend Pipeline (Python)  [mode={mode}]")
    print("=" * 60)
    for name, p in passes:
        try:
            rc = p.run(module)
            status = "ok" if rc == 0 else f"rc={rc}"
            print(f"  [pass] {name:22s} {status}")
        except Exception as e:
            print(f"  [pass] {name:22s} SKIP ({type(e).__name__}: {e})")

    if do_codegen:
        print("-" * 60)
        cg = MachineCodeGen(precision, out_bin, out_txt,
                            max_layers=max_layers)
        n = cg.run(module)
        print(f"  [codegen] machine_codegen      {n} instructions -> {out_bin}")
        # write a sidecar config the testbench reads to know mode + n_instr
        cfg_path = os.path.splitext(out_bin)[0] + ".cfg"
        with open(cfg_path, "w") as f:
            f.write(f"mode={mode}\n")
            f.write(f"n_instr={n}\n")
            f.write(f"max_layers={max_layers}\n")
            graph_cfg = getattr(module, "graph_json", {})
            moe_cfg = graph_cfg.get("runtime_free_moe", {})
            real_layer_cfg = graph_cfg.get("real_layer", {})
            seq_len = int(real_layer_cfg.get("tokens", moe_cfg.get(
                "tokens", 4 if mode == "prefill" else 1)))
            f.write(f"seq_len={seq_len}\n")
            f.write(f"runtime_free_moe={int(bool(moe_cfg))}\n")
            f.write(f"real_layer={int(bool(real_layer_cfg))}\n")
        print(f"  [codegen] wrote config -> {cfg_path}")
    print("=" * 60)


def main(argv=None):
    ap = argparse.ArgumentParser(description="NPU compiler backend (Python)")
    ap.add_argument("-i", "--input", required=True, help="graph.json (GIR export)")
    ap.add_argument("-o", "--output", default="ucode.bin", help="output ucode.bin")
    ap.add_argument("--txt", default=None, help="output ucode.txt (default: <output>.txt)")
    ap.add_argument("--precision", default="int8", choices=["int8", "fp16", "int4", "int3"])
    ap.add_argument("--core", type=int, default=1, help="core_num")
    ap.add_argument("--dsp", type=int, default=1024, help="DSP count (32x32=1024)")
    ap.add_argument("--no-codegen", action="store_true", help="skip codegen pass")
    ap.add_argument("--max-layers", type=int, default=0,
                    help="limit transformer layers emitted (0=all; prefill sim uses 1-2)")
    ap.add_argument("--mode", default="decode", choices=["decode", "prefill"],
                    help="decode=single-token, prefill=multi-token (affects CFG_M in TB)")
    args = ap.parse_args(argv)

    out_txt = args.txt or (os.path.splitext(args.output)[0] + ".txt")

    # arch
    arch = Arch()
    arch.core_num = args.core
    arch.dsp = args.dsp
    arch.multipliers = arch.dsp * (2 // arch.data_bytes)
    prec_map = {"int8": 8, "fp16": 16, "int4": 4, "int3": 3}
    arch.precision = prec_map[args.precision]

    # module + graph.json
    module = Module(arch)
    with open(args.input) as f:
        module.graph_json = json.load(f)

    prec_enum = {"int8": Precision.INT8, "fp16": Precision.FP16,
                 "int4": Precision.INT4, "int3": Precision.INT3}[args.precision]

    # override codegen precision
    import passes.machine_codegen as mcg
    run_pipeline(module, arch, do_codegen=not args.no_codegen,
                 out_bin=args.output, out_txt=out_txt, precision=prec_enum,
                 max_layers=args.max_layers, mode=args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
