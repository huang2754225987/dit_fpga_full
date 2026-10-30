"""CodeGen — MachineCodeGen.

Emits a 32-bit ``ucode.bin`` from the compiled BIR/DFG using the
double-buffered template that mirrors ``src/pass/MachineCodeGen.cpp``.

The C++ MachineCodeGen walks the lowered BIR (store instructions with
``post_ops``) to extract per-layer flags. This Python version reads the
same information directly from the DFG layer groups produced by Pass 1,
since the DFG already encodes the post-op chain per layer. The emitted
template is identical:

  Phase 0  preload constants (gamma/rope/act)  -> DMA_RD + VLD
  Phase 1  prefetch first weight tile         -> DMA_RD
  Phase 2  per-layer double-buffered pipeline:
            GLD_W(prec) -> BARRIER -> GLD_A -> BARRIER -> GEMM
            (parallel DMA_RD of next weight on non-final layers)
            -> BARRIER -> VLD(out, act_out)
            -> post-proc chain (RoPE/Softmax/SiLU/EltMul/VecAdd)
            -> VST(out) -> BARRIER -> DMA_WR
            -> prepare next layer input (RMSNorm or copy)
            cur_buf ^= 1   (toggle L1 weight buffer A/B)
  END

PostProc RF (v0..v7): V_INPUT=0 V_OUTPUT=1 V_AUX=2 V_ROPE_CS=3
                      V_UP=4 V_RESID=5 V_GAMMA1=6 V_GAMMA2=7
L1 cache: L1_W_BUF_A=0x0000 L1_W_BUF_B=0x1000 L1_ACT_IN=0x2000
          L1_ACT_OUT=0x2800 L1_GAMMA=0x3000 L1_ROPE=0x3100 L1_KV=0x3200
DDR:      WGT=0x0000 ACT=0x4000 GAMMA=0x8000 ROPE=0x8200
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional

import isa
from isa import UInstr, UOpcode, Precision
import layer0
import routed_moe

# Node types are imported lazily at the bottom (see _import_dataflow) to avoid
# an import-order dependency on the dataflow pass module name.


# =============================================================================
# Address map (mirrors MachineCodeGen.cpp constants)
# =============================================================================

def _grouped_moe_l1_fits(tokens: int, hidden: int, intermediate: int) -> bool:
    """Use the emitter's authoritative page/spill planner for grouped MoE."""
    return routed_moe.route_group_l1_fits(tokens, hidden, intermediate)
# PostProc RF indices (vd/vs1/vs2 field is 4 bits)
V_INPUT   = 0
V_OUTPUT  = 1
V_AUX     = 2
V_ROPE_CS = 3
V_UP      = 4
V_RESID   = 5
V_GAMMA1  = 6
V_GAMMA2  = 7

# L1 cache byte addresses (128-bit words * 16)
L1_W_BUF_A = 0x0000
L1_W_BUF_B = 0x1000
L1_ACT_IN  = 0x2000
L1_ACT_OUT = 0x2800
L1_GAMMA   = 0x3000
L1_ROPE    = 0x3100
L1_KV      = 0x3200

# DDR offsets
DDR_WGT_BASE   = 0x0000
DDR_ACT_BASE   = 0x4000
DDR_GAMMA_BASE = 0x8000
DDR_ROPE_BASE  = 0x8200


def _is_op(node, *types):
    return any(isinstance(node, t) for t in types)


class LayerInfo:
    """Per-layer flags + names, extracted from a DFG layer group."""

    def __init__(self, idx: int):
        self.idx = idx
        self.weight_name = ""
        self.input_name = ""
        self.output_name = ""
        self.output_dram_offset = 0
        self.has_rope = False
        self.has_resid = False      # eltadd (residual)
        self.has_silu = False
        self.has_eltmul = False
        self.has_softmax = False
        self.has_rmsnorm = False
        self.has_gemm = False
        self.is_bmm = False


class MachineCodeGen:
    """BIR/DFG -> 32-bit ucode binary."""

    def __init__(self, precision: Precision = Precision.INT8,
                 bin_path: str = "ucode.bin",
                 txt_path: str = "ucode.txt",
                 max_layers: int = 0):
        """max_layers>0 limits transformer layers emitted."""
        self.precision = precision
        self.bin_path = bin_path
        self.txt_path = txt_path
        self.max_layers = max_layers
        self.program: List[UInstr] = []
        self.programs: List[List[UInstr]] = []
        self.stage_paths: List[tuple[str, str]] = []
        self.layers: List[LayerInfo] = []

    # -- emit helpers (one per opcode, mirrors C++ emit*) ---------------
    def emit_end(self):
        self.program.append(isa.end())

    def emit_barrier(self):
        self.program.append(isa.barrier())

    def emit_gemm(self, acc=False, imm=0):
        self.program.append(isa.gemm(acc=acc, imm=imm))

    def emit_silu(self, vd, vs1):
        self.program.append(isa.silu(vd, vs1))

    def emit_sigmoid(self, vd, vs1):
        self.program.append(isa.sigmoid(vd, vs1))

    def emit_broadcast(self, vd, vs1, lane=0):
        self.program.append(isa.broadcast(vd, vs1, lane))

    def emit_vec_mul(self, vd, vs1, vs2):
        self.program.append(isa.mul(vd, vs1, vs2))

    def emit_rmsnorm(self, vd, vs1, vs2):
        self.program.append(isa.rmsnorm(vd, vs1, vs2))

    def emit_softmax(self, vd, vs1, valid_lanes=32):
        self.program.append(isa.softmax(vd, vs1, valid_lanes))

    def emit_rope(self, vd, vs1, vs2):
        self.program.append(isa.rope(vd, vs1, vs2))

    def emit_dma_read(self, addr):
        self.program.append(isa.dma_rd(addr))

    def emit_dma_write(self, addr):
        self.program.append(isa.dma_wr(addr))

    def emit_vec_add(self, vd, vs1, vs2):
        self.program.append(isa.add(vd, vs1, vs2))

    def emit_vld(self, vd, addr):
        self.program.append(isa.vld(vd, addr))

    def emit_vst(self, vs1, addr):
        self.program.append(isa.vst(vs1, addr))

    def emit_vld_token(self, vd, addr, row_stride_bytes, page=0):
        self.program.append(isa.vld_token(vd, addr, row_stride_bytes, page))

    def emit_vst_token(self, vs1, addr, row_stride_bytes, page=0):
        self.program.append(isa.vst_token(vs1, addr, row_stride_bytes, page))

    def emit_gld_w(self, prec, addr):
        self.program.append(isa.gld_w(prec, addr))

    def emit_gld_a(self, addr):
        self.program.append(isa.gld_a(addr))

    # -- layer info extraction from DFG --------------------------------
    def extract_layer_info(self, module) -> List[LayerInfo]:
        dfg = getattr(module, "dfg", None)
        if dfg is None:
            print("[MachineCodeGen] no DFG, nothing to emit")
            return []
        layers = []
        for li, layer_nodes in enumerate(dfg._layers):
            info = LayerInfo(li)
            for node in layer_nodes:
                if _is_op(node, GEMMNode, BMMNode):
                    info.has_gemm = True
                    info.is_bmm = isinstance(node, BMMNode)
                    # weight = rhs placeholder
                    if len(node.input_edges) > 1:
                        rhs = node.input_edges[1].src
                        if hasattr(rhs, "name"):
                            info.weight_name = rhs.name
                    if node.input_edges:
                        lhs = node.input_edges[0].src
                        if hasattr(lhs, "name"):
                            info.input_name = lhs.name
                elif _is_op(node, RoPENode):
                    info.has_rope = True
                elif _is_op(node, AddNode):
                    info.has_resid = True
                elif _is_op(node, SiLUNode):
                    info.has_silu = True
                elif _is_op(node, EltMulNode):
                    info.has_eltmul = True
                elif _is_op(node, SoftmaxNode):
                    info.has_softmax = True
                elif _is_op(node, RMSNormNode):
                    info.has_rmsnorm = True
            # find the layer's output placeholder (last placeholder in group)
            for node in reversed(layer_nodes):
                if hasattr(node, "is_output") and node.is_output:
                    info.output_name = node.name
                    break
            info.output_dram_offset = DDR_ACT_BASE + li * 0x400
            layers.append(info)
        if self.max_layers > 0:
            layers = layers[:self.max_layers]
        self.layers = layers
        return layers

    # -- the main codegen template ------------------------------------
    def run(self, module) -> int:
        self.program = []
        real_layer_cfg = getattr(module, "graph_json", {}).get("real_layer")
        if real_layer_cfg:
            return self.run_real_layer(real_layer_cfg)
        moe_cfg = getattr(module, "graph_json", {}).get("runtime_free_moe")
        if moe_cfg:
            return self.run_runtime_free_moe(moe_cfg)
        self.extract_layer_info(module)
        if not self.layers:
            return 0

        A = module.arch if hasattr(module, "arch") else None
        n_layers = len(self.layers)

        cur_buf = 0   # 0 -> L1_W_BUF_A, 1 -> L1_W_BUF_B
        cur_w_addr = [L1_W_BUF_A, L1_W_BUF_B]

        # ---- Phase 0: preload constants ----
        # gamma
        self.emit_dma_read(DDR_GAMMA_BASE)
        self.emit_barrier()
        self.emit_vld(V_GAMMA1, L1_GAMMA)
        self.emit_vld(V_GAMMA2, L1_GAMMA + 0x40)
        # rope
        self.emit_dma_read(DDR_ROPE_BASE)
        self.emit_barrier()
        self.emit_vld(V_ROPE_CS, L1_ROPE)
        # activation + residual
        self.emit_dma_read(DDR_ACT_BASE)
        self.emit_barrier()
        self.emit_vld(V_INPUT, L1_ACT_IN)
        self.emit_vld(V_RESID, L1_ACT_IN)
        # initial RMSNorm -> VST
        self.emit_rmsnorm(V_INPUT, V_INPUT, V_GAMMA1)
        self.emit_vst(V_INPUT, L1_ACT_IN)

        # ---- Phase 1: prefetch first weight ----
        self.emit_dma_read(DDR_WGT_BASE)
        self.emit_barrier()

        # ---- Phase 2: per-layer pipeline ----
        for i, layer in enumerate(self.layers):
            w_addr = cur_w_addr[cur_buf]

            # GLD_W (load weight to GEMM weight RF)
            self.emit_gld_w(self.precision, w_addr)
            self.emit_barrier()
            # GLD_A (load activation)
            self.emit_gld_a(L1_ACT_IN)
            self.emit_barrier()
            # GEMM
            self.emit_gemm()
            # prefetch next weight on non-final layers (overlap with GEMM)
            if i < n_layers - 1:
                self.emit_dma_read(DDR_WGT_BASE + (i + 1) * 0x400)
            self.emit_barrier()

            # VLD: move GEMM output (L1_ACT_OUT) into PostProc V_OUTPUT
            self.emit_vld(V_OUTPUT, L1_ACT_OUT)

            # ---- post-processing chain ----
            if layer.has_rope:
                self.emit_rope(V_OUTPUT, V_OUTPUT, V_ROPE_CS)
            if layer.has_softmax:
                self.emit_softmax(V_OUTPUT, V_OUTPUT)
            if layer.has_silu:
                self.emit_silu(V_UP, V_OUTPUT)  # save gate*up intermediate
            if layer.has_eltmul:
                self.emit_vec_mul(V_OUTPUT, V_OUTPUT, V_UP)
            if layer.has_resid:
                self.emit_vec_add(V_OUTPUT, V_OUTPUT, V_RESID)
            if layer.has_rmsnorm:
                self.emit_rmsnorm(V_OUTPUT, V_OUTPUT, V_GAMMA2)

            # VST: write PostProc output back to L1
            self.emit_vst(V_OUTPUT, L1_ACT_OUT)
            self.emit_barrier()
            # DMA_WR: write result to DDR
            self.emit_dma_write(layer.output_dram_offset)

            # prepare next layer input
            if i < n_layers - 1:
                next_layer = self.layers[i + 1]
                if next_layer.has_rmsnorm:
                    self.emit_rmsnorm(V_INPUT, V_INPUT, V_GAMMA1)
                    self.emit_vst(V_INPUT, L1_ACT_IN)
                # save up_proj intermediate for SwiGLU if needed
                # (simplified: just update residual pointer)
                self.emit_vld(V_RESID, L1_ACT_IN)

            # toggle double buffer
            cur_buf ^= 1

        self.emit_end()

        # ---- dump ----
        n = isa.dump_program(self.bin_path, self.txt_path, self.program)
        return n

    def run_runtime_free_moe(self, config: dict) -> int:
        """Lower a router-output-to-routed-FFN program without host routing.

        The graph JSON opts in with ``runtime_free_moe``. Router logits must
        already occupy the two ACC sets in the layout consumed by TOPK. This
        explicit contract prevents the generic layer template from pretending
        it has generated a numerically complete transformer layer.
        """
        tokens = int(config.get("tokens", 32))
        hidden = int(config.get("hidden_size", 2048))
        intermediate = int(config.get("intermediate_size", 1408))
        if config.get("router_logits_in_acc", False) is not True:
            raise ValueError("runtime_free_moe requires router_logits_in_acc=true")
        grouped_batches = config.get("grouped_batches", False)
        if grouped_batches is not False and grouped_batches is not True:
            raise ValueError("runtime_free_moe grouped_batches must be a boolean")
        if grouped_batches:
            # This is the compact multi-group path for full-size dimensions
            # that cannot keep the entire token batch in the L1 working set.
            # Each group is written to a distinct 64-KiB output window; keep
            # this opt-in so existing consumers retain their output layout.
            rows_per_group = config.get("rows_per_group")
            if rows_per_group is not None:
                rows_per_group = int(rows_per_group)
            moe_program = routed_moe.emit_all_experts_grouped_batches(
                tokens=tokens, rows_per_group=rows_per_group,
                hidden_size=hidden, intermediate_size=intermediate,
                num_experts=int(config.get("num_experts", 60)),
                cache_routes=True,
                scale_metadata_n_invariant=bool(
                    config.get(
                        "routed_expert_scale_metadata_n_invariant", False)))
            self.program = [*moe_program, isa.end()]
            if len(self.program) > 262144:
                raise ValueError(
                    f"runtime-free MoE program has {len(self.program)} instructions")
            return isa.dump_program(self.bin_path, self.txt_path, self.program)

        grouped_option = config.get("grouped_experts")
        if grouped_option is None:
            # The page-aware grouped path fits the current L1 partition for
            # reduced demos; retain the token-serial path for production-sized
            # dimensions that exceed the bounded staging regions. Grouped
            # execution scans all experts, so keep tiny batches on the direct
            # Top-4 path where cross-token weight reuse is too limited to
            # reliably repay that scan. At eight or more tokens, route overlap
            # amortizes expert staging; empty bodies fast-forward, and the
            # 32-token demo remains on the grouped path.
            grouped_option = (
                8 <= tokens <= 32 and
                _grouped_moe_l1_fits(tokens, hidden, intermediate))
        if grouped_option:
            grouped_experts = int(config.get("num_experts", 60))
            moe_program = routed_moe.emit_all_experts_grouped(
                tokens=tokens, hidden_size=hidden,
                intermediate_size=intermediate,
                num_experts=grouped_experts)
        else:
            moe_program = [
                *routed_moe.emit_router_cache(tokens),
                *routed_moe.emit_all_tokens(tokens, hidden, intermediate),
            ]
        self.program = [
            *moe_program,
            isa.end(),
        ]
        if len(self.program) > 262144:
            raise ValueError(f"runtime-free MoE program has {len(self.program)} instructions")
        return isa.dump_program(self.bin_path, self.txt_path, self.program)

    def run_real_layer(self, config: dict) -> int:
        """Lower a serialized manifest-driven real-weight Layer-0 timing run."""
        self.programs = []
        self.stage_paths = []
        manifest_path = config.get("manifest")
        if not manifest_path:
            raise ValueError("real_layer requires a manifest path")
        tokens = int(config.get("tokens", 1))
        if not 1 <= tokens <= 32:
            raise ValueError("real_layer tokens must be in [1,32]")
        manifest = layer0.load_manifest(manifest_path)
        decode_position = config.get("decode_position")
        if decode_position is not None:
            if tokens != 1:
                raise ValueError(
                    "decode_position compiles exactly one token; set tokens=1")
            if "input_byte_offset" not in config:
                raise ValueError(
                    "position-specialized decode requires input_byte_offset")
            self.program = layer0.emit_decode_token(
                manifest, position=int(decode_position),
                input_byte_offset=int(config["input_byte_offset"]),
                k_cache_window_base=int(config.get("k_cache_window_base", 0)),
                v_cache_window_base=int(config.get("v_cache_window_base", 2)),
                output_window_base=int(config.get("output_window_base", 4)))
            return isa.dump_program(self.bin_path, self.txt_path, self.program)
        available_tokens = int(manifest.get("activation_source", {}).get("tokens", 1))
        if config.get("prefill_layer0_staged") or config.get("prefill_layer0"):
            input_start_token = int(config.get("input_start_token", 0))
            required_tokens = input_start_token + tokens
            if available_tokens < required_tokens:
                raise ValueError(
                    f"manifest contains {available_tokens} activation tokens, "
                    f"but complete prefill needs tokens through {required_tokens - 1}")
            if config.get("prefill_input_is_rmsnorm", False):
                raise ValueError(
                    "complete Layer-0 prefill expects raw ACT rows and emits input RMSNorm")
            if config.get("prefill_layer0_staged"):
                self.programs = layer0.emit_prefill_layer0_programs(
                    manifest, tokens=tokens,
                    input_start_token=input_start_token,
                    attention_rows_per_group=(
                        None if config.get("attention_rows_per_group") is None
                        else int(config["attention_rows_per_group"])),
                    first_position=int(config.get(
                        "rope_position_start", input_start_token)))
                bin_path = Path(self.bin_path)
                txt_path = Path(self.txt_path)
                total_instructions = 0
                for index, program in enumerate(self.programs):
                    stage_bin = bin_path.with_name(
                        f"{bin_path.stem}.stage{index:02d}{bin_path.suffix}")
                    stage_txt = txt_path.with_name(
                        f"{txt_path.stem}.stage{index:02d}{txt_path.suffix}")
                    self.stage_paths.append((str(stage_bin), str(stage_txt)))
                    total_instructions += isa.dump_program(
                        str(stage_bin), str(stage_txt), program)
                self.program = []
                return total_instructions
            self.program = layer0.emit_prefill_layer0_stage(
                manifest, tokens=tokens,
                input_start_token=input_start_token,
                rows_per_group=int(config.get("rows_per_group", 8)),
                attention_rows_per_group=(
                    None if config.get("attention_rows_per_group") is None
                    else int(config["attention_rows_per_group"])),
                first_position=int(config.get(
                    "rope_position_start", input_start_token)))
            return isa.dump_program(self.bin_path, self.txt_path, self.program)
        if config.get("prefill_attention_stage"):
            input_start_token = int(config.get("input_start_token", 0))
            required_tokens = input_start_token + tokens
            if available_tokens < required_tokens:
                raise ValueError(
                    f"manifest contains {available_tokens} activation tokens, "
                    f"but prefill attention needs tokens through {required_tokens - 1}")
            norm_program = []
            input_source_windows = None
            if not config.get("prefill_input_is_rmsnorm", False):
                norm_window_base = int(config.get(
                    "prefill_norm_window_base", 12))
                norm_program = layer0.emit_prefill_input_rmsnorm(
                    manifest, tokens=tokens,
                    input_start_token=input_start_token,
                    output_window_base=norm_window_base)
                input_source_windows = [
                    (norm_window_base + token // 16, token % 16)
                    for token in range(0, tokens, 8)
                ]
            attention_program = layer0.emit_prefill_attention_stage(
                manifest, tokens=tokens, input_start_token=input_start_token,
                qkv_window_base=int(config.get("qkv_window_base", 0)),
                first_position=int(config.get(
                    "rope_position_start", input_start_token)),
                attention_output_window_base=int(
                    config.get("attention_output_window_base", 6)),
                output_window_base=int(config.get("o_projection_window_base", 10)),
                attention_rows_per_group=(
                    int(config["attention_rows_per_group"])
                    if config.get("attention_rows_per_group") is not None else
                    (32 if tokens > 16 else 16 if tokens >= 16 else 8)),
                input_source_windows=input_source_windows)
            self.program = [*norm_program, *attention_program]
            if len(self.program) > 262144:
                raise ValueError(
                    f"prefill attention program has {len(self.program)} instructions")
            return isa.dump_program(self.bin_path, self.txt_path, self.program)
        prefill_projection = config.get("prefill_fp16_projection")
        if prefill_projection is not None:
            input_start_token = int(config.get("input_start_token", 0))
            required_tokens = input_start_token + tokens
            if available_tokens < required_tokens:
                raise ValueError(
                    f"manifest contains {available_tokens} activation tokens, "
                    f"but projection needs tokens through {required_tokens - 1}")
            if not isinstance(prefill_projection, str) or not prefill_projection:
                raise ValueError("prefill_fp16_projection must name a projection suffix")
            self.program = layer0.emit_prefill_fp16_projection(
                manifest, prefill_projection, tokens=tokens,
                input_start_token=input_start_token)
        else:
            if available_tokens < tokens:
                raise ValueError(
                    f"manifest contains {available_tokens} activation tokens, requested {tokens}")
            self.program = layer0.emit_tokens(manifest, tokens=tokens)
        return isa.dump_program(self.bin_path, self.txt_path, self.program)


# late import: the dataflow module is named `dataflow` inside `passes`
def _import_dataflow():
    import importlib
    try:
        return importlib.import_module("passes.dataflow")
    except ImportError:
        return importlib.import_module("pass_dataflow")


# rebind the pass-internal names via lazy import so this module stands alone
_mod = _import_dataflow()
GEMMNode = _mod.GEMMNode
BMMNode = _mod.BMMNode
RMSNormNode = _mod.RMSNormNode
RoPENode = _mod.RoPENode
SoftmaxNode = _mod.SoftmaxNode
SiLUNode = _mod.SiLUNode
EltMulNode = _mod.EltMulNode
AddNode = _mod.AddNode
TransposeNode = _mod.TransposeNode
ReshapeNode = _mod.ReshapeNode
ConcatNode = _mod.ConcatNode
