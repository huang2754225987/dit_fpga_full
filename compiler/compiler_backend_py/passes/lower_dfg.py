"""Pass 2 - LowerDFG: lower DFG nodes into BIR loop nests.

Python reimplementation of ``src/pass/LowerDFG.{h,cpp}``.

Walks ``dfg._layers`` and, for each GEMM/BMM key node, emits four loop
nests (load A -> FB, load B -> WB, matmult -> PSUM, store -> DRAM) plus a
post-op chain (RMSNorm/RoPE/Softmax/SiLU/EltMul/EltAdd/Transpose). This
mirrors the C++ reference lowering; the resulting BIR is what the
optimization passes (partition/opt/unroll/allocator) transform.

For the single-core decode target the tiling factors collapse to the full
dimension (m_tile = M, k_tile = K, n_tile = N) so each nest is a single
perfect loop per dimension. The MachineCodeGen reads the DFG flags
directly for its double-buffered template, so the loops emitted here are
structural (they record the lowering plan) rather than the final ucode.
"""
from __future__ import annotations

from typing import List, Optional

from bir.module import Module, Function, BasicBlock
from bir.loop import LoopInst
from bir.instruction import (LoadInst, StoreInst, MatmultInst, VectorInst,
                             InstructionType, VectorType)
from bir.memory import MemoryLocation, MemoryType, DataType
from bir.access_pattern import AccessPattern
from bir.expr import VarExpr, ConstExpr

from passes.dataflow import (GEMMNode, BMMNode, RMSNormNode, RoPENode,
                             SoftmaxNode, SiLUNode, EltMulNode, AddNode,
                             TransposeNode, ReshapeNode, PlaceHolderNode)


# on-chip mloc cache per function
class _MlocCache:
    def __init__(self):
        self.by_name: dict = {}

    def get(self, name, mem_type, shape=None):
        key = (name, mem_type)
        if key not in self.by_name:
            self.by_name[key] = MemoryLocation(
                name=f"{name}_{mem_type.name}", mem_type=mem_type,
                tensor_shape=shape)
        return self.by_name[key]


class LowerDFGPass:
    """Pass: lower_dfg - DFG -> BIR loop nests."""

    def __init__(self, *args, **kwargs):
        self.fn: Optional[Function] = None
        self.mloc: _MlocCache = None

    def run(self, module: Module) -> int:
        dfg = getattr(module, "dfg", None)
        if dfg is None:
            print("[LowerDFG] no DFG, skipping")
            return 0

        # one function "main" with one BasicBlock per layer
        if hasattr(module, "add_function"):
            self.fn = module.add_function("main")
        else:
            self.fn = None
        self.mloc = _MlocCache()

        plan: List[dict] = []
        for li, layer_nodes in enumerate(dfg._layers):
            bb = self.fn.add_basic_block(f"layer_{li+1}") if self.fn is not None else None
            entry = self._lower_layer(bb, layer_nodes, li)
            plan.append(entry)

        module.lower_plan = plan
        print(f"[LowerDFG] lowered {len(plan)} layers into BIR BasicBlocks")
        return 0

    # ------------------------------------------------------------------
    def _lower_layer(self, bb: Optional[BasicBlock], nodes: list, li: int) -> dict:
        entry = {"layer": li, "ops": [], "loops_emitted": 0}
        keynode = None
        post_chain: list = []
        for node in nodes:
            if isinstance(node, (GEMMNode, BMMNode)):
                keynode = node
            elif not isinstance(node, PlaceHolderNode):
                post_chain.append(node)

        # lower the key GEMM/BMM
        if keynode is not None:
            if isinstance(keynode, GEMMNode):
                n_loops = self._lower_gemm(bb, keynode)
            else:
                n_loops = self._lower_bmm(bb, keynode)
            entry["ops"].append(keynode.type)
            entry["loops_emitted"] += n_loops

        # lower the post-op chain
        for node in post_chain:
            self._lower_post_op(bb, node, keynode)
            entry["ops"].append(node.type)

        return entry

    # ------------------------------------------------------------------
    def _lower_gemm(self, bb: BasicBlock, g: GEMMNode) -> int:
        """Emit load A -> FB, load B -> WB, matmult -> PSUM, store -> DRAM."""
        M, K, N = g.M, g.K, g.N
        n_loops = 0

        # access patterns: A@DRAM[i,k], B@DRAM[k,j], C@PSUM[i,j], C@DRAM[i,j]
        a_dram = self._mloc_for_edge(g.input_edges[0] if g.input_edges else None,
                                    MemoryType.DRAM)
        b_dram = self._mloc_for_edge(g.input_edges[1] if len(g.input_edges) > 1 else None,
                                    MemoryType.DRAM)
        ta_fb = MemoryLocation("TA", MemoryType.FB, tensor_shape=[M, K])
        tb_wb = MemoryLocation("TB", MemoryType.WB, tensor_shape=[K, N])
        tc_psum = MemoryLocation("TC", MemoryType.PSUM, tensor_shape=[M, N])
        c_dram = MemoryLocation("C_out", MemoryType.DRAM, tensor_shape=[M, N])

        # nest 1: load A -> TA@FB,  for i0 in [0,M) for k0 in [0,K)
        loop_a = self._emit_load_loop(bb, "i0", 0, M, "k0", 0, K,
                                      src=(a_dram, 2), dst=(ta_fb, 2))
        # nest 2: load B -> TB@WB,  for k1 in [0,K) for j1 in [0,N)
        loop_b = self._emit_load_loop(bb, "k1", 0, K, "j1", 0, N,
                                      src=(b_dram, 2), dst=(tb_wb, 2))
        # nest 3: matmult  for i2,j2,k2
        loop_m = self._emit_matmult_loop(bb, "i2", 0, M, "j2", 0, N, "k2", 0, K,
                                         a=(ta_fb, 2), b=(tb_wb, 2), psum=(tc_psum, 2))
        # nest 4: store  for i3 in [0,M) for j3 in [0,N)
        loop_s = self._emit_store_loop(bb, "i3", 0, M, "j3", 0, N,
                                       src=(tc_psum, 2), dst=(c_dram, 2))
        n_loops = 4
        # mark reduce loop
        if loop_m is not None:
            loop_m._is_matmult_reduce = True
            # set unroll annotations for output-stationary ordering
            loop_a.set_unroll(True)
            loop_b.set_unroll(True)
            loop_m._free = True
        return n_loops

    def _lower_bmm(self, bb: BasicBlock, b: BMMNode) -> int:
        """Like GEMM but with an outermost batch loop b in [0, Batch)."""
        B, M, K, N = b.batch, b.M, b.K, b.N
        # wrap the four GEMM nests in a batch loop (simplified: emit b loop + GEMM)
        outer = LoopInst("b", bb, 0, B, 1)
        outer._parallel = True
        outer.set_unroll(True)
        if bb is not None:
            bb.add_instruction(outer)
        # inner GEMM (reuse _lower_gemm on the outer's body)
        inner_bb = outer.body.body if hasattr(outer.body, "body") else None
        g = GEMMNode(0, M, K, N)
        g.input_edges = b.input_edges
        self._lower_gemm(inner_bb, g)
        return 5

    # ------------------------------------------------------------------
    def _lower_post_op(self, bb: BasicBlock, node, keynode):
        """Lower a post-op node into a VectorInst loop (single instruction, perfect)."""
        vtype, imm = self._post_op_kind(node)
        if vtype is None:
            return
        # single perfect loop nest: for m in [0,M) for n in [0,N)  VectorInst
        # (dimensions inferred from keynode; default M=N=hidden)
        M = N = self._infer_dim(node, keynode)
        loop_m = LoopInst("m", bb, 0, M, 1)
        loop_m.set_unroll(True)
        loop_n = LoopInst("n", loop_m.body.body, 0, N, 1)
        loop_n.set_spatial(True)
        # the VectorInst reads+writes psum in-place
        psum = MemoryLocation("psum", MemoryType.PSUM, tensor_shape=[M, N])
        vi = VectorInst("vop", vtype, loop_n.body.body)
        vi.imm = imm
        vi.input_aps.append(AccessPattern(access=[ConstExpr(0)], mloc=psum))
        vi.output_aps.append(AccessPattern(access=[ConstExpr(0)], mloc=psum))
        loop_n.body.body.add_instruction(vi) if hasattr(loop_n.body.body, "add_instruction") else None
        # attach extra operands
        self._attach_post_op_operands(node, vi, psum)
        if bb is not None:
            bb.add_instruction(loop_m)

    def _post_op_kind(self, node):
        """Map a DFG post-op node to (VectorType, imm)."""
        if isinstance(node, RMSNormNode):
            return VectorType.RMSNorm, float(getattr(node, "N", 0))
        if isinstance(node, SoftmaxNode):
            return VectorType.Softmax, 0.0
        if isinstance(node, RoPENode):
            return VectorType.RoPE, float(getattr(node, "head_dim", 0))
        if isinstance(node, SiLUNode):
            return VectorType.EltwiseSiLU, 0.0
        if isinstance(node, GELUNode := type(None)) or node.__class__.__name__ == "GELUNode":
            return VectorType.EltwiseGELU, 0.0
        if isinstance(node, EltMulNode):
            return VectorType.EltwiseMul, 0.0
        if isinstance(node, AddNode):
            return VectorType.EltwiseAdd, 0.0
        return None, 0.0

    def _infer_dim(self, node, keynode):
        if keynode is not None:
            return getattr(keynode, "N", 64) or 64
        return 64

    def _attach_post_op_operands(self, node, vi, psum):
        """Add the second operand (residue/gamma/rope/up) as a 2nd input AP."""
        if isinstance(node, (AddNode, EltMulNode)):
            rhs = MemoryLocation("rhs", MemoryType.RESIDUEBANK)
            vi.input_aps.append(AccessPattern(access=[ConstExpr(0)], mloc=rhs))
        elif isinstance(node, (RMSNormNode,)):
            gamma = MemoryLocation("gamma", MemoryType.ROWBANK)
            vi.input_aps.append(AccessPattern(access=[ConstExpr(0)], mloc=gamma))

    # ------------------------------------------------------------------
    # loop-nest emitters
    # ------------------------------------------------------------------
    def _emit_load_loop(self, bb, name0, lb0, ub0, name1, lb1, ub1,
                        src=None, dst=None):
        """for i0 in [lb0,ub0) for k0 in [lb1,ub1)  LoadInst src->dst."""
        loop0 = LoopInst(name0, bb, lb0, ub0, 1)
        loop0.set_unroll(True)
        inner_bb = loop0.body.body if hasattr(loop0.body, "body") else None
        loop1 = LoopInst(name1, inner_bb, lb1, ub1, 1)
        loop1.set_spatial(True)
        body_bb = loop1.body.body if hasattr(loop1.body, "body") else None
        ld = LoadInst("ld", body_bb)
        if src and dst:
            ld.input_aps.append(AccessPattern(access=[ConstExpr(0)], mloc=src[0]))
            ld.output_aps.append(AccessPattern(access=[ConstExpr(0)], mloc=dst[0]))
        if body_bb is not None and hasattr(body_bb, "add_instruction"):
            body_bb.add_instruction(ld)
        if bb is not None and hasattr(bb, "add_instruction"):
            bb.add_instruction(loop0)
        return loop0

    def _emit_store_loop(self, bb, name0, lb0, ub0, name1, lb1, ub1,
                         src=None, dst=None):
        loop0 = LoopInst(name0, bb, lb0, ub0, 1)
        loop0.set_unroll(True)
        inner_bb = loop0.body.body if hasattr(loop0.body, "body") else None
        loop1 = LoopInst(name1, inner_bb, lb1, ub1, 1)
        loop1.set_spatial(True)
        body_bb = loop1.body.body if hasattr(loop1.body, "body") else None
        st = StoreInst("st", body_bb)
        if src and dst:
            st.input_aps.append(AccessPattern(access=[ConstExpr(0)], mloc=src[0]))
            st.output_aps.append(AccessPattern(access=[ConstExpr(0)], mloc=dst[0]))
        if body_bb is not None and hasattr(body_bb, "add_instruction"):
            body_bb.add_instruction(st)
        if bb is not None and hasattr(bb, "add_instruction"):
            bb.add_instruction(loop0)
        return loop0

    def _emit_matmult_loop(self, bb, n0, lb0, ub0, n1, lb1, ub1, n2, lb2, ub2,
                           a=None, b=None, psum=None):
        loop0 = LoopInst(n0, bb, lb0, ub0, 1)
        loop0.set_unroll(True)
        bb1 = loop0.body.body if hasattr(loop0.body, "body") else None
        loop1 = LoopInst(n1, bb1, lb1, ub1, 1)
        loop1.set_unroll(True)
        bb2 = loop1.body.body if hasattr(loop1.body, "body") else None
        loop2 = LoopInst(n2, bb2, lb2, ub2, 1)
        loop2.set_spatial(True)
        body_bb = loop2.body.body if hasattr(loop2.body, "body") else None
        mm = MatmultInst("mm", body_bb)
        if a and b and psum:
            mm.input_aps.append(AccessPattern(access=[ConstExpr(0)], mloc=a[0]))
            mm.input_aps.append(AccessPattern(access=[ConstExpr(0)], mloc=b[0]))
            mm.input_aps.append(AccessPattern(access=[ConstExpr(0)], mloc=psum[0]))
            mm.output_aps.append(AccessPattern(access=[ConstExpr(0)], mloc=psum[0]))
        if body_bb is not None and hasattr(body_bb, "add_instruction"):
            body_bb.add_instruction(mm)
        if bb is not None and hasattr(bb, "add_instruction"):
            bb.add_instruction(loop0)
        return loop0

    def _mloc_for_edge(self, edge, mem_type):
        if edge is None:
            return MemoryLocation("unknown", mem_type)
        src = edge.src
        if isinstance(src, PlaceHolderNode) and src.mloc is not None:
            return src.mloc
        name = getattr(src, "name", "t")
        return MemoryLocation(name, mem_type)


__all__ = ["LowerDFGPass"]
