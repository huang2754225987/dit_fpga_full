"""Pass 1 — DataFlowGraph.

Builds a ``DataFlowGraph`` from a GIR ``graph.json`` export. The DFG is a
DAG of typed compute nodes (GEMM, BMM, Softmax, RoPE, RMSNorm, SiLU,
EltMul, EltAdd, Transpose, Reshape, Concat, ...) connected by edges that
carry ``MemoryLocation`` placement hints. PlaceHolders represent tensors
living in DRAM.

This is consumed by Pass 2 (LowerDFG), which walks the DFG and emits BIR
loop nests.

Python reimplementation of ``src/pass/DataFlowGraph.{h,cpp}``.
"""
from __future__ import annotations

import json
from typing import List, Dict, Optional, Any

from bir.memory import MemoryLocation, MemoryType, DataType


# =============================================================================
# DFG nodes
# =============================================================================
class DataFlowNode:
    """Base DFG node."""

    def __init__(self, nid: int):
        self.id = nid
        self.type = "base"
        self.input_edges: List["DataFlowEdge"] = []
        self.output_edges: List["DataFlowEdge"] = []
        self.input_shapes: List[list] = []
        self.output_shapes: List[list] = []
        self.step = 0          # decode step (for KV cache)
        self.layer_index = 0   # which transformer layer

    def __repr__(self):
        return f"{self.type}(id={self.id})"


class PlaceHolderNode(DataFlowNode):
    """A tensor living in DRAM (or on-chip if static)."""

    def __init__(self, nid: int, name: str):
        super().__init__(nid)
        self.type = "placeholder"
        self.name = name
        self.mloc: Optional[MemoryLocation] = None
        self.data_type: DataType = DataType.FIX16
        self.fraclen = 0
        self.is_static = False
        self.is_output = False
        self.is_input = False
        self.is_k_cache = False
        self.is_v_cache = False
        self.kv_cache_info: Dict = {"step": None, "depth": None}
        self.tensor_shape: list = []


class GEMMNode(DataFlowNode):
    def __init__(self, nid, M, K, N, sparse=False):
        super().__init__(nid)
        self.type = "gemm"
        self.batch, self.M, self.K, self.N = 1, M, K, N
        self.sparse = sparse


class BMMNode(DataFlowNode):
    def __init__(self, nid, Batch, M, K, N, SubBatch=1):
        super().__init__(nid)
        self.type = "bmm"
        self.batch, self.M, self.K, self.N = Batch, M, K, N
        self.sub_batch = SubBatch


class TransposeNode(DataFlowNode):
    def __init__(self, nid, dims):
        super().__init__(nid)
        self.type = "transpose"
        self.dims = dims


class SoftmaxNode(DataFlowNode):
    def __init__(self, nid, dim=-1, valid_length=0):
        super().__init__(nid)
        self.type = "softmax"
        self.dim = dim
        self.valid_length = valid_length
        self.inout_left_shift = (0, 0)


class RoPENode(DataFlowNode):
    def __init__(self, nid, seqlen, head_dim):
        super().__init__(nid)
        self.type = "rope"
        self.seqlen = seqlen
        self.head_dim = head_dim


class AddNode(DataFlowNode):
    def __init__(self, nid):
        super().__init__(nid)
        self.type = "eltadd"


class DivNode(DataFlowNode):
    def __init__(self, nid):
        super().__init__(nid)
        self.type = "divide"


class GELUNode(DataFlowNode):
    def __init__(self, nid):
        super().__init__(nid)
        self.type = "gelu"


class SiLUNode(DataFlowNode):
    def __init__(self, nid):
        super().__init__(nid)
        self.type = "silu"


class EltMulNode(DataFlowNode):
    def __init__(self, nid):
        super().__init__(nid)
        self.type = "eltmul"


class ConcatNode(DataFlowNode):
    def __init__(self, nid, dim=-1):
        super().__init__(nid)
        self.type = "concat"
        self.dim = dim


class SplitNode(DataFlowNode):
    def __init__(self, nid, split_dim, split_size):
        super().__init__(nid)
        self.type = "split"
        self.split_dim = split_dim
        self.split_size = split_size


class EmbeddingNode(DataFlowNode):
    def __init__(self, nid):
        super().__init__(nid)
        self.type = "embedding"


class ReshapeNode(DataFlowNode):
    def __init__(self, nid):
        super().__init__(nid)
        self.type = "reshape"


class ArgMaxNode(DataFlowNode):
    def __init__(self, nid):
        super().__init__(nid)
        self.type = "argmax"


class RMSNormNode(DataFlowNode):
    def __init__(self, nid, N):
        super().__init__(nid)
        self.type = "rmsnorm"
        self.N = N


class CausalMaskingNode(DataFlowNode):
    def __init__(self, nid, seqlen):
        super().__init__(nid)
        self.type = "causal_masking"
        self.seqlen = seqlen


class LayerNormNode(DataFlowNode):
    def __init__(self, nid, N):
        super().__init__(nid)
        self.type = "layernorm"
        self.N = N


class DataFlowEdge:
    def __init__(self, src: DataFlowNode, dst: DataFlowNode):
        self.src = src
        self.dst = dst
        self.mloc: Optional[MemoryLocation] = None

    def __repr__(self):
        return f"{self.src}->{self.dst}"


# =============================================================================
# DataFlowGraph container
# =============================================================================
class DataFlowGraph:
    """The whole DFG: nodes, edges, placeholder map, per-layer groups."""

    def __init__(self):
        self.nodes: List[DataFlowNode] = []
        self.edges: List[DataFlowEdge] = []
        self.ph_map: Dict[str, PlaceHolderNode] = {}
        self._layers: List[List[DataFlowNode]] = []
        self.qinfo: Dict = {}
        self.current_step = 0
        self.layers_per_step = 0

    # -- node/edge helpers (mirror C++ add_node/create_edge) ----------
    def add_node(self, node: DataFlowNode, layer_list: List[DataFlowNode]):
        node.id = len(self.nodes)
        self.nodes.append(node)
        layer_list.append(node)
        node.step = self.current_step
        node.layer_index = len(self._layers)
        return node

    def create_edge(self, src: DataFlowNode, dst: DataFlowNode):
        edge = DataFlowEdge(src, dst)
        self.edges.append(edge)
        src.output_edges.append(edge)
        dst.input_edges.append(edge)
        return edge

    def get_ph_by_name(self, name: str, shape=None) -> PlaceHolderNode:
        ph = self.ph_map.get(name)
        if ph is None:
            ph = PlaceHolderNode(len(self.nodes), name)
            ph.mloc = MemoryLocation(name=name, mem_type=MemoryType.DRAM)
            self.nodes.append(ph)
            self.ph_map[name] = ph
            if shape is not None:
                ph.tensor_shape = list(shape)
        return ph

    # -- main entry: build from GIR graph.json --------------------------
    def build(self, graph_json: dict, qinfo: Optional[Dict] = None):
        tensors = graph_json.get("tensors", [])
        ops = graph_json.get("ops", [])
        self.qinfo = qinfo or {}

        # Phase 1: tensors -> PlaceHolders
        for t in tensors:
            name = t["name"]
            ph = self.get_ph_by_name(name, t.get("shape"))
            ph.is_input = t.get("is_input", False)
            ph.is_output = t.get("is_output", False)
            ph.is_k_cache = t.get("is_k_cache", False)
            ph.is_v_cache = t.get("is_v_cache", False)
            kv = t.get("kv_cache_info") or {}
            ph.kv_cache_info = {"step": kv.get("step"), "depth": kv.get("depth")}
            # data type
            dt_str = t.get("dtype", "")
            ph.data_type = _dtype_from_str(dt_str)

        # Phase 2: ops -> typed nodes
        for op in ops:
            self._build_op(op)

        self.sort_nodes()
        self.infer_shape()
        return self

    def _build_op(self, op: dict):
        opcode = op.get("opcode", "")
        cls = op.get("class", "")
        attrs = op.get("attrs", {}) or {}
        inputs = op.get("inputs", []) or []
        outputs = op.get("outputs", []) or []
        params = op.get("params", {}) or {}

        layer_nodes = []
        self._layers.append(layer_nodes)

        def ph(name):
            return self.get_ph_by_name(name)

        # -- key node: linear / bmm --
        if opcode == "linear":
            N = attrs.get("N", 0)
            K = attrs.get("K", 0)
            M = attrs.get("M", 1)
            g = GEMMNode(0, M, K, N, sparse=False)
            self.add_node(g, layer_nodes)
            # edges lhs->gemm, rhs->gemm
            if inputs:
                e = self.create_edge(ph(inputs[0]), g)
                e.mloc = ph(inputs[0]).mloc
            if len(inputs) > 1:
                e = self.create_edge(ph(inputs[1]), g)
                e.mloc = ph(inputs[1]).mloc
            onode = g
        elif opcode == "bmm":
            Batch = attrs.get("Batch", 1)
            M = attrs.get("M", 1)
            K = attrs.get("K", 1)
            N = attrs.get("N", 1)
            SubBatch = attrs.get("SubBatch", 1)
            b = BMMNode(0, Batch, M, K, N, SubBatch)
            self.add_node(b, layer_nodes)
            if inputs:
                self.create_edge(ph(inputs[0]), b).mloc = ph(inputs[0]).mloc
            if len(inputs) > 1:
                self.create_edge(ph(inputs[1]), b).mloc = ph(inputs[1]).mloc
            onode = b
        else:
            # post-op / single node: output is first output placeholder
            onode = ph(outputs[0]) if outputs else None

        # -- post-op chain (for ops with a list of post ops) --
        # In GIR export, each op is standalone; the "chain" is the linear
        # sequence of ops feeding outputs forward. We model post-ops as
        # standalone nodes connected to the previous op's output placeholder.
        if opcode in ("RMSlayernorm", "RMSNorm", "rmsnorm"):
            n = RMSNormNode(0, attrs.get("N", 0))
            self.add_node(n, layer_nodes)
            if onode is not None and not isinstance(onode, PlaceHolderNode):
                self.create_edge(onode, n)
            elif inputs:
                self.create_edge(ph(inputs[0]), n)
            onode = n
        elif opcode == "RoPE" or cls == "RoPE":
            n = RoPENode(0, attrs.get("seqlen", 1), attrs.get("head_dim", 1))
            self.add_node(n, layer_nodes)
            if onode is not None and not isinstance(onode, PlaceHolderNode):
                self.create_edge(onode, n)
            elif inputs:
                self.create_edge(ph(inputs[0]), n)
            onode = n
        elif opcode in ("softmax",):
            n = SoftmaxNode(0, attrs.get("dim", -1))
            self.add_node(n, layer_nodes)
            if onode is not None and not isinstance(onode, PlaceHolderNode):
                self.create_edge(onode, n)
            elif inputs:
                self.create_edge(ph(inputs[0]), n)
            onode = n
        elif opcode == "silu":
            n = SiLUNode(0)
            self.add_node(n, layer_nodes)
            if onode is not None and not isinstance(onode, PlaceHolderNode):
                self.create_edge(onode, n)
            elif inputs:
                self.create_edge(ph(inputs[0]), n)
            onode = n
        elif opcode == "gelu":
            n = GELUNode(0)
            self.add_node(n, layer_nodes)
            if onode is not None and not isinstance(onode, PlaceHolderNode):
                self.create_edge(onode, n)
            elif inputs:
                self.create_edge(ph(inputs[0]), n)
            onode = n
        elif opcode == "eltadd":
            n = AddNode(0)
            self.add_node(n, layer_nodes)
            if onode is not None and not isinstance(onode, PlaceHolderNode):
                self.create_edge(onode, n)
            for inp in inputs[1:] if inputs else []:
                self.create_edge(ph(inp), n)
            onode = n
        elif opcode == "eltmul":
            n = EltMulNode(0)
            self.add_node(n, layer_nodes)
            if onode is not None and not isinstance(onode, PlaceHolderNode):
                self.create_edge(onode, n)
            for inp in inputs[1:] if inputs else []:
                self.create_edge(ph(inp), n)
            onode = n
        elif opcode == "transpose":
            n = TransposeNode(0, attrs.get("dims", []))
            self.add_node(n, layer_nodes)
            if onode is not None and not isinstance(onode, PlaceHolderNode):
                self.create_edge(onode, n)
            elif inputs:
                self.create_edge(ph(inputs[0]), n)
            onode = n
        elif opcode == "reshape":
            n = ReshapeNode(0)
            self.add_node(n, layer_nodes)
            if onode is not None and not isinstance(onode, PlaceHolderNode):
                self.create_edge(onode, n)
            elif inputs:
                self.create_edge(ph(inputs[0]), n)
            onode = n
        elif opcode == "concat":
            n = ConcatNode(0, attrs.get("dim", -1))
            self.add_node(n, layer_nodes)
            for inp in inputs:
                self.create_edge(ph(inp), n)
            onode = n
        elif opcode == "argmax":
            n = ArgMaxNode(0)
            self.add_node(n, layer_nodes)
            if onode is not None and not isinstance(onode, PlaceHolderNode):
                self.create_edge(onode, n)
            onode = n
        elif opcode == "divide":
            n = DivNode(0)
            self.add_node(n, layer_nodes)
            if onode is not None and not isinstance(onode, PlaceHolderNode):
                self.create_edge(onode, n)
            onode = n
        elif opcode in ("causal_masking",):
            n = CausalMaskingNode(0, attrs.get("seqlen", 1))
            self.add_node(n, layer_nodes)
            onode = n
        elif opcode == "embedding":
            n = EmbeddingNode(0)
            self.add_node(n, layer_nodes)
            if inputs:
                self.create_edge(ph(inputs[0]), n)
            onode = n

        # connect final node to output placeholders
        if outputs and onode is not None and not isinstance(onode, PlaceHolderNode):
            for out_name in outputs:
                self.create_edge(onode, ph(out_name))

    # -- topological sort by output edges (DFS) -------------------------
    def sort_nodes(self):
        # simple DFS topological order
        visited = set()
        order = []

        def dfs(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for e in node.input_edges:
                dfs(e.src)
            order.append(node)

        for n in self.nodes:
            dfs(n)
        for i, n in enumerate(order):
            n.id = i
        self.nodes = order

    # -- shape inference -----------------------------------------------
    def infer_shape(self):
        for n in self.nodes:
            if isinstance(n, PlaceHolderNode):
                n.output_shapes = [list(n.tensor_shape)] if n.tensor_shape else []
                continue
            if not n.input_edges:
                continue
            in_shape = n.input_edges[0].src.output_shapes[0] if \
                n.input_edges[0].src.output_shapes else []
            if isinstance(n, (GEMMNode,)):
                n.output_shapes = [[n.batch, n.M, n.N, 1]]
            elif isinstance(n, (BMMNode,)):
                n.output_shapes = [[n.batch, n.M, n.N, n.sub_batch]]
            elif isinstance(n, (AddNode, GELUNode, SiLUNode, EltMulNode,
                                SoftmaxNode, RMSNormNode, RoPENode, DivNode)):
                n.output_shapes = [list(in_shape)]
            elif isinstance(n, ConcatNode):
                n.output_shapes = [list(in_shape)]
            elif isinstance(n, TransposeNode):
                s = list(in_shape)
                if len(s) >= 2:
                    s[-1], s[-2] = s[-2], s[-1]
                n.output_shapes = [s]
            elif isinstance(n, ReshapeNode):
                n.output_shapes = [list(in_shape)]
            # propagate shape back to placeholder mloc
            for e in n.output_edges:
                if isinstance(e.dst, PlaceHolderNode):
                    e.dst.tensor_shape = list(n.output_shapes[0]) if n.output_shapes else []


def _dtype_from_str(s: str) -> DataType:
    s = (s or "").lower()
    if "float16" in s or "fp16" in s or "fix16" in s:
        return DataType.FIX16
    if "int8" in s or "fix8" in s or "fp8" in s:
        return DataType.FIX8
    if "int4" in s or "fix4" in s:
        return DataType.FIX8
    if "float32" in s or "fix32" in s:
        return DataType.FIX32
    return DataType.FIX16


# =============================================================================
# Pass wrapper
# =============================================================================
class DataFlowPass:
    """Pass 1: build the DFG from graph.json and attach to the Module."""

    def __init__(self, on_chip_decode: bool = True):
        self.on_chip_decode = on_chip_decode

    def run(self, module) -> int:
        # graph_json is stashed on the module by the driver before running
        graph_json = getattr(module, "graph_json", None)
        if graph_json is None:
            print("[DataFlowPass] no graph_json on module, skipping")
            return 0
        dfg = DataFlowGraph()
        dfg.build(graph_json, getattr(module, "qinfo", {}))
        module.dfg = dfg
        print(f"[DataFlowPass] built DFG: {len(dfg.nodes)} nodes, "
              f"{len(dfg.edges)} edges, {len(dfg._layers)} layers")
        return 0
