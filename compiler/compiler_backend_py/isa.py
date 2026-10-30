"""ISA - 32-bit instruction encoding for the NPU.

Instruction format (32 bits, little-endian 4 bytes):
  [31:28] opcode  (4 bits)
  [27:24] vd      (4 bits) - destination register / precision mode
  [23:20] vs1     (4 bits) - source register 1
  [19:16] vs2     (4 bits) - source register 2
  [15:0]  imm     (16 bits) - immediate / SRAM address

The encoding matches ``rtl/pkg/isa_pkg.sv``.
"""
import struct
from enum import IntEnum


class UOpcode(IntEnum):
    NOP     = 0x0
    VLD     = 0x1   # L1 cache -> PostProc RF
    VST     = 0x2   # PostProc RF -> L1 cache
    GLD_W   = 0x3   # L1 -> GEMM Weight RF (precision in vd[1:0])
    GLD_A   = 0x4   # L1 -> GEMM Activation RF
    GEMM    = 0x5   # Weight RF x Act RF -> Accumulator
    SOFTMAX = 0x6
    RMSNORM = 0x7
    ROPE    = 0x8
    SILU    = 0x9
    ADD     = 0xA   # vector add
    MUL     = 0xB   # vector multiply (element-wise)
    DMA_RD  = 0xC   # DDR -> L1 cache
    DMA_WR  = 0xD   # L1 cache -> DDR
    BARRIER = 0xE
    END     = 0xF


class Precision(IntEnum):
    FP16 = 0
    INT1 = 1
    INT2 = 2
    INT3 = 3
    # Backward-compatible aliases. The RTL encoding is FP16/INT1/INT2/INT3;
    # it has no native INT8/INT4 value in the two-bit instruction field.
    INT8 = INT1
    INT4 = INT2


class UInstr:
    """A single 32-bit microcode instruction."""

    def __init__(self, opcode, vd=0, vs1=0, vs2=0, imm=0):
        self.opcode = UOpcode(opcode)
        self.vd = vd & 0xF
        self.vs1 = vs1 & 0xF
        self.vs2 = vs2 & 0xF
        self.imm = imm & 0xFFFF

    def encode(self):
        """Pack into a 32-bit word."""
        return ((self.opcode & 0xF) << 28) | \
               ((self.vd & 0xF) << 24) | \
               ((self.vs1 & 0xF) << 20) | \
               ((self.vs2 & 0xF) << 16) | \
               (self.imm & 0xFFFF)

    def bytes_le(self):
        """Little-endian 4 bytes."""
        return struct.pack('<I', self.encode())

    def __repr__(self):
        return (f"UInstr({self.opcode.name}, vd={self.vd}, "
                f"vs1={self.vs1}, vs2={self.vs2}, imm=0x{self.imm:04x})")

    def to_text(self):
        """Human-readable single line for ucode.txt."""
        return (f"{self.opcode.name:8s} vd={self.vd} vs1={self.vs1} "
                f"vs2={self.vs2} imm=0x{self.imm:04x}  ; raw=0x{self.encode():08x}")


# =============================================================================
# Static factory functions (mirror ISA::* in C++)
# =============================================================================
def nop():           return UInstr(UOpcode.NOP)
def end():           return UInstr(UOpcode.END)
def barrier():       return UInstr(UOpcode.BARRIER)

NOP_SUBOP_SET_CFG_K = 0xE
NOP_SUBOP_SET_CFG_M = 0xF


def set_cfg_k(k):
    """Set the internal K tile width for following scheduled operations."""
    if k not in (32, 64, 128, 256):
        raise ValueError("CFG_K override must be 32, 64, 128, or 256")
    return UInstr(UOpcode.NOP, imm=(NOP_SUBOP_SET_CFG_K << 12) | k)


def set_cfg_m(m):
    """Set active rows; M=0 selects the current route-group population."""
    if not 0 <= m <= 32:
        raise ValueError("CFG_M override must be in [0,32]")
    return UInstr(UOpcode.NOP, imm=(NOP_SUBOP_SET_CFG_M << 12) | m)


def vld(vd, imm):    return UInstr(UOpcode.VLD, vd=vd, imm=imm)
def vst(vs1, imm):   return UInstr(UOpcode.VST, vs1=vs1, imm=imm)


def vld_page(vd, imm, page):
    """Load one vector from a selected L1 page (page 2 is the 16-KiB extension)."""
    if not 0 <= page <= 2 or imm < 0 or imm > 0xFFFF or imm % 32:
        raise ValueError("paged vector load requires page 0..2 and a 32-byte aligned offset")
    vs1 = 0x2 if page == 2 else page << 2
    return UInstr(UOpcode.VLD, vd=vd, vs1=vs1, imm=imm)


def vst_page(vs1, imm, page):
    """Store one vector to an L1 page (page 2 is the 16-KiB extension)."""
    if not 0 <= page <= 2 or imm < 0 or imm > 0xFFFF or imm % 32:
        raise ValueError("paged vector store requires page 0..2 and a 32-byte aligned offset")
    return UInstr(UOpcode.VST, vd=(0 if page == 2 else page << 2),
                  vs1=vs1, vs2=(1 if page == 2 else 0), imm=imm)


def _token_vector_stride_code(row_stride_bytes):
    """Encode a 32-byte-aligned L1 token-row stride in seven bits."""
    if row_stride_bytes < 64 or row_stride_bytes > 4096 or row_stride_bytes % 32:
        raise ValueError("vector token row stride must be 32-byte aligned in [64,4096]")
    return row_stride_bytes // 32 - 1


def vld_token(vd, imm, row_stride_bytes, page=0):
    """Load a vector at ``imm + current_token * stride`` from an L1 page.

    The low five address bits are otherwise unused by the 32-byte-aligned
    vector address, so token-stride instructions encode their L1 page there.
    """
    if imm < 0 or imm > 0xFFFF or imm % 32 or page not in (0, 1, 2):
        raise ValueError("token vector load address must be 32-byte aligned and encodable")
    code = _token_vector_stride_code(row_stride_bytes)
    return UInstr(UOpcode.VLD, vd=vd, vs1=0x8 | (code >> 4),
                  vs2=code & 0xF, imm=imm | page)


def vld_group_token(vd, imm, row_stride_bytes):
    """Load ``imm + route_group_token[row] * stride`` from the high L1 page."""
    return vld_token(vd, imm, row_stride_bytes)


def vst_token(vs1, imm, row_stride_bytes, page=0):
    """Store a vector at ``imm + current_token * stride`` into an L1 page."""
    if imm < 0 or imm > 0xFFFF or imm % 32 or page not in (0, 1, 2):
        raise ValueError("token vector store address must be 32-byte aligned and encodable")
    code = _token_vector_stride_code(row_stride_bytes)
    return UInstr(UOpcode.VST, vd=0x8 | (code >> 4), vs1=vs1,
                  vs2=code & 0xF, imm=imm | page)


def vst_group_token(vs1, imm, row_stride_bytes):
    """Store to the high-page output row selected by the active route group."""
    return vst_token(vs1, imm, row_stride_bytes)


def gld_w(prec, imm, group=0, expert_sel=0, proj=0):
    """GLD.W with compact or extended descriptor scale-group addressing.

    Expert-selected groups 8..15 use the extended form because vs1[3] in the
    compact encoding selects direct-weight precision and would discard the
    expert slot. Direct-weight groups retain their compact encoding.
    """
    if not 0 <= group < 64 or not 0 <= proj < 3:
        raise ValueError("GLD.W scale group/projection is outside the encoded range")
    if group < 16 and (expert_sel == 0 or group < 8):
        return UInstr(UOpcode.GLD_W, vd=(int(prec) | (proj << 2)),
                      vs1=group, vs2=expert_sel, imm=imm)
    if not 1 <= expert_sel <= 4 or imm < 0 or imm > 0xFFFF or imm % 4096:
        raise ValueError("extended GLD.W groups require an expert slot and bank-aligned L1 address")
    # vs1=0xF marks the six-bit scale-group form. The L1 bank remains in
    # imm[15:12], while imm[5:0] carries the group index. It also disambiguates
    # expert slots from direct-nbits encodings for groups 8..15.
    return UInstr(UOpcode.GLD_W, vd=(int(prec) | (proj << 2)),
                  vs1=0xF, vs2=expert_sel,
                  imm=(imm & 0xF000) | group)
def gld_a(imm, wait_postproc=False, allow_dma_overlap=False, page=0):
    """Load an activation tile, optionally asserting it is DMA-disjoint.

    ``allow_dma_overlap`` uses the otherwise-unused vs1 field as an explicit
    compiler assertion that this L1 read cannot alias the active weight DMA.
    """
    if page not in (0, 1):
        raise ValueError("scalar GLD.A source page must be 0 or 1")
    return UInstr(UOpcode.GLD_A,
                  vs1=((1 if allow_dma_overlap else 0) | (page << 1)),
                  vs2=int(wait_postproc), imm=imm)
def gld_a_batch(imm):
    """Load a K-major activation tile with one token row per ARF bank."""
    if imm < 0 or imm > 0xFFFF or imm % 32:
        raise ValueError("batched GLD.A address must be 32-byte aligned and encodable")
    return UInstr(UOpcode.GLD_A, vs1=0x8, imm=imm)
def gld_a_batch_transpose(imm, padded_rows=False, page=0,
                          wait_postproc=False, row_stride_rows=None,
                          allow_dma_overlap=False):
    """Transpose a batch tile, optionally waiting for prior vector stores.

    Page 2 uses the same instruction fields for extended addressing and cannot
    be combined with the PostProc wait option. In a hardware loop,
    ``row_stride_rows`` encodes the padded row capacity (1..15) for compact
    route groups; omit it for the legacy 32-row or CFG_M-derived stride.
    ``allow_dma_overlap`` marks a compiler-proven range disjoint from DMA
    staging; imm[1] is unused because the address is 32-byte aligned.
    ``wait_postproc`` uses imm[2], also ignored by the byte-address decoder,
    to defer this activation read until preceding vector stores retire.
    """
    if imm < 0 or imm > 0xFFFF or imm % 32 or page not in (0, 1, 2):
        raise ValueError("batched GLD.A address must be aligned and select page 0..2")
    if row_stride_rows is not None and not 1 <= row_stride_rows <= 15:
        raise ValueError("route-group row stride must be in [1,15]")
    if wait_postproc and (page == 2 or padded_rows or
                          row_stride_rows is not None):
        raise ValueError("PostProc wait cannot share batched GLD.A stride fields")
    return UInstr(UOpcode.GLD_A, vd=(1 if page == 2 else 0),
                  vs1=(0xC | (int(padded_rows or row_stride_rows is not None))) |
                      ((page & 1) << 1),
                  vs2=(row_stride_rows if row_stride_rows is not None else
                       int(wait_postproc)),
                  imm=(imm | (2 if allow_dma_overlap else 0)))


def gld_a_batch_transpose_strided(imm, row_stride_bytes, page=0,
                                  allow_dma_overlap=False,
                                  wait_postproc=False):
    """Transpose an MxK FP16 tile from token-major rows in L1.

    ``row_stride_bytes`` is the distance between source row starts. The
    encoded stride is measured in 32-byte L1 lines and supports page 0 or 1.
    The low immediate bit carries the ninth stride bit; source bases are
    32-byte aligned, so the bit is discarded by the L1 address decode.
    The optional DMA-disjoint marker uses imm[1], also below the address unit.
    """
    if (imm < 0 or imm > 0xFFFF or imm % 32 or page not in (0, 1) or
            row_stride_bytes < 64 or row_stride_bytes > 16384 or
            row_stride_bytes % 32):
        raise ValueError("strided batch transpose requires aligned page-0/1 rows")
    stride_lines = row_stride_bytes // 32
    stride_code = stride_lines - 1
    # vd==0, vs2==1 aliases the legacy PostProc-wait form; a nonzero vd
    # with vs2==0 aliases page 2. Other zero high/low stride fields are valid.
    if stride_code == 1 or (
            (stride_code >> 4) != 0 and (stride_code & 0xF) == 0):
        raise ValueError("row stride collides with a legacy GLD.A encoding")
    return UInstr(UOpcode.GLD_A, vd=(stride_code >> 4) & 0xF,
                  vs1=0xC | (page << 1), vs2=stride_code & 0xF,
                  imm=(imm | ((stride_code >> 8) & 1) |
                       (2 if allow_dma_overlap else 0) |
                       (4 if wait_postproc else 0)))


def gld_a_batch_transpose_split(imm, split_iter, spill_bank,
                                padded_rows=True, page=1):
    """Transpose looped K chunks, switching L1 page at ``split_iter``.

    The primary range uses ``imm`` and ``page``. Starting at the split
    iteration, chunks are read from the 4-KiB-aligned ``spill_bank`` in the
    opposite page. GLD.A's otherwise-unused vd/vs2 fields carry this mapping.
    """
    if (imm < 0 or imm > 0xFFFF or imm % 32 or page not in (0, 1) or
            not 1 <= split_iter <= 15 or not 1 <= spill_bank <= 15):
        raise ValueError("split batched GLD.A requires aligned addresses and encodable page fields")
    return UInstr(UOpcode.GLD_A, vd=spill_bank,
                  vs1=(0xC | (int(padded_rows))) | (page << 1),
                  vs2=split_iter, imm=imm)


def l1_clear_group_accumulator(output_bytes):
    """Clear the requested byte range in the 64-KiB high-page accumulator."""
    if output_bytes <= 0 or output_bytes > 65536 or output_bytes % 32:
        raise ValueError("group accumulator clear size must be 32-byte aligned in [32,65536]")
    words = output_bytes // 32
    return UInstr(UOpcode.NOP, vd=0xE, vs1=0xE, vs2=0xF, imm=words)


def gemm(acc=False, imm=32, prec=Precision.FP16, skip_store=False,
         expert_sel=0, proj=0, loop_managed=False):
    """GEMM encoding: vd[3]=skip, vd[2]=acc, vd[1:0]=precision."""
    vd = ((1 if skip_store else 0) << 3) | ((1 if acc else 0) << 2) | (int(prec) & 0x3)
    vs1 = 0xC if loop_managed else proj
    return UInstr(UOpcode.GEMM, vd=vd, vs1=vs1, vs2=expert_sel, imm=imm)
def softmax(vd, vs1, valid_lanes=32):
    """Normalize the first ``valid_lanes`` and zero the remaining lanes."""
    if not 1 <= valid_lanes <= 32:
        raise ValueError("softmax valid_lanes must be in [1,32]")
    return UInstr(UOpcode.SOFTMAX, vd=vd, vs1=vs1,
                  imm=0 if valid_lanes == 32 else valid_lanes)


def online_softmax(vd, vs1, alpha_vd, valid_lanes=32, reset=False,
                   state_slot=0, raw_output=False):
    """Update one 32-lane FlashAttention softmax tile.

    The softmax engine retains running max/sum state between online calls,
    writes globally normalized weights to ``vd`` by default, and broadcasts
    the prior output rescale factor to ``alpha_vd``. ``raw_output`` instead
    writes exp(score - running_max) and broadcasts exp(old_max - new_max),
    for attention schedules that defer output normalization until the final
    key tile. Set ``reset`` on the first K tile for each query row.
    """
    if not 1 <= valid_lanes <= 32:
        raise ValueError("online softmax valid_lanes must be in [1,32]")
    if not 0 <= state_slot < 128:
        raise ValueError("online softmax state slot must be in [0,127]")
    if state_slot < 64:
        slot_bits = state_slot << 8
        source_field = vs1
    else:
        if not 0 <= vs1 < 8:
            raise ValueError("extended online softmax source register must be in [0,7]")
        slot_bits = state_slot << 8
        source_field = vs1 | 0x8
    imm = (0 if valid_lanes == 32 else valid_lanes) | 0x40 | slot_bits
    if reset:
        imm |= 0x80
    if raw_output:
        imm |= 0x8000
    return UInstr(UOpcode.SOFTMAX, vd=vd, vs1=source_field,
                  vs2=alpha_vd, imm=imm)


def finalize_online_softmax(alpha_vd, state_slot=0):
    """Write the reciprocal running denominator for one online state slot."""
    if not 0 <= state_slot < 128:
        raise ValueError("online softmax state slot must be in [0,127]")
    if state_slot < 64:
        imm = 0x4040 | (state_slot << 8)
        source_field = 0
    else:
        imm = 0x4000 | (state_slot << 8)
        source_field = 0x8
    return UInstr(UOpcode.SOFTMAX, vd=0, vs1=source_field,
                  vs2=alpha_vd, imm=imm)
def rmsnorm(vd, vs1, vs2): return UInstr(UOpcode.RMSNORM, vd=vd, vs1=vs1, vs2=vs2)


def rmsnorm_accumulate(vs1, reset=False):
    """Accumulate one 32-lane x^2 chunk into the persistent RMS statistic."""
    return UInstr(UOpcode.RMSNORM, vs1=vs1,
                  imm=0x4000 | (0x2000 if reset else 0))


def rmsnorm_finalize(element_count):
    """Finalize a multi-chunk RMS statistic for ``element_count`` values."""
    if element_count < 32 or element_count > 4096 or element_count % 32:
        raise ValueError("RMSNorm element_count must be a multiple of 32 in [32,4096]")
    inv_count = struct.unpack('<H', struct.pack('<e', 1.0 / element_count))[0]
    return UInstr(UOpcode.RMSNORM, vd=inv_count >> 12,
                  vs1=inv_count >> 8, vs2=inv_count >> 4,
                  imm=0x8000 | (inv_count & 0xF))


def rmsnorm_apply(vd, vs1, vs2):
    """Apply the persistent full-vector inv_rms to one x/gamma chunk."""
    return UInstr(UOpcode.RMSNORM, vd=vd, vs1=vs1, vs2=vs2, imm=0xC000)


def rope(vd, vs1, vs2): return UInstr(UOpcode.ROPE, vd=vd, vs1=vs1, vs2=vs2)
def silu(vd, vs1):  return UInstr(UOpcode.SILU, vd=vd, vs1=vs1)
def sigmoid(vd, vs1): return UInstr(UOpcode.SILU, vd=vd, vs1=vs1, imm=1)
def add(vd, vs1, vs2): return UInstr(UOpcode.ADD, vd=vd, vs1=vs1, vs2=vs2)
def mul(vd, vs1, vs2): return UInstr(UOpcode.MUL, vd=vd, vs1=vs1, vs2=vs2)
def dma_rd(imm, l1_bank=0, expert_sel=0, proj=0):
    return UInstr(UOpcode.DMA_RD, vd=l1_bank, vs1=proj, vs2=expert_sel, imm=imm)


def dma_rd_kloop(imm, l1_bank=2, stride_bytes=2048):
    """Fetch one 4 KiB weight window per K-loop iteration into ping-pong L1 banks.

    The static-weight DMA encoding selects an iteration stride of 1 or 2 KiB;
    its bank ID alternates automatically while a hardware loop is active.
    """
    if stride_bytes not in (1024, 2048):
        raise ValueError("K-loop DMA stride must be 1024 or 2048 bytes")
    if not 0 <= l1_bank <= 14 or l1_bank % 2:
        raise ValueError("K-loop DMA must start at an even L1 bank in [0,14]")
    if imm < 0 or imm > 0xFFFF:
        raise ValueError("K-loop DMA immediate must fit 16 bits")
    vs1 = 0x4 | (1 if stride_bytes == 2048 else 0)
    return UInstr(UOpcode.DMA_RD, vd=l1_bank, vs1=vs1, vs2=0, imm=imm)
def dma_wr(imm, output_window=0, page=0):
    """Write 1 KiB from an L1 page to ``OUT + window*64KiB + imm``."""
    if not 0 <= output_window < 16:
        raise ValueError("DMA_WR output window must be in [0,15]")
    if not 0 <= imm <= 0xFFFF or page not in (0, 1):
        raise ValueError("DMA_WR needs a 16-bit offset and L1 page 0 or 1")
    return UInstr(UOpcode.DMA_WR, vd=output_window, vs1=page << 2, imm=imm)


def dma_wr_linear_burst(imm, transfer_bytes, output_window=0, page=0):
    """Write a contiguous aligned L1 range using a 256-byte length code."""
    if not 0 <= output_window < 16:
        raise ValueError("DMA_WR output window must be in [0,15]")
    if page not in (0, 1) or imm < 0 or imm > 0xFFFF or imm % 32:
        raise ValueError("linear DMA_WR needs an aligned L1 offset and page 0 or 1")
    if transfer_bytes < 256 or transfer_bytes > 4096 or transfer_bytes % 256:
        raise ValueError("linear DMA_WR size must be a 256-byte multiple in [256,4096]")
    length_code = transfer_bytes // 256 - 1
    # vs1[3]=1 selects an extended length; vs1[0]=1 distinguishes a single
    # linear burst from the existing per-token stride mode.
    return UInstr(UOpcode.DMA_WR, vd=output_window,
                  vs1=0x9 | (page << 2), vs2=length_code, imm=imm)


def _token_dma_len_code(hidden_size):
    byte_count = hidden_size * 2
    if hidden_size < 32 or hidden_size > 2048 or hidden_size % 32:
        raise ValueError("token DMA hidden_size must be a multiple of 32 in [32,2048]")
    # DMA token transactions are encoded in 256-byte units.  A 32-lane tile
    # therefore occupies one padded 256-byte resident token slot.
    return max(256, ((byte_count + 255) // 256) * 256) // 256 - 1


def dma_rd_token(byte_offset, hidden_size, l1_bank):
    """Load one token from ACT DDR using a hardware-derived token stride."""
    if byte_offset < 0 or byte_offset % 16 or byte_offset // 16 > 0xFFFF:
        raise ValueError("token activation byte offset must be 16-byte aligned and encodable")
    return UInstr(UOpcode.DMA_RD, vd=l1_bank, vs1=0x8,
                  vs2=_token_dma_len_code(hidden_size), imm=byte_offset // 16)


def dma_rd_out_token(byte_offset, hidden_size, l1_bank):
    """Load one token from an OUT DDR window using the active token-loop index."""
    if byte_offset < 0 or byte_offset % 16 or byte_offset // 16 > 0xFFFF:
        raise ValueError("OUT token byte offset must be 16-byte aligned and encodable")
    if not 0 <= l1_bank <= 15:
        raise ValueError("OUT token L1 bank must be in [0,15]")
    return UInstr(UOpcode.DMA_RD, vd=l1_bank, vs1=0x9,
                  vs2=_token_dma_len_code(hidden_size), imm=byte_offset // 16)


def dma_rd_token_group(byte_offset, hidden_size, l1_bank, tokens):
    """Load a token loop directly into consecutive L1 rows.

    Unlike ``dma_rd_token``, this encoding advances the L1 destination by one
    transfer per token as well as advancing the DDR source address.
    """
    if byte_offset < 0 or byte_offset % 16 or byte_offset // 16 > 0xFFFF:
        raise ValueError("token activation byte offset must be 16-byte aligned and encodable")
    if not 0 <= l1_bank < 16 or not 1 <= tokens <= 32:
        raise ValueError("invalid token-group L1 bank or token count")
    token_bytes = (_token_dma_len_code(hidden_size) + 1) * 256
    if l1_bank * 4096 + tokens * token_bytes > 65536:
        raise ValueError("token-group DMA exceeds the 64-KiB L1 page")
    return UInstr(UOpcode.DMA_RD, vd=l1_bank, vs1=0xA,
                  vs2=_token_dma_len_code(hidden_size), imm=byte_offset // 16)


def dma_rd_out_token_group(byte_offset, transfer_bytes, l1_bank, tokens):
    """Gather a fixed-width field from 4-KiB token rows in OUT DDR.

    The instruction must be enclosed by ``token_loop_begin(tokens)`` and
    ``token_loop_end()``. Source rows advance by 4 KiB while L1 destinations
    advance by the transfer length, so a 256-byte attention head becomes a
    compact token-major matrix without rereading the rest of each hidden row.
    ``byte_offset`` is relative to OUT DDR and may include a projection window
    and the first token's offset within that projection. The selected base
    window may be OUT, ACT, or WGT DDR.
    """
    if byte_offset < 0 or byte_offset % 16 or byte_offset >= (2 << 20):
        raise ValueError("OUT token-group offset must be 16-byte aligned and encodable")
    if not 0 <= l1_bank < 16 or not 1 <= tokens <= 32:
        raise ValueError("invalid OUT token-group L1 bank or token count")
    if (transfer_bytes < 256 or transfer_bytes > 4096 or
            transfer_bytes % 256):
        raise ValueError("OUT token-group transfer must be a 256-byte multiple in [256,4096]")
    if (l1_bank * 4096 + tokens * transfer_bytes > 65536 or
            byte_offset + (tokens - 1) * 4096 + transfer_bytes > 2 << 20):
        raise ValueError("OUT token-group range exceeds its L1 or DDR window")
    length_code = transfer_bytes // 256 - 1
    source_encoding = 0xE
    if byte_offset >> 20:
        if length_code > 7:
            raise ValueError("upper-window OUT gathers support at most 2 KiB")
        source_encoding = 0xB
        length_code |= 0x8
    return UInstr(UOpcode.DMA_RD, vd=l1_bank, vs1=source_encoding,
                  vs2=length_code, imm=(byte_offset >> 4) & 0xFFFF)


def dma_rd_head_token_group(byte_offset, transfer_bytes, l1_bank, tokens,
                            source="out"):
    """Gather one head field per token from OUT, ACT, or WGT base DDR.

    Each issued DMA reads from a 4-KiB-strided token row and packs the selected
    contiguous field into the L1 destination. The source base is selected by
    vs1=6 (ACT), vs1=7 (WGT), or vs1=D (OUT). Attention gathers use 256-byte
    immediate units, extending the address window to 16 MiB without changing
    the token stride. vs2[3] marks the gather; vs2[2:0] encodes 256..2048 B.
    """
    source_encoding = {"out": 0xD, "act": 0x6, "wgt": 0x7}.get(source)
    if source_encoding is None:
        raise ValueError("head-token gather source must be out, act, or wgt")
    if byte_offset < 0 or byte_offset % 256 or byte_offset >= (1 << 24):
        raise ValueError("head token-group offset must be 256-byte aligned and encodable")
    if not 0 <= l1_bank < 16 or not 1 <= tokens <= 32:
        raise ValueError("invalid head token-group L1 bank or token count")
    if (transfer_bytes < 256 or transfer_bytes > 2048 or
            transfer_bytes % 256):
        raise ValueError("head token-group transfer must be a 256-byte multiple in [256,2048]")
    length_code = transfer_bytes // 256 - 1
    length_code |= 0x8
    token_bytes = transfer_bytes
    if l1_bank * 4096 + tokens * token_bytes > 65536:
        raise ValueError("head token-group DMA exceeds the 64-KiB L1 page")
    return UInstr(UOpcode.DMA_RD, vd=l1_bank, vs1=source_encoding,
                  vs2=length_code, imm=(byte_offset >> 8) & 0xFFFF)


def dma_rd_route_group(byte_offset=0, l1_bank=4, proj=0, source="act"):
    """Gather cached route-group activations into K-chunk-major L1 staging.

    The hardware derives K geometry from the selected expert descriptor,
    gathers live rows, and zero-pads each non-empty K chunk to 32 rows so its
    fixed batch-GLD stride is preserved. Empty groups stage one zero row. The
    source offset is relative to ACT DDR.
    """
    if byte_offset < 0 or byte_offset % 16 or byte_offset // 16 > 0xFFFF:
        raise ValueError("route-group activation offset must be 16-byte aligned")
    if not 1 <= l1_bank <= 15 or proj not in (0, 1):
        raise ValueError("invalid route-group L1 bank or projection")
    if source not in ("act", "out"):
        raise ValueError("route-group source must be 'act' or 'out'")
    # 0xF is reserved for OUT route-group gathers. Keeping the source marker
    # out of vs2=slot avoids colliding with descriptor-indirect expert DMAs.
    return UInstr(UOpcode.DMA_RD, vd=l1_bank,
                  vs1=(0xF if source == "out" else 0xC | proj),
                  vs2=(proj if source == "out" else 0),
                  imm=byte_offset // 16)


def dma_wr_token(l1_addr, hidden_size, output_window=0, page=0):
    """Store one token to an output window in contiguous token-major order."""
    if l1_addr < 0 or l1_addr > 0xFFFF or l1_addr % 32:
        raise ValueError("token output L1 address must be 32-byte aligned and encodable")
    if not 0 <= output_window < 16:
        raise ValueError("DMA_WR output window must be in [0,15]")
    if page not in (0, 1):
        raise ValueError("token output source page must be 0 or 1")
    return UInstr(UOpcode.DMA_WR, vd=output_window, vs1=0x8 | (page << 2),
                  vs2=_token_dma_len_code(hidden_size), imm=l1_addr)


def dma_wr_group_token(l1_addr, hidden_size, output_window=0):
    """Store one grouped token row to a selected output window."""
    if l1_addr < 0 or l1_addr > 0xFFFF or l1_addr % 32:
        raise ValueError("group token output L1 offset must be 32-byte aligned")
    if not 0 <= output_window < 16:
        raise ValueError("DMA_WR output window must be in [0,15]")
    return UInstr(UOpcode.DMA_WR, vd=output_window, vs1=0xC,
                  vs2=_token_dma_len_code(hidden_size), imm=l1_addr)


def dma_wr_group_tokens(l1_addr, hidden_size, tokens, output_window=0):
    if not 1 <= tokens <= 32:
        raise ValueError("group output DMA supports 1..32 tokens")
    return [token_loop_begin(tokens),
            dma_wr_group_token(l1_addr, hidden_size, output_window),
            # TOKEN_END is all-idle gated, so it retires the DMA before the
            # loop repeats without a separate global BARRIER.
            token_loop_end()]


def dma_wr_head_group_scatter(hidden_size, tokens, output_window,
                              output_offset, page=0,
                              source_l1_offset=0):
    """Write compact head rows into token-major hidden rows with a 4-KiB stride.

    The source is a compact ``[tokens, head_dim]`` matrix at page-0/1 L1
    address zero, or at a 8-KiB-aligned page offset for later M32 query tiles.
    ``output_offset`` selects the head's feature slice within each
    2048-wide output row; ``output_window`` selects the first 16-token DDR
    window. The dedicated DMA_WR mode keeps the source stride compact while
    scattering destinations at the full-hidden token stride.
    """
    if not 1 <= tokens <= 32:
        raise ValueError("head scatter supports 1..32 tokens")
    if page not in (0, 1):
        raise ValueError("head scatter source page must be 0 or 1")
    if hidden_size < 32 or hidden_size > 2048 or hidden_size % 32:
        raise ValueError("head scatter width must be a multiple of 32 in [32,2048]")
    if not 0 <= output_window < 16:
        raise ValueError("DMA_WR output window must be in [0,15]")
    if (output_offset < 0 or output_offset % 32 or
            output_offset + hidden_size * 2 > 4096):
        raise ValueError("head scatter feature range must fit a 4-KiB token row")
    if (source_l1_offset < 0 or source_l1_offset > 0xE000 or
            source_l1_offset % 0x2000):
        raise ValueError("head scatter source offset must be 8-KiB aligned in [0,0xE000]")
    encoded_offset = (source_l1_offset | output_offset
                      if source_l1_offset >= 0x4000 else output_offset)
    return [token_loop_begin(tokens),
            UInstr(UOpcode.DMA_WR, vd=output_window,
                   vs1=0xA | (page << 2) |
                        (1 if source_l1_offset == 0x2000 else 0),
                   vs2=_token_dma_len_code(hidden_size), imm=encoded_offset),
            token_loop_end()]


# =============================================================================
# Extended NOP sub-opcodes (imm[15:12])
# =============================================================================
NOP_SUBOP_TOPK = 0x1
NOP_SUBOP_SCALE = 0x2
NOP_SUBOP_GROUP_WEIGHT = 0x3
NOP_SUBOP_LOOP = 0x4
NOP_SUBOP_AC2V = 0x5
NOP_SUBOP_ROUTE = 0x6
NOP_SUBOP_RWEIGHT = 0x7
NOP_SUBOP_TOKEN_BEGIN = 0x8
NOP_SUBOP_TOKEN_END = 0x9
NOP_SUBOP_SET_WPAGE = 0xA
NOP_SUBOP_STATIC_SCALE = 0xB
NOP_SUBOP_TGLD = 0xC
NOP_SUBOP_BCAST = 0xD


def nop_scale(ntile, current_expert_only=False):
    """Load expert scale metadata for an N tile.

    ``current_expert_only`` is reserved for grouped execution, where the
    route-group descriptor maps expert slot 1 to the active expert.
    """
    if not 0 <= ntile < 256:
        raise ValueError("scale N-tile index must be in [0,255]")
    return UInstr(UOpcode.NOP, vd=int(current_expert_only),
                  imm=(NOP_SUBOP_SCALE << 12) | ntile)


def hw_loop(count, body_len, stride_code=0):
    """Repeat a body; GLD_A advances by 64 << stride_code bytes per iteration."""
    if not 1 <= count <= 0xFFF or not 1 <= body_len <= 15 or not 0 <= stride_code <= 3:
        raise ValueError("invalid hardware loop geometry")
    return UInstr(UOpcode.NOP, vd=stride_code << 2, vs1=body_len,
                  imm=(NOP_SUBOP_LOOP << 12) | count)


def static_cached_wide_loop(count, body_len=4):
    """Replay a static K=128 body from loop_controller's local buffer."""
    if not 1 <= count <= 0xFFF or not 1 <= body_len <= 15:
        raise ValueError("invalid cached static-loop geometry")
    # vd=F selects loop_controller; vs2=F reserves the K=128 static-weight
    # address/scale update mode. A small count keeps imm[15:12] at NOP_NONE,
    # so ucode_fetch does not also create a second loop controller.
    return UInstr(UOpcode.NOP, vd=0xF, vs1=body_len, vs2=0xF, imm=count)


def transpose_load(l1_addr, row_stride_bytes, transpose=True):
    """Load one 32x32 FP16 tile from L1 into the WRF.

    The default transpose maps source columns to WRF K entries and source rows
    to output lanes (QK's [key, feature] layout). ``transpose=False`` preserves
    source rows as WRF K entries and columns as output lanes (PV's
    [key, feature] layout).
    """
    if l1_addr < 0 or l1_addr % 32 or l1_addr // 32 >= 2048:
        raise ValueError("transpose-load L1 address must be 32-byte aligned")
    if row_stride_bytes < 32 or row_stride_bytes > 2048 or row_stride_bytes % 32:
        raise ValueError("transpose-load row stride must be 32..2048 bytes")
    stride_code = row_stride_bytes // 32 - 1
    return UInstr(UOpcode.NOP, vd=stride_code & 0xF,
                  vs1=(0 if transpose else 1),
                  vs2=(stride_code >> 4) & 0x3,
                  imm=(NOP_SUBOP_TGLD << 12) | (l1_addr // 32))


def topk(token):
    if not 0 <= token < 32:
        raise ValueError("router token must be in [0,31]")
    return UInstr(UOpcode.NOP, imm=(NOP_SUBOP_TOPK << 12) | token)


def topk_current_token():
    """Cache Top-4 for the token selected by the active hardware token loop."""
    return UInstr(UOpcode.NOP,
                  imm=(NOP_SUBOP_TOPK << 12) | (1 << 5))


def topk_current_token_from_row(source_row, token_base, tokens):
    """Cache each loop token from a fixed ACC row into its global route slot."""
    if not 0 <= source_row < 32:
        raise ValueError("Top-K source ACC row must be in [0,31]")
    if (not 1 <= tokens <= 32 or not 0 <= token_base < 32 or
            token_base + tokens > 32):
        raise ValueError("Top-K token range must fit the 32-entry route cache")
    return UInstr(
        UOpcode.NOP, vd=source_row & 0xF,
        vs1=0x2 | (((source_row >> 4) & 1) << 2),
        imm=(NOP_SUBOP_TOPK << 12) | (1 << 5) | token_base)


def topk_acc_row(source_row, token):
    """Cache one global route slot from an explicitly selected ACC row."""
    if not 0 <= source_row < 32:
        raise ValueError("Top-K source ACC row must be in [0,31]")
    if not 0 <= token < 32:
        raise ValueError("Top-K token must be in [0,31]")
    return UInstr(
        UOpcode.NOP, vd=source_row & 0xF,
        vs1=0x2 | (((source_row >> 4) & 1) << 2),
        imm=(NOP_SUBOP_TOPK << 12) | token)


def route(token):
    """Restore this token's cached Top-4 IDs and weights, without host routing."""
    if not 0 <= token < 32:
        raise ValueError("route token must be in [0,31]")
    return UInstr(UOpcode.NOP, imm=(NOP_SUBOP_ROUTE << 12) | token)


def route_current_token():
    return UInstr(UOpcode.NOP, imm=(NOP_SUBOP_ROUTE << 12) | (1 << 5))


def route_group(expert_id, tokens=32, token_start=0):
    """Collect cached Top-K rows routed to expert_id, in token order."""
    if not 0 <= expert_id < 60:
        raise ValueError("routed expert ID must be in [0,59]")
    if not 1 <= tokens <= 32:
        raise ValueError("route grouping supports 1..32 cached tokens")
    if not 0 <= token_start < 32 or token_start + tokens > 32:
        raise ValueError("route-group token range must fit the 32-token cache")
    return UInstr(UOpcode.NOP, vd=(token_start >> 4) & 1,
                  vs1=token_start & 0xF,
                  imm=(NOP_SUBOP_ROUTE << 12) | (1 << 6) |
                      ((tokens - 1) << 7) | expert_id)


def route_group_current_expert(tokens=32, token_start=0):
    """Select the expert named by the active expert-loop iteration."""
    if not 1 <= tokens <= 32:
        raise ValueError("route grouping supports 1..32 cached tokens")
    if not 0 <= token_start < 32 or token_start + tokens > 32:
        raise ValueError("route-group token range must fit the 32-token cache")
    return UInstr(UOpcode.NOP, vd=(token_start >> 4) & 1,
                  vs1=token_start & 0xF,
                  imm=(NOP_SUBOP_ROUTE << 12) | (1 << 6) |
                      ((tokens - 1) << 7) | 0x3F)


def token_loop_begin(count, allow_overlap=False, allow_gemm_overlap=False):
    if not 1 <= count <= 32:
        raise ValueError("token loop supports 1..32 tokens")
    if allow_overlap and allow_gemm_overlap:
        raise ValueError("select one TOKEN_BEGIN overlap mode")
    # vd[0] overlaps only an unrelated PostProc drain. vd[1] additionally
    # permits a live GEMM/RF read while the body DMA writes another L1 tile.
    overlap_mode = 2 if allow_gemm_overlap else int(allow_overlap)
    return UInstr(UOpcode.NOP, vd=overlap_mode,
                  imm=(NOP_SUBOP_TOKEN_BEGIN << 12) | count)


def token_loop_end():
    return UInstr(UOpcode.NOP, imm=NOP_SUBOP_TOKEN_END << 12)


def expert_loop_begin(count):
    if not 1 <= count <= 60:
        raise ValueError("expert loop supports 1..60 routed experts")
    # Keep vd != 0xF: loop_controller reserves vd=0xF for K-tile LOOP.
    return UInstr(UOpcode.NOP, vd=0xE, vs1=0xE, vs2=0xD, imm=count)


def expert_loop_end():
    return UInstr(UOpcode.NOP, vd=0xE, vs1=0xD, vs2=0xE)


def set_weight_page(page):
    if not 0 <= page < 4096:
        raise ValueError("static weight page must fit 12 bits")
    return UInstr(UOpcode.NOP, imm=(NOP_SUBOP_SET_WPAGE << 12) | page)


def static_scale(page_offset, group_count=1):
    """Load 1..16 contiguous 128-byte scale/zero groups from this page."""
    if page_offset < 0 or page_offset >= 0x10000 or page_offset % 16:
        raise ValueError("static scale offset must be 16-byte aligned within its page")
    if not 1 <= group_count <= 16:
        raise ValueError("static scale group_count must be in [1,16]")
    return UInstr(UOpcode.NOP, vd=group_count - 1,
                  imm=(NOP_SUBOP_STATIC_SCALE << 12) | (page_offset // 16))


def static_dma(weight_addr, l1_bank=1, fp16=False, loop_stride_bytes=None,
               transfer_bytes=None):
    """Return page-select + 4-KiB static weight DMA instructions.

    For a quantized hardware-loop body, encode the DDR tile stride and
    transfer size (128/256/512/1024 bytes) in vs2. A non-looped compact
    transfer may combine adjacent tiles into a supported larger DMA burst.
    Legacy calls retain the existing 4-KiB transfer behavior.
    """
    if weight_addr < 0 or weight_addr % 16:
        raise ValueError("static weight address must be non-negative and 16-byte aligned")
    if loop_stride_bytes is not None and transfer_bytes is not None:
        raise ValueError("static DMA cannot encode loop stride and burst size together")
    stride_code = 0
    if loop_stride_bytes is not None:
        if fp16:
            raise ValueError("compact quantized loop stride cannot be used for FP16 DMA")
        stride_codes = {128: 1, 256: 2, 512: 3, 1024: 4}
        if loop_stride_bytes not in stride_codes:
            raise ValueError(
                "quantized static loop stride must be 128, 256, 512, or 1024 bytes")
        stride_code = stride_codes[loop_stride_bytes]
    if transfer_bytes is not None:
        if fp16 and transfer_bytes != 2048:
            raise ValueError("compact FP16 static DMA currently supports 2048-byte tiles")
        transfer_codes = {
            128: 1, 256: 2, 512: 3, 1024: 4,
            2048: 5, 3072: 6, 4096: 7,
        }
        if transfer_bytes not in transfer_codes:
            raise ValueError("unsupported compact static DMA burst size")
        stride_code = transfer_codes[transfer_bytes]
    return [set_weight_page(weight_addr >> 16),
            UInstr(UOpcode.DMA_RD, vd=l1_bank, vs1=0x5 if fp16 else 0x4,
                   vs2=stride_code, imm=(weight_addr & 0xFFFF) // 16)]


def static_scale_at(scale_addr, group_count=1):
    if scale_addr < 0 or scale_addr % 16:
        raise ValueError("static scale address must be non-negative and 16-byte aligned")
    if not 1 <= group_count <= 16:
        raise ValueError("static scale group_count must be in [1,16]")
    return [set_weight_page(scale_addr >> 16),
            static_scale(scale_addr & 0xFFFF, group_count)]


def static_gld_w(nbits, l1_addr, group=0):
    if not 1 <= nbits <= 4:
        raise ValueError("static quantized precision must be in [1,4]")
    if not 0 <= group < 16:
        raise ValueError("static scale group must be in [0,15]")
    return UInstr(UOpcode.GLD_W, vd=int(Precision.INT1),
                  vs1=0x8 | (group & 0x7),
                  vs2=((group >> 3) << 3) | nbits, imm=l1_addr)


def static_gemm(nbits, acc=False, skip_store=False, imm=32, loop_managed=False):
    """Encode compact INT GEMM controls, including loop-entry ACC flags."""
    if not 1 <= nbits <= 4:
        raise ValueError("static quantized precision must be in [1,4]")
    vd = ((1 if skip_store else 0) << 3) | ((1 if acc else 0) << 2) | int(Precision.INT1)
    return UInstr(UOpcode.GEMM, vd=vd, vs1=0xC if loop_managed else 0x8,
                  vs2=nbits, imm=imm)


def rweight(vd, slot):
    """Broadcast a selected FP16 routing weight to all 32 PRF lanes."""
    if not 0 <= vd < 16 or not 1 <= slot <= 4:
        raise ValueError("invalid route-weight register or Top-4 slot")
    return UInstr(UOpcode.NOP, vd=vd, vs2=slot, imm=NOP_SUBOP_RWEIGHT << 12)


def group_weight(vd, row):
    """Broadcast the cached route weight for one compact group row."""
    if not 0 <= vd < 16 or not 0 <= row < 32:
        raise ValueError("invalid route-group weight destination or row")
    return UInstr(UOpcode.NOP, vd=vd,
                  imm=(NOP_SUBOP_GROUP_WEIGHT << 12) | row)


def group_skip_if_empty(instruction_count):
    """Skip following source instructions when the selected expert is empty."""
    if not 1 <= instruction_count <= 0xFFFF:
        raise ValueError("group skip count must fit a non-zero 16-bit value")
    return UInstr(UOpcode.NOP, vd=0xE, vs1=0xF,
                  vs2=(instruction_count >> 12) & 0xF,
                  imm=instruction_count & 0x0FFF)


def group_skip_if_row_inactive(row, instruction_count):
    """Skip one padded-row body when the selected group has no such row."""
    if not 0 <= row < 32 or not 1 <= instruction_count <= 7:
        raise ValueError("group row skip requires row 0..31 and 1..7 instructions")
    # Keep NOP_SUBOP_NOP so this control marker cannot trigger another unit.
    # vs1=0xD distinguishes it from other control markers; row[4] is imm[3],
    # row[3:0] is vs2, and the following source length is imm[2:0].
    return UInstr(UOpcode.NOP, vd=0xE, vs1=0xD, vs2=row & 0xF,
                  imm=((row >> 4) << 3) | instruction_count)


def broadcast(vd, vs1, lane=0):
    """Broadcast one FP16 lane from a PRF register to all 32 lanes."""
    if not 0 <= vd < 16 or not 0 <= vs1 < 16 or not 0 <= lane < 32:
        raise ValueError("invalid PRF broadcast operands")
    return UInstr(UOpcode.NOP, vd=vd, vs1=vs1,
                  imm=(NOP_SUBOP_BCAST << 12) | lane)


def expert_dma(nt, kt, k_tiles, slot, proj, l1_bank=0,
               transfer_k=None):
    """Compact descriptor addressing: logical 32-K tile, never a byte offset.

    k_tiles is the projection's full K dimension divided by 32. During a
    hardware loop, decode advances this logical index by cfg_K/32 per
    iteration, so cfg_K=64 reads pairs of adjacent tiles. The descriptor
    supplies the byte stride after hardware selects the expert. When
    ``transfer_k`` is provided, the extended grouped-DMA form carries its
    transfer width independently of CFG_K so a differently sized next chunk
    can be prefetched while the current GEMM is active.
    """
    if not (0 <= nt < 256 and 0 <= kt < k_tiles <= 255 and 1 <= slot <= 4 and 0 <= proj < 3):
        raise ValueError("invalid expert tile coordinates")
    logical_tile = nt * k_tiles + kt
    if transfer_k is None:
        return dma_rd(logical_tile, l1_bank, slot, proj)
    transfer_code = {32: 1, 64: 2, 128: 3, 256: 0}.get(transfer_k)
    if transfer_code is None or logical_tile > 0x3FFF:
        raise ValueError("explicit expert DMA K must be legal and tile index fit 14 bits")
    # vs1=0xF with vs2 in [2,13] is the extended expert-transfer form.
    # The packed selector avoids the existing vs1=0xE OUT-token-gather
    # encoding, which otherwise aliases descriptor-driven Down transfers.
    # vs2=0/1 remains reserved for OUT route-group gathers. imm[15:14] stores
    # the transfer width (0=256, 1=32, 2=64, 3=128) and imm[13:0] the logical
    # descriptor tile, expanding the address range without changing legacy
    # non-override DMA instructions.
    selector = 2 + (slot - 1) * 3 + proj
    return UInstr(UOpcode.DMA_RD, vd=l1_bank, vs1=0xF,
                  vs2=selector, imm=(transfer_code << 14) | logical_tile)


def weighted_expert_chunk(slot, output_reg, sum_reg, weight_reg, first=False):
    """Combine a computed 32-lane expert output with its runtime route weight.

    The first contribution overwrites sum_reg; later contributions add to it.
    Registers must be distinct so the broadcast cannot destroy an input.
    """
    regs = (output_reg, sum_reg, weight_reg)
    if len(set(regs)) != 3 or any(not 0 <= r < 16 for r in regs):
        raise ValueError("combine needs three distinct PRF registers")
    program = [rweight(weight_reg, slot)]
    if first:
        program.append(mul(sum_reg, output_reg, weight_reg))
    else:
        program.extend([mul(output_reg, output_reg, weight_reg),
                        add(sum_reg, sum_reg, output_reg)])
    return program

def ac2v(vd, row, set=0):
    """NOP_AC2V: acc_buffer 一行 (32xfp32) -> fp16 -> PostProc RF vd。

    imm[15:12]=4'h5 (子码), imm[4:0]=acc 行号, imm[5]=set 选择
    (0=非活动集/刚 store 完的结果集, 1=活动集)。
    每条搬 1 行；搬一个 32x32 tile 需 32 条。
    """
    imm = (NOP_SUBOP_AC2V << 12) | ((set & 0x1) << 5) | (row & 0x1F)
    return UInstr(UOpcode.NOP, vd=vd, imm=imm)


# =============================================================================
# Serialization
# =============================================================================
def write_binary(path, program):
    """Write a list of UInstr to a binary file (4 bytes each, little-endian)."""
    with open(path, 'wb') as f:
        for instr in program:
            f.write(instr.bytes_le())
    return len(program)


def write_text(path, program):
    """Write a human-readable text dump of the program."""
    with open(path, 'w') as f:
        f.write(f"# {len(program)} instructions\n")
        f.write(f"# {'idx':>4} {'opcode':8s} {'vd':>2} {'vs1':>3} {'vs2':>3} {'imm':>6}  raw\n")
        for i, instr in enumerate(program):
            f.write(f"{i:4d}  {instr.to_text()}\n")
    return len(program)


def dump_program(bin_path="ucode.bin", txt_path="ucode.txt", program=None):
    """Convenience: write both binary and text dumps."""
    n1 = write_binary(bin_path, program)
    n2 = write_text(txt_path, program)
    print(f"[MachineCodeGen] wrote {n1} instructions -> {bin_path} ({n1*4} bytes)")
    print(f"[MachineCodeGen] wrote text dump -> {txt_path}")
    return n1


if __name__ == '__main__':
    # Self-test: encode/decode a few instructions
    prog = [
        dma_rd(0x1000),
        barrier(),
        gld_w(Precision.INT8, 0x0000),
        barrier(),
        gld_a(0x2000),
        barrier(),
        gemm(),
        barrier(),
        end(),
    ]
    print(f"Self-test: {len(prog)} instructions")
    for i, inst in enumerate(prog):
        print(f"  {i}: {inst.to_text()}  bytes={inst.bytes_le().hex()}")
    dump_program("_test_ucode.bin", "_test_ucode.txt", prog)
    print("OK")
