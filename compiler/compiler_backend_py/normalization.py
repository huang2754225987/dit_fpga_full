"""Full-hidden-dimension normalization microcode scheduling."""
try:
    from . import isa
except ImportError:
    import isa

TILE = 32
VECTOR_BYTES = TILE * 2


def emit_rmsnorm(x_base, gamma_base, out_base, hidden_size=2048,
                 x_reg=0, gamma_reg=1, out_reg=2,
                 wait_after_store=True):
    """Emit two-pass RMSNorm across the complete hidden dimension.

    The first pass preserves one global sum-of-squares across every 32-value
    chunk. After a single reciprocal-square-root finalize command, the second
    pass applies the shared statistic and each chunk's gamma values.
    ``wait_after_store=False`` is safe only when the next L1 consumer uses a
    GLD.A instruction marked to wait for the PostProc writer.
    """
    if hidden_size < TILE or hidden_size > 4096 or hidden_size % TILE:
        raise ValueError("hidden_size must be a multiple of 32 in [32,4096]")
    if len({x_reg, gamma_reg, out_reg}) != 3:
        raise ValueError("RMSNorm scheduling requires three distinct PRF registers")
    if any(reg < 0 or reg > 15 for reg in (x_reg, gamma_reg, out_reg)):
        raise ValueError("PRF register indices must be in [0,15]")

    chunks = hidden_size // TILE
    program = []
    for chunk in range(chunks):
        offset = chunk * VECTOR_BYTES
        program.extend([
            isa.vld(x_reg, x_base + offset),
            isa.rmsnorm_accumulate(x_reg, reset=chunk == 0),
        ])

    program.extend(emit_rmsnorm_apply(
        x_base, gamma_base, out_base, hidden_size,
        x_reg=x_reg, gamma_reg=gamma_reg, out_reg=out_reg,
        finalize=True, wait_after_store=wait_after_store))
    return program


def emit_rmsnorm_apply(x_base, gamma_base, out_base, hidden_size=2048,
                       x_reg=0, gamma_reg=1, out_reg=2, finalize=False,
                       wait_after_store=True):
    """Finalize RMSNorm; deferred stores require a marked GLD.A consumer."""
    if hidden_size < TILE or hidden_size > 4096 or hidden_size % TILE:
        raise ValueError("hidden_size must be a multiple of 32 in [32,4096]")
    if len({x_reg, gamma_reg, out_reg}) != 3:
        raise ValueError("RMSNorm scheduling requires three distinct PRF registers")
    if any(reg < 0 or reg > 15 for reg in (x_reg, gamma_reg, out_reg)):
        raise ValueError("PRF register indices must be in [0,15]")

    chunks = hidden_size // TILE
    program = [isa.rmsnorm_finalize(hidden_size)] if finalize else []
    for chunk in range(chunks):
        offset = chunk * VECTOR_BYTES
        program.extend([
            isa.vld(x_reg, x_base + offset),
            isa.vld(gamma_reg, gamma_base + offset),
            isa.rmsnorm_apply(out_reg, x_reg, gamma_reg),
            isa.vst(out_reg, out_base + offset),
        ])
    # A marked GLD.A can retire this writer before reading L1. Keep the
    # default fence for standalone schedules with other consumers.
    if wait_after_store:
        program.append(isa.barrier())
    return program
