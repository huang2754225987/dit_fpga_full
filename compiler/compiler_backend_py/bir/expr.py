"""BIR affine expression system.

Python reimplementation of the C++ BIR Expr hierarchy (Expr.h / Expr.cpp).

The expression model supports:
  * ``VarExpr``   - a loop induction variable, optionally bound to a LoopAxis
  * ``ConstExpr`` - an integer (float-stored, like the C++ side) constant
  * ``RangeExpr`` - a half-open range ``[start, end)`` with a stride
  * ``ArithExpr`` - a binary/leaf arithmetic node (``+ * - < > == & | %``)

Pythonic niceties layered on top:
  * structural ``__eq__`` / ``__hash__`` (so Exprs are usable as dict keys)
  * arithmetic operator overloading (``Expr + Expr``, ``Expr * int``, ...)
  * ``substitute(var, replacement)`` and ``evaluate(env)``
  * ``to_string()`` for debugging
"""

from __future__ import annotations

import math
from enum import Enum
from typing import List, Dict, Optional, Tuple, Union


class ExprOp(Enum):
    """Arithmetic / logic operator carried by an ArithExpr node."""
    AddExpr = 1
    SubExpr = 2
    MulExpr = 3
    GtExpr = 4
    LtExpr = 5
    EqExpr = 6
    AndExpr = 7
    OrExpr = 8
    ModExpr = 9
    IdentityExpr = 10


_OP_SYMBOL = {
    ExprOp.AddExpr: "+",
    ExprOp.SubExpr: "-",
    ExprOp.MulExpr: "*",
    ExprOp.GtExpr: ">",
    ExprOp.LtExpr: "<",
    ExprOp.EqExpr: "==",
    ExprOp.AndExpr: "&",
    ExprOp.OrExpr: "|",
    ExprOp.ModExpr: "%",
    ExprOp.IdentityExpr: "",
}

_LOGIC_OPS = {ExprOp.EqExpr, ExprOp.GtExpr, ExprOp.LtExpr, ExprOp.AndExpr, ExprOp.OrExpr}


class Expr:
    """Base class for all BIR expressions.

    The C++ hierarchy uses RTTI (``llvm::isa``). In Python we just use
    ``isinstance`` plus the helper predicates below.
    """

    def __init__(self, op: Optional[ExprOp] = None,
                 operands: Optional[List["Expr"]] = None):
        self._op = op
        self._operands: List["Expr"] = operands if operands is not None else []

    # -- introspection -------------------------------------------------
    @property
    def op(self) -> Optional[ExprOp]:
        return self._op

    @property
    def operands(self) -> List["Expr"]:
        return self._operands

    def get_operand(self, index: int) -> "Expr":
        return self._operands[index]

    def num_operands(self) -> int:
        return len(self._operands)

    def is_arith_expr(self) -> bool:
        return not self.is_logic_expr()

    def is_logic_expr(self) -> bool:
        return self._op in _LOGIC_OPS

    # -- structural equality / hashing ---------------------------------
    def _key(self):
        return (type(self).__name__, self._op, tuple(o._key() for o in self._operands))

    def __eq__(self, other):
        if not isinstance(other, Expr):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    # -- the core transforms every subclass must honour ---------------
    def substitute(self, var: "VarExpr", replacement: "Expr") -> "Expr":
        """Return a new Expr with ``var`` replaced by ``replacement``.

        Mirrors ``Expr::replace`` in C++ but is cleaner because each node
        owns its own implementation. The base class handles ArithExpr.
        """
        if not self._operands:
            return self
        new_ops = [o.substitute(var, replacement) for o in self._operands]
        return ArithExpr(self._op, new_ops)

    def evaluate(self, env: Optional[Dict["VarExpr", int]] = None) -> int:
        """Evaluate to an integer using ``env`` to resolve VarExprs."""
        env = env or {}
        if self._op is None or not self._operands:
            raise ValueError(f"cannot evaluate leaf expr {self!r}")
        if self.is_logic_expr():
            raise ValueError("evaluate() is for arithmetic expressions; "
                             "use evaluate_bool() for logic exprs")
        if len(self._operands) == 1:
            val = self._operands[0].evaluate(env)
            if self._op == ExprOp.IdentityExpr:
                return val
            raise ValueError(f"unknown unary op {self._op}")
        a = self._operands[0].evaluate(env)
        b = self._operands[1].evaluate(env)
        if self._op == ExprOp.AddExpr:
            return a + b
        if self._op == ExprOp.SubExpr:
            return a - b
        if self._op == ExprOp.MulExpr:
            return a * b
        if self._op == ExprOp.ModExpr:
            return a % b
        raise ValueError(f"cannot evaluate op {self._op}")

    def evaluate_bool(self, env: Optional[Dict["VarExpr", int]] = None) -> bool:
        env = env or {}
        if not self.is_logic_expr():
            raise ValueError(f"{self!r} is not a logic expression")
        a = self._operands[0].evaluate(env)
        b = self._operands[1].evaluate(env)
        if self._op == ExprOp.EqExpr:
            return a == b
        if self._op == ExprOp.GtExpr:
            return a > b
        if self._op == ExprOp.LtExpr:
            return a < b
        if self._op == ExprOp.AndExpr:
            return self._operands[0].evaluate_bool(env) and self._operands[1].evaluate_bool(env)
        if self._op == ExprOp.OrExpr:
            return self._operands[0].evaluate_bool(env) or self._operands[1].evaluate_bool(env)
        raise ValueError(f"cannot evaluate logic op {self._op}")

    # -- cloning -------------------------------------------------------
    def clone(self) -> "Expr":
        if not self._operands:
            return self
        return ArithExpr(self._op, [o.clone() for o in self._operands])

    # -- variable collection ------------------------------------------
    def collect_vars(self) -> List["VarExpr"]:
        result: List[VarExpr] = []
        self._collect_vars_into(result)
        return result

    def _collect_vars_into(self, out: List["VarExpr"]) -> None:
        for o in self._operands:
            o._collect_vars_into(out)

    # -- debugging -----------------------------------------------------
    def to_string(self) -> str:
        if not self._operands:
            sym = _OP_SYMBOL.get(self._op, "?")
            return sym
        if len(self._operands) == 1:
            return f"{_OP_SYMBOL.get(self._op, '?')}({self._operands[0].to_string()})"
        sym = _OP_SYMBOL.get(self._op, "?")
        return f"({self._operands[0].to_string()} {sym} {self._operands[1].to_string()})"

    def __repr__(self):
        return self.to_string()

    def __str__(self):
        return self.to_string()

    # -- arithmetic operator overloading ------------------------------
    def _binop(self, op: ExprOp, other) -> "ArithExpr":
        other = _coerce(other)
        return ArithExpr(op, [self, other])

    def __add__(self, other):
        return self._binop(ExprOp.AddExpr, other)

    def __radd__(self, other):
        return _coerce(other)._binop(ExprOp.AddExpr, self)

    def __sub__(self, other):
        return self._binop(ExprOp.SubExpr, other)

    def __rsub__(self, other):
        return _coerce(other)._binop(ExprOp.SubExpr, self)

    def __mul__(self, other):
        return self._binop(ExprOp.MulExpr, other)

    def __rmul__(self, other):
        return _coerce(other)._binop(ExprOp.MulExpr, self)

    def __mod__(self, other):
        return self._binop(ExprOp.ModExpr, other)

    def __lt__(self, other):
        # NOTE: this overrides rich-comparison for Expr objects. We keep
        # __eq__ structural, so `<` returns an Expr (predicate) rather than a
        # bool, which matches the C++ ``Expr::operator<`` semantics.
        return self._binop(ExprOp.LtExpr, other)

    def __gt__(self, other):
        return self._binop(ExprOp.GtExpr, other)

    def __and__(self, other):
        return self._binop(ExprOp.AndExpr, other)

    def __or__(self, other):
        return self._binop(ExprOp.OrExpr, other)

    # -- simplification (best-effort, mirrors Expr::simplify) ---------
    def simplify(self) -> "Expr":
        if not self._operands:
            return self
        if self.is_logic_expr():
            return ArithExpr(self._op, [o.simplify() for o in self._operands])
        # binary arithmetic
        if len(self._operands) == 1:
            return ArithExpr(self._op, [self._operands[0].simplify()])
        a, b = self._operands[0].simplify(), self._operands[1].simplify()
        if isinstance(a, ConstExpr) and isinstance(b, ConstExpr):
            if self._op == ExprOp.AddExpr:
                return ConstExpr(a.value + b.value)
            if self._op == ExprOp.SubExpr:
                return ConstExpr(a.value - b.value)
            if self._op == ExprOp.MulExpr:
                return ConstExpr(a.value * b.value)
            if self._op == ExprOp.ModExpr:
                return ConstExpr(a.value % b.value) if b.value != 0 else ConstExpr(0)
        # identity folding
        if self._op in (ExprOp.AddExpr, ExprOp.SubExpr) and isinstance(a, ConstExpr) and a.value == 0:
            return b if self._op == ExprOp.AddExpr else ArithExpr(ExprOp.SubExpr, [ConstExpr(0), b])
        if self._op == ExprOp.MulExpr and isinstance(a, ConstExpr) and a.value == 1:
            return b
        if self._op == ExprOp.MulExpr and isinstance(a, ConstExpr) and a.value == 0:
            return ConstExpr(0)
        if self._op == ExprOp.MulExpr and isinstance(b, ConstExpr) and b.value == 1:
            return a
        if self._op == ExprOp.MulExpr and isinstance(b, ConstExpr) and b.value == 0:
            return ConstExpr(0)
        if self._op == ExprOp.AddExpr and isinstance(b, ConstExpr) and b.value == 0:
            return a
        # distribution: c * (x + d) -> c*x + c*d
        if self._op == ExprOp.MulExpr and isinstance(a, ConstExpr) \
                and isinstance(b, ArithExpr) and b._op == ExprOp.AddExpr \
                and isinstance(b._operands[1], ConstExpr):
            c = a
            x, d = b._operands[0], b._operands[1]
            return ArithExpr(ExprOp.AddExpr,
                             [ArithExpr(ExprOp.MulExpr, [c, x]).simplify(),
                              ArithExpr(ExprOp.MulExpr, [c, d]).simplify()])
        return ArithExpr(self._op, [a, b])


class ArithExpr(Expr):
    """A binary arithmetic / logic node (the C++ generic ``Expr``)."""
    def __init__(self, op: ExprOp, operands: List[Expr]):
        super().__init__(op=op, operands=list(operands))

    def to_string(self) -> str:
        sym = _OP_SYMBOL.get(self._op, "?")
        if len(self._operands) == 1:
            return f"{sym}({self._operands[0].to_string()})"
        return f"({self._operands[0].to_string()} {sym} {self._operands[1].to_string()})"


class ConstExpr(Expr):
    """An integer constant.

    The C++ side stores a ``float _value`` to mirror quantisation fractions; we
    keep a plain ``int`` since the BIR passes that touch this reimplementation
    only ever use integer constants.
    """

    def __init__(self, value):
        self.value = int(value)

    def _key(self):
        return ("ConstExpr", self.value)

    def __eq__(self, other):
        return isinstance(other, ConstExpr) and other.value == self.value

    def __hash__(self):
        return hash(("ConstExpr", self.value))

    def substitute(self, var, replacement):
        return self

    def evaluate(self, env=None):
        return self.value

    def clone(self):
        return ConstExpr(self.value)

    def _collect_vars_into(self, out):
        pass

    def to_string(self) -> str:
        return str(self.value)

    def __repr__(self):
        return f"ConstExpr({self.value})"


class VarExpr(Expr):
    """A loop induction variable.

    A VarExpr is identified by its bound ``LoopAxis`` (preferred) or, when no
    axis is set, by its ``name``. Two VarExprs are equal when they share the
    same axis *object* (or, lacking one, the same name).
    """

    def __init__(self, name: str = "", axis=None):
        self.name = name or (axis.name if axis is not None else "")
        self.axis = axis  # type: ignore[assignment]
        if axis is not None:
            axis.var = self

    def _key(self):
        # Identity of a VarExpr follows the owning axis when present.
        if self.axis is not None:
            return ("VarExpr-axis", id(self.axis))
        return ("VarExpr-name", self.name)

    def __eq__(self, other):
        if not isinstance(other, VarExpr):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def substitute(self, var, replacement):
        if self == var:
            return replacement
        return self

    def evaluate(self, env=None):
        env = env or {}
        if self in env:
            return env[self]
        if self.axis is not None:
            return self.axis.lb
        raise ValueError(f"cannot evaluate unbound VarExpr {self.name!r}")

    def clone(self) -> "VarExpr":
        # VarExprs are identity-stable (like the C++ ``VarExpr::clone`` which
        # returns ``this``); cloning yields the same object.
        return self

    def _collect_vars_into(self, out):
        out.append(self)

    def to_string(self) -> str:
        return self.name

    def __repr__(self):
        return f"VarExpr({self.name!r})"


class RangeExpr(Expr):
    """A half-open range ``[start, end)`` accessed with ``stride``.

    Mirrors the C++ ``RangeExpr`` which is used by AccessPattern to denote a
    contiguous span of a tensor dimension.
    """

    def __init__(self, start: int, end: int, stride: int = 1,
                 vars: Optional[List] = None):
        self.start = start
        self.end = end
        self.stride = stride
        self.vars: List = list(vars) if vars is not None else []

    def _key(self):
        return ("RangeExpr", self.start, self.end, self.stride)

    def __eq__(self, other):
        return isinstance(other, RangeExpr) \
            and other.start == self.start and other.end == self.end \
            and other.stride == self.stride

    def __hash__(self):
        return hash(self._key())

    def substitute(self, var, replacement):
        # Ranges are constants once materialised.
        return self

    def evaluate(self, env=None):
        return self.start

    def clone(self) -> "RangeExpr":
        return RangeExpr(self.start, self.end, self.stride, list(self.vars))

    def _collect_vars_into(self, out):
        pass

    def overlap(self, other: "RangeExpr") -> bool:
        """True if the two half-open ranges touch at least one element."""
        if self.stride != 1 or other.stride != 1:
            # strided ranges: fall back to a conservative overlap check on the
            # [start,end) span (good enough for the dependency checks the BIR
            # passes rely on).
            pass
        return (self.start <= other.start < self.end) or \
               (other.start <= self.start < other.end) or \
               (self.start < other.end <= self.end) or \
               (other.start < self.end <= self.end)

    def to_string(self) -> str:
        s = f"[{self.start}:{self.end}"
        if self.stride != 1:
            s += f":{self.stride}"
        s += ")"
        return s

    def __repr__(self):
        return f"RangeExpr({self.start}, {self.end}, {self.stride})"


def _coerce(value) -> Expr:
    """Coerce a python int / float into a ConstExpr; pass Exprs through."""
    if isinstance(value, Expr):
        return value
    if isinstance(value, bool):
        return ConstExpr(int(value))
    if isinstance(value, (int, float)):
        return ConstExpr(value)
    raise TypeError(f"cannot coerce {type(value).__name__} into an Expr")


def make_range_from_expr(e: Expr) -> RangeExpr:
    """Build a RangeExpr covering the full iteration span of ``e``.

    Mirrors the C++ ``RangeExpr(Expr& e)`` constructor: it parses ``e`` as an
    affine expression ``sum(k_i * var_i) + c`` and turns each var's axis into a
    [lb, ub) range, then expands accordingly.
    """
    if isinstance(e, ConstExpr):
        return RangeExpr(e.value, e.value + 1, 1)
    if isinstance(e, RangeExpr):
        return e.clone()
    if isinstance(e, VarExpr):
        if e.axis is None:
            return RangeExpr(0, 1, 1)
        return RangeExpr(e.axis.lb, e.axis.ub, e.axis.stride, [e.axis])
    varmap: Dict = {}
    const_term = 0
    _parse_affine(e, varmap, const_term)
    # rebuild the lambda since python closures over primitives need mutable refs
    start = 0
    end = 0
    stride = 0
    const_val = const_term
    # recompute properly
    start, end, stride = _affine_range(varmap, const_val)
    return RangeExpr(start, end, stride, list(varmap.keys()))


def _parse_affine(e: Expr, varmap: Dict, const_term: int) -> int:
    """Recursive helper: write k per axis into varmap, return const term."""
    if isinstance(e, ConstExpr):
        return e.value
    if isinstance(e, VarExpr):
        if e.axis is not None and e.axis not in varmap:
            varmap[e.axis] = 1
        return 0
    if isinstance(e, ArithExpr):
        if e._op == ExprOp.AddExpr:
            return _parse_affine(e._operands[0], varmap, const_term) + \
                   _parse_affine(e._operands[1], varmap, const_term)
        if e._op == ExprOp.SubExpr:
            return _parse_affine(e._operands[0], varmap, const_term) - \
                   _parse_affine(e._operands[1], varmap, const_term)
        if e._op == ExprOp.MulExpr:
            a, b = e._operands[0], e._operands[1]
            # expect one const, one var
            if isinstance(a, ConstExpr) and isinstance(b, VarExpr):
                if b.axis is not None:
                    varmap[b.axis] = varmap.get(b.axis, 0) + a.value
                return 0
            if isinstance(b, ConstExpr) and isinstance(a, VarExpr):
                if a.axis is not None:
                    varmap[a.axis] = varmap.get(a.axis, 0) + b.value
                return 0
            # fall through: nested mul -> recurse on the var side, fold const
            for x, y in ((a, b), (b, a)):
                if isinstance(x, ConstExpr):
                    sub_varmap: Dict = {}
                    sub_c = _parse_affine(y, sub_varmap, 0)
                    for axis, k in sub_varmap.items():
                        varmap[axis] = varmap.get(axis, 0) + k * x.value
                    return sub_c * x.value
    return 0


def _affine_range(varmap: Dict, const_val: int) -> Tuple[int, int, int]:
    start = const_val
    end = const_val
    stride = 0
    for axis, k in varmap.items():
        start += axis.lb * k
        end += (axis.ub - 1) * k
        if axis.stride > 1:
            stride = axis.stride * k
    if stride == 0:
        stride = 1
    end += 1  # half-open
    return start, end, stride
