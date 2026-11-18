"""Directed MPFR arithmetic for auditing generated fixed-point coefficient words."""

from dataclasses import dataclass
import operator

import gmpy2 as mp


def down(fn, *args):
    with mp.context(mp.get_context(), round=mp.RoundDown):
        return fn(*args)


def up(fn, *args):
    with mp.context(mp.get_context(), round=mp.RoundUp):
        return fn(*args)


def interval(x):
    return x if isinstance(x, Interval) else Interval(mp.mpfr(x), mp.mpfr(x))


@dataclass(frozen=True)
class Interval:
    lo: mp.mpfr
    hi: mp.mpfr

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __add__(self, other):
        rhs = interval(other)
        return Interval(
            down(operator.add, self.lo, rhs.lo), up(operator.add, self.hi, rhs.hi)
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + -interval(other)

    def __rsub__(self, other):
        return interval(other) + -self

    def __mul__(self, other):
        rhs = interval(other)
        pairs = [(a, b) for a in (self.lo, self.hi) for b in (rhs.lo, rhs.hi)]
        return Interval(
            min(down(operator.mul, a, b) for a, b in pairs),
            max(up(operator.mul, a, b) for a, b in pairs),
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        rhs = interval(other)
        if rhs.lo <= 0 <= rhs.hi:
            raise ZeroDivisionError("denominator interval includes zero")
        return self * Interval(
            down(operator.truediv, 1, rhs.hi), up(operator.truediv, 1, rhs.lo)
        )

    def __rtruediv__(self, other):
        return interval(other) / self

    def __pow__(self, n):
        if not (isinstance(n, int) and n >= 0 and self.lo >= 0):
            raise ValueError(
                "coefficient generator needs only nonnegative integer powers"
            )
        return Interval(down(operator.pow, self.lo, n), up(operator.pow, self.hi, n))


class DirectedMath:
    mpfr = staticmethod(interval)

    def __getattr__(self, name):
        if name not in {
            "exp",
            "expm1",
            "atan",
            "asinh",
            "tanh",
            "erf",
            "sqrt",
            "log",
            "log2",
            "log10",
        }:
            raise AttributeError(name)
        fn = getattr(mp, name)

        def monotone(x):
            x = interval(x)
            return Interval(down(fn, x.lo), up(fn, x.hi))

        return monotone

    @staticmethod
    def sin_cos(x):
        if x.lo != x.hi:
            raise ValueError("trig grid arguments must be exact dyadics")
        return tuple(Interval(down(fn, x.lo), up(fn, x.lo)) for fn in (mp.sin, mp.cos))

    @staticmethod
    def const_pi():
        return Interval(down(mp.const_pi), up(mp.const_pi))

    @staticmethod
    def get_exp(x):
        assert mp.get_exp(x.lo) == mp.get_exp(x.hi)
        return mp.get_exp(x.hi)
