"""Math built-in functions for MatPy."""

import math
import numpy as np
from matpy.builtins import register
from matpy.runtime.types import Mat
from matpy.runtime.matrix import to_mat


def _wrap_math(func):
    def wrapper(*args):
        unwrapped = []
        for a in args:
            if isinstance(a, Mat):
                unwrapped.append(a.data)
            else:
                unwrapped.append(a)
        result = func(*unwrapped)
        return to_mat(result)

    # Preserve docstring from original function
    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__
    return wrapper


register("sin", _wrap_math(np.sin))
register("cos", _wrap_math(np.cos))
register("tan", _wrap_math(np.tan))
register("asin", _wrap_math(np.arcsin))
register("acos", _wrap_math(np.arccos))
register("atan", _wrap_math(np.arctan))
register("atan2", _wrap_math(np.arctan2))
register("sinh", _wrap_math(np.sinh))
register("cosh", _wrap_math(np.cosh))
register("tanh", _wrap_math(np.tanh))
register("asinh", _wrap_math(np.arcsinh))
register("acosh", _wrap_math(np.arccosh))
register("atanh", _wrap_math(np.arctanh))

register("exp", _wrap_math(np.exp))
register("log", _wrap_math(np.log))
register("log2", _wrap_math(np.log2))
register("log10", _wrap_math(np.log10))
register("sqrt", _wrap_math(np.sqrt))

register("ceil", _wrap_math(np.ceil))
register("floor", _wrap_math(np.floor))
register("fix", _wrap_math(np.fix))
register("round", _wrap_math(np.round))

register("abs", _wrap_math(np.abs))
register("angle", _wrap_math(np.angle))
register("sign", _wrap_math(np.sign))
register("conj", _wrap_math(np.conj))


@register("min")
def _min(*args):
    arrs = [a.data if isinstance(a, Mat) else np.array(a) for a in args]
    if len(arrs) == 1:
        return to_mat(np.min(arrs[0]))
    elif len(arrs) == 2:
        return to_mat(np.minimum(arrs[0], arrs[1]))
    return to_mat(np.min(arrs))


@register("max")
def _max(*args):
    arrs = [a.data if isinstance(a, Mat) else np.array(a) for a in args]
    if len(arrs) == 1:
        return to_mat(np.max(arrs[0]))
    elif len(arrs) == 2:
        return to_mat(np.maximum(arrs[0], arrs[1]))
    return to_mat(np.max(arrs))


@register("sum")
def _sum(x, dim=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if dim is not None:
        return to_mat(np.sum(data, axis=int(dim) - 1))
    return to_mat(np.sum(data))


@register("prod")
def _prod(x, dim=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if dim is not None:
        return to_mat(np.prod(data, axis=int(dim) - 1))
    return to_mat(np.prod(data))


@register("cumsum")
def _cumsum(x, dim=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if dim is not None:
        return to_mat(np.cumsum(data, axis=int(dim) - 1))
    return to_mat(np.cumsum(data))


@register("cumprod")
def _cumprod(x, dim=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if dim is not None:
        return to_mat(np.cumprod(data, axis=int(dim) - 1))
    return to_mat(np.cumprod(data))


@register("diff")
def _diff(x, n=1):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return to_mat(np.diff(data, n=int(n)))


register("mod", _wrap_math(np.mod))
register("rem", _wrap_math(np.remainder))


@register("pi")
def _pi():
    return math.pi


@register("inf")
def _inf():
    return float("inf")


@register("nan")
def _nan():
    return float("nan")


# Additional math functions
register("expm1", _wrap_math(np.expm1))
register("log1p", _wrap_math(np.log1p))
register("cbrt", _wrap_math(np.cbrt))
register("square", _wrap_math(np.square))
register("reciprocal", _wrap_math(np.reciprocal))
register("gcd", _wrap_math(np.gcd))
register("lcm", _wrap_math(np.lcm))
register("hypot", _wrap_math(np.hypot))
register("rad2deg", _wrap_math(np.degrees))
register("deg2rad", _wrap_math(np.radians))


@register("factor")
def _factor(n):
    """Prime factorization."""
    n = int(n)
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return Mat(np.array(factors))


@register("isprime")
def _isprime(n):
    """Check if number is prime."""
    n = int(n)
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


@register("primes")
def _primes(n):
    """Generate list of primes up to n."""
    n = int(n)
    if n < 2:
        return Mat(np.array([]))
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return Mat(np.array([i for i in range(2, n + 1) if sieve[i]]))


@register("factorial")
def _factorial(n):
    """Factorial function."""
    n = int(n)
    if n < 0:
        return float("nan")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


@register("nchoosek")
def _nchoosek(n, k):
    """Binomial coefficient."""
    n, k = int(n), int(k)
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


@register("perms")
def _perms(x):
    """All permutations."""
    from itertools import permutations

    data = x.data if isinstance(x, Mat) else np.array(x)
    items = list(data.flatten())
    perms = list(permutations(items))
    return Mat(np.array(perms))


@register("permn")
def _permn(n, k):
    """Number of permutations."""
    n, k = int(n), int(k)
    if k < 0 or k > n:
        return 0
    result = 1
    for i in range(k):
        result *= n - i
    return result


@register("nextpow2")
def _nextpow2(n):
    """Next higher power of 2."""
    n = abs(int(n) if not isinstance(n, Mat) else int(n.data.flat[0]))
    if n == 0:
        return 0
    return int(np.ceil(np.log2(n)))


@register("cart2pol")
def _cart2pol(x, y):
    """Convert Cartesian to polar coordinates."""
    x = x.data if isinstance(x, Mat) else np.array(x)
    y = y.data if isinstance(y, Mat) else np.array(y)
    theta = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    return Mat(theta), Mat(r)


@register("pol2cart")
def _pol2cart(theta, r):
    """Convert polar to Cartesian coordinates."""
    theta = theta.data if isinstance(theta, Mat) else np.array(theta)
    r = r.data if isinstance(r, Mat) else np.array(r)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return Mat(x), Mat(y)


@register("log2")
def _log2(x):
    """Base-2 logarithm."""
    if isinstance(x, Mat):
        return Mat(np.log2(x.data))
    return np.log2(x)


@register("pow2")
def _pow2(x):
    """Power of 2."""
    if isinstance(x, Mat):
        return Mat(2.0**x.data)
    return 2.0**x


@register("realpow")
def _realpow(x, y):
    """Real power."""
    x = x.data if isinstance(x, Mat) else np.array(x)
    y = y.data if isinstance(y, Mat) else np.array(y)
    return Mat(x**y)


@register("hypot")
def _hypot(x, y):
    """Hypotenuse function."""
    x = x.data if isinstance(x, Mat) else np.array(x)
    y = y.data if isinstance(y, Mat) else np.array(y)
    return Mat(np.sqrt(x**2 + y**2))


# ── Special Functions ─────────────────────────────────────────


@register("airy")
def _airy(k, x):
    """Airy functions."""
    from scipy.special import airy as scipy_airy

    x_data = x.data if isinstance(x, Mat) else np.array(x)
    k = int(k)
    ai, aip, bi, bip = scipy_airy(x_data)
    if k == 0:
        return Mat(ai)
    elif k == 1:
        return Mat(aip)
    elif k == 2:
        return Mat(bi)
    elif k == 3:
        return Mat(bip)
    return Mat(ai)


@register("besselj")
def _besselj(nu, x):
    """Bessel function of the first kind."""
    from scipy.special import jv

    nu = float(nu.data.flat[0]) if isinstance(nu, Mat) else float(nu)
    x_data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(jv(nu, x_data))


@register("bessely")
def _bessely(nu, x):
    """Bessel function of the second kind."""
    from scipy.special import yv

    nu = float(nu.data.flat[0]) if isinstance(nu, Mat) else float(nu)
    x_data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(yv(nu, x_data))


@register("besseli")
def _besseli(nu, x):
    """Modified Bessel function of the first kind."""
    from scipy.special import iv

    nu = float(nu.data.flat[0]) if isinstance(nu, Mat) else float(nu)
    x_data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(iv(nu, x_data))


@register("besselk")
def _besselk(nu, x):
    """Modified Bessel function of the second kind."""
    from scipy.special import kv

    nu = float(nu.data.flat[0]) if isinstance(nu, Mat) else float(nu)
    x_data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(kv(nu, x_data))


@register("besselh")
def _besselh(nu, k, x):
    """Bessel function of the third kind (Hankel function)."""
    from scipy.special import hankel1, hankel2

    nu = float(nu.data.flat[0]) if isinstance(nu, Mat) else float(nu)
    x_data = x.data if isinstance(x, Mat) else np.array(x)
    k = int(k)
    if k == 1:
        return Mat(hankel1(nu, x_data))
    else:
        return Mat(hankel2(nu, x_data))


@register("ellipj")
def _ellipj(u, m):
    """Jacobi elliptic functions."""
    from scipy.special import ellipj as scipy_ellipj

    u_data = u.data if isinstance(u, Mat) else np.array(u)
    m_data = float(m.data.flat[0]) if isinstance(m, Mat) else float(m)
    sn, cn, dn, ph = scipy_ellipj(u_data, m_data)
    return Mat(sn), Mat(cn), Mat(dn)


@register("ellipke")
def _ellipke(m):
    """Complete elliptic integrals."""
    from scipy.special import ellipk, ellipe

    m_data = float(m.data.flat[0]) if isinstance(m, Mat) else float(m)
    return float(ellipk(m_data)), float(ellipe(m_data))


@register("ellipticK")
def _ellipticK(m):
    """Complete elliptic integral of the first kind."""
    from scipy.special import ellipk

    m_data = float(m.data.flat[0]) if isinstance(m, Mat) else float(m)
    return float(ellipk(m_data))


@register("ellipticE")
def _ellipticE(m):
    """Complete elliptic integral of the second kind."""
    from scipy.special import ellipe

    m_data = float(m.data.flat[0]) if isinstance(m, Mat) else float(m)
    return float(ellipe(m_data))
