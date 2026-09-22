r"""Geometric function $\rho(z,x,y)$."""

from pttools.speedup import njit
from pttools.ssm.const import CS0, CS0_2
from pttools.type_hints import FloatOrArr


@njit(cache=True)
def rho[T: FloatOrArr](z: T, x: T | float, y: T | float) -> T:
    r"""Geometric function $\rho(z,x,y)$
    $$\rho(z,x,y) = \frac{
    \left( y^2 - (x-z)^2 \right)^2
    \left( (x + z)^2 - y^2 \right)^2
    }{16 x y z^2}$$
    :giombi_2024_cs:`\ ` eq. 2.37
    :giombi_2026:`\ ` eq. 2.42
    This is included in
    :gw_pt_ssm:`\ ` eq. 3.47.
    """
    return (y**2 - (x - z)**2)**2 * ((x + z)**2 - y**2)**2 / (16 * x * y * z**2)  # pyrefly: ignore[bad-return]


@njit(cache=True)
def rho_delta[T: FloatOrArr](z: T, x: T | float, xp: T | float, xm: T | float, cs2: T | float = CS0_2) -> T:
    r"""$\rho(z,x)$
    $$\rho(z,x) = z^2 \left( \frac{1 - c_s^2}{c_s^2} \right)^2 \frac{(x - x_+)^2(x - x_{-})^2}{x(x_+ + x_{-} - x)}$$
    :giombi_2024_cs:`\ ` eq. B.21
    This is obtained by integrating $y$ (aka. $\tilde{x}$) out using a delta function in the kernel.
    """
    return z**2 * rho_delta_factor(cs2) * rho_delta_frac(x=x, xp=xp, xm=xm)  # pyrefly: ignore[bad-return]


@njit(cache=True)
def rho_delta_factor[T: FloatOrArr](cs2: T = CS0_2) -> T:  # type: ignore[assignment]
    r"""The $c_s^2$ factor in $\rho(z,x)$
    $$\left( \frac{1 - c_s^2}{c_s^2} \right)^2$$
    :giombi_2024_cs:`\ ` eq. B.21.
    """
    # typing.cast() is not used below, since Numba cannot compile it.
    return ((1 - cs2) / cs2)**2  # pyrefly: ignore[bad-return]


@njit(cache=True)
def rho_delta_frac[T: FloatOrArr](x: T, xp: T | float, xm: T | float) -> T:
    r"""The $(x,x_+, x_{-})$ fraction in $\rho(z,x)$
    $$\frac{(x - x_+)^2(x - x_{-})^2}{x(x_+ + x_{-} - x)}$$
    :giombi_2024_cs:`\ ` eq. B.21.
    """
    return (x - xp)**2 * (x - xm)**2 / (x * (xp + xm - x))  # pyrefly: ignore[bad-return]


@njit(cache=True)
def rho_pm[T: FloatOrArr](z: T, x: T | float, cs: T | float = CS0) -> T:
    r"""Geometric function $\rho(z,x,y)$
    Here $y \equiv x_+ + x_{-} - x$
    :giombi_2026:`\ ` p. 25.
    """
    return rho(z=z, x=x, y=x_plus(z=z, cs=cs) + x_minus(z=z, cs=cs) - x)  # pyrefly: ignore[bad-return]


@njit(cache=True)
def x_minus[T: FloatOrArr](z: T, cs: T | float = CS0) -> T:
    r"""$x_-$
    $$x_- = \frac{1 - c_s}{2 c_s} z$$
    :giombi_2024_cs: \ ` p. 13
    :giombi_2026: \ ` p. 25
    This is denoted as $z_-$ on :gw_pt_ssm:`\ ` p. 12.
    """
    return (1 - cs) / (2 * cs) * z  # pyrefly: ignore[bad-return]


@njit(cache=True)
def x_plus[T: FloatOrArr](z: T, cs: T | float = CS0) -> T:
    r"""$x_+$
    $$x_+ = \frac{1 + c_s}{2 c_s} z$$
    :giombi_2024_cs: \ ` p. 13
    :giombi_2026: \ ` p. 25
    This is denoted as $z_+$ on :gw_pt_ssm:`\ ` p. 12.
    """
    return (1 + cs) / (2 * cs) * z  # pyrefly: ignore[bad-return]
