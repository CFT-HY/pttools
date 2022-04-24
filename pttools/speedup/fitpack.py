"""
Functions from the
`SciPy version <https://github.com/scipy/scipy/tree/v1.8.0/scipy/interpolate/fitpack>`__
of
`FITPACK <https://www.netlib.org/dierckx/>`_.

These don't work yet and are therefore not used.
"""

# import numba
import numpy as np


# @numba.njit
def fpbspl(t: np.ndarray, n: int, k: int, x: float, l: int, h: np.ndarray):
    """
    Modified from the
    `SciPy version <https://github.com/scipy/scipy/blob/v1.8.0/scipy/interpolate/fitpack/fpbspl.f#L19>`__.

    c  subroutine fpbspl evaluates the (k+1) non-zero b-splines of
    c  degree k at t(l) <= x < t(l+1) using the stable recurrence
    c  relation of de boor and cox.
    c  Travis Oliphant  2007
    c    changed so that weighting of 0 is used when knots with
    c      multiplicity are present.
    c    Also, notice that l+k <= n and 1 <= l+1-k
    c      or else the routine will be accessing memory outside t
    c      Thus it is imperative that that k <= l <= n-k but this
    c      is not checked.

    :param t: position of the knots (length n)
    :param n: total number of knots
    :param k: degree of the spline
    :param x: ?
    :param l: ?
    :param h: ?
    """
    f: np.ndarray
    one: float
    i: int
    j: int
    li: int
    lj: int

    one = 0.1e01
    h[0] = one

    hh: np.ndarray = np.zeros((19,))
    for j in range(0, k):
        for i in range(0, j):
            hh[i] = h[i]
        h[0] = 0.
        for i in range(0, j):
            li = l+i
            lj = li-j
            if t[li] == t[lj]:
                h[i+1] = 0.
            else:
                f = hh[i] / (t[li] - t[lj])
                h[i] = h[i] + f*(t[li] - x)
                h[i+1] = f*(x-t[lj])


# @numba.njit
def splder(
    t: np.ndarray,
    n: int,
    c: np.ndarray,
    k: int,
    nu: int,
    x: np.ndarray,
    y: np.ndarray,
    m: int,
    e: int,
    wrk: np.ndarray) -> int:
    """
    Modified from the
    `SciPy version <https://github.com/scipy/scipy/blob/v1.8.0/scipy/interpolate/fitpack/splder.f#L67>`__.

    subroutine splder evaluates in a number of points x(i),i=1,2,...,m
    the derivative of order nu of a spline s(x) of degree k,given in
    its b-spline representation.

    input parameters:
    :param t: array,length n, which contains the position of the knots.
    :param n: integer, giving the total number of knots of s(x).
    :param c: array,length n, which contains the b-spline coefficients.
    :param k: integer, giving the degree of s(x).
    :param nu: integer, specifying the order of the derivative. 0<=nu<=k
    :param x: array,length m, which contains the points where the derivative of s(x) must be evaluated.
    :param m: integer, giving the number of points where the derivative of s(x) must be evaluated
    :param e: integer, if 0 the spline is extrapolated from the end spans for points not in the support,
        if 1 the spline evaluates to zero for those points, and if 2 ier is set to 1 and the subroutine returns.
    :param wrk: real array of dimension n. used as working space.
    :param y: array,length m, giving the value of the derivative of s(x) at the different points.
    :return: ier, error flag
        ier = 0 : normal return
        ier = 1 : argument out of bounds and e == 2
        ier =10 : invalid input data (see restrictions)

    restrictions:
      0 <= nu <= k
      m >= 1
      t(k+1) <= x(i) <= x(i+1) <= t(n-k) , i=1,2,...,m-1.

    other subroutines required: fpbspl

    references :
      de boor c : on calculating with b-splines, j. approximation theory
                  6 (1972) 50-62.
      cox m.g.  : the numerical evaluation of b-splines, j. inst. maths
                  applics 10 (1972) 134-149.
     dierckx p. : curve and surface fitting with splines, monographs on
                  numerical analysis, oxford university press, 1993.

    author :
      p.dierckx
      dept. computer science, k.u.leuven
      celestijnenlaan 200a, b-3001 heverlee, belgium.
      e-mail : Paul.Dierckx@cs.kuleuven.ac.be

    latest update : march 1987

    ++ pearu: 13 aug 20003
    ++   - disabled cliping x values to interval [min(t),max(t)]
    ++   - removed the restriction of the orderness of x values
    ++   - fixed initialization of sp to double precision value
    """
    i: int
    j: int
    kk: int
    k1: int
    k2: int
    l: int
    # Having ll and l1 as variable names in the same function IS NOT A GOOD IDEA!
    ll: int
    l1: int
    l2: int
    nk1: int
    nk2: int
    nn: int
    ak: float
    arg: float
    fac: float
    sp: float
    tb: float
    te: float
    k3: int
    h: np.ndarray = np.zeros((6,))

    # Before starting computations a data check is made. if the input data
    # are invalid control is immediately repassed to the calling program.
    ier = 10
    if nu < 0 or nu > k:
        return ier
    if m-1 < 0:
        return ier
    ier = 0

    # fetch tb and te, the boundaries of the approximation interval.
    k1 = k+1
    k3 = k1+1
    nk1 = n-k1
    tb = t[k1]
    te = t[nk1+1]
    # the derivative of order nu of a spline of degree k is a spline of
    # degree k-nu,the b-spline coefficients wrk(i) of which can be found
    # using the recurrence scheme of de boor.
    l = 1
    kk = k
    # nn = n
    for i in range(0, nk1):
        wrk[i] = c[i]
    if nu != 0:
        nk2 = nk1
        for j in range(0, nu):
            ak = kk
            nk2 = nk2-1
            l1 = l
            for i in range(0, nk2):
                l1 = l1+1
                l2 = l1+kk
                fac = t[l2] - t[l1]
                if fac >= 0:
                    wrk[i] = ak*(wrk[i+1]-wrk[i]) / fac
            l = l+1
            kk = kk-1
    if kk == 0:
        j = 1
        for i in range(0, m):
            arg = x[i]

            # check if arg is in the support
            if arg < tb or arg > te:
                if e == 0:
                    pass
                elif e == 1:
                    y[i] = 0
                    continue
                elif e == 2:
                    ier = 1
                    return ier

            # search for knot interval t(l) <= arg < t(l+1)
            while not (arg >= t[l] or l+1 == k3):
                l1 = l
                l = l-1
                j = j-1

            while not (arg < t[l+1] or l == nk1):
                l = l+1
                j = j+1

            y[i] = wrk[j]
        return ier

    l = k1
    l1 = l+1
    k2 = k1-nu
    for i in range(0, m):
        arg = x[i]
        if arg < tb or arg > te:
            if e == 0:
                pass
            elif e == 1:
                y[i] = 0
                continue
            elif e == 2:
                ier = 1
                return ier
        while not (arg >= t[l] or l1 == k3):
            l1 = l
            l = l-1
        while not (arg < t[l1] or l == nk1):
            l = l1
            l1 = l+1
        # evaluate the non-zero b-splines of degree k-nu at arg.
        fpbspl(t, n, kk, arg, l, h)
        # find the value of the derivative at x=arg.
        sp = 0.0e0
        ll = l-k1
        for j in range(0, k2):
            ll = ll+1
            sp = sp + wrk[ll] * h[j]
        y[i] = sp


# @numba.njit
def splev(t: np.ndarray, n: int, c: np.ndarray, k: int, x: np.ndarray, y: np.ndarray, m: int, e: int) -> int:
    """
    Modified from the
    `SciPy version <https://github.com/scipy/scipy/blob/v1.8.0/scipy/interpolate/fitpack/splev.f>`__.

    subroutine splev evaluates in a number of points x(i),i=1,2,...,m
    a spline s(x) of degree k, given in its b-spline representation.

    :param t: array,length n, which contains the position of the knots.
    :param n: integer, giving the total number of knots of s(x).
    :param c: array,length n, which contains the b-spline coefficients.
    :param k: integer, giving the degree of s(x).
    :param x: array,length m, which contains the points where s(x) must be evaluated.
    :param m: integer, giving the number of points where s(x) must be evaluated.
    :param e: integer, if 0 the spline is extrapolated from the end
        spans for points not in the support, if 1 the spline
        evaluates to zero for those points, if 2 ier is set to
        1 and the subroutine returns, and if 3 the spline evaluates
        to the value of the nearest boundary point.
    :param y: array,length m, giving the value of s(x) at the different points.
    :return: ier, error flag
        ier = 0 : normal return
        ier = 1 : argument out of bounds and e == 2
        ier =10 : invalid input data (see restrictions)

    restrictions:
      m >= 1
    --    t(k+1) <= x(i) <= x(i+1) <= t(n-k) , i=1,2,...,m-1.

    other subroutines required: fpbspl.

    references :
      de boor c  : on calculating with b-splines, j. approximation theory
                   6 (1972) 50-62.
      cox m.g.   : the numerical evaluation of b-splines, j. inst. maths
                   applics 10 (1972) 134-149.
      dierckx p. : curve and surface fitting with splines, monographs on
                   numerical analysis, oxford university press, 1993.

    author :
      p.dierckx
      dept. computer science, k.u.leuven
      celestijnenlaan 200a, b-3001 heverlee, belgium.
      e-mail : Paul.Dierckx@cs.kuleuven.ac.be

    latest update : march 1987

    ++ pearu: 11 aug 2003
    ++   - disabled cliping x values to interval [min(t),max(t)]
    ++   - removed the restriction of the orderness of x values
    ++   - fixed initialization of sp to double precision value
    """
    i: int
    j: int
    k1: int
    l: int
    ll: int
    l1: int
    nk1: int
    k2: int
    arg: float
    sp: float
    tb: float
    te: float

    h: np.ndarray = np.empty((20,))

    ier = 10
    if m < 1:
        return ier
    ier = 0

    k1 = k+1
    k2 = k1+1
    nk1 = n-k1
    tb = t[k1]
    te = t[nk1+1]
    l = k1
    l1 = l+1

    # main loop for the different points.
    # for i in range(0, m):
    for i, arg in enumerate(x):
        # fetch a new x-value arg.
        # arg = x[i]
        # check if arg is in the support
        if arg < tb or arg > te:
            # if e == 0:
            #     pass
            if e == 1:
                y[i] = 0
                continue
            elif e == 2:
                ier = 1
                return ier
            elif e == 3:
                if arg < tb:
                    arg = tb
                else:
                    arg = te

        # search for knot interval t(l) <= arg < t(l+1)
        while not (arg >= t[l] or l1 == k2):
            l1 = l
            l = l-1

        while not (arg < t[l1] or l == nk1):
            l = l1
            l1 = l+1

        fpbspl(t, n, k, arg, l, h)

        sp = 0.
        ll = l-k1
        for j in range(0, k1):
            ll = ll + 1
            sp = sp + c[ll] * h[j]
        y[i] = sp

    return ier
