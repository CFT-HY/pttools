"""This module provides function for working with significant figures.

Modified from
https://bitbucket.org/hindmars/sound-shell-model/
"""

import re
import typing as tp

import numpy as np

EPAT = re.compile(r'^([^e]+)e(.+)$')


def round_sig(x: float, n: int) -> str:
    """round floating point x to n significant figures"""
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    try:
        x = float(x)
    except ValueError:
        raise TypeError("x must be a floating point object")
    form = "%0." + str(n - 1) + "e"
    st = form % x
    num, expo = EPAT.findall(st)[0]
    expo = int(expo)
    #   fs = string.split(num,'.')
    fs = num.split('.')
    if len(fs) < 2:
        fs = [fs[0], ""]
    if expo == 0:
        return num
    elif expo > 0:
        if len(fs[1]) < expo:
            fs[1] += "0" * (expo - len(fs[1]))
        st = fs[0] + fs[1][0:expo]
        if len(fs[1][expo:]) > 0:
            st += '.' + fs[1][expo:]
        return st
    else:
        expo = -expo
        if fs[0][0] == '-':
            fs[0] = fs[0][1:]
            sign = "-"
        else:
            sign = ""
        return sign + "0." + "0" * (expo - 1) + fs[0] + fs[1]


def round_sig_signed(x: float, n: int) -> str:
    """round floating point x to n significant figures"""
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    try:
        x = float(x)
    except ValueError:
        raise TypeError("x must be a floating point object")
    form = "%+0." + str(n - 1) + "e"
    st = form % x
    num, expo = EPAT.findall(st)[0]
    expo = int(expo)
    #   fs = string.split(num,'.')
    fs = num.split('.')
    if len(fs) < 2:
        fs = [fs[0], ""]
    if expo == 0:
        return num
    elif expo > 0:
        if len(fs[1]) < expo:
            fs[1] += "0" * (expo - len(fs[1]))
        st = fs[0] + fs[1][0:expo]
        if len(fs[1][expo:]) > 0:
            st += '.' + fs[1][expo:]
        return st
    else:
        expo = -expo
        if fs[0][0] == '-':
            fs[0] = fs[0][1:]
            sign = "-"
        else:
            sign = ""
        return sign + "0." + "0" * (expo - 1) + fs[0] + fs[1]


def round_sig_error(x: float, ex: float, n: int, paren: bool = False) -> tp.Union[str, tp.Tuple[str, str]]:
    """Find ex rounded to n sig-figs and make the floating point x
   match the number of decimals.  If [paren], the string is
   returned as quantity(error) format"""
    stex = round_sig(ex, n)
    if stex.find('.') < 0:
        extra_zeros = len(stex) - n
        sigfigs = len(str(int(x))) - extra_zeros
        stx = round_sig(x, sigfigs)
    else:
        num_after_dec = len(stex.split('.')[1])
        stx = ("%%.%df" % num_after_dec) % x
    if paren:
        if stex.find('.') >= 0:
            stex = stex[stex.find('.') + 1:]
        return "%s(%s)" % (stx, stex)
    return stx, stex


def format_table(
        cols: tp.List[np.ndarray],
        errors: tp.List[np.ndarray],
        n: int,
        labels: tp.List[str] = None,
        headers: tp.List[str] = None,
        latex: bool = False):
    """Format a table such that the errors have n significant
   figures.  [cols] and [errors] should be a list of 1D arrays
   that correspond to data and errors in columns.  [n] is the number of
   significant figures to keep in the errors.  [labels] is an optional
   column of strings that will be in the first column.  [headers] is
   an optional list of column headers.  If [latex] is true, format
   the table so that it can be included in a LaTeX table """
    if len(cols) != len(errors):
        raise ValueError("Error:  cols and errors must have same length")

    n_cols = len(cols)
    n_rows = len(cols[0])

    if headers is not None:
        if labels is not None:
            if len(headers) == n_cols:
                headers = [""] + headers
            elif len(headers) == n_cols + 1:
                pass
            else:
                raise ValueError("length of headers should be %d" % (n_cols + 1))
        else:
            if len(headers) != n_cols:
                raise ValueError("length of headers should be %d" % n_cols)

    if labels is not None:
        if len(labels) != n_rows:
            raise ValueError("length of labels should be %d" % n_rows)

    str_cols = []
    for col, error in zip(cols, errors):
        str_cols.append([])
        str_cols.append([])
        for i in range(n_rows):
            val, err = round_sig_error(col[i], error[i], n)
            str_cols[-2].append(val)
            str_cols[-1].append(err)

    lengths = [max([len(item) for item in strcol]) for strcol in str_cols]
    fmt = ""
    if labels is not None:
        fmt += "%%%ds " % (max(map(len, labels)))
        if latex:
            fmt += "& "
    for length in lengths:
        fmt += "%%%ds " % length
        if latex:
            fmt += "& "
    if latex:
        fmt = fmt[:-2] + " \\\\"
    output = []
    if headers:
        if labels:
            hs = [headers[0]]
            for head in headers[1:]:
                hs.append(head)
                hs.append('+/-')
        else:
            hs = []
            for head in headers:
                hs.append(head)
                hs.append('+/-')
        output.append(fmt % tuple(hs))
    for i in range(n_rows):
        if labels is not None:
            output.append(fmt % tuple([labels[i]] + [strcol[i] for strcol in str_cols]))
        else:
            output.append(fmt % tuple([strcol[i] for strcol in str_cols]))
    return output


def round_sig_error2(x: float, ex1: float, ex2: float, n: int) -> tp.Tuple[str, str, str]:
    """Find min(ex1,ex2) rounded to n sig-figs and make the floating point x
   and max(ex,ex2) match the number of decimals."""
    min_err = min(ex1, ex2)
    min_stex = round_sig(min_err, n)
    if min_stex.find('.') < 0:
        extra_zeros = len(min_stex) - n
        sigfigs = len(str(int(x))) - extra_zeros
        stx = round_sig(x, sigfigs)
        max_stex = round_sig(max(ex1, ex2), sigfigs)
    else:
        num_after_dec = len(min_stex.split('.')[1])
        stx = ("%%.%df" % num_after_dec) % x
        max_stex = ("%%.%df" % num_after_dec) % (max(ex1, ex2))
    if ex1 < ex2:
        return stx, min_stex, max_stex
    return stx, max_stex, min_stex
