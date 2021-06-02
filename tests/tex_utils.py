import numpy as np

import sigfig as sf


def tex_sf(x,n=2,sci_notn_threshold=2, mult='\\times'):
    expon = np.ceil(np.log10(abs(x))) - 1
    if np.abs(expon) > sci_notn_threshold:
        manti = x*10**(-expon)
        manti_str = sf.round_sig(manti,n)
        if manti_str == "10":
            return ("$ " + "10^{{{0:.0f}}} $" ).format(expon+1)
        else:
            return ("$ " + sf.round_sig(manti,n) + mult + " 10^{{{0:.0f}}} $" ).format(expon)            
    else:
        return "$ " + sf.round_sig(x,n) + " $"

def tex_sf_signed(x,n,sci_notn_threshold=2):
    expon = np.ceil(np.log10(abs(x))) - 1
    if np.abs(expon) > sci_notn_threshold:
        manti = x*10**(-expon)
        manti_str = sf.round_sig(manti,n)
        if manti_str == "10":
            return ("$ " + sf.round_sig_signed(manti,n) + mult + " 10^{{{0:.0f}}} $" ).format(expon)
        else:
            return ("$ " + sf.round_sig_signed(manti,n) + mult + " 10^{{{0:.0f}}} $" ).format(expon)
    else:
        return "$ " + sf.round_sig_signed(x,n) + " $"

