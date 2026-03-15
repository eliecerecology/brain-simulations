from numpy.polynomial.polynomial import polyfit, polyval
import numpy as np

def FA_metric(phasor, scales):
    """ y: cummulative sum, and scales should be defined"""
    y = calc_detrened(phasor)
    F_fa = np.zeros(len(scales))
    
    for i, s in enumerate(scales):

        diffs = y[s:] - y[:-s]
        F_fa[i] = np.sqrt(np.mean(diffs**2))

    # Fit log–log slope
    coeff_fa = np.polyfit(np.log2(scales), np.log2(F_fa), 1)
    alpha_fa = coeff_fa[0]
    fit_fa = 2 ** np.polyval(coeff_fa, np.log2(scales))

    # plt.figure(figsize=(7, 5))
    # plt.loglog(scales, F_fa, 'o', label="FA Data")
    # plt.loglog(scales, fit_fa, 'r--', label=f"Fit: α = {alpha_fa:.2f}")
    # plt.title("Fluctuation Analysis (FA)")
    # plt.xlabel("log₂(Window size s)")
    # plt.ylabel("log₂(Fluctuation F(s))")
    # plt.legend()
    # plt.grid(True, which="both", ls="--")
    # plt.show()

    print(f"Estimated FA exponent α = {alpha_fa:.3f}")

    return fit_fa, alpha_fa

def calc_detrened(data):
    x = np.abs(data)
    y = np.cumsum(x - np.mean(x))

    return y

def dfa_rms(y, scale):
    n_windows = len(y) // scale

    if n_windows == 0:
        return np.nan  # scale too large
        
    shape = (n_windows, scale)
    Y = np.lib.stride_tricks.as_strided(y, shape=shape)
    rms = np.zeros(n_windows)
    scale_axis = np.arange(scale)

    for i, window in enumerate(Y):
        coeff = np.polyfit(scale_axis, window, 1)
        trend = np.polyval(coeff, scale_axis)
        rms[i] = np.sqrt(np.mean((window - trend) ** 2))
    return np.mean(rms)


def dfa_scales(min_exp=5, max_exp=9, step=0.25):
    """
    Logarithmic scales: 2^5 ... 2^9 in reasonable increments.
    Ensures scales are strictly increasing and unique.
    """
    scales = np.round(2 ** np.arange(min_exp, max_exp, step)).astype(int)
    scales = np.unique(scales)
    return scales

def DFA(data):
    """
    Full DFA1 for a complex phasor time series.
    Returns (alpha, scales, F)
    """
    y = calc_detrened(data)
    scales = dfa_scales()

    F = []
    for s in scales:
        rms_val = dfa_rms(y, s)
        if not np.isnan(rms_val):
            F.append(rms_val)
        else:
            F.append(np.nan)

    F = np.array(F)

    # Remove invalid scales
    mask = ~np.isnan(F)
    scales = scales[mask]
    F = F[mask]

    coeff = np.polyfit(np.log2(scales), np.log2(F), 1)
    alpha = coeff[0]

    return alpha, scales, F

def plv_matrix_vectorized(inst_theta):
    X = np.exp(1j * inst_theta)          # (T, N)
    M = np.dot(X.conj().T, X) / X.shape[0]
    return np.abs(M)