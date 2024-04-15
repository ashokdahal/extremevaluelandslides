import numpy as np

# import tensorflow_probability as tfp


def eGPD_ppf(p, sig, k=0.8118067, xi=0.4919825):
    """
    Calculate the quantile for a Extended Generalized Pareto Distribution.

    Parameters:
        p (float): Probability level (0 < p < 1).
        xi (float): Shape parameter.
        sig (float): Scale parameter.
        k (float): Parameter k.

    Returns:
        float: Quantile value.
    """
    return (sig / xi) * (((1 - (p ** (1 / k))) ** (-xi)) - 1)


def eGPD_cdf(y, k, sig, xi):
    """
    Calculate the CDF for a Extended Generalized Pareto Distribution.

    Parameters:
        xi (float): Shape parameter.
        sig (float): Scale parameter.
        k (float): Parameter k.
        y (float): Exceedence Level of the value

    Returns:
        p (float): Probability level (0 < p < 1).
    """
    return (1 - (1 + xi * y / sig) ** (-1 / xi)) ** k


def getquantiles(sig, x, k=0.8118067, xi=0.4919825, area=1):
    """
    Calculate the quantiles for a Extended Generalized Pareto Distribution compared with exponential distribution.

    Parameters:
        p (float): Probability level (0 < p < 1).
        xi (float): Shape parameter.
        sig (float): Scale parameter.
        k (float): Parameter k.

    Returns:
        float: Quantile value.
    """
    xi = xi
    sigma = (sig + 0.2) * area
    kappa = k
    if xi <= 0:
        return 1e10

    dat = x[x > 0]

    exp_dat = eGPD_cdf(dat, k=kappa, sig=sigma[x > 0], xi=xi)
    exp_dat = expon.ppf(exp_dat)

    p_min = 0
    n_p = len(exp_dat) * (1 - p_min)
    ps = p_min + np.arange(1, int(n_p) + 1) / (n_p + 1) * (1 - p_min)
    return np.quantile(exp_dat, ps), expon.ppf(ps)


def eGPD_areadensity(sig, p, k=0.8118067, xi=0.4919825):
    """
    Calculate the CDF for a Extended Generalized Pareto Distribution.

    Parameters:
        p (float): Probability level (0 < p < 1).
        xi (float): Shape parameter.
        sig (float): Scale parameter.
        k (float): Parameter k.


    Returns:
        q (float): Exceedence Level of the value
    """
    # return (sig / xi) * (((1-(1-((p)**(-1/k))))**(-xi))-1)
    return (sig / xi) * (((1 - p ** (1 / k)) ** (-xi)) - 1)


def inferenceLH(model, model_weights, xdata, rp):
    model.load_weights(model_weights)
    trained_model = model
    ypred = trained_model.predict(xdata, batch_size=64)

    prob_occ = ypred[0]

    sigma = ypred[1] + 0.2
    p = rp * 1  # prob_occ
    intensity = eGPD_areadensity(sig=sigma, p=p)

    frequency = 1 / rp

    landslide_hazard = prob_occ * intensity  # * frequency

    return landslide_hazard


def eGPD_exceedenceprob(sig, y, k=0.8118067, xi=0.4919825):
    """
    Calculate the quantile for a Extended Generalized Pareto Distribution.

    Parameters:
        y (float): exceedence threshold.
        xi (float): Shape parameter.
        sig (float): Scale parameter.
        k (float): Parameter k.

    Returns:
        float: Quantile value.
    """
    # return (sig / xi) * (((1-(1-((p)**(-1/k))))**(-xi))-1)
    return 1 - ((1 - (1 + ((xi * y) / sig)) ** (-1 / xi)) ** k)


def inferenceLHProb(model, model_weights, xdata, ep):
    model.load_weights(model_weights)
    trained_model = model
    ypred = trained_model.predict(xdata, batch_size=64)

    prob_occ = ypred[0]

    sigma = ypred[1] + 0.2
    intensity = eGPD_exceedenceprob(sig=sigma, y=ep)

    # frequency = 1/rp

    landslide_hazard = prob_occ * intensity  # * frequency

    return landslide_hazard
