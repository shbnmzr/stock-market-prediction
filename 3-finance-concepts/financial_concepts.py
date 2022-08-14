import numpy as np
from scipy import stats


def compute_daily_return(data):
    data['Daily Return'] = data['Adj Close'].pct_change(1)
    data.dropna(inplace=True)
    return data


def compute_daily_sharpe_ratio(data, risk_free=0):
    daily_return = compute_daily_return(data)['Daily Return']
    mean_return = daily_return.mean()
    std_return = daily_return.std()

    sharpe_ratio = (mean_return - risk_free) / std_return
    return sharpe_ratio


def compute_annual_sharpe_ratio(data, risk_free=0):
    daily_sharpe_ratio = compute_daily_sharpe_ratio(data, risk_free)
    annual_sharpe_ratio = daily_sharpe_ratio * np.sqrt(252)
    return annual_sharpe_ratio


def compute_probabilistic_sr(data, benchmark=0, risk_free=0):
    daily_sharpe_ratio = compute_daily_sharpe_ratio(data, risk_free)
    daily_return = compute_daily_return(data)['Daily Return']
    daily_return = np.array(daily_return)

    skew = stats.skew(daily_return)
    kurtosis = stats.kurtosis(daily_return, fisher=True)
    n = len(data)

    std_sr = np.sqrt((1 / (n - 1)) * (1 + (0.5 * np.square(daily_sharpe_ratio)) - (skew * daily_sharpe_ratio) + (kurtosis / 4) * np.square(daily_sharpe_ratio)))
    ratio = (daily_sharpe_ratio - benchmark) / std_sr
    psr = stats.norm.cdf(ratio)

    return psr


def compute_annual_psr(data, benchmark=0, risk_free=0):
    daily_psr = compute_probabilistic_sr(data, benchmark, risk_free)
    annual_psr = daily_psr * np.sqrt(252)
    return annual_psr


def compute_daily_sortino_ratio(data, threshold=0, risk_free=0):
    daily_return = compute_daily_return(data)['Daily Return']
    mean_return = daily_return.mean()
    downsides = daily_return[daily_return < threshold]
    std_return = downsides.std()

    sartino_ratio = (mean_return - risk_free) / std_return
    return sartino_ratio


def compute_annual_sortino_ratio(data, threshold=0, risk_free=0):
    daily_ratio = compute_daily_sortino_ratio(data, threshold, risk_free)
    annual_ratio = daily_ratio * np.sqrt(252)
    return annual_ratio
