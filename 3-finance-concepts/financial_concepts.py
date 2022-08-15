import numpy as np
from scipy import stats

durations = {
        'daily': 1,
        'weekly': 5,
        'monthly': 21,
        'annual': 252
    }


def compute_daily_return(data):
    data['Daily Return'] = data['Adj Close'].pct_change(1)
    data.dropna(inplace=True)
    return data


def compute_sharpe_ratio(data, risk_free_rate=0, duration='daily'):
    daily_return = compute_daily_return(data)['Daily Return']
    mean_return = daily_return.mean()
    std_return = daily_return.std()

    sharpe_ratio = ((mean_return - risk_free_rate) / std_return) * np.sqrt(durations[duration])
    return sharpe_ratio


def compute_probabilistic_sharpe_ratio(data, benchmark=0, risk_free_rate=0, duration='daily'):
    daily_sharpe_ratio = compute_sharpe_ratio(data, risk_free_rate)
    daily_return = compute_daily_return(data)['Daily Return']
    daily_return = np.array(daily_return)

    skew = stats.skew(daily_return)
    kurtosis = stats.kurtosis(daily_return, fisher=True)
    n = len(data)

    std_sr = np.sqrt((1 / (n - 1)) * (1 + (0.5 * np.square(daily_sharpe_ratio)) - (skew * daily_sharpe_ratio) + (kurtosis / 4) * np.square(daily_sharpe_ratio)))
    ratio = (daily_sharpe_ratio - benchmark) / std_sr
    psr = stats.norm.cdf(ratio) * np.sqrt(durations[duration])

    return psr


def compute_sortino_ratio(data, threshold=0, risk_free_rate=0, duration='daily'):
    daily_return = compute_daily_return(data)['Daily Return']
    mean_return = daily_return.mean()
    downsides = daily_return[daily_return < threshold]
    std_return = downsides.std()

    sartino_ratio = ((mean_return - risk_free_rate) / std_return) * np.sqrt(durations[duration])
    return sartino_ratio
