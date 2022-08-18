import numpy as np
from scipy import stats

durations = {
        'daily': 1,
        'weekly': 5,
        'monthly': 21,
        'annual': 252
}


def compute_moving_average(data, duration='monthly'):
    moving_average = data.rolling(durations[duration]).mean()
    return moving_average


def compute_daily_return(data):
    daily_returns = data.pct_change(1)
    daily_returns.dropna(axis=0, inplace=True)
    return daily_returns


def compute_change(data):
    adj_close = data['Adj Close'].copy()
    change = (adj_close[-1] - adj_close[0]) / adj_close[0]
    return change


def compute_percent_change(data):
    change = compute_change(data)
    percent_change = change * 100
    return percent_change


def compute_sharpe_ratio(data, risk_free_rate=0, duration='daily'):
    daily_return = compute_daily_return(data['Adj Close'])
    mean_return = daily_return.mean()
    std_return = daily_return.std()

    sharpe_ratio = ((mean_return - risk_free_rate) / std_return) * np.sqrt(durations[duration])
    return sharpe_ratio


def compute_probabilistic_sharpe_ratio(data, benchmark=0, risk_free_rate=0, duration='daily'):
    daily_sharpe_ratio = compute_sharpe_ratio(data, risk_free_rate)
    daily_return = compute_daily_return(data['Adj Close'])
    daily_return = np.array(daily_return)

    skew = stats.skew(daily_return)
    kurtosis = stats.kurtosis(daily_return, fisher=True)
    n = len(data)

    std_sr = np.sqrt((1 / (n - 1)) * (1 + (0.5 * np.square(daily_sharpe_ratio)) - (skew * daily_sharpe_ratio) + (kurtosis / 4) * np.square(daily_sharpe_ratio)))
    ratio = (daily_sharpe_ratio - benchmark) / std_sr
    psr = stats.norm.cdf(ratio) * np.sqrt(durations[duration])

    return psr


def compute_sortino_ratio(data, threshold=0, risk_free_rate=0, duration='daily'):
    daily_return = compute_daily_return(data['Adj Close'])
    mean_return = daily_return.mean()
    downsides = daily_return[daily_return < threshold]
    std_return = downsides.std()

    sartino_ratio = ((mean_return - risk_free_rate) / std_return) * np.sqrt(durations[duration])
    return sartino_ratio


def compute_cumulative_return(data):
    cum_return = (1 + data).cumprod() - 1
    return cum_return


def compute_cumulative_percent_return(data):
    cum_return = compute_cumulative_return(data)
    cum_perc_return = cum_return * 100
    return cum_perc_return
