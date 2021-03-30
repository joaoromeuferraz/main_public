from constants.data import FUTURES_MONTH_CODES_CKEY, FUTURES_MONTH_CODES_MKEY
from constants.portfolio import DATES_PATH
import numpy as np
from scipy.stats import norm
from scipy.interpolate import CubicSpline
import pandas as pd
import datetime as dt


def find_maturity(i, codes):
    """

    :param i: which maturity to return (1, 2, 3, etc)
    :param codes: list of maturity codes
    :return: maturity code correspoding to ith maturity
    """

    assert i <= len(codes), f'{i}th maturity does not exist'
    mtys = [dt.datetime(int(c[-2:]), FUTURES_MONTH_CODES_CKEY[c[0]], 1) for c in codes]
    ans = np.sort(mtys)[i - 1]

    return FUTURES_MONTH_CODES_MKEY[ans.month] + str(ans.year)


def black_scholes(S, K, T, r, sig, call=True):
    """

    :param S: spot price
    :param K: strike price
    :param T: time to maturity (years)
    :param r: risk-free rate
    :param sig: volatility
    :param call: if it is a call option
    :return:
    """

    d1 = (np.log(S / K) + ((r + ((sig ** 2) / 2)) * T)) / (sig * np.sqrt(T))
    d2 = d1 - (sig * np.sqrt(T))

    if call:
        p = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    else:
        p = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))

    return p


def exponential_smoothing(series, com):
    lmbda = 1 / (com + 1)
    W = np.arange(len(series))
    W = np.exp(-lmbda * W)
    C = np.sum(W, axis=0)
    W = W / C
    W = np.flip(W)

    mu = np.mean(series)
    vol = np.sqrt(np.sum(W * ((series - mu) ** 2)))

    return mu, vol


def loc_nearest(date, weekday=None, which_weekday=None, return_dt=True):
    """
    Returns closest business date to a given date. If weekday is given, then return the closes business day
    that is closest to the given day and matches the given weekday.

    :param date: reference date
    :param weekday: weekday (either 'M', 'T', 'W', 'R', 'F')
    :param return_dt: returns date in datetime
    :param which_weekday: which weekday of the month (1, 2, 3, or 4)
    :return: date
    """
    mapper = {
        'M': 0, 'T': 1, 'W': 2, 'R': 3, 'F': 4
    }

    date = date if isinstance(date, dt.datetime) else dt.datetime.strptime(date, '%Y-%m-%d')

    if weekday and not which_weekday:
        days_ahead = mapper[weekday] - date.weekday()
        if days_ahead >= -3:
            date = date + dt.timedelta(days_ahead)
        else:
            date = date + dt.timedelta(days_ahead + 7)
    elif weekday and which_weekday:
        days_ahead = mapper[weekday] - date.weekday()
        next_day = date + dt.timedelta(days_ahead + 7) if days_ahead < 0 else date + dt.timedelta(days_ahead)
        date = next_day + dt.timedelta(7 * (which_weekday - 1))

    dates = pd.read_csv(DATES_PATH)
    nearest = dates[dates['date'] >= date.strftime('%Y-%m-%d')]['date'].iloc[0]
    nearest = dt.datetime.strptime(nearest, '%Y-%m-%d') if return_dt else nearest
    return nearest


def du(start_date, end_date):
    dates = pd.read_csv(DATES_PATH)
    dates_sub = dates[(dates['date'] >= start_date) & (dates['date'] <= end_date)]
    return len(dates_sub) - 1


def dc(start_date, end_date):
    assert isinstance(start_date, str) or isinstance(start_date, dt.datetime), 'DATE MUST BE STR OR DATETIME'
    assert isinstance(end_date, str) or isinstance(end_date, dt.datetime), 'DATE MUST BE STR OR DATETIME'

    start_date = dt.datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
    end_date = dt.datetime.strptime(end_date, '%Y-%m-%d') if isinstance(end_date, str) else end_date

    delta = end_date - start_date
    return delta.days


def interpolate_curve(series, method='cubic_spline'):
    """

    :param series: series with curve. index is T and values are rates
    :param method: interpolatino method. default: cubic spline
    :return: interpolated series
    """
    if method == 'cubic_spline':
        cs = CubicSpline(series.index, series.values)
    else:
        raise Exception(f"Unsupported interpolation: {method}")
    return cs

