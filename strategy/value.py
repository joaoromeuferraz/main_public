from strategy import StrategyBase
from sklearn.linear_model import LinearRegression
import numpy as np
from helpers import *
from constants.asset import FUTURES_MULTIPLIERS
import datetime as dt
import json


class Strategy(StrategyBase):
    def __init__(self, quantiles, fractional=False, lb=250, long_only=False, **kwargs):
        """

        :param quantiles: quantiles that determine size and value portfolios
        >>> quantiles = {'size': (0.5, 0.5), 'value': (0.3, 0.7)}
        :param fractional:
        :param lb:
        :param long_only:
        :param kwargs:
        """
        super().__init__(**kwargs)

        self.quantiles = quantiles
        self.fractional = fractional
        self.lb = lb
        self.long_only = long_only

    def initialization(self):
        self.settings.start_date = '2016-01-05'
        self.settings.end_date = '2020-12-21'
        self.settings.initial_capital = 100000
        self.settings.trading_cost = None
        self.settings.auto_roll_futures = True
        self.settings.rebal_frequency = 'M'
        self.settings.fractional = self.fractional
        self.settings.price_table = 'quotes_bbg'

        if not self.fractional:
            self.settings.standard_lot['EQUITY'] = 1
        else:
            self.settings.standard_lot = None

    def on_day_close(self):
        if self.rebalance:
            date = self.portfolio.current_date

            tickers = self.data.get_valid_tickers(date, table='quotes_bbg')
            fields = ['TICKER', 'VALUE']
            size = self.data.get_metrics_equity(tickers, 'mkt_cap', date=date, table='metrics', fields=fields)
            value = self.data.get_metrics_equity(tickers, 'pb', date=date, table='metrics', fields=fields)

            size = size.set_index('TICKER', drop=True)
            value = value.set_index('TICKER', drop=True)
            value = 1/value  # converting to b/p

            size_q = tuple(size.quantile(q=self.quantiles['size'])['VALUE'])
            value_q = tuple(value.quantile(q=self.quantiles['value'])['VALUE'])

            small, big = size[size < size_q[0]].dropna().index, size[size >= size_q[1]].dropna().index
            low, high = value[value < value_q[0]].dropna().index, value[value >= value_q[1]].dropna().index

            small_low = list(set(small).intersection(low))
            big_low = list(set(big).intersection(low))

            small_high = list(set(small).intersection(high))
            big_high = list(set(big).intersection(high))

            long = small_high + big_high
            short = small_low + big_low

            long_weights = np.array([size['VALUE'].loc[t] for t in long])
            short_weights = np.array([size['VALUE'].loc[t] for t in short])

            if self.long_only:
                long_weights /= np.sum(long_weights)
            else:
                long_weights /= (2*np.sum(long_weights))
                short_weights /= (-2*np.sum(short_weights))

            prices = self.data.get_quotes_equity(long+short, date=date, table='quotes_bbg', fields=['TICKER', 'CLOSE'])
            prices = prices.set_index('TICKER', drop=True)

            nav = self.portfolio.nav
            long_qty, short_qty = {}, {}

            for i in range(len(long)):
                t = long[i]
                long_qty[t] = nav*long_weights[i]/float(prices.loc[t])
                long_qty[t] = np.round(long_qty[t], 0) if self.fractional else long_qty[t]

            if not self.long_only:
                for i in range(len(short)):
                    t = short[i]
                    short_qty[t] = nav*short_weights[i]/float(prices.loc[t])
                    short_qty[t] = np.round(short_qty[t], 0) if self.fractional else short_qty[t]

            order = {**long_qty, **short_qty} if not self.long_only else long_qty
            print(f"Trade Order: {order}")
            self.portfolio.trade(date=self.portfolio.current_date, order=order, target=True, when='CLOSE')

    def on_exit(self):
        self.portfolio.plot('nav')


if __name__ == '__main__':
    quantiles = {'size': (0.5, 0.5), 'value': (0.35, 0.65)}
    fractional = True
    lb = 250
    strat = Strategy(quantiles=quantiles, fractional=fractional, lb=lb)
    strat.run()
