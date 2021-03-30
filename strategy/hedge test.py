from strategy import StrategyBase
from sklearn.linear_model import LinearRegression
import numpy as np
from helpers import *
from constants.asset import FUTURES_MULTIPLIERS
import datetime as dt
import json


class Strategy(StrategyBase):
    def __init__(self, total_multiplier=5, fractional=True, **kwargs):
        super().__init__(**kwargs)

        self.total_multiplier = total_multiplier
        self.fractional = fractional
        self.lb = 250
        self.prev_ref = None

    def initialization(self):
        self.settings.start_date = '2019-02-01'
        self.settings.end_date = '2020-09-24'
        self.settings.initial_capital = 35000
        self.settings.trading_cost = None
        self.settings.auto_roll_futures = True
        self.settings.rebal_frequency = 'W-MON'
        self.settings.fractional = self.fractional

        if not self.fractional:
            self.settings.standard_lot['EQUITY'] = 1
        else:
            self.settings.standard_lot = None

    def on_day_close(self):
        if self.rebalance:
            ticker = 'IVVB11'
            futures = 'WSP', 'WDO'

            end = dt.datetime.strptime(self.portfolio.current_date, '%Y-%m-%d')
            start = end - dt.timedelta(days=10)
            start = loc_nearest(start).strftime('%Y-%m-%d')
            end = end.strftime('%Y-%m-%d')

            quotes = self.data.get_quotes_equity(ticker=ticker, date=(start, end)).set_index('DATE')
            isp = self.data.get_quotes_futures_by_mty(end, futures[0], 1, du(start, end))
            dol = self.data.get_quotes_futures_by_mty(end, futures[1], 1, du(start, end))

            data = pd.concat((quotes['CLOSE'], isp['CLOSE'], dol['CLOSE']), axis=1)
            data.columns = ['IVVB11', futures[0], futures[1]]
            data = data.dropna(axis=0, how='any')
            data['IVVB11'] = data['IVVB11'].astype(float)
            data[futures[0]] *= FUTURES_MULTIPLIERS[futures[0]]
            data[futures[1]] *= FUTURES_MULTIPLIERS[futures[1]]

            X = data[['IVVB11', futures[1]]].values
            y = data[futures[0]].values
            y = y[:, np.newaxis]

            model = LinearRegression()
            model.fit(X, y)

            print(f"Model coefficients: {model.coef_[0][0]}, {model.coef_[0][1]}")

            q = np.array([1, -model.coef_[0][0], -model.coef_[0][1]]) * self.total_multiplier

            if not self.fractional:
                q = np.array([int(np.ceil(x)) if x > 0 else int(np.floor(x)) for x in q])
            mtys = [self.data.get_mty_code(self.portfolio.current_date, f, 1) for f in futures]

            order = {(futures[0], mtys[0]): q[0], 'IVVB11': q[1], (futures[1], mtys[1]): q[2]}
            print(f"Trade Order: {order}")
            self.portfolio.trade(date=self.portfolio.current_date, order=order, target=True, when='CLOSE')

    def on_exit(self):
        self.portfolio.plot('nav')


if __name__ == '__main__':

    run_all = False

    if run_all:

        multipliers = [1, 2, 3, 4, 5]
        all_navs = pd.DataFrame()
        all_returns = pd.DataFrame()

        for m in multipliers:
            strat = Strategy(total_multiplier=m)
            strat.run()
            all_navs = pd.concat((all_navs, pd.DataFrame(strat.portfolio.nav_series, columns=[str(m)])), axis=1) \
                if not all_navs.empty else pd.DataFrame(strat.portfolio.nav_series, columns=[str(m)])
            all_returns = pd.concat((all_returns, pd.DataFrame(strat.portfolio.return_series, columns=[str(m)])), axis=1) \
                if not all_returns.empty else pd.DataFrame(strat.portfolio.return_series, columns=[str(m)])


