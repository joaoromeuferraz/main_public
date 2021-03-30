from strategy import StrategyBase
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import numpy as np
from helpers import *
from constants.asset import FUTURES_MULTIPLIERS
import datetime as dt
import json


class Strategy(StrategyBase):
    def __init__(self, top=50, lb=250, vol_target=0.20, restricted_tickers=None, z_max=8, **kwargs):
        super().__init__(**kwargs)
        self.top = top
        self.lb = lb
        self.vol_target = vol_target
        self.restricted_tickers = restricted_tickers
        self.z_max = z_max
        self.current_tickers = None

        self.prices = None
        self.tickers = None

    def initialization(self):
        self.settings.start_date = '2012-01-02'
        self.settings.end_date = '2020-10-16'
        self.settings.initial_capital = 100000
        self.settings.trading_cost = None
        self.settings.auto_roll_futures = True
        self.settings.rebal_frequency = 'M'
        self.settings.fractional = False
        self.settings.standard_lot['EQUITY'] = 1

    def risk_parity_optimization(self):
        returns = self.prices.pct_change().dropna(how='all', axis=0)
        returns = returns.dropna(how='any', axis=1)

        norm_ret = (returns - returns.mean(axis=0))/returns.std(axis=0, ddof=1)
        abs_norm_ret = np.abs(norm_ret)
        mask = (abs_norm_ret < self.z_max).all(axis=0)

        for t in mask.index:
            if self.restricted_tickers and t in self.restricted_tickers:
                mask[t] = False

        returns = returns[returns.columns[mask]]

        self.tickers = np.array(returns.columns)
        Q = returns.cov().values
        vol_target = (self.vol_target / np.sqrt(250))**2
        w_init = 1 / np.sqrt(np.diagonal(Q))
        w_init /= np.sum(w_init)

        def objective(x):
            return -np.sum(np.log(x))

        def constraint(x):
            return vol_target - np.dot(np.dot(x.T, Q), x)

        cons = {'type': 'ineq', 'fun': constraint}
        sol = minimize(fun=objective, x0=w_init, method='SLSQP', constraints=cons)
        w = np.array(sol.x)
        return pd.Series(w, index=self.tickers)

    def on_day_close(self):
        if self.rebalance:
            cur_date = self.portfolio.current_date
            self.tickers = self.data.get_tickers_by_liquidity(cur_date, top=self.top)

            prices = None
            for t in self.tickers:
                price_t = self.data.get_quotes_equity_lb(cur_date, ticker=t, lb=self.lb).set_index('DATE')['CLOSE']
                price_t = pd.Series(price_t, name=t)
                prices = prices.join(price_t, how='left') if prices is not None else pd.DataFrame(price_t)
            self.prices = prices.astype(float)
            w = self.risk_parity_optimization()
            q = self.portfolio.nav * w / prices.iloc[-1][self.tickers].astype(float)
            q = q.round(0).astype(int)

            if self.current_tickers is not None:
                zeros = [t for t in self.current_tickers if t not in self.tickers]
                order = {t: 0 for t in zeros}
                self.portfolio.trade(date=cur_date, order=order, target=True, when='CLOSE')

            order = dict(zip(q.index, q.values))
            print(f"Trade Order: {order}")
            self.portfolio.trade(date=self.portfolio.current_date, order=order, target=True, when='CLOSE')
            self.current_tickers = self.tickers.copy()

    def on_exit(self):
        self.portfolio.plot('nav')


if __name__ == '__main__':
    strat = Strategy(top=50, lb=250, vol_target=0.20, restricted_tickers=None)
    strat.run()
