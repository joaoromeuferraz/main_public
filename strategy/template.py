from strategy import StrategyBase
from sklearn.linear_model import LinearRegression
import numpy as np
from helpers import *
from constants.asset import FUTURES_MULTIPLIERS
import datetime as dt


class Strategy(StrategyBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.lb = 250
        self.prev_ref = None

    def initialization(self):
        self.settings.start_date = '2016-01-04'
        self.settings.end_date = '2020-09-24'
        self.settings.initial_capital = 1000000
        self.settings.trading_cost = None
        self.settings.auto_roll_futures = True
        self.settings.rebal_frequency = 'W-FRI'
        self.settings.fractional = True
        self.settings.standard_lot = None

    def on_day_close(self):
        if self.rebalance:
            tickers = ['BOVA11', 'IVVB11']
            end = dt.datetime.strptime(self.portfolio.current_date, '%Y-%m-%d')
            start = dt.datetime(end.year - 1, month=end.month, day=end.day)
            start = loc_nearest(start).strftime('%Y-%m-%d')
            end = end.strftime('%Y-%m-%d')

            quotes = self.data.get_quotes_equity(ticker=['BOVA11', 'IVVB11'], date=(start, end))
            dol = self.data.get_reference_rate((start, end), 'USD') * 1000

            ivvb = quotes[quotes['TICKER'] == 'IVVB11'][['DATE', 'CLOSE']].set_index('DATE').astype(float)
            bova = quotes[quotes['TICKER'] == 'BOVA11'][['DATE', 'CLOSE']].set_index('DATE').astype(float)

            data = pd.concat((ivvb, dol, bova), axis=1)
            data.columns = pd.Index(['IVVB', 'DOL', 'BOVA'])
            data = data.dropna(axis=0, how='any')

            X = data[['IVVB', 'DOL']].values
            y = data['BOVA'].values[:, np.newaxis]

            model = LinearRegression()
            model.fit(X, y)
            residuals = model.predict(X) - y

            print(f'Current residual: {residuals[-1, 0]}')
            # ref_rate = self.data.get_reference_rate(self.portfolio.current_date, 'USD')

            # if ref_rate == 0:
            #    ref_rate = self.prev_ref

            signal = np.sign(residuals[-1, 0])

            q = np.array([signal, -signal * model.coef_[0, 0], -signal * model.coef_[0, 1]])
            mv = q * np.array([y[-1, 0], X[-1, 0], X[-1, 1]])
            w = mv/np.sum(mv)
            mv = w * self.portfolio.nav * 0.50
            q = mv / np.array([y[-1, 0], X[-1, 0], X[-1, 1]])
            q = np.round(q, 0).astype(int)

            # wsp_qty = -ind_qty * model.coef_[0] / FUTURES_MULTIPLIERS['WSP'] / ref_rate

            # ind_mty = self.data.get_mty_code(self.portfolio.current_date, 'IND', 1)
            # wsp_mty = self.data.get_mty_code(self.portfolio.current_date, 'WSP', 1)
            wdo_mty = self.data.get_mty_code(self.portfolio.current_date, 'WDO', 1)

            # order = {('IND', ind_mty): ind_qty, ('WSP', wsp_mty): wsp_qty, ('WDO', wdo_mty): wdo_qty}
            order = {'BOVA11': q[0], 'IVVB11': q[1], ('WDO', wdo_mty): q[2] / FUTURES_MULTIPLIERS['WDO']}
            self.portfolio.trade(date=self.portfolio.current_date, order=order, target=True, when='CLOSE')

            # self.prev_ref = ref_rate

    def on_exit(self):
        self.portfolio.plot('nav')


if __name__ == '__main__':
    strat = Strategy()
    strat.run()
