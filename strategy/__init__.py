from portfolio import *
from settings.strategy import *
from data import *
from functools import wraps
import datetime as dt


class StrategyBase:
    def __init__(self, settings=None):
        self.settings = settings or Settings
        self.data = Data()
        self.portfolio = None

        self.rebalance = None
        self.rebal_dates = None

    def _pre_run(self):
        self.initialization()
        self.portfolio = Portfolio(start_date=self.settings.start_date, settings=self.settings)

        self.rebal_dates = self.portfolio.dates.set_index('date')
        self.rebal_dates = self.rebal_dates.loc[self.settings.start_date:self.settings.end_date]
        self.rebal_dates.index = pd.to_datetime(self.rebal_dates.index)
        self.rebal_dates = self.rebal_dates.resample(self.settings.rebal_frequency).last().index
        self.rebal_dates = np.array([loc_nearest(d) for d in self.rebal_dates])

    def _post_run(self):
        self.on_exit()

    def _run_wrapper(func):
        @wraps(func)
        def wrapper(self):
            self._pre_run()
            func(self)
            self._post_run()
        return wrapper

    def initialization(self):
        pass

    def on_day_close(self):
        pass

    def on_exit(self):
        pass

    @_run_wrapper
    def run(self):
        while self.portfolio.current_date <= self.settings.end_date:
            self.rebalance = dt.datetime.strptime(self.portfolio.current_date, '%Y-%m-%d') in self.rebal_dates
            self.data.max_date = self.portfolio.current_date
            self.on_day_close()
            self.portfolio.next_day()
