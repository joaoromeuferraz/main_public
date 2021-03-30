from constants.asset import *
import pandas as pd
from helpers import loc_nearest
import datetime as dt
from constants.data import FUTURES_MONTH_CODES_CKEY


class Asset:
    """
    Class that represents a particular asset

    functions:
    - trade(date, qty, price, fees)

    attributes:
    - trade_history
    - country
    """

    def __init__(self, ticker, currency='brl'):
        self.ticker = ticker
        self.currency = currency
        self.trade_history = pd.DataFrame(columns=TRADE_HISTORY_COLUMNS)
        self.quantity = 0

    def trade(self, date, qty, price, fees):
        row = dict(zip(TRADE_HISTORY_COLUMNS, [date, qty, price, fees]))
        self.trade_history = self.trade_history.append([row], ignore_index=True)
        self.quantity += qty


class Equity(Asset):
    daily_settlement = False
    atype = 'EQUITY'
    settlement = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dividend_history = pd.DataFrame(columns=DIVIDEND_HISTORY_COLUMNS)

    def dividend(self, pay_date, div_type, amount):
        row = dict(zip(DIVIDEND_HISTORY_COLUMNS, [pay_date, div_type, amount]))
        self.dividend_history = self.dividend_history.append([row], ignore_index=True)


class Futures(Asset):
    daily_settlement = True
    settlement = 0
    atype = 'FUTURES'

    def __init__(self, mty, **kwargs):
        super().__init__(**kwargs)
        self.mty = mty
        self.multiplier = FUTURES_MULTIPLIERS[self.ticker]
        self.currency = FUTURES_CURRENCIES[self.ticker]

        mty_year = int('20' + mty[1:3])
        mty_month = FUTURES_MONTH_CODES_CKEY[mty[0]]
        loc_info = FUTURES_MATURITIES[self.ticker]

        self.maturity_date = loc_nearest(date=dt.datetime(mty_year, mty_month, loc_info['day']),
                                         weekday=loc_info['weekday'], which_weekday=loc_info['which_weekday'])


class Option(Asset):
    daily_settlement = False
    atype = 'OPTION'

    def __init__(self, mty, underlying=None, rf=None, op_type='EURO', **kwargs):
        super().__init__(**kwargs)
        self.mty = mty
        self.underlying = underlying or OPTIONS_UNDERLYING[self.ticker]
        self.rf = rf or OPTIONS_REFERENCE[self.ticker]
        self.mty_date = OPTIONS_MATURITIES[self.ticker][self.mty]
        self.op_type = op_type
