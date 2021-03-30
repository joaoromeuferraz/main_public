from data import *
from constants.portfolio import *
from asset import *
from settings.portfolio import Settings
from exceptions import *
import numpy as np
from matplotlib import pyplot as plt
import datetime as dt


class Portfolio:
    """

    Cash-flow based porfolio
    """

    def __init__(self, start_date, dates=None, settings=None):
        self.start_date = start_date
        dates = dates or pd.read_csv(DATES_PATH)
        dates = dates[dates['date'] >= start_date]
        self.dates = dates.reset_index(drop=True)
        self.current_date = self.start_date
        self.current_date_idx = 0

        self.settings = settings or Settings
        self.data = Data()

        self.historical_holdings = pd.DataFrame(columns=HISTORICAL_HOLDINGS_COLUMNS)
        self.trade_log = pd.DataFrame(columns=TRADE_LOG_COLUMNS)
        self.active_holdings = {}
        self.notionals = {}

        self.cash = 0.0
        self.asset_value = 0.0
        self.nav = 0.0

        self.deposits = pd.Series(dtype=float)
        self.cash_series = pd.Series(dtype=float)
        self.nav_series = pd.Series(dtype=float)
        self.pnl_series = pd.Series(dtype=float)
        self.return_series = pd.Series(dtype=float)

        if self.settings.initial_capital:
            self.add_deposit(self.settings.initial_capital)

    def _handle_qty(self, prices, order):
        order_f = {**{(k, None): v for k, v in order.items() if isinstance(k, str)},
                   **{k: v for k, v in order.items() if isinstance(k, tuple)}}
        index = pd.MultiIndex.from_tuples(list(order_f.keys()))
        rows = [{'QUANTITY': v} for _, v in order_f.items()]
        qty = pd.concat((prices, pd.DataFrame(rows, index=index)), axis=1)

        # TO-DO: Assert sufficient liquidity (considering futures in other currencies)
        # assert (qty[when] * qty['QUANTITY']).sum() <= self.nav, 'ERROR - TRADE QUANTITY EXCEEDS NAV'
        qty = qty['QUANTITY']

        if not self.settings.fractional:
            qty = np.floor(qty).astype(int)

        if self.settings.standard_lot:
            for i in range(len(qty.index)):
                atype = self.data.parse_asset(qty.index[i])
                assert qty.iloc[i] % self.settings.standard_lot[atype] == 0, \
                    'ERROR - ASSET QTY DOES NOT CONFORM TO STD LOT'

        return qty

    def _get_asset(self, ticker):
        if ticker in self.active_holdings.keys():
            return self.active_holdings[ticker]
        else:
            atype = self.data.parse_asset(ticker[0]) if pd.isna(ticker[1]) else self.data.parse_asset(ticker)
            if atype == 'EQUITY':
                return Equity(ticker=ticker)
            elif atype == 'FUTURES':
                return Futures(ticker=ticker[0], mty=ticker[1])
            elif atype == 'OPTION':
                return Option(ticker=ticker[0], mty=ticker[1])
            else:
                raise ValueError(f'{atype} not supported')

    def _get_prices(self, tickers, date, when):
        base = date + "_"
        ids = [base + t[0] + t[1] if isinstance(t, tuple) else base + t for t in tickers]
        fields = ['TICKER', 'MTY', when]
        res = self.data.get_quotes_by_id(ids, fields=fields, table=self.settings.price_table)
        rows = [{when: d[when]} for d in res]
        index = [(d['TICKER'], d['MTY']) if d['MTY'] is not None else (d['TICKER'], None) for d in res]
        index = pd.MultiIndex.from_tuples(index) if index else None
        df = pd.DataFrame(rows, index=index) if index is not None else None
        return df

    def _update_nav(self):
        self.asset_value = 0
        tickers = [k[0] if pd.isna(k[1]) else k for k, _ in self.active_holdings.items()]
        prices = self._get_prices(tickers, self.current_date, 'CLOSE')

        for t, asset in self.active_holdings.items():
            try:
                price = float(prices['CLOSE'][t])
            except (KeyError, TypeError):
                if asset.atype == 'OPTION':
                    if asset.mty_date >= self.current_date:
                        price = self.data.get_black_scholes_price(self.current_date, t[0], t[1])
                        price = np.round(price, 2)
                    else:
                        price = 0.0
                else:
                    price = 0.0
                    # raise Exception(f'ERROR - UNABLE TO FIND PRICE FOR ASSET {asset.ticker}')
            tick_name = t[0] if pd.isna(t[1]) else t[0] + t[1]
            if asset.quantity != 0:
                print(f"Closing price of {tick_name}: {price}")
            if asset.daily_settlement:
                notional = self.notionals.get(t)
                if notional is not None and not notional == 0:
                    self.notionals[t] = asset.quantity * price
                    net_multiplier = asset.multiplier
                    if not asset.currency == 'BRL':
                        net_multiplier *= self.data.get_reference_rate(self.current_date, asset.currency)
                    net_adjustment = np.round((self.notionals[t] - notional) * net_multiplier, 2)
                    print(f"Adjustment of {t[0] + t[1]}: {net_adjustment}")
                    self.cash += net_adjustment
            else:
                cur_mtm = np.round(asset.quantity * price, 2)
                self.asset_value += cur_mtm

        self.nav = np.round(self.asset_value + self.cash, 2)
        rows = [{'Date': self.current_date, 'Asset': k, 'Quantity': v.quantity}
                for k, v in self.active_holdings.items() if not v.quantity == 0]
        self.historical_holdings = pd.concat((self.historical_holdings, pd.DataFrame(rows)), ignore_index=True, axis=0)

    def _update_series(self):
        self.cash_series = self.cash_series.append(pd.Series(self.cash, index=[self.current_date]))
        self.nav_series = self.nav_series.append(pd.Series(self.nav, index=[self.current_date]))

        prev_date = self.dates['date'].iloc[self.current_date_idx - 1] if not self.current_date_idx == 0 else None

        prev_nav = self.nav_series.loc[prev_date] if prev_date else 0.0
        prev_deposits = self.deposits.loc[:prev_date].sum() if prev_date else 0.0

        cur_nav = self.nav_series.loc[self.current_date]
        cur_deposits = self.deposits.loc[:self.current_date].sum()

        prev_pnl = prev_nav - prev_deposits
        cur_pnl = cur_nav - cur_deposits

        dly_pnl = np.round(cur_pnl - prev_pnl, 2)
        dly_ret = np.round((cur_nav/(prev_nav + cur_deposits - prev_deposits) - 1), 4)

        self.pnl_series = self.pnl_series.append(pd.Series(dly_pnl, index=[self.current_date]))
        self.return_series = self.return_series.append(pd.Series(dly_ret, index=[self.current_date]))

    def _handle_settlements(self):
        cur_date = dt.datetime.strptime(self.current_date, '%Y-%m-%d')
        for t, asset in list(self.active_holdings.items()):
            qty = asset.quantity
            if asset.atype == 'FUTURES' and cur_date == asset.maturity_date and qty != 0:
                self.trade(self.current_date, order={t: 0}, target=True, when='CLOSE')
                if self.settings.auto_roll_futures:
                    next_mty = self.data.get_quotes_futures_by_mty(self.current_date, t[0], 2)['MTY'].iloc[0]
                    self.trade(self.current_date, order={(t[0], next_mty): qty}, target=True, when='CLOSE')

    def trade(self, date, order, prices=None, trade_cost=None, target=True, when='CLOSE'):
        """

        :param date: trade date
        :param order: dictionary with keys as tickers and values as order quantities
            - if asset is a futures/option contract, use a tuple as a key specifying ticker and maturity
            >>> order = {'PETR4': 200, ('WDO', 'H20'): -1}
        :param prices: trade prices dictionary (if None, then closing prices of current date)
            >>> prices = {'PETR4': 14.38, ('WDO', 'H20'): 5332.0}
        :param trade_cost: fees associated with each trade (dictionary or float)
        :param target: set weights/quantities as target portfolio weights/quantities
        :param when: when trade is made ('OPEN', 'CLOSE', 'AVG', etc...)
        :return: trade id
        """

        if prices is not None:
            when = 'INTRA'

        trade_cost = trade_cost or self.settings.trading_cost
        tickers = list(order.keys())
        if prices:
            rows = [{when: v} for _, v in prices.items()]
            index = [(k, None) if isinstance(k, str) else k for k, _ in prices.items()]
            index = pd.MultiIndex.from_tuples(index)
            prices = pd.DataFrame(rows, index=index)
        else:
            prices = self._get_prices(tickers, date, when)

        qty = self._handle_qty(prices, order)

        for t in qty.index:
            asset = self._get_asset(t)
            net_qty = qty[t] - asset.quantity if target else qty[t]
            if trade_cost is None:
                fees = 0
            else:
                if isinstance(trade_cost, float):
                    fees = trade_cost
                else:
                    fees = trade_cost[t[0]] if pd.isna(t[1]) else trade_cost[t]

            asset.trade(date=date, qty=net_qty, price=float(prices[when][t]), fees=fees)

            net_value = net_qty * float(prices[when][t])
            if asset.daily_settlement:
                if self.notionals.get(t):
                    self.notionals[t] += net_value
                else:
                    self.notionals[t] = net_value
            else:
                self.asset_value += net_value
                self.cash += -net_value

            self.cash -= fees

            trade_info = dict(zip(TRADE_LOG_COLUMNS, [date, t[0], t[1], net_qty, prices[when][t], fees]))
            self.trade_log = pd.concat((self.trade_log, pd.DataFrame([trade_info])), axis=0, ignore_index=True)

            self.active_holdings[t] = asset

            print(f"Trading {net_qty} units of {t[0]} at {prices[when][t]}")

        print("")

    def next_day(self):
        print(f"Current date: {self.current_date}")
        self._handle_settlements()
        self._update_nav()

        print(f"Current NAV: {self.nav}")
        print("")

        self._update_series()

        self.current_date_idx += 1
        self.current_date = self.dates['date'][self.current_date_idx]

    def add_deposit(self, amount):
        self.deposits = self.deposits.append(pd.Series(amount, index=[self.current_date]))
        self.cash += amount
        self.nav += amount

    def add_dividend(self, ticker, div_type, amount):
        asset = self._get_asset((ticker, np.nan))
        asset.dividend(self.current_date, div_type, amount)

        if div_type == 'cash':
            self.cash += amount
        else:
            raise ValueError(f'Only cash dividends are supported')

        self.active_holdings[(ticker, np.nan)] = asset

    def plot(self, series, start_date=None, end_date=None, twinx=False, fig=None, ax=None):
        """

        :param series: list of series (str) to plot
            - either cash, nav, pnl, or returns
        :param start_date: plot start date
        :param end_date: plot end date
        :param twinx: separate axes
        :param fig: figure
        :param ax: axes
        :return: plot
        """

        if isinstance(series, str):
            series = [series]

        if twinx:
            assert len(series) <= 2, 'For separate axes, plot at most two series'

        mapper = {
            'cash': self.cash_series,
            'nav': self.nav_series,
            'pnl': self.pnl_series,
            'return': self.return_series
        }

        start = start_date if start_date else self.nav_series.index[0]
        end = end_date if end_date else self.nav_series.index[-1]

        dates = self.nav_series.loc[start:end].index
        dates = [dt.datetime.strptime(d, '%Y-%m-%d') for d in dates]
        if not fig:
            fig, ax = plt.subplots()
        num_ax = 0

        for name in series:
            try:
                s = mapper[name].loc[start:end]
                if name == 'pnl':
                    s = s.cumsum()
                elif name == 'return' or name == 'returns':
                    s = (1 + s).cumprod() - 1
            except KeyError as e:
                print(f"Unrecognized series name {e}")
                print(f"Supported series: {list(mapper.keys())}")
                return None

            if num_ax == 0:
                ax.plot(dates, s.values, label=name)
            else:
                if twinx:
                    twin = ax.twinx()
                    twin.plot(dates, s.values, label=name, color='orange')
                else:
                    ax.plot(dates, s.values, label=name)
            num_ax += 1

        ax.legend(loc='upper left')
        if twinx:
            twin.legend(loc='upper right')
        fig.autofmt_xdate()
        plt.show()

    def summary(self, start_date=None, end_date=None):
        start = start_date if start_date else self.nav_series.index[0]
        end = end_date if end_date else self.nav_series.index[-1]

        df = pd.concat((self.cash_series.loc[start:end], self.nav_series.loc[start:end],
                        self.pnl_series[start:end], self.return_series[start:end]),
                       keys=['cash', 'nav', 'pnl', 'returns'], axis=1)

        df['pnl'] = df['pnl'].cumsum()
        df['returns'] = (1 + df['returns']).cumprod() - 1

        dates = [dt.datetime.strptime(d, '%Y-%m-%d') for d in df.index]
        df.index = dates

        return df

    def run_until(self, date):
        while self.current_date <= date:
            self.next_day()


def generate_portfolio(start_date, end_date, deposits, dividends, trades):
    port_settings = Settings
    port_settings.initial_capital = 0.0

    portfolio = Portfolio(start_date=start_date, settings=port_settings)

    while portfolio.current_date <= end_date:
        dly_deposits = deposits[deposits['DATE'] == portfolio.current_date]
        dly_dividends = dividends[dividends['DATE'] == portfolio.current_date]
        dly_trades = trades[trades['DATE'] == portfolio.current_date]

        if not dly_deposits.empty:
            for i in dly_deposits.index:
                deposit_info = {'amount': dly_deposits['AMOUNT'][i]}
                portfolio.add_deposit(**deposit_info)

        if not dly_dividends.empty:
            for i in dly_dividends.index:
                dividend_info = {'ticker': dly_dividends['TICKER'][i], 'div_type': 'cash',
                                 'amount': dly_dividends['AMOUNT'][i]}
                portfolio.add_dividend(**dividend_info)

        if not dly_trades.empty:
            for i in dly_trades.index:
                if pd.isna(dly_trades['MTY'][i]):
                    ticker = dly_trades['TICKER'][i]
                else:
                    ticker = (dly_trades['TICKER'][i], dly_trades['MTY'][i])

                order = {ticker: float(dly_trades['QTY'][i])}
                prices = {ticker: float(dly_trades['PRICE'][i])}
                trade_cost = {ticker: float(dly_trades['FEES'][i])}
                target = False
                when = 'INTRA'

                trade_info = {'date': portfolio.current_date, 'order': order, 'prices': prices,
                              'trade_cost': trade_cost,
                              'target': target, 'when': when}

                portfolio.trade(**trade_info)

        portfolio.next_day()

    return portfolio


if __name__ == '__main__':
    pass
    # from personal_portfolio import *
    # start, end = '2018-06-14', '2020-11-13'
    # portfolio = generate_portfolio(start, end, get_deposits(), get_dividends(), get_trades())
