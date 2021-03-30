from constants.data import *
from constants.asset import *
from data.db_api import *
from helpers import *
import datetime as dt
from matplotlib import pyplot as plt
from matplotlib import animation


class Data:
    """

    Retrieves data from database
    """

    def __init__(self, max_date=None, connect_info=None):
        connect_info = connect_info or DEFAULT_CONNECT_INFO
        self.conn = get_new_connection(**connect_info)
        self.max_date = max_date
        self.dates = pd.read_csv(DATES_PATH)
        self.api = DbAPI(connect_info=connect_info)

    def parse_asset(self, ticker):
        if isinstance(ticker, tuple):
            key = ticker[0] + ticker[1] if not pd.isna(ticker[1]) else ticker[0]
        else:
            key = ticker

        fields = ['TYPE']
        tmp_and = {'=': {'ID': key}}
        data = self._select(DEFAULT_TABLES['assets'], tmp_and=tmp_and, fields=fields)
        assert len(data) > 0, 'NON-EXISTENT TICKER'

        return data[0]['TYPE']

    def get_black_scholes_price(self, date, ticker, mty, r=None, vol=None, lb=250, com=60):
        """

        :param date: reference date
        :param ticker: option ticker
        :param mty: mty code of option
            >>> ticker, mty = 'BOVA', 'Q70'
        :param r: risk-free rate
        :param vol: return volatility
        :param lb: lookback (for estimates)
        :param com: center of mass used for estimates
        :return:
        """

        underlying = OPTIONS_UNDERLYING[ticker]
        mty_date = OPTIONS_MATURITIES[ticker][mty]
        strike = OPTIONS_STRIKE[ticker][mty]
        call = OPTIONS_STRIKE[ticker][mty] == 'CALL'

        tmp_and = {'=': {'TICKER': underlying}, '<=': {'DATE': date}}
        fields = ['DATE', 'CLOSE']

        data = self._select(DEFAULT_TABLES['quotes'], tmp_and=tmp_and, fields=fields)
        data = pd.DataFrame(data)
        data = data[data['DATE'] <= date].reset_index(drop=True)
        data = data.iloc[-lb:].reset_index(drop=True)
        data['CLOSE'] = data['CLOSE'].astype(float)
        S = float(data['CLOSE'].iloc[-1])

        if not vol:
            series = data['CLOSE'].pct_change().dropna().values
            ewma, ewstd = exponential_smoothing(series, com)
            vol = ewstd * np.sqrt(250)

        if not r:
            r = self.get_di(date, 1)

        T = du(date, mty_date) / 250

        price = black_scholes(S, strike, T, r, vol, call=call)

        return price

    def get_cc(self, date, maturity=1):
        """

        :param date: reference date
        :param maturity: CC maturity
        :return: CC rate
        """

        assert isinstance(date, str), 'DATE MUST BE A STRING'
        if self.max_date:
            assert date <= self.max_date, 'DATE MUST BE BEFORE CURRENT DATE'

        tmp_and = {'=': {'DATE': date, 'TICKER': 'DDI'}}
        fields = ['CLOSE', 'MTY']
        data = self._select(DEFAULT_TABLES['quotes'], tmp_and=tmp_and, fields=fields)
        data = pd.DataFrame(data)
        mtys = data['MTY'].values
        mty = find_maturity(maturity, mtys)
        ref_price = float(data[data['MTY'] == mty]['CLOSE'].iloc[0])
        loc_info = FUTURES_MATURITIES['DDI'].copy()
        loc_info['date'] = dt.datetime(int('20' + mty[1:3]), FUTURES_MONTH_CODES_CKEY[mty[0]], loc_info.pop('day'))
        mty_date = loc_nearest(**loc_info)
        mty_date = mty_date.strftime('%Y-%m-%d')
        t = dc(date, mty_date) / 360

        if t == 0:
            mty = find_maturity(maturity + 1, mtys)
            ref_price = float(data[data['MTY'] == mty]['CLOSE'].iloc[0])
            loc_info = FUTURES_MATURITIES['DDI'].copy()
            loc_info['date'] = dt.datetime(int('20' + mty[1:3]), FUTURES_MONTH_CODES_CKEY[mty[0]], loc_info.pop('day'))
            mty_date = loc_nearest(**loc_info)
            mty_date = mty_date.strftime('%Y-%m-%d')
            t = dc(date, mty_date) / 360

        cc = ((100000.0 / float(ref_price)) - 1) / t

        return cc

    def get_di(self, date, maturity=1):
        """

        :param date: reference date
        :param maturity: DI maturity
        :return: DI rate
        """

        assert isinstance(date, str), 'DATE MUST BE A STRING'
        if self.max_date:
            assert date <= self.max_date, 'DATE MUST BE BEFORE CURRENT DATE'

        tmp_and = {'=': {'DATE': date, 'TICKER': 'DI1'}}
        fields = ['CLOSE', 'MTY']

        data = self._select(DEFAULT_TABLES['quotes'], tmp_and=tmp_and, fields=fields)
        data = pd.DataFrame(data)
        mtys = data['MTY'].values

        mty = find_maturity(maturity, mtys)
        ref_price = float(data[data['MTY'] == mty]['CLOSE'].iloc[0])

        mty_date = dt.datetime(int('20' + mty[1:3]), FUTURES_MONTH_CODES_CKEY[mty[0]], 1)
        mty_date = loc_nearest(mty_date)
        mty_date = mty_date.strftime('%Y-%m-%d')

        ttm = du(date, mty_date)
        if ttm == 0:
            mty = find_maturity(maturity + 1, mtys)
            ref_price = float(data[data['MTY'] == mty]['CLOSE'].iloc[0])
            mty_date = dt.datetime(int('20' + mty[1:3]), FUTURES_MONTH_CODES_CKEY[mty[0]], 1)
            mty_date = loc_nearest(mty_date)
            mty_date = mty_date.strftime('%Y-%m-%d')
            ttm = du(date, mty_date)

        di = (100000.0 / ref_price) ** (252 / float(ttm)) - 1

        return di

    def get_cc_curve(self, date):
        assert isinstance(date, str), 'DATE MUST BE A STRING'
        if self.max_date:
            assert date <= self.max_date, 'DATE MUST BE BEFORE CURRENT DATE'

        tmp_and = {'=': {'DATE': date, 'TICKER': 'DDI'}}
        fields = ['CLOSE', 'MTY']
        data = self._select(DEFAULT_TABLES['quotes'], tmp_and=tmp_and, fields=fields)
        data = pd.DataFrame(data).set_index('MTY')
        res_data = pd.DataFrame(columns=['CC', 'MTY_DATE', 'T'])

        for m in data.index:
            loc_info = FUTURES_MATURITIES['DDI'].copy()
            loc_info['date'] = dt.datetime(int('20' + m[1:3]), FUTURES_MONTH_CODES_CKEY[m[0]], loc_info.pop('day'))
            mty_date = loc_nearest(**loc_info)
            mty_date = mty_date.strftime('%Y-%m-%d')
            t = dc(date, mty_date) / 360
            cc = ((100000.0 / float(data.loc[m])) - 1) / t if not t == 0 else 0.0
            res = pd.DataFrame([{'CC': cc, 'MTY_DATE': mty_date, 'T': t}], index=[m])
            res_data = pd.concat((res_data, res), axis=0)

        data = pd.concat((data, res_data), axis=1)
        data['MTY_DATE'] = pd.to_datetime(data['MTY_DATE'])

        return data.reset_index().set_index('MTY_DATE').sort_index()

    def get_di_curve(self, date):
        assert isinstance(date, str), 'DATE MUST BE A STRING'
        if self.max_date:
            assert date <= self.max_date, 'DATE MUST BE BEFORE CURRENT DATE'

        tmp_and = {'=': {'DATE': date, 'TICKER': 'DI1'}}
        fields = ['CLOSE', 'MTY']
        data = self._select(DEFAULT_TABLES['quotes'], tmp_and=tmp_and, fields=fields)
        data = pd.DataFrame(data).set_index('MTY')
        res_data = pd.DataFrame(columns=['DI', 'MTY_DATE', 'T'])

        for m in data.index:
            loc_info = FUTURES_MATURITIES['DI1'].copy()
            loc_info['date'] = dt.datetime(int('20' + m[1:3]), FUTURES_MONTH_CODES_CKEY[m[0]], loc_info.pop('day'))
            mty_date = loc_nearest(**loc_info)
            mty_date = mty_date.strftime('%Y-%m-%d')
            t = du(date, mty_date) / 252
            di = (100000.0 / float(data.loc[m])) ** (1 / float(t)) - 1 if not t == 0 else 0.0
            res = pd.DataFrame([{'DI': di, 'MTY_DATE': mty_date, 'T': t}], index=[m])
            res_data = pd.concat((res_data, res), axis=0)

        data = pd.concat((data, res_data), axis=1)
        data['MTY_DATE'] = pd.to_datetime(data['MTY_DATE'])

        return data.reset_index().set_index('MTY_DATE').sort_index()

    def animate_di_curve(self, start_date, end_date, curves=None):
        """
        ONLY AVAIALBLE IN JUPYTER NOTEBOOK
        USE: HTML(anim.to_jshtml())
        """
        if not curves:
            dates = self.dates[(self.dates['date'] >= start_date) & (self.dates['date'] <= end_date)]['date'].values
            curves = []
            for d in dates:
                curve = self.get_di_curve(d)
                curve = curve['DI']
                curves.append(curve)

        fig, ax = plt.subplots()
        fig.suptitle('DI Term Structure')
        line, = ax.plot([], [], label='Term Structure', color='blue')
        legend = ax.legend(loc='upper right', frameon=False)
        ax.margins(0.05)
        ax.set_ylim((0, 15))
        ax.set_ylabel('DI Rate')
        ax.set_xlabel('Time to Maturity')

        def init():
            line.set_data([], [])
            return line,

        def animate(i):
            y = list(curves[i].values * 100)
            x = list(curves[i].index)
            ax.set_xlim((0, int(max(x)) + 1))
            line.set_data(x, y)
            line.set_label(dates[i])
            ax.legend(loc='upper right', frameon=False)
            return line,

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=100, interval=100,
                                       blit=True)
        return anim

    def get_reference_rate(self, date, rate):
        """

        :param date: reference date
        :param rate: reference rate ('USD', etc.)
        :return: reference rate
        """
        assert isinstance(date, str) or isinstance(date, tuple), 'DATE MUST BE A STRING OR TUPLE'

        if isinstance(date, tuple):
            start = date[0]
            end = date[1]
        else:
            start = end = date

        if self.max_date:
            assert start <= end <= self.max_date, 'DATE MUST BE BEFORE CURRENT DATE'

        tmp_and = {'=': {'CODE': rate}, '>=': {'DATE': start}, '<=': {'DATE': end}}
        fields = ['DATE', 'RATE']

        data = self._select(DEFAULT_TABLES['reference_rates'], tmp_and=tmp_and, fields=fields)
        data = pd.DataFrame(data).set_index('DATE')
        if len(data) == 1:
            data = data['RATE'].iloc[0]
        else:
            data[data['RATE'] == 0.0] = np.nan
            data = data.fillna(method='ffill')

        return data

    def get_futures_adjustment(self, date, ticker, mty):
        assert isinstance(date, str), 'DATE MUST BE A STRING'
        if self.max_date:
            assert date <= self.max_date, 'DATE MUST BE BEFORE CURRENT DATE'

        fields = ['ADJ']
        tmp_and = {'=': {'date': date, 'ticker': ticker, 'mty': mty}}
        data = self._select(DEFAULT_TABLES['quotes'], tmp_and=tmp_and, fields=fields)
        assert len(data) > 0, 'NON-EXISTENT TICKER'

        return data[0]['ADJ']

    def get_mty_code(self, date, ticker, mty):
        tmp_and = {'=': {'DATE': date, 'TICKER': ticker}}
        fields = ['MTY']
        data = self._select(DEFAULT_TABLES['quotes'], tmp_and=tmp_and, fields=fields)
        data = pd.DataFrame(data)
        mtys = data['MTY'].values
        mty_cod = find_maturity(mty, mtys)
        return mty_cod

    def get_quotes_futures_by_mty(self, date, ticker, mty, lb=None):
        """

        :param date: reference date
        :param ticker: futures ticker (e.g., DOL, ISP, IND, etc.)
        :param mty: which maturity (1st, 2nd, 3rd, etc.)
        :param lb: lookback days
        :return: quote
        """
        # time_1 = dt.datetime.now()
        # print(f"RETRIEVING QUOTES")

        assert isinstance(date, str), 'DATE MUST BE A STRING'
        if self.max_date:
            assert date <= self.max_date, 'DATE MUST BE BEFORE CURRENT DATE'

        if lb:
            cur_idx = self.dates[self.dates['date'] == date].index[0] - lb
            assert cur_idx >= 0, 'NOT ENOUGH DATA FOR LOOKBACK'
            cur_date = self.dates['date'].iloc[cur_idx]
            end_date = date
        else:
            cur_idx = self.dates[self.dates['date'] == date].index[0]
            cur_date, end_date = date, date

        # time_2 = dt.datetime.now()
        # print(f"SETUP TIME: {time_2 - time_1}")

        fields = ['DATE', 'TICKER', 'MTY', 'CLOSE']
        tmp_and = {'=': {'TICKER': ticker}, '>=': {'DATE': cur_date}, '<=': {'DATE': end_date}}
        all_data = self._select(DEFAULT_TABLES['quotes'], tmp_and=tmp_and, fields=fields)
        all_data = pd.DataFrame(all_data)

        # time_3 = dt.datetime.now()
        # print(f"QUERY TIME: {time_3 - time_2}")

        df = pd.DataFrame(columns=['DATE', 'TICKER', 'MTY', 'CLOSE'])

        while cur_date <= end_date:
            data = all_data[all_data['DATE'] == cur_date]
            data = pd.DataFrame(data)
            mtys = data['MTY'].values
            mty_cod = find_maturity(mty, mtys)

            data = data[data['MTY'] == mty_cod]
            data = pd.DataFrame(data)
            data['CLOSE'] = data['CLOSE'].astype(float)
            df = pd.concat((df, data), axis=0, ignore_index=True)

            cur_idx += 1
            cur_date = self.dates['date'].iloc[cur_idx]

        # time_4 = dt.datetime.now()
        # print(f"LOOP TIME: {time_4 - time_3}")
        # print("")

        return df.set_index('DATE', drop=True)

    def get_tickers_by_liquidity(self, date, top=50, schema=None, table=None):
        """

        :param date: reference date
        :param top: number of tickers to return
        :param schema: schema to query
        :param table: table to query
        :return: return list of tickers ordered by daily volume (descending)
        """
        schema = schema or DEFAULT_SCHEMA
        table = table or DEFAULT_TABLES['quotes']

        q = f"SELECT ticker FROM {schema}.{table} WHERE date = '{date}' ORDER BY vol DESC"
        res, data = self.api.query(q, args=None, fetch=True, commit=False)
        data = pd.DataFrame(data)
        return data['ticker'].values[:top]

    def get_quotes_equity(self, ticker, date=None, fields=None, table=None, schema=None):
        assert isinstance(ticker, str) or isinstance(ticker, list) or isinstance(ticker, np.ndarray), \
            'TICKER MUST BE STR OR LIST-LIKE'
        if date is not None:
            assert isinstance(date, str) or isinstance(date, dict) or isinstance(date, list) or isinstance(date, tuple)
            if self.max_date:
                err_msg = 'Date must be before than current date'
                if isinstance(date, str):
                    assert date <= self.max_date, err_msg
                elif isinstance(date, dict):
                    assert (date['start'] <= self.max_date) & (date['end'] <= self.max_date), err_msg
                else:
                    assert (date[0] <= self.max_date) & (date[0] <= self.max_date), err_msg
        else:
            if self.max_date:
                date = {'start': self.dates.iloc[0], 'end': self.max_date}

        fields = fields or DEFAULT_FIELDS
        res = self._quote_search(ticker, date=date, fields=fields, table=table, schema=schema)
        df = pd.DataFrame(res)

        return df

    def get_metrics_equity(self, ticker, metric, date=None, fields=None, table=None, schema=None):
        assert isinstance(ticker, str) or isinstance(ticker, list) or isinstance(ticker, np.ndarray), \
            'TICKER MUST BE STR OR LIST-LIKE'
        if date is not None:
            assert isinstance(date, str) or isinstance(date, dict) or isinstance(date, list) or isinstance(date, tuple)
            if self.max_date:
                err_msg = 'Given date must be before current date'
                if isinstance(date, str):
                    assert date <= self.max_date, err_msg
                    date = {'start': date, 'end': date}
                elif isinstance(date, dict):
                    assert (date['start'] <= self.max_date) & (date['end'] <= self.max_date), err_msg
                else:
                    assert (date[0] <= self.max_date) & (date[0] <= date[1]), err_msg
                    date = {'start': date[0], 'end': date[1]}
        else:
            if self.max_date:
                date = {'start': self.dates.iloc[0], 'end': self.max_date}

        fields = fields or DEFAULT_FIELDS_METRICS
        table = table or DEFAULT_TABLES['metrics']
        metric = [metric] if isinstance(metric, str) else metric
        tmp_and = {'=': {'METRIC': metric}, '>=': {'DATE': date['start']}, '<=': {'DATE': date['end']}}
        tmp_and['='] = {**tmp_and['='], 'TICKER': ticker} if isinstance(ticker, str) else tmp_and['=']
        tmp_in = {'TICKER': ticker} if not isinstance(ticker, str) else None
        data = self._select(table, tmp_and=tmp_and, tmp_in=tmp_in, fields=fields, schema=schema)
        data = pd.DataFrame(data)
        data['VALUE'] = data['VALUE'].astype(float)
        return data

    def get_quotes_equity_lb(self, date, ticker, lb=None, fields=None, table=None, schema=None):
        """

        :param date: reference date
        :param ticker: ticker or list of tickers
        :param lb: how many days to return (if None, 0)
        :param fields: fields to return
        :param table: table to query
        :param schema: schema to query
        :return: Dataframe with quotes
        """

        assert isinstance(date, str), 'DATE MUST BE STR'
        assert isinstance(ticker, str) or isinstance(ticker, list) or isinstance(ticker, np.ndarray), \
            'TICKER MUST BE LIST-LIKE'

        date = {'start': self.dates['date'][self.dates['date'] <= date].values[-lb], 'end': date} if lb else date
        return self.get_quotes_equity(ticker, date, fields, table, schema)

    def get_quotes_by_id(self, id, fields=None, table=None, schema=None):
        assert isinstance(id, str) or isinstance(id, list), 'ID MUST BE STR OR LIST'

        fields = fields or DEFAULT_FIELDS
        tmp_in = {'id': id} if isinstance(id, list) else {'id': [id]}
        table = table or DEFAULT_TABLES['quotes']
        return self._select(table, tmp_in=tmp_in, fields=fields, schema=schema)

    def get_spot(self, date, currency='USD'):
        """

        :param date: reference date
        :param currency: reference currency
        :return: forward implied spot rate
        """
        if not currency == 'USD':
            raise Exception(f"Currency {currency} is unsupported")

        tmp_and = {'=': {'DATE': date, 'TICKER': 'DOL'}}
        fields = ['MTY', 'CLOSE']
        data = pd.DataFrame(self._select(DEFAULT_TABLES['quotes'], tmp_and=tmp_and, fields=fields))
        mtys = data['MTY'].values

        mty_num = 1
        m = find_maturity(mty_num, mtys)
        loc_info = FUTURES_MATURITIES['DOL'].copy()
        loc_info['date'] = dt.datetime(int('20' + m[1:3]), FUTURES_MONTH_CODES_CKEY[m[0]], loc_info.pop('day'))
        mty_date = loc_nearest(**loc_info)
        mty_date = mty_date.strftime('%Y-%m-%d')

        if mty_date == date:
            mty_num = 2
            m = find_maturity(mty_num, mtys)
            loc_info = FUTURES_MATURITIES['DOL'].copy()
            loc_info['date'] = dt.datetime(int('20' + m[1:3]), FUTURES_MONTH_CODES_CKEY[m[0]], loc_info.pop('day'))
            mty_date = loc_nearest(**loc_info)
            mty_date = mty_date.strftime('%Y-%m-%d')

        t_dc, t_du = dc(date, mty_date) / 360, du(date, mty_date) / 252

        di, cc = self.get_di(date, mty_num), self.get_cc(date, mty_num)
        dol = float(data['CLOSE'][data['MTY'] == m].iloc[0])

        spot = dol * (1 + (cc * t_dc)) / ((1 + di) ** t_du)
        return spot

    def get_valid_tickers(self, date, table=None, schema=None):
        schema = schema or DEFAULT_SCHEMA
        table = table or DEFAULT_TABLES['quotes_bbg']
        sql = "SELECT DISTINCT TICKER from " + schema + "." + table + " WHERE DATE = '" + str(date) + "'"
        _, data = self.api.query(sql)
        data = pd.DataFrame(data)
        return data['TICKER'].values

    def _quote_search(self, ticker, date=None, mty=None, fields=None, table=None, schema=None):
        tmp_and, tmp_in = None, None
        if date:
            if isinstance(ticker, str):
                if isinstance(date, str):
                    tmp_and = {'=': {'date': date, 'ticker': ticker}}
                elif isinstance(date, dict):
                    tmp_and = {'=': {'ticker': ticker}, '>=': {'date': date['start']}, '<=': {'date': date['end']}}
                elif isinstance(date, list) or isinstance(date, tuple):
                    tmp_and = {'=': {'ticker': ticker}, '>=': {'date': date[0]}, '<=': {'date': date[1]}}
                else:
                    raise ValueError(f'ERROR - GET QUOTE - date must be of type str, dict, list, or tuple')
                tmp_and['='] = {**tmp_and['='], 'mty': mty} if mty else tmp_and['=']
            elif isinstance(ticker, list) or isinstance(ticker, np.ndarray):
                if isinstance(date, str):
                    tmp_and = {'=': {'date': date}}
                    tmp_and['='] = {**tmp_and['='], 'mty': mty} if mty else tmp_and['=']
                    tmp_in = {'ticker': ticker}
                elif isinstance(date, dict):
                    tmp_and = {'>=': {'date': date['start']}, '<=': {'date': date['end']}}
                    if mty:
                        tmp_and['='] = {'mty': mty}
                    tmp_in = {'ticker': list(ticker)}
                elif isinstance(date, list) or isinstance(date, tuple):
                    tmp_and = {'>=': {'date': date[0]}, '<=': {'date': date[1]}}
                    if mty:
                        tmp_and['='] = {'mty': mty}
                    tmp_in = {'ticker': ticker}
                else:
                    raise ValueError(f'ERROR - GET QUOTE - date must be of type str, dict, list, or tuple')
            else:
                raise ValueError(f'ERROR - GET QUOTE - ticker must be of type str or list')
        else:
            if isinstance(ticker, str):
                tmp_and = {'=': {'ticker': ticker}}
                tmp_and['='] = {**tmp_and['='], 'mty': mty} if mty else tmp_and['=']
            elif isinstance(ticker, list):
                tmp_in = {'ticker': ticker}
                if mty:
                    tmp_and['='] = {'mty': mty}
            else:
                raise ValueError(f'ERROR - GET QUOTE - ticker must be of type str or list')

        table = table or DEFAULT_TABLES['quotes']
        return self._select(table, tmp_and, tmp_in, fields, schema)

    def _select(self, table, tmp_and=None, tmp_in=None, fields=None, schema=None):
        schema = schema or DEFAULT_SCHEMA
        sql, args = create_select(schema, table, tmp_and, tmp_in, fields)
        res, data = self.api.query(sql, args, fetch=True, commit=False)
        return data
