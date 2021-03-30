# http://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_D14082020.ZIP
from constants.data import *
from data.db_api import *
import os
import dload
import pandas as pd
from lxml import html
import requests
from helpers import *
from constants.asset import *


def push_sql(data, date_f=None, connect_info=None, schema=None, table=None):
    """

    :param data: data to be pushed to SQL
    :param date_f: formatted date
    :param connect_info: dictionary with SQL connection info (host, user, password)
    :param schema: schema name to insert rows
    :param table: table name to insert rows
    :return: count of rows inserted
    """

    schema = schema or DEFAULT_SCHEMA
    table = table or DEFAULT_TABLES['quotes']
    api = DbAPI(connect_info=connect_info)

    ins_rows = 0
    for i in data.index:
        row = data.iloc[i].to_dict()
        row = {str(k): str(v) for k, v in row.items()}
        if not row.get('DATE') is None:
            row['DATE'] = date_f if date_f else row['DATE']
        else:
            row['DATE'] = date_f

        if not row.get('ID'):
            row['ID'] = row['DATE'] + "_" + row['TICKER']
            row['ID'] += row['MTY'] if row.get('MTY') else ""
        sql, args = create_insert(schema, table, row)
        res, _ = api.query(sql, args)
        ins_rows += res

    return ins_rows


class EquityImportBr:
    def __init__(self, date, download=True, daily=True):
        """

        :param date: import date in format ddmmyyyy
        """

        self.date_f = date[4:] + "-" + date[2:4] + "-" + date[:2] if download else None

        names = ['TIPREG', 'DATE', 'CODBDI', 'TICKER', 'TPMERC', 'NAME', 'ESPECI', 'PRAZOT', 'MODREF', 'OPEN_NUM',
                 'OPEN_DEC', 'MAX_NUM', 'MAX_DEC', 'MIN_NUM', 'MIN_DEC', 'AVG_NUM', 'AVG_DEC', 'CLOSE_NUM', 'CLOSE_DEC',
                 'BID_NUM', 'BID_DEC', 'ASK_NUM', 'ASK_DEC', 'NEG', 'QTY', 'VOL_NUM', 'VOL_DEC', 'PREEXE_NUM',
                 'PREEXE_DEC', 'INDOPC', 'DATVEN', 'FATCOT', 'PTOEXE_NUM', 'PTOEXE_DEC', 'CODISI', 'DISMES']

        final_names = ['DATE', 'TICKER', 'NAME', 'OPEN_NUM', 'OPEN_DEC', 'MAX_NUM', 'MAX_DEC', 'MIN_NUM', 'MIN_DEC',
                       'AVG_NUM', 'AVG_DEC', 'CLOSE_NUM', 'CLOSE_DEC', 'BID_NUM', 'BID_DEC', 'ASK_NUM', 'ASK_DEC',
                       'NEG', 'QTY', 'VOL_NUM', 'VOL_DEC']

        colspecs = [(0, 2), (2, 10), (10, 12), (12, 24), (24, 27), (27, 39), (39, 49), (49, 52), (52, 56), (56, 67),
                    (67, 69), (69, 80), (80, 82), (82, 93), (93, 95), (95, 106), (106, 108), (108, 119), (119, 121),
                    (121, 132), (132, 134), (134, 145), (145, 147), (147, 152), (152, 170), (170, 186), (186, 188),
                    (188, 199), (199, 201), (201, 202), (202, 210), (210, 217), (217, 224), (224, 230), (230, 242),
                    (242, 245)]

        cache_path = 'data/cache'
        file_name = 'COTAHIST_D' + date if daily else 'COTAHIST_A' + date

        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)

        if download:
            url = os.path.join(EQUITY_BASE_URL, file_name + ".ZIP")
            dload.save_unzip(url, cache_path, delete_after=True)

        path = os.path.join(cache_path, file_name + ".TXT")
        df = pd.read_fwf(path, names=names, colspecs=colspecs, dtype='str')
        df.drop(df.tail(1).index, inplace=True)
        df.drop(df.head(1).index, inplace=True)

        if download:
            os.remove(os.path.join(cache_path, file_name + ".TXT"))

        df = df.where((pd.notnull(df)), None)
        df = df[pd.isnull(df['PRAZOT'])]
        df = df[final_names]

        join_cols = [t[:t.index('_DEC')] for t in list(df.columns) if '_DEC' in t]

        for c in join_cols:
            df[c] = df[c + "_NUM"].astype(str) + "." + df[c + "_DEC"].astype(str)
            df[c] = df[c].astype(float)
            df.drop([c + "_NUM", c + "_DEC"], axis=1, inplace=True)

        df['QTY'] = df['QTY'].astype(float)
        df['NEG'] = df['NEG'].astype(float)
        df['DATE'] = df['DATE'].apply(lambda x: x[:4] + "-" + x[4:6] + "-" + x[6:8])

        self.data = df.reset_index(drop=True)

    def push_sql(self, connect_info=None, schema=None, table=None):
        """

        :param connect_info: dictionary with SQL connection info (host, user, password)
        :param schema: schema name to insert rows
        :param table: table name to insert rows
        :return: count of rows inserted
        """

        return push_sql(self.data, self.date_f, connect_info, schema, table)


class FuturesImportBr:
    def __init__(self, date):
        """

        :param date: import date in format ddmmyyyy
        """
        self.date_f = date[-4:] + "-" + date[2:4] + "-" + date[:2]
        date = date[:2] + "/" + date[2:4] + "/" + date[-4:]
        self.conn = None
        try:
            link = FUTURES_BASE_URL + "?dData1=" + str(date)
            r = requests.get(link)
            self._tree = html.fromstring(r.content)
            tr_elements = self._tree.xpath('//tr')

            headers = []
            for t in tr_elements[0]:
                name = t.text_content()
                headers.append(str(name.strip()))

            tbl = {}
            for i in headers:
                tbl[i] = []

            for i in range(1, len(tr_elements)):
                c = 0
                for t in tr_elements[i]:
                    content = t.text_content()
                    content = content.strip()
                    tbl[headers[c]].append(content)
                    c += 1

            for i in range(len(tbl[headers[0]])):
                if not tbl[headers[0]][i] == "":
                    spl = tbl[headers[0]][i].split(" ")
                    tbl[headers[0]][i] = spl[0]

                if not i == 0:
                    if tbl[headers[0]][i] == "":
                        last = tbl[headers[0]][i - 1]
                        tbl[headers[0]][i] = last

            for i in range(len(tbl[headers[1]])):
                for j in range(2, len(headers)):
                    old = tbl[headers[j]][i]
                    new = old.replace(".", "")
                    new = new.replace(",", ".")
                    new = str(new)
                    tbl[headers[j]][i] = new

            df = pd.DataFrame.from_dict(tbl)

            new_col_names = ["TICKER", "MTY", "PREV_CLOSE", "CLOSE", "DIFF", "ADJ"]

            mapper = dict(zip(headers, new_col_names))

            self.data = df.rename(columns=mapper)

            r.close()
        except IndexError:
            self.data = None

    def push_sql(self, connect_info=None, schema=None, table=None):
        """

        :param connect_info: dictionary with SQL connection info (host, user, password)
        :param schema: schema name to insert rows
        :param table: table name to insert rows
        :return: count of rows inserted
        """
        self.data = self.data[['TICKER', 'MTY', 'CLOSE', 'ADJ']]

        return push_sql(self.data, self.date_f, connect_info, schema, table)


class ReferenceRates(FuturesImportBr):
    def __init__(self, ref_ticker='WSP', **kwargs):
        """

        :param date: reference ticker to calculate rate
        """
        super().__init__(**kwargs)
        sub = self.data[self.data['TICKER'] == ref_ticker]
        maturities = sub['MTY'].values
        first = find_maturity(1, maturities)
        ref = sub[sub['MTY'] == first].iloc[0]

        try:
            ref = np.abs(float(ref.loc['ADJ']) / float(ref.loc['DIFF']) / FUTURES_MULTIPLIERS[ref_ticker])
        except ZeroDivisionError:
            ref = 0.0
        row = {'ID': self.date_f + "_" + 'USD', 'DATE': self.date_f, 'CODE': 'USD', 'RATE': ref}
        self.data = pd.DataFrame([row])

    def push_sql(self, connect_info=None, schema=None, table=None):
        """

        :param connect_info: dictionary with SQL connection info (host, user, password)
        :param schema: schema name to insert rows
        :param table: table name to insert rows
        :return: count of rows inserted
        """
        table = table or DEFAULT_TABLES['reference_rates']

        if self.data is not None:
            return push_sql(self.data, self.date_f, connect_info, schema, table)
        else:
            return 0


def get_all(start, end=None):
    dates_path = 'data/dates/br.csv'
    if end:
        dates = pd.read_csv(dates_path)
        dates = dates[(dates['date'] >= start) & (dates['date'] <= end)]
        dates = dates.reset_index(drop=True)
        dates = dates['date'].values
    else:
        dates = [start]

    for d in dates:
        print(f'Current date: {d}')
        d_f = d[-2:] + d[5:7] + d[:4]
        imp_eq = EquityImportBr(date=d_f)
        imp_fut = FuturesImportBr(date=d_f)
        ref_rates = ReferenceRates(date=d_f)
        push_eq = imp_eq.push_sql()
        push_fut = imp_fut.push_sql()
        push_rates = ref_rates.push_sql()

        print(f"{push_eq} equity quotes inserted")
        print(f"{push_fut} futures quotes inserted")
        print(f"{push_rates} reference rates inserted")
        print("")


if __name__ == '__main__':

    ys = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
    for y in ys:
        print(f"Year: {y}")
        eq = EquityImportBr(y, download=False, daily=False)
        rows = eq.push_sql()
        print(f"{rows} equity quotes inserted")
