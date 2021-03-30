import pandas as pd
from constants.data import DEFAULT_TABLES, DEFAULT_SCHEMA, DEFAULT_CONNECT_INFO, LOCAL_CONNECT_INFO
from data.db_api import *


def import_px(path, index=None, table=None, schema=None, connect_info=None):
    data = pd.read_csv(path)
    index = index or data.index
    iter_idx = [index] if not hasattr(index, '__iter__') else index
    table = table or DEFAULT_TABLES['quotes_bbg']
    schema = schema or DEFAULT_SCHEMA
    connect_info = connect_info or DEFAULT_CONNECT_INFO
    api = DbAPI(connect_info=connect_info)

    for i in iter_idx:
        ins_rows = 0
        dly_data = data.iloc[i]
        date = str(dly_data['DATE'])
        for t in dly_data.index:
            close = dly_data[t]
            if not t == 'DATE' and not pd.isna(close):
                id = date + "_" + t
                row = {'ID': id, 'DATE': date, 'TICKER': t, 'CLOSE': str(close)}
                sql, args = create_insert(schema, table, row)
                res, _ = api.query(sql, args)
                ins_rows += res
        print(f"{date}: {ins_rows} quotes inserted")


def import_metrics(path, metric, index=None, table=None, schema=None, connect_info=None):
    data = pd.read_csv(path)
    index = index or data.index
    iter_idx = [index] if not hasattr(index, '__iter__') else index
    table = table or DEFAULT_TABLES['metrics']
    schema = schema or DEFAULT_SCHEMA
    connect_info = connect_info or DEFAULT_CONNECT_INFO
    api = DbAPI(connect_info=connect_info)

    for i in iter_idx:
        ins_rows = 0
        dly_data = data.iloc[i]
        date = str(dly_data['DATE'])
        for t in dly_data.index:
            value = dly_data[t]
            if not t == 'DATE' and not pd.isna(value):
                id = date + "_" + t + "_" + metric
                row = {'ID': id, 'DATE': date, 'METRIC': metric, 'TICKER': t, 'VALUE': str(value)}
                sql, args = create_insert(schema, table, row)
                res, _ = api.query(sql, args)
                ins_rows += res
        print(f"{date}: {ins_rows} quotes inserted")


def csv_format(inpath, outpath, metric, index=None):
    data = pd.read_csv(inpath)
    index = index or data.index
    iter_idx = [index] if not hasattr(index, '__iter__') else index
    out = pd.DataFrame(columns=['ID', 'DATE', 'METRIC', 'TICKER', 'VALUE'])

    for i in iter_idx:
        ins_rows = 0
        dly_data = data.iloc[i]
        date = str(dly_data['DATE'])
        for t in dly_data.index:
            value = dly_data[t]
            if not t == 'DATE' and not pd.isna(value):
                id = date + "_" + t + "_" + metric
                row = {'ID': id, 'DATE': date, 'METRIC': metric, 'TICKER': t, 'VALUE': str(value)}
                row = pd.DataFrame([row])
                out = out.append(row)
                ins_rows += 1
        print(f"{date}: {ins_rows} quotes inserted")
    out.to_csv(outpath)
