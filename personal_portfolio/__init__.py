import pandas as pd

deposits_path = 'personal_portfolio/deposits.csv'
dividends_path = 'personal_portfolio/dividends.csv'
trades_path = 'personal_portfolio/trades.csv'


def get_deposits():
    return pd.read_csv(deposits_path)


def get_dividends():
    return pd.read_csv(dividends_path)


def get_trades():
    return pd.read_csv(trades_path)


def add_trade(date, ticker, mty, qty, price, fees):
    df = pd.read_csv(trades_path)
    row = {'DATE': date, 'TICKER': ticker, 'MTY': mty, 'QTY': qty, 'PRICE': price, 'FEES': fees}
    df = pd.concat((df, pd.DataFrame([row])), axis=0, ignore_index=True)
    df.to_csv(trades_path, index=False)


def add_deposit(date, amount):
    df = pd.read_csv(deposits_path)
    row = {'DATE': date, 'AMOUNT': amount}
    df = pd.concat((df, pd.DataFrame([row])), axis=0, ignore_index=True)
    df.to_csv(deposits_path, index=False)


def add_dividend(date, ticker, amount):
    df = pd.read_csv(dividends_path)
    row = {'DATE': date, 'TICKER': ticker, 'AMOUNT': amount}
    df = pd.concat((df, pd.DataFrame([row])), axis=0, ignore_index=True)
    df.to_csv(dividends_path, index=False)
