DEFAULT_CONNECT_INFO = {
    'host': '127.0.0.1',
    'user': 'dbuser',
    'password': 'dbuserdbuser'
}

LOCAL_CONNECT_INFO = {
    'host': '127.0.0.1',
    'user': 'dbuser',
    'password': 'dbuserdbuser'
}

DEFAULT_SCHEMA = 'dbmaster'

DEFAULT_TABLES = {
    'quotes': 'quotes',
    'assets': 'assets',
    'reference_rates': 'reference_rates',
    'quotes_bbg': 'quotes_bbg',
    'metrics': 'metrics'
}

EQUITY_BASE_URL = "http://bvmf.bmfbovespa.com.br/InstDados/SerHist/"
FUTURES_BASE_URL = "http://www2.bmf.com.br/pages/portal/bmfbovespa/lumis/lum-ajustes-do-pregao-ptBR.asp"

DEFAULT_FIELDS = ['DATE', 'TICKER', 'CLOSE']
DEFAULT_FIELDS_METRICS = ['DATE', 'METRIC', 'TICKER', 'VALUE']

FUTURES_MONTH_CODES_CKEY = {
    'F': 1,
    'G': 2,
    'H': 3,
    'J': 4,
    'K': 5,
    'M': 6,
    'N': 7,
    'Q': 8,
    'U': 9,
    'V': 10,
    'X': 11,
    'Z': 12
}

FUTURES_MONTH_CODES_MKEY = {
    1: 'F',
    2: 'G',
    3: 'H',
    4: 'J',
    5: 'K',
    6: 'M',
    7: 'N',
    8: 'Q',
    9: 'U',
    10: 'V',
    11: 'X',
    12: 'Z'
}