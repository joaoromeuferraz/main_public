TRADE_HISTORY_COLUMNS = ['Date', 'Quantity', 'Price', 'Fees']
DIVIDEND_HISTORY_COLUMNS = ['Payment Date', 'Type', 'Amount']

DEFAULT_RESTRICTED = ['BFRE11', 'GMAT3', 'BFRE12', 'BSAN30', 'MGLU3', 'CIEL3', 'KROT3',
                      'BBOV11', 'PETZ3', 'OGXP3', 'BOAS3', 'ELDO11B', 'KLBN11', 'TIMS3',
                      'BMGB11', 'HBSA3', 'PCAR5', 'VIVO4', 'OIBR3', 'PDGR3', 'CRUZ3', 'TCSL4',
                      'GFSA3', 'OIBR4', 'STRX11', 'UGPA4', 'LAVV3', 'HRTP3', 'SEQL3', 'CURY3',
                      'PLPL3', 'PGMN3', 'AMIL3', 'ENEV3', 'IRBR9', 'IRBR1', 'ECOD3', 'SIMH3',
                      'SINC11', 'RSID3', 'MMXM3', 'OHLB3', 'GOOG35', 'MELK3', 'TCSA3',
                      'TCSL3', 'TNLP3', 'HBOR3', 'CVCB1', 'RBAG12', 'BPAC13', 'MILK11',
                      'JSLG11', 'GFSA11', 'VAGR3', 'PDGR12', 'NTCO1', 'BBDC2', 'LOGN3',
                      'BBRK3', 'AQLL11B', 'BRIN3', 'OSXB3', 'OIBR9', 'MAGG3', 'FTCE11B',
                      'BPHA3', 'BRIM11', 'RZTR11', 'SEBB11', 'RECR11', 'RBPD13', 'KNCR11',
                      'KNRI11', 'XPIN12', 'DOVL11B', 'UNIP6', 'OIBR1', 'PRML3', 'INPR3',
                      'LLIS3', 'PMAM3', 'GESE11B', 'VULC3', 'BRCR12', 'CMIG2', 'XPCI12']

FUTURES_MATURITIES = {
    'DOL': {'day': 1, 'weekday': None, 'which_weekday': None},
    'WDO': {'day': 1, 'weekday': None, 'which_weekday': None},
    'IND': {'day': 15, 'weekday': 'W', 'which_weekday': None},
    'WIN': {'day': 15, 'weekday': 'W', 'which_weekday': None},
    'ISP': {'day': 1, 'weekday': 'F', 'which_weekday': 3},
    'WSP': {'day': 1, 'weekday': 'F', 'which_weekday': 3},
    'DI1': {'day': 1, 'weekday': None, 'which_weekday': None},
    'DDI': {'day': 1, 'weekday': None, 'which_weekday': None}
}

FUTURES_MULTIPLIERS = {
    'DOL': 50.0,
    'IND': 1.0,
    'ISP': 50.0,
    'WDO': 10.0,
    'WIN': 0.20,
    'WSP': 2.5
}

FUTURES_CURRENCIES = {
    'DOL': 'BRL',
    'IND': 'BRL',
    'ISP': 'USD',
    'WDO': 'BRL',
    'WIN': 'BRL',
    'WSP': 'USD'
}

OPTIONS_UNDERLYING = {
    'BOVA': 'BOVA11'
}

OPTIONS_REFERENCE = {
    'BOVA': 'DI'
}

OPTIONS_MATURITIES = {
    'BOVA': {'Q70': '2020-05-18', 'Q75': '2020-05-18', 'M100': '2021-01-18'}
}

OPTIONS_STRIKE = {
    'BOVA': {'Q70': 70.0, 'Q75': 75.0, 'M100': 100.0}
}

OPTIONS_TYPE = {
    'BOVA': {'Q70': 'PUT', 'Q75': 'PUT', 'M100': 'PUT'}
}
