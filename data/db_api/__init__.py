import pymysql
from constants.data import DEFAULT_CONNECT_INFO
import cryptography


def get_new_connection(host, user, password):
    """

    :param host: SQL host name
    :param user: SQL user name
    :param password: SQL password
    :return: connection
    """
    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False
    )
    return conn


def create_select(schema, table, tmp_and=None, tmp_in=None, fields=None):
    """

    :param schema: Schema name
    :param table: Table name
    :param tmp_and: Dictionary that represents elements of where clause (connected with AND):
        - keys: symbols representing equality/inequality sign (=, ≥, >, ≤, <)
        - values: dictionaries representing where clause for that symbol
        - example:
            >>> template = {'=': {'ticker': 'PETR4'}, '>=': {'date': '2019-01-01'}, '<=': {'date': '2019-12-31'}}
    :param tmp_in: Dictionary that represents elements of where in clause
        - keys: field name
        - values: lists with elements
        - example:
            >>> template = {'ticker': ['ITSA4', 'PETR4', 'VALE3']}
    :param fields: Which columns to return
    :return: SQL formatted select statement
    """

    field_list = " " + ",".join(fields) + " " if fields else " * "

    terms_and = [" " + x + k + "%s " for k, v in tmp_and.items() for x, _ in tmp_and[k].items()] if tmp_and else []
    args_and = [y for k, v in tmp_and.items() for x, y in tmp_and[k].items()] if tmp_and else None

    terms_in = [" " + k + " in ('" + "', '".join(v) + "')" for k, v in tmp_in.items()] if tmp_in else []

    terms = terms_and + terms_in
    args = args_and

    w_clause = " WHERE " + "AND".join(terms) if terms else ""
    sql = "select " + field_list + " from " + schema + "." + table + " " + w_clause

    return sql, args


def create_insert(schema, table, row):
    """

    :param schema: Schema name
    :param table: Table name
    :param row: Row to be inserted (dict):
        - keys: field_name
        - values: element
        - example:
        >>> row = {'date': '20200814', 'ticker': 'PETR4', 'price': 18.08}
    :return: SQL formatted insert statement
    """
    sql = "insert into " + schema + "." + table + " "

    cols = list(row.keys())
    cols = ",".join(cols)
    col_clause = "(" + cols + ") "

    args = list(row.values())

    s_stuff = ["%s"] * len(args)
    s_clause = ",".join(s_stuff)
    v_clause = " values(" + s_clause + ")"

    sql += " " + col_clause + " " + v_clause

    return sql, args


class DbAPI:
    def __init__(self, connect_info=None):
        connect_info = connect_info or DEFAULT_CONNECT_INFO
        self.conn = get_new_connection(**connect_info)
        self.cur = self.conn.cursor()

    def query(self, sql, args=None, fetch=True, commit=True):
        res = self.cur.execute(sql, args=args)
        data = self.cur.fetchall() if fetch else None
        if commit:
            self.conn.commit()
        return res, data

    def close(self):
        self.conn.close()
