'''
Helper functions for connecting or writing to various databases.
'''

def db_connect(conn_str, flav, schema=''):
    if flav == 'postgresql':
        cnx = db_connect_pg(conn_str, schema)
    elif flav == 'sqlite':
        cnx = db_connect_sqlite(conn_str)
    elif flav == 'hive':
        cnx = db_connect_hive(conn_str)
    else:
        raise ValueError('flav %s unsupported' % (flav,))
    return cnx

def db_connect_pg(conn_str, schema=''):
    import psycopg2 as pg
    cnx = pg.connect(**conn_str)
    #execute('SET search_path TO ' + schema + ', public;', cnx)
    return cnx

def db_connect_sqlite(conn_str):
    import sqlite3 as lite
    cnx = lite.connect(conn_str)
    return cnx

def db_connect_hive(conn_str):
    import pyhs2
    cnx = pyhs2.connect(**eval(conn_str)) # conn_str is a string containing a dict of parameters
    return cnx

def execute(query, cnx, verbose=True):
    if verbose:
        print query
    from sql import execute
    return execute(query, cnx)
