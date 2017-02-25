#!/usr/bin/env python

'''
Example of using a PostGres DB
'''

import psycopg2 as pg
import sys
from pandas.io.sql import read_sql, to_sql, execute

con = None
try:
    params = {'host':'admosphereanalysis.cyp44ziydlot.us-east-1.rds.amazonaws.com', 
        'database':'admosphere_product_labs', 'user':'admosphere_product', 'port':'5432', 'password':'aJJ3fZndR97YRDda'}
    con = pg.connect(**params) 
    cur = con.cursor()
    cur.execute('SELECT version()')          
    ver = cur.fetchone()
    print ver    
except pg.DatabaseError, e:
    print 'Error %s' % e    
    sys.exit(1)
finally:
    if con:
        con.close()