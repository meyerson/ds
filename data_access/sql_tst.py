#!/usr/bin/env python

'''
Example of using a sqlite DB.
'''

import sys
import argparse
from data_load import *
from pandas.io.sql import read_sql
from pandas.io.sql import write_frame

def main(db, query):
    import sqlite3 as lite
    cnx = lite.connect(db)
            
    df = read_sql(query, cnx)
    print df.describe()
    import pdb; pdb.set_trace()

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='read csv, write distributions to db',argument_default=[])
   parser.add_argument('db', help='sqlite3 database file to use')

   try:
      args = parser.parse_args()
   except Exception as e:
       print str(e)
       print parser.print_help()
       exit(0)
   print args

   query = 'select up_or_not, record_date, water_E30_P,gas_E7_P,value_E30_P,water_b7_30_D,water_E7_D,water_E30,gas_E7_D,water_E30_D,value_E30_D,water_E7_P,gas_b7_30_R,oil_b7_30_R,gas_E30_P,value_b7_30_R,oil_E7_D,gas_E30_D,value_E7_D,oil_b7_30_D,gas_b7_30_D,value_E7_P,water_b7_30_R,oil_E7_P,value_b7_30_D,oil,oil_E30,gas_E7,gas,oil_E30_P,water_E7,oil_E7,oil_E30_D,water,value_E7,gas_E30,value,value_E30 from recoverable_xy_pairs ORDER BY RANDOM() limit 100000 ;'

   main(args.db, query)
