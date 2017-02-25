#!/usr/bin/env python

import pyhs2
import pandas as pd
import tempfile
import sql_pandas_help
import sys
import time

class Hive:
    def __init__(self, conn_str):
        self.cnx = pyhs2.connect(**eval(conn_str))
 
 
    def read_frame(self, query, parse_dates=[], drop_prefixes=True):
        ''' Performs a query on a hive DB and returns the results as a DataFrame.
        '''
        with self.cnx.cursor() as cur:
            print 'Executing query:', query
            sys.stdout.flush()
            starttime=time.time()
            cur.execute(query)
            print '... finished in %.1f secs' % (time.time()-starttime,)
            sys.stdout.flush()

            if drop_prefixes:
                column_names = [a['columnName'].split('.')[-1] for a in cur.getSchema()]
            else:
                column_names = [a['columnName'] for a in cur.getSchema()]
            if column_names is None:
                raise Exception('No column names returned for ' + query)

            print 'Fetching: ', query, column_names
            starttime=time.time()
            sys.stdout.flush()
            rows = cur.fetch()
            print 'fetched %d rows in %.1f secs' % (len(rows), time.time()-starttime)
            sys.stdout.flush()
            output = None
            if len(rows) > 0:
                output = pd.DataFrame(data=rows, columns=column_names)
                for _ in parse_dates:
                    output[_] = pd.to_datetime(output[_])
                    #output.set_index(index, inplace=True) # creates a multi-index
                    #output.sortlevel(inplace=True) # using a datetime in an index does NOT sort the dataframe

                print 'read ', output.shape, ' rows x columns'
            else:
                print '\n\n\NO DATA RETURNED FROM QUERY.\n\n'
            
        return output

    
    def write_frame(self, df, table_name, prefix='temp_file', if_exists='append'):
        ''' Writes a DataFrame into a Hive table named <table_name>
        '''
        starttime=time.time()
        print 'Writing', df.shape, 'rows x columns to', table_name
        
        if table_name[0] == '_':
            raise ValueError('Hive table names cannot begin with underscore: ' + table_name)
            
        # write to csv on local filesystem
        csv_file = tempfile.NamedTemporaryFile(delete=True, prefix=prefix)
        df.to_csv(path_or_buf=csv_file.name, header=False, index=False)
        
        # upload into HDFS (this would be totally unnecessary if the LOAD DATA LOCAL INPATH didn't throw "file not found" exceptions)
        from subprocess import call
        call(["hdfs", "dfs", "-put", csv_file.name, "/tmp"])
        
        # need to remove the local path information from the "name" of the file and get the actual name of the file.
        import os
        dir, file_name = os.path.split(csv_file.name)
        hdfs_path = os.path.join("/tmp", file_name)
        
        column_definitions = sql_pandas_help.make_create_hive_statement_for_dataframe(df)
        with self.cnx.cursor() as cur:
            if if_exists != 'append':
                overwrite = 'OVERWRITE'
                # must drop table since schema definitions can with the addition of new features and OVERWRITE won't change schema
                query = 'DROP TABLE IF EXISTS %s' % table_name; 
                print query; 
                cur.execute(query)
            else:
                overwrite = ''
            
            query = ("CREATE TABLE IF NOT EXISTS %s (%s) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' " % (table_name, column_definitions))
            print query; 
            cur.execute(query)
            
            #query = "set hive.mapred.supports.subdirectories=true; LOAD DATA INPATH '%s' %s INTO TABLE %s" % (hdfs_path, overwrite, table_name)
            query = "LOAD DATA INPATH '%s' %s INTO TABLE %s" % (hdfs_path, overwrite, table_name)
            print query
            cur.execute(query)

        # remove temp file from HDFS
        call(["hdfs", "dfs", "-rm", hdfs_path])

        print 'Wrote %d rows in %.1f secs' % (df.shape[0], time.time()-starttime)

        
 
if __name__ == '__main__':
    conn_str = "{'host':'azuscdbdm01', 'port':10000, 'authMechanism':'PLAIN', 'database':'data_science', 'user':'bigdata', 'password':'B1Gd@t@', }"
    d = {'one' : pd.Series([10., 20., 30.], index=['aa', 'bb', 'cc']),'two' : pd.Series([11., 22., 33.], index=['aa', 'bb', 'cc'])}
    df = pd.DataFrame(d)
    hv = Hive(conn_str)
    table_name = 'delete_me_unit_test_hs2'
    hv.write_frame(df, table_name, if_exists='overwrite')
    query = "SELECT * FROM %s" % (table_name,)
    print query
    df2 = hv.read_frame(query)
    assert(df.shape == df2.shape)
    rows, cols = df.shape
    for i in range(rows):
        for j in range(cols):
            assert(df.iloc[i, j] == df2.iloc[i, j])
