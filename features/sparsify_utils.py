#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
pd.set_option('display.width', 200)
import sys

from pandas.io.sql import read_sql, execute
from sql import write_frame
from sql_help import *
from sql_pandas_help import TableWriter
from data_load import *

def main(args):
    ''' Given an input file containing the names of input fields and desired EWMA spans,
        read raw input fields from a database, compute the relevant EWMA spans for each
        input field and differences between EWMAs of differing spans.

        Persist these newly created features along with the raw input field to a database or file.
    '''
    params = read_input_defs(args.config)
    inputs = params['inputs'] # inputs is dict of [source_field_name, dict(params)]
    as_cols = ["%s AS %s" % (k,v.get('as',k),) for (k,v) in inputs.iteritems()]
    inputs = dict([(v.get('as',k).lower(),v) for (k,v) in inputs.iteritems()])

    date_col = params['date_col']
    groupby = params.get('groupby',[])
    prefix = params.get('prefix', '')

    in_flav = params['in_flav']
    multi_index_cols = [date_col] + groupby

    query_cols = set(multi_index_cols + as_cols)
    query = 'SELECT %s %s ' % (','.join(query_cols), params['raw_inputs_table'],  )

    if args.n:
      query += ' LIMIT %s' % (args.n,)
    print query

    if in_flav == 'postgresql':
        cnx = db_connect_pg(params['in_conn'], params.get('schema', ''))
        query += ';'
        df = read_sql(query, cnx, parse_dates=[date_col])
    elif in_flav == 'sqlite':
        cnx = db_connect_sqlite(params['in_conn'])
        query += ';'
        df = read_sql(query, cnx, parse_dates=[date_col])
    elif in_flav == 'hive':
        import hs2
        df = hs2.Hive(params['in_conn']).read_frame(query, parse_dates=[date_col])
        print 'last date seen:',df.recorddate.max()
    elif in_flav == 'csv':
        df, input_features = prepare_model_inputs(params['filename'], nrows=args.n, index_col=multi_index_cols,
            header=0, usecols=params['inputs'].keys())
    else:
        raise Exception("in_flav unknown" + in_flav)

    df.set_index(multi_index_cols, inplace=True)
    df.replace(r'', np.nan, regex=True, inplace=True)
    df.replace('NULL', np.nan, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # convert nan_to_zero columns from nan to 0
    nan_to_zero_cols = [k for (k,v) in inputs.iteritems() if v.get('nan_to_zero') == True]
    for col in nan_to_zero_cols:
      df[col].fillna(0, inplace=True)

    from sparsify import sparsify
    print df.shape,len(inputs),type(inputs)
    #print df.columns

    X, X_names = sparsify(df, inputs)
    df = pd.DataFrame(data=X.todense(), columns=X_names, index=df.index)

    # reset index just before writing out:
    df.reset_index(inplace=True)

    ### output to output_location (or csv)
    out_flav = params['out_flav']
    if out_flav not in ['postgresql']:
        # 'sqlite3 cannot store date/datetime/timestamp/etc so use strings'
        df[date_col] = df[date_col].apply(lambda x: str(x.date()))

    if prefix:
        new_names = dict([(k,prefix+k) for k in df.columns])
        df.rename(columns = new_names, inplace=True)

    output_location = params.get('output_location')
    print 'Writing to %s:%s shape=%d' % (out_flav, output_location, len(df.shape))
    if out_flav == 'csv':
        df.to_csv(path_or_buf=output_location, header=False, index=False)
    else:
        print 'last date seen later:',df.recorddate.max()
        tw = TableWriter(params['out_conn_str'], output_location, flav=out_flav, if_exists='overwrite')
        tw(df)

        index_query = 'create index '+output_location+'_index ON ' + output_location+' (' + ','.join([prefix+_ for _ in multi_index_cols]) + ')'
        print 'NO INDEX ON HIVE: ' + index_query
        #execute(index_query, out_cnx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='read db, generate new table of emas in db',argument_default=[])
    parser.add_argument('config', help='name of (python) configuration file')
    parser.add_argument('-n', type=int, default=None, help='nrows to load')

    try:
      args = parser.parse_args()
    except Exception as e:
       print str(e)
       print parser.print_help()
       exit(0)
    print args

    main(args)
