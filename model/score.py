#!/usr/bin/env python

import argparse
import sys
from data_load import *
from sparsify import sparsify

import pandas as pd
import numpy as np
from pandas.io.sql import read_sql
from sql_help import *
from sql_pandas_help import TableWriter

def main(args):
    ''' Generate and persist predictions on "scoring" rows in database, using a previously
        trained model.

        Read <input_dict> by looking up them up in a database (using model_id as a key)
        then allow the passed-in args.inputs file to override what was found in the database.
    '''
    model_id = args.model_id
    model_base_name = 'model_binary.' + str(model_id) # 'model_binary.' comes from tree_tools.py
    params = read_input_defs(args.inputs)

    from pandas.io.sql import execute
    model_table_name = 'models'
    in_flav = params['in_flav']
    in_conn_str = params['in_conn_str']
    skip_sparse = args.skip_sparse
    
    in_cnx = db_connect(in_conn_str, in_flav, params.get('schema'))
    query = 'SELECT input_dict FROM %s WHERE id=%d' % (model_table_name, model_id,)
    #print in_flav,in_conn_str
    #print('query:',query[0:-1],in_cnx)
    #print(dir(in_cnx))
    print query
    trained_model_input_dict = eval(execute(query, in_cnx).fetchone()[0])
    print(trained_model_input_dict.keys())
    #print(trained_model_input_dict['joblib_dump_dir'])
    
    if type(trained_model_input_dict) == type(''):
        trained_model_input_dict = dict(trained_model_input_dict)
    metric =  trained_model_input_dict['metric']
    usecols = trained_model_input_dict['inputs'].keys()
    groupby = trained_model_input_dict.get('groupby',[])
    date_col = trained_model_input_dict['date_col']
    multi_index_cols = groupby + [date_col]

    data_to_score_flav = trained_model_input_dict['in_flav']
    print('data_to_score_flav: ',   data_to_score_flav)
    print(date_col)
    

    if data_to_score_flav in ['postgresql', 'sqlite', 'hs2','hive']:
        named_cols = [k for (k,v) in trained_model_input_dict['inputs'].iteritems() if not v.get('query')]
        query_cols = ["%s AS %s" % (v.get('query'),k) for (k,v) in trained_model_input_dict['inputs'].iteritems() if v.get('query')]
        cols = list(set(multi_index_cols + named_cols + query_cols))
        query = 'select %s, %s %s ' % (metric, ','.join(cols), trained_model_input_dict['query'])
        if args.n:
            query += 'limit ' + str(args.n)
        print(query)

    if data_to_score_flav == 'postgresql':
        cnx = db_connect_pg(in_cnx, trained_model_input_dict.get('schema', ''))
        query += ';'
        df = read_sql(query, cnx, parse_dates=[date_col])
    elif data_to_score_flav == 'sqlite':
        from os import environ
        environ['TMPDIR'] = trained_model_input_dict.get('TMPDIR', '/tmp') # this is needed because sometimes sqlite needs a big temp space
        cnx = db_connect_sqlite(in_conn_str)
        query += ';'
        df = read_sql(query, cnx, parse_dates=[date_col])
    elif data_to_score_flav == 'hs2'or data_to_score_flav == 'hive':
        import hs2
        df = hs2.Hive(in_conn_str).read_frame(query, parse_dates=[date_col])

    elif data_to_score_flav== 'joblib':
        from sklearn.externals import joblib
        df = joblib.load(trained_model_input_dict['joblib_dump_dir'])
        #input_features = df.columns
    elif data_to_score_flav=='csv':
        sep = '\t'
        datafile_csv = trained_model_input_dict['in_conn'].replace('csv.gz','txt')
        df = pd.read_csv(open(datafile_csv, 'r'),index_col=False,header=0 ,
                         sep=sep, parse_dates=['recoverable_xy_pairs.'+date_col])
        df.columns = [x.replace('recoverable_xy_pairs.','') for x in df.columns]
    else:
        raise Exception("data_to_score_flav unknown" + data_to_score_flav)

    df = df.set_index([date_col])
    input_features = df.columns
    #print sorted(input_features)

    # ensure that query returned what we needed:
    missing = [col for col in usecols if col not in df.columns]
    if missing:
        print 'Missing columns:', '\n'.join(missing), ' ... aborting.'
        exit()

    dropped = df.dropna(how='any', inplace=True)
    if dropped:
        print 'DROPPING ROWS FROM SCORING: ', dropped

    print df[metric].describe()

    # index will have been removed, and possibly metric, so just limit to existing input_features
    input_defs = {k:v for k,v in trained_model_input_dict['inputs'].iteritems() if k in input_features}
    print 'debug info'
    print df.shape,len(input_defs),type(input_defs)
    #print df.columns

    if skip_sparse:
        X,X_names = df[input_defs.keys()].values,input_defs.keys()
    else:
        X, X_names = sparsify(df, input_defs)
    # this code is necessary when the index is not date only:
    #format='%Y%m%d%H'
    #df.index = pd.to_datetime(df.index.astype(str), format=trained_model_input_dict.get('date_format',format))

    # ingest trained model:
    model_binaries_dir = trained_model_input_dict['model_binaries_dir'] # required for both training (where to output model) and scoring (from where to load model)
    #model_base_name = trained_model_input_dict['model_base_name'] # model base name is auto-generated during training
    model_base_name = params['model_base_name']
    from sklearn.externals import joblib
    import os
    clf = joblib.load(os.path.join(model_binaries_dir, model_base_name))

    if skip_sparse:
        df['predicted_'+metric] = clf.predict(X)
    else:
        df['predicted_'+metric] = clf.predict(X.todense())

      
    df['model_id'] = model_id
    cols_to_write = ['model_id'] + groupby + [metric,'predicted_'+metric]
    df_write = df[cols_to_write].reset_index()


    # where to persist predictions:
    out_flav = params['out_flav']
    out_conn_str = params['out_conn_str']
    print out_conn_str
    out_cnx = db_connect(out_conn_str, out_flav, params.get('schema'))
    tw = TableWriter(out_cnx, 'scoring_preds', flav=out_flav, if_exists='replace')
    if out_flav in ['sqlite', 'hive']:
        # 'sqlite3 & hive cannot handle pandas datetime so use strings'
        df_write[date_col] = df_write[date_col].apply(lambda x: x.date()) # datetime can't be stored so convert to date & it will store as strings
    tw(df_write)


if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='read csv, write distributions to db',argument_default=[])
   parser.add_argument('model_id', type=int, help='id of trained model stored in input db')
   parser.add_argument('inputs', help='input definitions')
   parser.add_argument('-skip_sparse',action="store_true", default=False)
   parser.add_argument('-n', type=int, default=None, help='nrows to score')
   

   try:
      args = parser.parse_args()
   except Exception as e:
       print str(e)
       print parser.print_help()
       exit(0)
   print args

   main(args)
