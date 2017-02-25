#!/usr/bin/env python

import argparse
import numpy as np
np.set_printoptions(threshold=100000, linewidth=200, precision=2, suppress=True, nanstr="NA")

import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 500)
from pandas.io.sql import read_sql, execute
from pandas.io.sql import write_frame
from sql_help import *
from sql_pandas_help import TableWriter
from data_load import *

def add_label_level(df, name):
    ''' given a DataFrame and <name>, add <name> as a level "above" existing column labels '''
    if name in df.columns:
        raise ValueError('dataframe already contains column with name "%s", would overwrite all values in that column with the string "%"' % (name, name))
    df[name] = name
    df = df.set_index(name, append=True, inplace=False).unstack().swaplevel(0,1,axis=1)
    return df


MAX_RATIO=999
def compute_ewma_and_td(pvt, name, span, diff_types):
    ''' <name> names a column in pvt; compute emas and diffs on that column and return a df of ema&diff '''
    print name,name in pvt
    srs = pvt[name]
    ema = pd.ewma(srs, span=span,ignore_na=True)
    ema_name = '%s_e%d' % (name, span)

    diffs = []
    for dt in diff_types:
        if dt == 'raw':
            diff = srs - ema.shift(1)
            diff_prefix = 'd'
        else:
            # Ratio: [-MAX_RATIO, MAX_RATIO]
            diff = np.clip(srs / ema.shift(1), -MAX_RATIO, MAX_RATIO)
            diff_prefix = 'p'

        diff_name = '%s_e%d_%s' % (name, span, diff_prefix)
        diff = add_label_level(diff, diff_name)
        diffs.append(diff)

    ema = add_label_level(ema, ema_name) # do this AFTER computing diffs, otherwise levels get stacked
    for diff in diffs:
        ema = ema.join(diff)
    return ema

def compute_ewma_diff(df, name, span1, span2, diff_type):
    ''' Given EWMA's of two different spans, compute a difference between them '''
    ema1_name = '%s_e%d' % (name, span1)
    ema2_name = '%s_e%d' % (name, span2)
    d_name = '%s_b%d_%d' % (name, span1, span2)
    if diff_type == 'raw':
        d_name = d_name + '_d'
        diff = df[ema1_name] - df[ema2_name]
    else:
        d_name = d_name + '_r'
        diff = (1+df[ema1_name])/(1+df[ema2_name])
    return add_label_level(diff, d_name)


def main(args):
    ''' Given an input file containing the names of input fields and desired EWMA spans,
        read raw input fields from a database, compute the relevant EWMA spans for each
        input field and differences between EWMAs of differing spans.

        Persist these newly created features along with the raw input field to a database or file.
    '''
    params = read_input_defs(args.config)
    inputs = params['inputs'] # inputs is dict of [source_field_name, dict(params)]
    passthrough = params['passthrough'] # passthrough is list of field names to ingest and output, but not compute EWMAs on
    spans = sorted(list(set(params['spans']))) # use set to remove duplicates
    groupby = params.get('groupby')
    groupas = params.get('groupas')
    date_col = params['date_col']
    prefix = params.get('prefix', '')

    in_flav = params['in_flav']
    multi_index_cols = [date_col] + groupby
    uncomputed_cols = [k for (k,v) in params['inputs'].iteritems() if not v.get('computed') == True]
    # rename columns when 'as' is specified
    # not used since "inputs.keys()" gets broken: as_names = dict([(k,v.get('as')) for (k,v) in params['inputs'].iteritems() if v.get('as') is not None])

    query_cols = set(multi_index_cols + uncomputed_cols + passthrough)
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
    elif in_flav == 'csv':
        df, input_features = prepare_model_inputs(params['filename'], nrows=args.n, index_col=multi_index_cols,
            header=0, usecols=inputs.keys())
    else:
        raise Exception("in_flav unknown" + in_flav)

    print 'succeeded.'

    #df.set_index(multi_index_cols, inplace=True)
    df.replace(r'', np.nan, regex=True, inplace=True)
    df.replace('NULL', np.nan, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # convert nan_to_zero columns from nan to 0
    nan_to_zero_cols = [k for (k,v) in params['inputs'].iteritems() if v.get('nan_to_zero') == True]
    for col in nan_to_zero_cols:
      df[col].fillna(0, inplace=True)

    # now that "NULL" strings, -inf's, nan have been dealt with, convert all columns to numbers (some still might be strings)
    df = df.convert_objects(convert_numeric=True)
    print df.describe().T.sort(['count','mean'])

    nan_cols = df.columns[(df.isnull().any(axis=0))]
    #df.columns[(df.isnull().any(axis=0))]
    if len(nan_cols) > 0:
      print nan_cols
      print '\n\n\n  WARNING! NAN columns\n\n\n'
      #print df[nan_cols].describe().T.sort(['count','mean'])
      print df[nan_cols].isnull().sum()
      print 'WARNING: NOT ! DROPPING THE FOLLOWING COLUMNS - due to the presence of NaNs'
      print 'will be ignoring nans in computing ewmas'
      print '\n'.join(sorted(nan_cols))
      
      ##df.fillna(-99, inplace=True)
      df.dropna(how='all', inplace=True, axis=1)

      #in nan_cols but not in df.columns
      nan_cols = set(nan_cols).difference(df.columns)
      for col in nan_cols:
          del(inputs[col])

    if groupas:
        renamed = dict(zip(groupby, groupas))
        df.rename(columns=renamed, inplace=True)
        groupby = groupas

    print len(df.columns)
    pvt = pd.pivot_table(df,index=date_col,columns=groupby[0])
    del df
    #set(df.columns).difference((list(pvt.columns.levels[0]))

    #print pvt.columns
    #exit()
    print "Computing individual EWMAs and today's difference from those EWMAs."
    emas = []
    for span in spans:
      for (col,dct) in inputs.iteritems(): # inputs is dict of [source_field_name, dict(params)]
        emas.append(compute_ewma_and_td(pvt, col, span, dct['types'])) # returns list of series
    jnd = emas[0]
    for _ in emas[1:]:
      jnd = jnd.merge(_, left_index=True, right_index=True)
    del emas

    print  'max_obstemp_b2_30_r' in jnd.columns
    print  'waterproduction_b2_30_r' in jnd.columns
    print  'max_obstemp' in jnd.columns
    #exit()

    print "Computing diffs between EWMAs of different spans."
    ema_diffs = []
    for (col,dct) in inputs.iteritems(): # inputs is dict of [source_field_name, dict(params)]
      for i in range(0,len(spans)-1):
         for j in range(i+1, len(spans)):
            for diff_type in dct['types']:
                ema_diffs.append(compute_ewma_diff(jnd, col, spans[i], spans[j], diff_type))
                
    print  'max_obstemp_b2_30_r' in jnd.columns
    print  'waterproduction_b2_30_r' in jnd.columns
    print  'max_obstemp' in jnd.columns

    print "Merging computations by index."
    for _ in ema_diffs:
      jnd = jnd.merge(_, left_index=True, right_index=True)
    del ema_diffs
    print  'max_obstemp','max_obstemp' in jnd.columns
    
    print "Moving the groupby's back into the row index (and sort by groupby, date)."
    jnd = jnd.reindex_axis(sorted(jnd.columns), axis=1).stack().swaplevel(0,1,axis=0).sortlevel()

    print "Joining raw inputs with new features table."
    pvt = pvt.reindex_axis(sorted(pvt.columns), axis=1).stack().swaplevel(0,1,axis=0).sortlevel()
    jnd = pvt.merge(jnd, left_index=True, right_index=True)
    del pvt
    print  'max_obstemp','max_obstemp' in jnd.columns
    # reset index just before writing out:
    jnd.reset_index(inplace=True)

    ### output to output_location (or csv)
    out_flav = params['out_flav']
    if out_flav in ['sqlite', 'hive']:
        # 'sqlite3 cannot handle pandas datetime so use strings'
        jnd[date_col] = jnd[date_col].apply(lambda x: x.date()) # datetime can't be stored so convert to date & it will store as strings

    if prefix:
        new_names = dict([(k,prefix+k) for k in jnd.columns])
        jnd.rename(columns = new_names, inplace=True)

    output_location = params.get('output_location')
    if out_flav == 'csv':
        jnd.to_csv(path_or_buf=output_location, header=False, index=False)
    else:

        print  'max_obstemp_b2_30_r' in jnd.columns
        print  'waterproduction_b2_30_r' in jnd.columns
        print  'max_obstemp' in jnd.columns
        jnd.fillna(-99,inplace=True)
        tw = TableWriter(params['out_conn_str'], output_location, flav=out_flav, if_exists='overwrite')
        tw(jnd)

        index_query = 'create index '+output_location+'_index ON ' + output_location+' (' + ','.join([prefix+_ for _ in multi_index_cols]) + ');'
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

    #import cProfile
    #cProfile.run("main(args)", __file__ + '.profile')
    #import sys
    #sys.exit(main(args))
    main(args)
