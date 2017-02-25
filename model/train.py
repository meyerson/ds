#!/usr/bin/env python

from tree_tools import dump_tree
import argparse
import datetime
import sys
from data_load import prepare_model_inputs, read_input_defs
from sparsify import sparsify
from time_series_help import TSCV, weight_decay

import pandas as pd
pd.set_option('display.width', 200)
import numpy as np
from pandas.io.sql import read_sql
from sql_help import *
from sql_pandas_help import TableWriter

from sklearn.metrics import f1_score, precision_score

class FminHelper:
    def __init__(self, true, preds):
        self.true = true
        self.preds = preds

    def __call__(self, t_arr):
        # fmin deals with arrays, but we just need one threshold value
        t_preds = self.preds - t_arr[0]
        return 1 - precision_score(np.sign(self.true), np.sign(t_preds))


def optimize_threshold(tr_y, preds):
    fm = FminHelper(tr_y, preds)
    from scipy.optimize import fmin
    x0 = [tr_y.mean()]
    t_opt = fmin(fm, x0, xtol=0.01)
    print 'avg true: %5.5f   avg pred: %5.5f   t_opt: %5.5f  precision: %5.5f' % (
        tr_y.mean(), preds.mean(), t_opt, fm([t_opt]))
    return t_opt


def walk_forward_train_and_test(model_creator, df, inputs, metric, weights=None,
        dense=False, num_models=96, tr_days=None, ts_days=0, model_id=0, extra_index_cols=['date'],
        model_dir='', prediction_persister=None):
    ''' Given the <model_creater> and data in <df>, create training windows (of size
        <tr_days>) and test on subsequent testing windows (of size <ts_days>)

        If ts_days == 0, then just train and output a model to <model_dir>
    '''
    y = df[metric]
    n = len(df)
    available_days = (max(df.index) - min(df.index)).days - 1

    
    if not tr_days:
      tr_days = max(0, available_days - ts_days)
    print tr_days, ts_days
    cv_tuples = TSCV(df.index, tr_days=tr_days, ts_days=ts_days)
    if len(cv_tuples) == 0:
        raise Exception('Insufficient history for training window, data starts on %s but '
            'should start on %s in order to capture %d training and %d testing days.\n\nMax '
            'days available: %d' %
            (min(df.index),
            max(df.index) + pd.tseries.offsets.relativedelta(days=-(tr_days+ts_days)),
            tr_days, ts_days, available_days))


    for tr_rows, ts_rows, tr_beg, ts_beg in cv_tuples.next(1):
        # Out of training history for this cv set?
        if tr_rows == [] or ((ts_beg-tr_beg).days < tr_days):
            break

        # Want test days but none in this window?
        if ts_days > 0 and len(ts_rows) == 0:
            continue

        tr = df.loc[tr_rows]
        ts = df.loc[ts_rows] if ts_rows else []
        trts = df.loc[tr_rows+ts_rows]
        if dense:
            X, X_names = tr[inputs.keys()].values, inputs.keys()
        else:
            if weights is not None:
                wts = weights.loc[tr_rows].values.flatten()
            else:
                wts = None

            # !This code is cheating since it uses test data to guide encoding of train data!
            print 'Feature extraction ... ',
            sys.stdout.flush()
            all_raw = all([_['type'] == 'raw' for _ in inputs.values()])
            if all_raw:
                X_names = inputs.keys()
                X = trts[X_names].values
            else:
                X, X_names = sparsify(trts, inputs, verbose=False)
                X = X.todense()
            print 'done.'
            sys.stdout.flush()
            tr_X = X[:len(tr)]
            tr_y = y.loc[tr_rows]
            ts_X = X[-len(ts):]
            ts_y = y.loc[ts_rows] if ts_rows else []

        printed = False
        model_num = 0
        while model_num < num_models:
            clf = model_creator()
            if not printed:
                print clf
                print 'Training model',
                printed = True
            print '.',
            sys.stdout.flush()
            clf.fit(tr_X, tr_y, sample_weight=wts)
            print 'done.'
            sys.stdout.flush()
            model_num += clf.n_estimators

            # If no testing window, serialize model.
            if ts_days == 0:
                #fn_base = '%d.%d.%s_%s.mod' % (model_id, model_num, tr_beg.date(), ts_beg.date())
                fn_base = '%d' % (model_id,)
                dump_tree(clf, X_names, fn_base, model_dir, fop='a')

            # Otherwise, write out predictions.
            elif prediction_persister:
                if hasattr(clf, 'predict_proba'):
                    tr_preds = clf.predict_proba(tr_X)
                    ts_preds = clf.predict_proba(ts_X)
                else:
                    tr_preds = clf.predict(tr_X)
                    ts_preds = clf.predict(ts_X)
                tr_preds = tr_preds[:,-1] # only using last prediction from each sample
                ts_preds = ts_preds[:,-1] # only using last prediction from each sample
                trts['raw_predicted_'+metric] = np.append(tr_preds, ts_preds, axis=0)

                # optimize threshold using only training predictions
                t_opt = optimize_threshold(tr_y, tr_preds)

                # abs shouldn't be necessary, but otherwise negative zero occurs.
                trts['predicted_'+metric] = abs(np.ceil(trts['raw_predicted_'+metric] - t_opt))

                trts['test_set']  = [0 if i < len(tr) else 1 for i in range(len(tr)+len(ts))]
                trts['date'] = trts.index.date
                trts['model_id'] = model_id
                persist_cols = extra_index_cols + ['date','model_id','test_set',metric,'predicted_'+metric,'raw_predicted_'+metric]
                prediction_persister(trts[persist_cols], index_cols=['date']+extra_index_cols)

                # now compute some measures of goodness & other statistics for test-set performance:
                preds = clf.predict(ts_X)

                from roc_curve import roc_curve
                curve = roc_curve(pd.Series(ts_y), pd.Series(preds))

                pny = zip(preds, ts_y)
                j = int(len(pny)/4.0) # top 25%
                srt = np.array(sorted(pny, key=lambda p_t: (p_t[0], -p_t[1]))) # worst case sorting of ties
                top_pct = srt[-j:,1]
                bot_pct = srt[:j,1]

                mean_tr_y = np.mean(tr_y)
                mean_ts_y = np.mean(ts_y)
                mean_top = np.mean(top_pct)
                mean_bot = np.mean(bot_pct)
                ratio = mean_bot/mean_top
                print
                print "%10s %10s %-13s %-13s %8s %8s %8s %8s %8s %5s %8s" % (
                    'beginDate', 'endDate', 'trn_N:trn_P', 'tst_N:tst_P', 'trn_mean',
                    'tst_mean','mean_bot','mean_top','ratio','sum_top','ROC auc')
                print "%10s %10s %6d:%-6d %6d:%-6d %8.4f %8.4f %8.4f %8.4f %8.2f %8.2f %8.2f" % (
                    tr_beg.date(), ts_beg.date(), len(tr_y), sum(tr_y), len(pny), sum(ts_y), mean_tr_y, mean_ts_y, mean_bot,
                    mean_top, ratio, np.sum(top_pct), roc_auc)
                print '--------------------------------------------------------------------------------'


class RegressorFactory:
    ''' A simple class that produces Regressor instances that are ready to train.
    '''
    def __init__(self, min_weight, explainable):
        self.min_weight = min_weight
        self.explainable = explainable

    def __call__(self):
       if self.explainable:
          max_depth=3
          n_estimators=1
          max_features=1.0
          n_jobs = 1
       else:
          max_depth=None
          max_features=0.2
          n_estimators=96

       n_jobs = min(3,n_estimators)
       from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
       return RandomForestClassifier(bootstrap=(not self.explainable), n_estimators=n_estimators, max_features=max_features,
                                    min_samples_leaf=self.min_weight, min_samples_split=self.min_weight, max_depth=max_depth, n_jobs=n_jobs)


def main(args):
    ''' Reads training rows from a database or csv file; trains and optionally tests
        models against those rows.

        If <ts_days> is zero (its default), train a single model and output its trained
        parameters.

        If <ts_days> is greater than zero, train models using a sliding window and test
        against subsequent data, outputting the predictions on the test data to a database.
    '''
    params = read_input_defs(args.inputs)
    usecols = params['inputs'].keys()
    metric = params['metric'].lower()
    groupby = params.get('groupby',[])
    date_col = params['date_col']
    multi_index_cols = groupby + [date_col]

    
    custom_weight = params.get('custom_weight')
    
    weight = [params.get('weight')]
    if weight == [None]:
        weight = []

    in_flav = params['in_flav']
    
    if in_flav != 'csv':
        null_to_zero_metric = metric #"case when %s !='' then %s else 0 end as %s" % (metric, metric, metric)
        named_cols = [k for (k,v) in params['inputs'].iteritems() if not v.get('query')]
        query_cols = ["%s AS %s" % (v.get('query'),k) for (k,v) in params['inputs'].iteritems() if v.get('query')]
        cols = list(set(multi_index_cols + named_cols + query_cols + weight))
        query = 'select %s, %s %s ' % (null_to_zero_metric, ','.join(cols), params['query'])
        if args.n:
            query += 'limit ' + str(args.n)

    if in_flav == 'postgresql':
        cnx = db_connect_pg(params['in_conn'], params.get('schema', ''))
        query += ';'
        df = read_sql(query, cnx, parse_dates=[date_col])
    elif in_flav == 'sqlite':
        from os import environ
        # TMPDIR is needed because sometimes sqlite needs a big temp space:
        environ['TMPDIR'] = params.get('TMPDIR', '/tmp')
        cnx = db_connect_sqlite(params['in_conn'])
        query += ';'
        df = read_sql(query, cnx, parse_dates=[date_col])
    elif in_flav == 'hive':
        import hs2
        print 'max_obstemp_b2_30_r' in query
        
        text_file = open("train_query.debug", "w")
        text_file.write(query)
        text_file.close()
        #exit()
        #df = hs2.Hive(params['in_conn']).read_frame(query+' limit 1000')
        df = hs2.Hive(params['in_conn']).read_frame(query, parse_dates=[date_col])
        #print df.shape
        #exit()
    elif in_flav == 'csv':
        df, input_features = prepare_model_inputs(params['in_conn'], nrows=args.n, de_dup_cols=args.dup, header=0,
                                                comp=args.comp, usecols=list(set(usecols + multi_index_cols + [metric])))

        if custom_weight:
            df[weight[0]] = (eval(custom_weight))*params.get('min_weight',1000)
            #df[weight[0]] = (100.0*df[metric]+1.0)*params.get('min_weight',1000)
       
    elif in_flav == 'joblib':
        from sklearn.externals import joblib
        df = joblib.load(params['joblib_dump_dir'])
        input_features = df.columns
    else:
        raise Exception("in_flav unknown: " + in_flav)

    if in_flav != 'joblib' and params.get('joblib_dump_dir'):
        from sklearn.externals import joblib
        joblib.dump(df, params['joblib_dump_dir'])

    format = "%Y-%m-%d"
    df[date_col] = pd.to_datetime(df[date_col], format=format)
    df.set_index(date_col, inplace=True)
    input_features = df.columns - weight
    
    
    # ensure that query returned what we needed:
    missing = [col for col in usecols if col not in df.columns]
    if missing:
        print 'Missing columns:', '\n'.join(missing), ' ... aborting.'
        exit()

    print df[metric].describe()

    df['decays'] = weight_decay(df.index)

    # drop rows with missing output
    df = df[pd.notnull(df[metric])]

    # drop rows with missing required values
    df.dropna(subset=params.get('required'), how='any', inplace=True)

    # index will have been removed, and possibly metric, so just limit to existing input_features
    input_defs = {k:v for k,v in params['inputs'].iteritems() if k in input_features}

    # where to persist model information
    model_output_flav = params['model_output_flav']
    model_output_conn_str = params['model_output_conn_str']
    model_output_cnx = db_connect(model_output_conn_str, model_output_flav, params.get('schema'))
    model_table_name = 'models'

    # does the <model_table_name> table exist?
    if model_output_flav == 'hive':
        model_db = eval(model_output_conn_str)['database'] # database is a required part of the connection string
        #query = "CREATE TABLE IF NOT EXISTS %s.%s (id STRING, run_date STRING)" % (model_db, model_table_name)
        query = "SHOW TABLES IN %s LIKE '%s'"  % (model_db, model_table_name)
    elif model_output_flav == 'sqlite':
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name='%s'" % (model_table_name,);
    exists = execute(query, model_output_cnx).fetchall()

    if not exists or exists == [None] or exists == '':
        model_id = 0
    else: # table exists:
        new_id_query = 'SELECT max(id) FROM ' + model_table_name
        max_model_id = execute(new_id_query, model_output_cnx).fetchone()
        if max_model_id is None or max_model_id == [None] or max_model_id == '':
            model_id = 0
        else:
            model_id = int(max_model_id[0]) + 1

    # ideally, record exact code revisions of this run so that it can be perfectly recreated:
    #from subprocess import call
    #git_hash = call(["git", "rev-parse", "HEAD"])
    # ^^^ this won't quite work right unless it's run from the correct directories.

    # persist

    tw = TableWriter(model_output_conn_str, model_table_name, flav=model_output_flav, if_exists='append')
    print tw.flav
    print 'conn_str:', tw.conn_str
    # df = pd.DataFrame(data={'id':model_id, 'run_date':str(datetime.date.today()), 'input_dict':params}, index=np.arange(1))
    # print df.shape
    # print params.keys()
    # exit()
    tw(pd.DataFrame(data={'id':model_id, 'run_date':str(datetime.date.today()), 'input_dict':str(params)}, index=np.arange(1)))

    tw = TableWriter(model_output_conn_str, 'training_preds', flav=model_output_flav, if_exists='append')

    # Where to store the trained models; required for both training (where to output model) and scoring (from where to load model)
    model_binaries_dir = params['model_binaries_dir']
    
    walk_forward_train_and_test(RegressorFactory(params.get('min_weight',1000), args.fast), df, input_defs, metric,
        df[weight]*df.decays if weight else df.decays, tr_days=args.tr_days, ts_days=args.ts_days,
        num_models=args.num_models, model_id=model_id, extra_index_cols=groupby, model_dir=model_binaries_dir, prediction_persister=tw)
    # walk_forward_train_and_test(RegressorFactory(params.get('min_weight',1000), args.fast), df, input_defs, metric, # 
    #     df[weight]*df.decays, tr_days=args.tr_days, ts_days=args.ts_days,
    #     num_models=args.num_models, model_id=model_id, extra_index_cols=groupby, model_dir=model_binaries_dir, prediction_persister=tw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='read csv, write distributions to db',argument_default=[])
    parser.add_argument('inputs', help='input definitions')
    parser.add_argument('-comp', default='gzip', help='is file gzipped')
    parser.add_argument('-n', type=int, default=None, help='nrows to load')
    parser.add_argument('-tr_days', type=int, help='training window days')
    parser.add_argument('-ts_days', type=int, default=0, help='testing window days, if > 0 then do a walkforward analysis')
    parser.add_argument('-dup', default=[], help='which columns to consider when de-duping')
    parser.add_argument('-fast', action='store_true', help='fast=shallow models')
    parser.add_argument('-num_models', type=int, default=96, help='number of random models to generate per time window')

    try:
        args = parser.parse_args()
    except Exception as e:
        print str(e)
        print parser.print_help()
        exit(0)
    print args

    #import cProfile
    #cProfile.run("main", __file__ + '.profile')
    sys.exit(main(args))
