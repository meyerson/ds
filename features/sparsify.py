#ex!/usr/bin/env python

import pandas as pd
import numpy as np
np.set_printoptions(threshold=10000, linewidth=200)

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
DEFAULT_BANNED = list(ENGLISH_STOP_WORDS)
#['for','of','or','and','the','a']

def qcut(x, q_bins, labels=None, retbins=False, precision=3):
    ''' custom implementation of qcut that doesn't toss duplicates or blow up on nan
    '''
    import pandas.core.common as com
    import pandas.core.algorithms as algos
    import pandas.tools.tile as tile

    if com.is_integer(q_bins):
        quantiles = np.linspace(0, 1, q_bins + 1)
    else:
        quantiles = q_bins
    bins = np.unique(algos.quantile(x, quantiles))
    if len(bins) < 3: # bins contain first left-most endpoint and last right-most endpoint
        from scipy.stats import itemfreq
        # what % of samples fall into each bin?
        breaks = 1.0 * np.cumsum([count for (key,count) in itemfreq(x)]) / len(x)
        dist_from_mid = np.abs(breaks - 0.5)
        most_mid_break = breaks[dist_from_mid.argmin()]
        quantiles = np.array((0, most_mid_break, 1))
        bins = np.unique(algos.quantile(x, quantiles))
    return tile._bins_to_cuts(x, bins, labels=labels, retbins=retbins,
                             precision=precision, include_lowest=True)


def sparsify(df, inputs, q_bins=3, output_features_file=None, verbose=True):
    ''' Given a DataFrame df, which could contain text strings as well as integer "code" fields,
        create an array (i.e. matrix) of numbers by binarizing the text ngrams and codes as well
        as quantizing numbers if desired.
    '''
    import scipy.sparse as sps
    import itertools
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction import DictVectorizer

    keep = []
    mats = []
    feature_names = []
    bins = []

    N = len(df)*1.0

    names_lookup = {2:['low','high'], 3:['low','medium','high']}
    
    # for col_name, col_info in inputs.iteritems():
    #     print 'col_name:', col_name
    # exit()
    for col_name, col_info in inputs.iteritems():
      
        col_type = col_info.get('type')

        NN = sum(df[col_name].notnull())
        vc = df[col_name].value_counts()
        uniq = list(vc.index.values) # sorted in decreasing order of prevalence
        U = len(uniq)
        if verbose:
            print '%35s %7d %8d %10s' % (col_name, NN, U, col_type),

        if U < 2 and (NN < 0.001*N or NN > 0.999*N):
             if verbose:
                print 'verbose'
             continue

        if col_type in ['string', 'ngram']:
            #if df[col_name].dtype is np.dtype('object'): # ngram unique values after tokenizing:
            if col_type == 'string':
               ngram_range = (1,1)
               token_pattern = '.*' # match entire field
            else: # ngram
               ngram_range = (1,3)
               token_pattern = '[^ ]+' # anything separated by spaces
            ngram_range   = col_info.get('ngram_range',ngram_range) # feature def can override
            token_pattern = col_info.get('token_pattern', token_pattern)
            banned = col_info.get('banned',[]) + DEFAULT_BANNED
            vctrizr = CountVectorizer(min_df=max(2,int(0.00015*N)), ngram_range=ngram_range,
                                      stop_words=banned, token_pattern=token_pattern)
            data = df[col_name].fillna('')  #apply(lambda x: "%s" % (int(x) if not np.isnan(x) else "" ,)) # gymnastics to prevent integers from being converted to floats
            X = vctrizr.fit_transform(data) # returns CSC format
            uniq = vctrizr.get_feature_names()
            names = [col_name+'__'+str(_) for _ in uniq]
            U = len(names)
            feature_names.extend(names)
        elif col_type == 'unique numbers' or col_type == 'quantile_cols':
            if col_type == 'unique numbers':
                vocab = dict(itertools.izip(uniq,range(1,U+1)))  # mapping from raw value to integer label (which will become column number...)
                val_list = [vocab.get(_,0) for _ in df[col_name]] # for each raw value, lookup its integer label
                names = [col_name+'__'+str(_) for _ in ['nan']+["%s" % (int(u) if not np.isnan(u) else '',) for u in uniq]] # gymnastics to prevent integers from being converted to floats
            else:
                qts = qcut(df[col_name], q_bins) # ranges that each element falls into
                #val_list = len(qts.levels)- qts.labels # assigned levels for each element [3, 1, 0, 0, 2, ...]   0 means 'nan'
                val_list = 1 + qts.labels # assigned levels for each element [3, 1, 0, 0, 2, ...]   0 means 'nan'
                #names = [col_name+'__'+str(_) for _ in ['nan'] + list(qts.levels.values)]
                names = [col_name+'__'+str(_) for _ in ['nan']+names_lookup[min(len(qts.levels),q_bins)]]

            store_dtype = np.uint8
            if U > 256:
              store_dtype = np.uint16
            if U >= 65536:
              raise ValueError("Too many unique values to encode as unique columns, column_name: %s  #unique values:%d" % (col_name, U))

            ''' create coo_matrix:
            coo_matrix((data, (i, j)), [shape=(M, N)])
            to construct from three arrays:
               data[:] the entries of the matrix, in any order
               i[:] the row indices of the matrix entries
               j[:] the column indices of the matrix entries
               Where A[i[k], j[k]] = data[k]. When shape is not specified, it is inferred from the index arrays '''
            X = sps.coo_matrix((np.ones(len(val_list), dtype=store_dtype), (range(len(val_list)), val_list))).tocsc() # (1's tall, # labels wide, data)
            feature_names.extend(names)
        elif col_type == 'quantiles':
            qts = qcut(df[col_name], q_bins) # ranges that each element falls into
            val_list = len(qts.levels)- qts.labels # assigned levels for each element [3, 1, 0, 0, 2, ...]   0 means 'nan'
            X = sps.csc_matrix(np.array(val_list)).T  # csc_matrix() creates a row vector initially; must be transposed to get column vector
            feature_names.extend([col_name])
            bins.extend([None for _ in qts.levels]) # not sure what this is for...
        elif col_type == 'raw':
            # not sure what the point of creating sparse array is...: X = sps.csc_matrix(np.array(df[col_name].values)).T
            X = df[col_name].values.T
            feature_names.extend([col_name])
            
        keep.append(col_name)
        mats.append(X)

        # _.shape[1] throws an invalid index for raw coltypes
        assert(len(feature_names) == np.sum([_.shape[1] for _ in mats]))
        if verbose:
            print '%15s %-15s %20s %20s' % (col_type, X.shape, str(min(uniq)), str(max(uniq)))

    res = sps.hstack(mats).tocsc() # I think hstack returns a non-csc format
    print res.shape

    if output_features_file:
        with open(output_features_file,'w') as f:
            print >> f, '\n'.join("%8s %s" % (s,nm) for s,nm in
                zip(res.sum(axis=0).tolist()[0],feature_names))
        f.close()

    return res, feature_names


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='given a file describing the sparsification parameters, sparsify and write results to DB')
    parser.add_argument('config', help='name of (python) configuration file')
    parser.add_argument('-n', type=int, default=None, help='nrows to load')
    try:
      args = parser.parse_args()
    except Exception as e:
       print str(e)
       print parser.print_help()
       exit(0)
    print args


if __name__ == '__main__':
    main()

