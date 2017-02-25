import pandas as pd
import time

def read_input_defs(filename):
    ''' 
    Reads input parameters and definitions from a .py file.
    
    Input defs are a dict of feature:info mappings
     '''
    with open(filename,'r') as f:
        s = f.read()
        return eval(s)

def prepare_model_inputs(filename, delimiter=',', parse_dates=False, nrows=None, index_col=None, 
                        names=None, header=0, usecols=None, comp=None, ignore=[], 
                        de_dup_cols=[], to_bool=set()):
    '''
    Reads input rows from a flat file (csv, tsv, etc.).
    '''

    print parse_dates, nrows, index_col,names, header, usecols,
    print comp,ignore,de_dup_cols,to_bool
    start_time = time.time()
    print 'Loading data from %s' % (filename,),
    
    if names is not None:
        header=None
        
    infer_datetime_format = parse_dates
        
    # if index_col is None, this means that an index should not be used;
    # read_csv then expects index_col=False
    print usecols
    if usecols:
        df = pd.read_csv(open(filename, 'r'), parse_dates=parse_dates, infer_datetime_format=infer_datetime_format, sep=delimiter, nrows=nrows, compression=comp, index_col=False, names=names, header=header, usecols=usecols)
        #print df.shape
        
    else:
        
        df = pd.read_csv(open(filename, 'r'), parse_dates=parse_dates, infer_datetime_format=infer_datetime_format, sep=delimiter, nrows=nrows, compression=comp, index_col=False, names=names, header=header)

    df.drop(ignore, axis=1, inplace=True)
    #print 'before reindex'
    #print df.describe()
    
    
    #df = df.reindex(index=df[index_col])#method='pad') # do a reindex because read_csv is retarded and can't create a multiindex with a date.
    print df.shape
    # iterate through the columns and ignore the ignore cols
    if de_dup_cols:
      df.drop_duplicates(cols=de_dup_cols, inplace=True)
    for f in to_bool:
      df['has_' + f] = df[f].notnull()   
    input_features = list( to_bool | set(['has_' + _ for _ in to_bool]) | set(df.columns))

    print '... %s DataFrame loaded in %0.1f secs' % (df.shape, time.time() - start_time)
    
    return df, input_features
