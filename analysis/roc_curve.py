#!/usr/bin/env python

print(__doc__)

import numpy as np
from pandas import DataFrame
from pandas.io.sql import read_sql
from sql_help import *
from sql_pandas_help import TableWriter

def roc_curve(y_actual, y_score, plot=False):
    ''' Compute ROC stats and plot Receiver Operating Curve (if <plot> == True)
    '''
    # Binarize the output
    #from sklearn.preprocessing import label_binarize
    #y = label_binarize(y, classes=[0, 1, 2])
    #n_classes = y.shape[1]
    n_classes = len(np.unique(y_actual.values))
    y_actual = np.column_stack((y_actual.values, 1-y_actual.values))
    y_score = np.column_stack((y_score.values, 1-y_score.values))

    # Compute ROC curve and ROC area-under-curve for each class
    from sklearn.metrics import roc_curve, auc
    FPR = dict() # False Positive Rate
    TPR = dict() # True Positive Rate
    THRES = dict() # True Positive Rate
    roc_auc = dict()
    print "class ROC_AUC"
    for i in range(n_classes):
        FPR[i], TPR[i], THRES[i]  = roc_curve(y_actual[:, i], y_score[:, i])
        roc_auc[i] = auc(FPR[i], TPR[i])
        print "%5d %8.4f" % (i, roc_auc[i])

    print
    # Compute micro-average ROC curve and ROC area
    FPR["micro"], TPR["micro"], THRES["micro"] = roc_curve(y_actual.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(FPR["micro"], TPR["micro"])
    print "micro: %8.4f" % (roc_auc["micro"])
    
    if plot:
        print "Plotting {0} classes".format(n_classes)
        # Plot of a ROC curve for a specific class
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # Plot of a ROC curve for a specific class
        # fig = plt.figure()
        # plt.plot(FPR[2], TPR[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
        # plt.plot([0, 1], [0, 1], 'k--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.legend(loc="lower right")
        # plt.show()
        #fig.savefig("roc.png")

        # Plot ROC curve
        fig = plt.figure()
        plt.plot(FPR["micro"], TPR["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]))
        for i in range(n_classes):
            plt.plot(FPR[i], TPR[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        # plt.show()
        fig.savefig("roc.png")
        
    result = DataFrame()
    #np.column_stack((TPR[0], FPR[0]))
    result['TPR']   = TPR[0]
    result['FPR']   = FPR[0]
    result['THRES'] = THRES[0]
    return result
    
def main(args):
    from data_load import read_input_defs
    params = read_input_defs(args.inputs)
    
    in_flav = params['in_flav']
    in_conn = params['in_conn']
    in_query = params['in_query'] # " from <table_name> inner join <t2> on blah blah where blah blah "
    
    out_flav = params['out_flav']
    out_conn_str = params['out_conn_str']
    out_table = params['out_table']
    
    field_actual = params['field_actual']
    field_predicted = params['field_predicted']

    query = 'SELECT %s, %s %s;' % (field_actual, field_predicted, in_query)
    if in_flav == 'sqlite':
        cnx = db_connect_sqlite(in_conn)
        df = read_sql(query, cnx)
    elif in_flav == 'hs2':
        import hs2
        df = hs2.Hive('in_conn').read_frame(query)
    
    curve = roc_curve(df[field_actual], df[field_predicted], args.plot)
    tw = TableWriter(out_conn_str, out_table, flav=out_flav, if_exists='replace')
    tw(curve)
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute Receiver Operating Curves for table',argument_default=[])
    parser.add_argument('inputs', help='input definitions')
    parser.add_argument('-plot', action='store_true', help='plot the ROC curves')

    try:
        args = parser.parse_args()
    except Exception as e:
        print str(e)
        print parser.print_help()
        exit(0)
    print args

    main(args)
