#!/usr/bin/env python

'''
Utilities for parsing, outputing and understanding sklearn Trees.
'''
import sys
from copy import deepcopy
from sklearn.tree import _tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
from sklearn.externals import joblib
from sklearn import tree as TreeObj

from sklearn import tree as TreeObj

__all__ = ["DecisionTreeClassifier",
           "DecisionTreeRegressor",
           "ExtraTreeClassifier",
           "ExtraTreeRegressor"]

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CLASSIFICATION = {
    "gini": _tree.Gini,
    "entropy": _tree.Entropy,
}

REGRESSION = {
    "mse": _tree.MSE,
}

LTE_OP = '<='
GT_OP  = '> '
ONE_HOT_MARK = '::'

from collections import defaultdict


class Pattern:
    '''
    Used to simplify rules encoded in trees. For instance, the rule "X > 10 and X > 2"
    will be simplified to "X > 2".
    '''
    def __init__(self):
      self.features = []
      self.one_hot_includes = defaultdict(set) # map feature name to set of values
      self.one_hot_excludes = defaultdict(set) # map feature name to set of values
      self.mins = {} # map feature name to a min (possible non existent)
      self.maxs = {} # map feature name to a max (possible non existent)

    def __repr__(self):
      rep = ''
      for f in self.features:
         vs = self.one_hot_includes.get(f)
         if vs is not None:
            rep += ' and %s in %s' % (f, list(vs))

         ''' only had excludes if the include list is empty '''
         if not self.one_hot_includes:
            vs = self.one_hot_excludes.get(f)
            if vs is not None:
               rep += ' and %s not in %s' % (f, list(vs))

         mn = self.mins.get(f)
         if mn is not None:
            if int(mn) == mn:
               mn_rep = mn
            else:
               mn_rep = '%0.2f' % (mn,)
            rep += ' and %s%s%s' % (f, GT_OP, mn_rep)

         mx = self.maxs.get(f)
         if mx is not None:
            if int(mx) == mx:
               mx_rep = mx
            else:
               mx_rep = '%0.2f' % (mx,)
            rep += ' and %s%s%s' % (f, LTE_OP, mx_rep)

      if rep[:4] == ' and':
         rep = rep[4:]
      rep = rep.replace('[','(')
      rep = rep.replace(']',')')
      return rep


    def add_condition(self, feature, op, t):
      if ONE_HOT_MARK in feature:
         ''' special handling for one-hot encoded variables
             if comp is '<=' then put in <exclude> set, 
             otherwise put in <include> set
         '''
         k,v = feature.split(ONE_HOT_MARK)
         if k not in self.features: # make sure order of insertion is maintained
            self.features.append(k)
         if op == LTE_OP:
            self.one_hot_excludes[k].add(v)
         else:
            self.one_hot_includes[k].add(v)
      else:
         if feature not in self.features: # make sure order of insertion is maintained
            self.features.append(feature)
         if op == LTE_OP:
            mx = self.maxs.get(feature)
            if mx is None:
               self.maxs[feature] = t
            else:
               self.maxs[feature] = min(t, mx)
         else:
            mn = self.mins.get(feature)
            if mn is None:
               self.mins[feature] = t
            else:
               self.mins[feature] = max(t, mn)
      return self


def tree_str(clf, feature_names=None, outfile=sys.stdout, full=True,json_store = None):

   '''
   Parameters
   ----------
   clf : decision tree classifier
   The decision tree to be exported to graphviz.
   
   feature_names : list of strings, optional (default=None)
        Names of each of the features.
   
   Returns
   -------
   out_str: string
   The string to which the tree was printed.
   '''
   import json

   tree_id = None
   
   def node_to_str(tree, node_id,return_num=False):
      value = tree.value[node_id]
      impurity = tree.impurity[node_id]
      nns = tree.n_node_samples[node_id]
      #print value, impurity, nns
      if tree.n_outputs == 1:
         value = value[0, :]

      if hasattr(value, '__iter__') and len(value) > 1:
         # classifier impurity is returned, as well as value == [negatives, positives]
         if value[0] == 0 and value[1] == 0:
            rate = -1
         else:
            rate = value[1] / value.sum()

         val_str = ''.join("%8d" % _ for _ in value)
         # impurity monotonically increases with rate; not useful to see.
         # perf = "%8.6f %20s %6s  " % (rate, val_str, nns)
         
         perf = "%8.6f %6s  " % (rate, nns)
      else:
         perf = "%12.6f %6s  " % (value, nns)
         
         
      return perf

   def format_rule(feature, comp, t):
      if int(t) == t: # use int formatting
         return ["%s%2s%d" % (feature, comp, int(t))]
      return ["%s%2s%0.2f" % (feature, comp, t)]

   def recurse(tree, node_id, parent_id=None, pattern=Pattern()):
       if node_id == _tree.TREE_LEAF:
           raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

       
       left_child_id  = tree.children_left[node_id]
       right_child_id = tree.children_right[node_id]
      
        
       perf = node_to_str(tree, node_id)

       if left_child_id == _tree.TREE_LEAF or full == True: 
           print >> outfile, perf + str(pattern)

           if isinstance(json_store,list)
               tree_dict = pattern.__dict__
               tree_dict['perf'] = map(float,perf.split())

              
               tree_dict['tree_id'] = tree_id
               #print(dir(tree))
               json_store.append(json.dumps(tree_dict))
               #print(type(tree_id))
               #print 'ok'
               
  
       if left_child_id != _tree.TREE_LEAF:  # and right_child_id != _tree.TREE_LEAF
         if feature_names is not None:
            feature = feature_names[tree.feature[node_id]]
            used_features.add(feature)
         else:
            feature = "X[%s]" % tree.feature[node_id]
         t = tree.threshold[node_id]

         rule = format_rule(feature, LTE_OP, t)
         recurse(tree, left_child_id, node_id, deepcopy(pattern).add_condition(feature, LTE_OP, t))  # DFS (follow left-hand branches until leaf is encountered)

         rule = format_rule(feature, GT_OP,  t)
         recurse(tree, right_child_id, node_id, deepcopy(pattern).add_condition(feature, GT_OP, t))

   used_features = set()

   
   if isinstance(clf, _tree.Tree):
       
       recurse(clf, 0, Pattern())
       
   elif isinstance(clf, RandomForestRegressor) or isinstance(clf, RandomForestClassifier):
      for _ in clf.estimators_:
         print >> outfile, '<tree>'

         #print _.random_state
         try:
             tree_id = hash(tuple(_.random_state.get_state()[1]))#tree_id has a larger scope - can be seen by all in dump_tree
         except:
             tree_id = _.random_state
         #tree_id =matree_id[1])
         #print tree_id
         #print type(''.join(list(tree_id)))
         #exit()
         recurse(_.tree_, 0, Pattern())

         #tree_json = treeToJson(_.tree_)
         print >> outfile, '</tree>'
   elif hasattr(clf, "tree_"):
      recurse(clf.tree_, 0, Pattern())
   else:
    print "No 'tree_' member found for ", clf
   

   # def node_to_str(tree, node_id):
   #    value = tree.value[node_id]
   #    impurity = tree.impurity[node_id]
   #    nns = tree.n_node_samples[node_id]
   #    if tree.n_outputs == 1:
   #       value = value[0, :]

   #    if hasattr(value, '__iter__') and len(value) > 1:
   #       # classifier impurity is returned, as well as value == [negatives, positives]
   #       if value[0] == 0 and value[1] == 0:
   #          rate = -1
   #       else:
   #          rate = value[1] / value.sum()

   #       val_str = ''.join("%8d" % _ for _ in value)
   #       # impurity monotonically increases with rate; not useful to see.
   #       # perf = "%8.6f %20s %6s  " % (rate, val_str, nns)
   #       perf = "%8.6f %6s  " % (rate, nns)
   #    else:
   #       perf = "%12.6f %6s  " % (value, nns)
   #    return perf

   # def format_rule(feature, comp, t):
   #    if int(t) == t: # use int formatting
   #       return ["%s%2s%d" % (feature, comp, int(t))]
   #    return ["%s%2s%0.2f" % (feature, comp, t)]

   # def recurse(tree, node_id, parent_id=None, pattern=Pattern()):
   #    if node_id == _tree.TREE_LEAF:
   #       raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

   #    left_child_id  = tree.children_left[node_id]
   #    right_child_id = tree.children_right[node_id]

   #    perf = node_to_str(tree, node_id)

   #    if left_child_id == _tree.TREE_LEAF or full == True:  # and right_child_id != _tree.TREE_LEAF
   #       #print >> outfile, perf + '  '.join(prefix)
   #       print >> outfile, perf + str(pattern)


   #    if left_child_id != _tree.TREE_LEAF:  # and right_child_id != _tree.TREE_LEAF
   #       if feature_names is not None:
   #          feature = feature_names[tree.feature[node_id]]
   #          used_features.add(feature)
   #       else:
   #          feature = "X[%s]" % tree.feature[node_id]
   #       t = tree.threshold[node_id]

   #       rule = format_rule(feature, LTE_OP, t)
   #       recurse(tree, left_child_id, node_id, deepcopy(pattern).add_condition(feature, LTE_OP, t))  # DFS (follow left-hand branches until leaf is encountered)

   #       rule = format_rule(feature, GT_OP,  t)
   #       recurse(tree, right_child_id, node_id, deepcopy(pattern).add_condition(feature, GT_OP, t))

   # used_features = set()

   # if isinstance(clf, _tree.Tree):
   #    recurse(clf, 0, Pattern())
   # elif isinstance(clf, RandomForestRegressor) or isinstance(clf, RandomForestClassifier):
   #    for _ in clf.estimators_:
   #       print >> outfile, '<tree>'
   #       recurse(_.tree_, 0, Pattern())
   #       print >> outfile, '</tree>'
   # elif hasattr(clf, "tree_"):
   #    recurse(clf.tree_, 0, Pattern())
   # else:
   #  print "No 'tree_' member found for ", clf


def get_feature_importances(clf, feature_names):
    total_importance = np.sum(clf.feature_importances_)
    imp_weights = clf.feature_importances_ / total_importance
    return sorted(zip(imp_weights, feature_names), reverse=True)



# def tree_from_db(database='pguser_db',host='127.0.0.1',
#                      password='',user='pguser'):
#     import psycopg2,jsonpickle

#     con = psycopg2.connect(database=database,host=host,
#                                 user=user,password=password)

#     cur = con.cursor()
#     cur.execute("select * from tree_json")

#     rows = cur.fetchall()

#     print len(rows)
    
#     for row in rows:
#         print((row[2]).keys())
#         try:
#             result = jsonpickle.decode(row[2])
#         except:
#             print('nope')
    
#     con.close()
    

#an example of how to store json rep. of a tree in a database (psql)
##def bar_to_db(x_array,y_array,db_name='basic_viz'):

    
    # import sqlite3 as lite

    # print 'dump bar chart to db'
    # con = lite.connect(db_name)
    # cur = con.cursor()  
    # cur.execute('SELECT SQLITE_VERSION()')
    # cur.execute("CREATE TABLE ftr_bar (Id INT, Name TEXT, ftr_name TEXT,x REAL,y REAL)")
    
    # cur.execute("INSERT INTO ftr_bar VALUES(?,?,?,?,?)",(0,'awesome_bar','magic',x,y))
    # con.commit()
    
    # con.close()

    
def trees_to_db(json_store,db_name='forest.db',
               use_sqlite=False,password='',user='pguser',
               database = 'pguser_db',host='127.0.0.1'):

    import sqlite3 as lite
    import os
    import cPickle as pkl
    import psycopg2
   
    if use_sqlite:
        con = lite.connect(db_name)
        cur = con.cursor()  
        cur.execute('SELECT SQLITE_VERSION()')
     
        cur.execute("INSERT INTO tree VALUES(1,?,?)",('forest0',s))

        

    else:
         con = psycopg2.connect(database=database,host=host,user=user,
                                password=password) 
         cur = con.cursor()
         cur.execute('SELECT version()')
         cur.execute("drop table if exists forest_db")
         cur.execute("CREATE TABLE forest_db(Id INTEGER PRIMARY KEY,model json );")
         query = "INSERT INTO forest_db (Id, model) VALUES (%s, %s)"
    
         for i,elem in enumerate(json_store): 
             cur.execute(query, (i,elem))
         
    con.commit()
    con.close()

#    
def clf_to_db(clf,db_name='forest.db',use_sqlite=False,
              password='',user='pguser',database = 'pguser_db',
              host='127.0.0.1',base_dir='./'):

    import psycopg2
    import cPickle as pkl
    
    con = psycopg2.connect(database=database,host=host,user=user,
                           password=password)
    cur = con.cursor()
    cur.execute('SELECT version()')
    cur.execute("drop table if exists RF_clf")
    cur.execute("CREATE TABLE RF_clf(Id INTEGER PRIMARY KEY,clf bytea );")
    query = "INSERT INTO RF_clf (Id, clf) VALUES (%s, %s)"
    cur.execute(query, (1,psycopg2.Binary(pkl.dumps(clf))))

    cur.execute('Select %s',(psycopg2.Binary(pkl.dumps(clf,-1)), ))
    result, = cur.fetchone()
    print type(str(result))
    test = pkl.loads(str(result))
    
    joblib.dump(clf,base_dir+'RF_clf/RF_clf_00.pkl')
    
    con.commit()
    con.close()

    print '~~~~~~~~~~'
    print(type(test))
    print(type(clf))

    
#def h    

    
def dump_tree(clf, feature_names, fn='forest.txt', dir='',
              fop='w', binaries=True,json_store = None):
    '''
    Persists a trained Tree (e.g. RandomForest) to text and numpy binary files.
    '''
    import os

    fn = str(fn)

    if binaries:
        
        joblib.dump(clf, os.path.join(dir, 'model_binary.' + fn))
        joblib.dump(feature_names, os.path.join(dir, 'model_feature_names.' + fn))

    full_path = os.path.join(dir, 'model_features_imps.' + fn + '.txt')
    with open(full_path, fop) as f:
        imps = get_feature_importances(clf, feature_names)
        print >> f, '\n'.join("%0.5f %s" % (imp,name) for imp, name in imps if imp >= 0)

    full_path = os.path.join(dir, 'model_trees.' + fn)
    with open(full_path, fop) as f:
        tree_str(clf, feature_names=feature_names, outfile=f, full=True,
                 json_store=json_store)
        


def tree_hist(json_store):
    '''
    Pools together all rules from a RF json_store of nodes into an arrangement usefull for plotting simple histograms, json_store must be a dictionary at this point - not a string
    '''
    all_feature_vals = {}
    maxs = {}
    mins = {}

    for j in json_store:
        if j['maxs']:
            #print j['maxs']
            for k,v in j['maxs'].items():
                if k in all_feature_vals:
                    all_feature_vals[k].add(v)
                else:
                    all_feature_vals[k] = set([v])
                if k in maxs:
                    maxs[k].add(v)
                else:
                    maxs[k] = set([v])    
        if j['mins']:
            for k,v in j['mins'].items():
                if k in all_feature_vals:
                    all_feature_vals[k].add(v)
                else:
                    all_feature_vals[k] = set([v])
                if k in mins:
                    mins[k].add(v)
                else:
                    mins[k] = set([v])
                    

    return all_feature_vals,maxs,mins


def decision_bound(clf,df,ax,test_col='test',cmap=None,x=None,y=None,
                   z=None,unseen_dims =1,pretty=False,test_zoom = True,
                   bndry_from_test=True,smooth_viz = False,show_background=True,
                   train_and_test=True,smart_zoom=False,retrain=False,newRF=False):
    '''
    plot a decision boundary -assume that columns are ranked by importance 
    '''
    import matplotlib.pyplot as plt
    from scipy import interpolate
    import matplotlib.tri as tri
    from collections import namedtuple, Counter
    import copy, random,itertools
    from matplotlib.colors import LogNorm
    
    #grab feature importances 
    clf_i  = list(np.argsort(clf.feature_importances_)[::-1])

    clf_new = copy.copy(clf) #we want to have the flexibility to retrain 
    df = df.fillna(df.mean())
    
    
    
    #columns
    if z==None:
        z = df.columns[-2]
    if x==None:
        x = df.columns[clf_i[0]]
    if y==None:
        y = df.columns[clf_i[1]]

    val_obs = df[z]
    #print 'xy:',x,y,df.columns

    #the data should have a col. that label train vs test data
    x_i= np.where(df.columns==x)[0][0]
    y_i= np.where(df.columns==y)[0][0]

    
    #dataframe for all positive test points - used for red contour splat
    if train_and_test:
        #print df.columns
        print z in df.columns
        df_1 = df[(df[z]==1)]
        bndry_from_test = False
        
    else:
        df_1 = df[(df[z]==1) & (df[test_col]==1)]

    #dataframe from which the boundary is drawn - will include both positve and neg.
    df_b =  df
    if bndry_from_test: #only use the test points to generate bndry
        df_b = df[df[test_col]==1]
    
    if retrain: # we just want to retrain on the pictured features - no unseen features
        if newRF:
            clf_new = RandomForestClassifier(max_features=None,n_estimators=10,n_jobs=-1)
            
        row_count = df_b.shape[0]
        len_pos = len(df_b[df_b[z]==1])
        random_i = random.sample(range(row_count),5*len_pos)

        df_train = df_1.append(df_b.iloc[random_i,:],ignore_index=True) #all failed states + a few more random ones

        
        y_train = np.array(list(df_1[z].values)+list(df_b[z].values[random_i]))
        print df_train[[x,y]].shape,y_train.shape
        
        clf_new.fit(df_train[[x,y]].values, y_train)
        print 'accuracy train:', clf_new.score(df_train[[x,y]].values, y_train)
        
        print 'accuracy test:', clf_new.score(df_1[[x,y]].values,df_1[z].values) #y_obs[y_obs==1].values)#,model.score(df_fail[names[indices][0:10]].values, y_obs[y_obs==1].values)

        #precision_recall_fscore_support(df_b[z], , average='micro')
        
    if show_background:
        im = ax.hexbin(df[x],df[y],cmap=plt.cm.Blues,gridsize =8)
        plt.colorbar(im, ax=ax)

    # histogram to contour plot
    xx = copy.copy(df_1[x].values)
    yy = copy.copy(df_1[y].values)

    #throow out bottom and top 1%
    xends,yends = map(lambda u:(np.percentile(u,1.0),np.percentile(u,99.0)),[xx,yy])

    x_b = df_b[x].values
    y_b = df_b[y].values
    nbins = 40
    if smart_zoom: #
        common_val_x = Counter(xx).most_common(3)
        common_val_y= Counter(yy).most_common(3)

        

        if len(common_val_x) != 1:
            if len(common_val_y) == 1:
                common_val_y = common_val_x
        print len(common_val_x), common_val_x

        
        min_delta_x = np.min([np.abs(common_val_x[0][0]- common_val_x[1][0]),xx.std()/5.0])
        
        min_delta_y =  np.min([np.abs(common_val_y[0][0]- common_val_y[1][0]),yy.std()/5.0])

    
        print min_delta_x,min_delta_y
        
        print 'xends ',xends

        def nice_bin(x_in,n_big_bins=5,n_lin_bins=5):
            xends = [np.max(x_in),np.min(x_in)]
            edges = np.array(list(set(map(lambda u:np.percentile(x_in,u),(100/n_big_bins)*np.arange(n_big_bins+1.)))))
            #print edges
            edges.sort()
            dx = edges[1:]-edges[0:-1]
            edges = itertools.chain([list(edges[0:-1]+i*dx/n_lin_bins) for i in range(0,n_lin_bins)])
            edges = list(edges)
            edges = [elem for subl in list(edges) for elem in subl]
            #print edges
            edges = list(set(edges))#+list(xends)
            edges.sort()
            edges.append(np.max(x_in))
            return edges

        print 'nice bins'
        xedges = nice_bin(xx)
        print 'xedges:',xedges
            
        # except:
        #     xedges = np.linspace(xx.mean()-min_delta_x,xx.mean()+min_delta_x,nbins+1)

        #try:
        yedges = nice_bin(yy)
        xedges = sorted(xedges)
        yedges.sort()
            
        print xedges,yedges
        
        if true_pos:
        
            H, xedges, yedges, img = ax.hist2d(xx, yy, bins=[xedges,yedges],norm=LogNorm(),cmap=plt.cm.jet)
            #H, xedges, yedges, img = ax.hexbin(xx, yy, bins=[xedges,yedges],norm=LogNorm(),cmap=plt.cm.jet)
            #H, xedges, yedges, img = ax.hexbin(xx, yy, bins=nbins,norm=LogNorm(),cmap=plt.cm.jet,xscale='symlog')
            plt.colorbar(img, ax=ax)
            x_thres = map(lambda u:np.percentile(xx,u),[25,50,75])

            
            # if np.min(xx) < 0.0 and np.max(xx) >0.0:
            #     ax.set_xscale('symlog',linthreshx=np.max(np.abs(x_thres)))
            # else:
            #     ax.set_xscale('symlog',linthreshx=.0000001)

            # if np.min(yy) < 0.0  and np.max(yy) >0.0:
            #     ax.set_yscale('symlog',linthreshy=min_delta_y)
            # else:
            #     ax.set_yscale('symlog',linthreshy=.0000001)

                
            # ax.set_yscale('symlog')
        else:
            H, xedges, yedges =  np.histogram2d(xx,yy,normed=False,
                                                bins=[xedges,yedges])   
            im = plt.imshow(H.T, interpolation='nearest',cmap=plt.cm.jet, norm=LogNorm(),
                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto',origin='lower')
            print [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            xtick_pos = ax.get_xticks()
            
            xtick_i = np.linspace(0,len(xedges),len(xtick_pos))
            
            new_xlabels = map(lambda u:"{:.2e}".format(u),interpolate.griddata(np.arange(len(xedges)),np.array(xedges) ,xtick_i,method='nearest'))
            ax.set_xticklabels(new_xlabels,rotation=30)
            
            ytick_pos = ax.get_yticks()
            
            ytick_i = np.linspace(0,len(yedges),len(ytick_pos))
            
            new_ylabels = map(lambda u:"{:.2e}".format(u),interpolate.griddata(np.arange(len(yedges)),np.array(yedges) ,ytick_i,method='nearest'))
            ax.set_yticklabels(new_ylabels,rotation=30)
        
        
            plt.colorbar(im, ax=ax)

       
            
    else:
        #H, xedges, yedges =  np.histogram2d(xx,yy,normed=False,bins=nbins)
        xedges =np.linspace(xends[0],xends[1],nbins)
        yedges =np.linspace(yends[0],yends[1],nbins)
        
        H, xedges, yedges, img = ax.hist2d(xx, yy, bins=[xedges,yedges],norm=LogNorm(),cmap=plt.cm.jet)

        plt.colorbar(img, ax=ax)
        
    col_to_drop = [z]
    
    if test_col in df.columns:
        col_to_drop.append(test_col)
    
    df_1.drop(col_to_drop,axis=1,inplace=True)
    df_b.drop(col_to_drop,axis=1,inplace=True)

    n_features = df_1.shape[1]
 
    xx = list(0.5*(xedges[1:]+xedges[:-1]))
    xx.append(xedges[-1])
    xx.insert(0,xedges[0])
    
    yy = list(0.5*(yedges[1:]+yedges[:-1]))
    yy.append(yedges[-1])
    yy.insert(0,yedges[0])
   
    xx, yy = np.meshgrid(np.array(xx),np.array(yy))

  
    # #generate some leading additional dimensions
    d_i = clf_i
    d_i.remove(x_i)
    d_i.remove(y_i)

    if retrain:
        
        # len_pos = len(val_obs[val_obs==1])
        # random_i = random.sample(range(row_count),5*len_pos)

        # df_train = df_1(df_b.iloc[random_i,:],ignore_index=True) #all failed states + a few more random ones
        # y_train = np.array(list(val_obs[val_obs==1])+list(val_obs[random_i]))

        # #retrain_vals hodl eb some less 1 fraction of df
        # clf_new.fit(df_train[[x,y]].values, y_train)
        
        # unseen_dims=0 #overide unseen dims
        # print 'retrain on ONLY the pictured features - no hidden feature dims'
        # print row_count, len_pos
        # exit()
        data_len = len(xx.ravel())
        dummy_data = np.tile(df_b[[x,y]].mean() ,(data_len,1))
        dummy_data[:,0] = xx.ravel()
        dummy_data[:,1] = yy.ravel()
        
    else:
         # dummy data to draw decision boundary
        data_len = len(xx.ravel())
        dummy_data = np.tile(df_b.mean() ,(data_len,1))
        dummy_data[:,x_i] = xx.ravel()
        dummy_data[:,y_i] = yy.ravel()
        
        if pretty:
            if d_i[0] == 0:
                unseen_dims = 1
        
        if unseen_dims:
            for d in d_i[0:unseen_dims]:
                print 'd:', d
                dummy_data[:,d] = (interpolate.griddata((df_b[x], df_b[y]), df_b.iloc[:,d], (xx, yy), method='linear',fill_value=df_b.iloc[:,d].mean())).ravel()
        
    Z = clf_new.predict(dummy_data)
    
    #we MUST recompute that feature importanc - can't assume that column order implies this . . .
    print Z.shape,xx.shape
    Z = Z.reshape(xx.shape)

    
    if smart_zoom and true_pos==False:
        # ax.contourf(xx,yy,Z,alpha=.2,nlevels=10,linewidth=.20)
        # ax.contour(xx,yy,Z,alpha=1,levels = [.1,.5,.9],colors='k',linewidths=.5)
        
        ax.contourf(xx,yy,Z,alpha=.2,nlevels=10,linewidth=.20)
        ax.contour(xx,yy,Z,alpha=1,levels = [.1,.5,.9],colors='k',linewidths=.5)
    else:
        ax.contourf(xx, yy, Z,alpha=.5,nlevels=3,linewidth=.20)
        ax.contour(xx, yy, Z,alpha=1,levels = [.1,.5,.9],colors='k',linewidths=.5)

        # if true_pos and smart_zoom:
        #     x_thres = map(lambda u:np.percentile(xx,u),[25,50,75])

            
        #     if np.min(xx) < 0.0 and np.max(xx) >0.0:
        #         ax.set_xscale('symlog',linthreshx=np.max(np.abs(x_thres)))
        #     else:
        #         ax.set_xscale('symlog',linthreshx=.0000001)

        #     if np.min(yy) < 0.0  and np.max(yy) >0.0:
        #         ax.set_yscale('symlog',linthreshy=min_delta_y)
        #     else:
        #         ax.set_yscale('symlog',linthreshy=.0000001)
    


    ax.set_xlabel(str(x))
    ax.set_ylabel(str(y))
    ax.set_title('Decision Boundaries w/ Downtime Count')

    if test_zoom:
        ax.set_xlim(xedges.min(), xedges.max())
        ax.set_ylim(yedges.min(), yedges.max())

def tree_hist(json_store):
    '''
    Pools together all rules from a RF json_store of nodes into an arrangement usefull for plotting simple histograms, json_store must be a dictionary at this point - not a string
    '''
    all_feature_vals = {}
    maxs = {}
    mins = {}

    for j in json_store:
        if j['maxs']:
            #print j['maxs']
            for k,v in j['maxs'].items():
                if k in all_feature_vals:
                    all_feature_vals[k].add(v)
                else:
                    all_feature_vals[k] = set([v])
                if k in maxs:
                    maxs[k].add(v)
                else:
                    maxs[k] = set([v])    
        if j['mins']:
            for k,v in j['mins'].items():
                if k in all_feature_vals:
                    all_feature_vals[k].add(v)
                else:
                    all_feature_vals[k] = set([v])
                if k in mins:
                    mins[k].add(v)
                else:
                    mins[k] = set([v])
                    

    return all_feature_vals,maxs,mins
    
def plot_tree(clf,ax,cmap=None,label_downbranch=False,sample_size=True):
    '''
    Plot a decision tree using graphviz + networkx modules
    '''
    
    import networkx as nx
    import re,ast
    import matplotlib.pyplot as plt

    if cmap ==None:
        cmap = plt.get_cmap('Reds')
    export_file = TreeObj.export_graphviz(clf, out_file='temp.dot')
    #export_file.close()
    G = nx.read_dot('temp.dot')
    

    node_dict = {}
    edge_dict = {}
    
    for k,v in G.node.iteritems():
        v['label'] = v['label'].replace('\\n',',').replace('"','')
        
        style_dict = map(lambda x: 'rule: "'+ x +'"' if hasattr(re.search(r"<|>",x),'group') else x.replace('=',':') ,v['label'].split(','))
        style_dict = map(lambda x: x+',',style_dict)
        style_dict = map(lambda x: x.split(':'),style_dict)
        style_dict = map(lambda x: '"'+x[0].strip()+'"'+':'+x[1],style_dict)
       
        style_dict = '{'+''.join(style_dict)+'}'
        find_array = re.search(r'\[(.*)\]', style_dict)
        if hasattr(find_array,'group'):
            arr_str = find_array.group()
            new_arr_str = '['+','.join(arr_str[1:-1].strip().split())+']'
            style_dict = style_dict.replace(arr_str,new_arr_str)
            

        node_dict[k]  = ast.literal_eval(style_dict)
        
    pos_dict = nx.graphviz_layout(G)#,prog='dot')
    #rescale position using min,max and mean
    #pos_mean = np.mean(map(u[]
    pos_mean =map(np.mean,zip(*pos_dict.values()))
    pos_max = map(np.abs, np.array(pos_dict.values()).T - np.array([pos_mean] * len(pos_dict)).T)

    colors = []
    colors_dict = {}
    alphas = []
    sizes = []
    label_dict = {}


    #get probablity for all but the end nodes - store in color_dict
    def grab_vals(parent_k):#,node_dict):
        v =  node_dict[parent_k]
        
        child_k = [k for k in G.successors(parent_k)]

        if child_k:
        
            try:
                c_vals = [node_dict[k]['value'] for k in child_k]
                
                c_vals = np.array(c_vals)
                colors_dict[parent_k] = (np.sum(c_vals,axis=0)/np.sum(c_vals))[1]
                print colors_dict[parent_k]
            except:
           
                c_vals = [grab_vals(x) for x in child_k]
                result = []
                for i,sub_l in enumerate(c_vals):
                    if type(sub_l) == type(np.array([])):
                        c_vals[i] = [list(val) for val in sub_l]
                        result.extend(c_vals[i])
                    else:
                        result.extend([c_vals[i]])
                 
                    
                c_vals = result
                c_vals = np.array(c_vals)
                
                colors_dict[parent_k] = (np.sum(c_vals,axis=0)/np.sum(c_vals))[1]
            
        else:
            c_vals = node_dict[parent_k]['value']
            
        return c_vals

    print grab_vals('0') #start at the root node and work down


    child = []
    #pick up probabilities for the end nodes - generate node labels
    for k,v in node_dict.iteritems():
        #colors.append(cmap(v['gini']*2.0))

        alphas.append(v['samples'])
        #print k,v
        if 'value' in v:
            colors_dict[k] = (v['value'][1]/np.sum(v['value']))
            child.append(k)
      
        colors.append(colors_dict[k])

        if label_downbranch:
            if 'rule' in v:
                for i,k_child in enumerate(G.successors(k)):
                    label_dict[k_child]  = (k_child+' \n '+v['rule'])
                    
                    if i==1:
                        label_dict[k_child]  = label_dict[k_child].replace('<=','>')
       
        else:
            if 'rule' in v:
                label_dict[k] =(v['rule'].replace('<=',':').replace('>=',':').replace('=',':').split(':'))[0]
            
            else:
                label_dict[k] = k

    label_dict['0'] = (node_dict['0']['rule'].replace('<=',':').replace('>=',':').replace('=',':').split(':'))[0]
                
    alphas = np.array(alphas)

    sizes = alphas
    sizes = np.log(sizes.astype(float)/sizes.min()+1.0)
    sizes = 1200*sizes.astype(float)/sizes.max()

    alphas = alphas.astype('float')/alphas.max()
 
    
    plt.setp( ax.get_yticklabels(), visible=False)
    plt.setp( ax.get_xticklabels(), visible=False)

    cbar_nodes = dict([(k,colors_dict[str(k)]) for k in child])
    range_c = np.max(cbar_nodes.values())-np.min(cbar_nodes.values())
    min_c = np.min(cbar_nodes.values())
    cbar_nodes_n = {key: (value-min_c)/range_c for (key, value) in cbar_nodes.iteritems()}

    colors = map(lambda x: cmap((x-min_c)/range_c),colors)

    if sample_size:
        nodes= nx.draw_networkx_nodes(G,pos_dict,ax=ax,node_color=colors,
                                      node_size = sizes,linewidth=3.0)
    else:
        nodes= nx.draw_networkx_nodes(G,pos_dict,ax=ax,node_color=colors,
                                      linewidth=3.0)
    #facecolors = nodes.get_facecolors()
    #facecolors[:,3] = alphas
    
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.normalize(vmin=min_c,vmax=min_c + range_c))
    sm._A = []
    cbar = plt.colorbar(sm,ax=ax)
    
    for k in cbar_nodes_n:
        cbar.ax.hlines(cbar_nodes_n[str(k)], 0, 1, colors = 'k')#
        
  
    cbar.set_ticks(cbar_nodes.values())
    cbar.set_ticklabels(map(lambda x: x[0]+', P=' +str(x[1])[0:3], cbar_nodes.items()))
   

    
    nx.draw_networkx_labels(G,pos_dict,ax=ax,labels=label_dict,
                            font_size=6)  #takes labels opt keyword

    edge_colors = []
    edge_labels ={}
    for edge in G.edges():
        
        edge_colors.append(float(node_dict[edge[1]]['samples'])/node_dict[edge[0]]['samples'])
        #edge_labels[edge] = str(100*edge_colors[-1])[0:3] + '\n'+node_dict[edge[0]]['rule']
        for i,k_child in enumerate(G.successors(edge[0])):
            edge_labels[edge] = node_dict[edge[0]]['rule']
                    
            if i==1:
                edge_labels[(edge[0],k_child)] = node_dict[edge[0]]['rule'].replace('<=','>')
                #label_dict[k_child]  = label_dict[k_child].replace('<=','>')
        # edge_labels[edge] = node_dict[edge[0]]['rule']
        
        # if i==1
    edge_colors = np.array(edge_colors).astype('float')
    print edge_colors
    edge_colors = (edge_colors - edge_colors.min())/(edge_colors.max() - edge_colors.min())
    print edge_colors
    

    #weights = [G[u][v]['weight'] for u,v in G.edges]

     
    nx.draw_networkx_edges(G,pos_dict,alpha=.8,label='edges',
                           edge_cmap=cmap,edge_color =edge_colors,width = 10*edge_colors,arrows=False)
    nx.draw_networkx_edges(G,pos_dict)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    nx.draw_networkx_edge_labels(G,pos_dict,font_size=4,edge_labels=edge_labels,rotate = False,bbox=props)
    #nx.draw(G,nx.graphviz_layout(G),alpha=.8,
     #       edge_cmap=cmap,edge_color =edge_colors,width = 100*edge_colors)

    #ax.patch.set_visible(False)

    ax.patch.set_facecolor('None')    


    #print dir(nodes),nodes.get_alpha()
    #return nodes

    # l, b, w, h = cbar.ax.get_position().bounds
    # # cbar.ax.set_aspect("auto")
    # w = 0.03
    # #ax2 = ax.twinx()
    # ax2 = plt.twinx(ax=cbar.ax)
    # ax2.grid(False)
    # #ax2.set_yticks()
    # #cbar.set_ticks(colors_dict.values())
    # #ax2 =plt.twinx(ax=cbar.ax)
    # cbar.ax.set_position([l, b, w, h])
    # ax2.set_position([l, b, w, h])
    # #cbar.ax.set_ylim(0, 1)
    # #ax2.set_ylim(-10, 10)
    # ax2.set_yticks(colors_dict.values()[0:8])
    # ax2.set_yticklabels(colors_dict.keys()[0:8])

def test_tree():
    '''
    Generate a tree for testing.
    '''
    # xn features
    xn = 5
    features = ['F'+ONE_HOT_MARK+'%d' % (_,) for _ in range(xn)]

    # first [0,xn/2] features contribute as i^2, last xn/2 don't contribute
    values = [_ if _ < xn/2 else 0 for _ in range(1,xn+1)]
    #values   = [pow(2,_) if _ < xn/2 else 0 for _ in range(xn)]

    print zip(features,values)

    n = 1000
    xs = np.zeros((n, xn))
    ys = np.zeros((n,))
    for i in range(n):
      bits = np.random.choice(values, size=xn/2+1, replace=False)
      for b in bits:
         xs[i][b-1] = 1
         ys[i] += values[b-1]
         #ys[i] += values[b-1]

    ys = np.array([np.sign(_) for _ in ys])

    for i in range(n):
      print "_ %5d%s" % (ys[i], xs[i])
    clf = RandomForestClassifier(n_estimators=1, min_samples_leaf=2, min_samples_split=2, bootstrap=True )
    #clf = RandomForestRegressor(n_estimators=1, min_samples_leaf=1, min_samples_split=1, )
    return clf.fit(xs, ys), features


def main():
   import argparse
   parser = argparse.ArgumentParser(description='read tree from sklearn model file, write to txt file')
   parser.add_argument('path_to_model')
   parser.add_argument('path_to_feature_names')
   parser.add_argument('--test_tree', action='store_true')
   args = parser.parse_args()

   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      exit(0)
   print args

   from sklearn.externals import joblib

   if args.test_tree:
      model, names = test_tree()
  
   else:
      model = joblib.load(args.path_to_model)
      names = joblib.load(args.path_to_feature_names)

   dump_tree(model, names, 'test_tree')

if __name__ == '__main__':
    main()
