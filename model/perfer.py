import sys

from numpy import array, empty, mean, median, min, sqrt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, BayesianRidge, LassoLars
from sklearn.svm import SVR
from sklearn.semi_supervised import label_propagation

from sklearn.grid_search import GridSearchCV


class PredictMean:
    def __init__(self, score=None):
        self._score = score

    def get_params(self, deep=False):
        return {'score':self._score}
        
    def set_params(self, params={}):
        self._score = params.get('score')
        return self
        
    def fit(self, inputs, outputs):
        self._score = mean(outputs)
        
    def predict(self, inputs):
        a=empty(len(inputs))
        a.fill(self._score)
        return a

    def score(self, inputs, outputs):
        preds = self.predict(inputs)
        return 1/(1+mean( (preds-outputs)**2))


class SSL:
    def __init__(self, estimator): # , label_strategy [extremes, most confident, only highest, only lowest, extremeX%, topX%, bottomX%]
        self._estimator = estimator

    def get_params(self, deep=False):
        return {'estimator':self._estimator.copy()}
        
    def set_params(self, params={}):
        self._estimator = params.get('estimator').copy()
        return self
        
    def fit(self, inputs, outputs):
        # loop until X% of unlabeled samples have a predicted label (y != -1)
        # ? until predictions on unlabeled samples are "strong"
        # ? until steady-state is reached
    
        labeled = outputs != -1
        trn_X = i
        # fit on labeled
        self._estimator.fit(inputs, outputs)
        
        # predict unlabeled
        
        # ? add labels to choice unlabeled samples
        
        
        
        
    def predict(self, inputs):
        a=empty(len(inputs))
        a.fill(self._estimator)
        return a

    def score(self, inputs, outputs):
        preds = self.predict(inputs)
        return 1/(1+mean( (preds-outputs)**2))

        
        
        
class Algs:
   def __init__(self, num_features, fast=False):
      n_estimators=100
      n_jobs=7
      min_samples_split=50
      min_samples_leaf=50
      max_depth=20
      n_neighbors=5
   
      if fast:
         n_estimators=5
         max_depth=7
         min_samples_split=500
         min_samples_leaf=500
         n_neighbors=5
      
      self.clfs = [
         ( "PredictMean",PredictMean(), {}), 
         ( "LinearReg", LinearRegression(normalize=True), {}), 
         ( "LassoLars", LassoLars(normalize=True), {'alpha':[0.10, 0.15, 0.20]}), 
         ( "BayesianRidge", BayesianRidge(normalize=True), 
                {'alpha_1':[5e-07, 7e-07, 1e-06], 'alpha_2':[1e-06, 5e-06, 1e-05], 'lambda_1':[5e-07, 1e-06, 5e-06], 'lambda_2':[5e-07, 1e-06, 5e-06],}), 
         ( "knn_C", KNeighborsClassifier(n_neighbors=n_neighbors, warn_on_equidistant=False), 
                {'n_neighbors':[4,8], 'weights':['uniform', 'distance'], 'p':[1,2,3]}),
         ( "knn_R", KNeighborsRegressor (n_neighbors=n_neighbors, warn_on_equidistant=False), 
                {'n_neighbors':[4,8], 'weights':['uniform', 'distance'], 'p':[1,2,3]}),
         ( "SGD", SGDRegressor(loss='huber'), {}), 
         ( "ShallowRF", RandomForestRegressor(n_estimators=100, n_jobs=1, max_depth=1, max_features=num_features), 
                {'max_depth':[1,2], 'min_samples_leaf':[1,2,3], 'max_features':(array((0.05, 0.1, 0.2, 0.5))*num_features).astype(int)} ), 
         ( "RF_R", RandomForestRegressor(n_estimators=n_estimators, n_jobs=3, 
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_depth=max_depth, max_features=num_features),
                {'max_depth':[1,2,3], 'min_samples_leaf':[1,3,5], 'max_features':(array((0.1,1.0))*num_features).astype(int)} ), 
         ( "RF_C", RandomForestClassifier(n_estimators=n_estimators, n_jobs=3, min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, max_depth=max_depth, max_features=num_features),     
                {'max_depth':[1,2,3], 'min_samples_leaf':[1,2,3], 'max_features':(array((0.01,0.05,0.1,0.2))*num_features).astype(int)} ), 
         ( "GradientBoost", GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=num_features), 
                {'max_depth':[2,4,8,16], 'min_samples_leaf':[5,8,10,15], 'max_features':(array((0.2, 0.35, 0.5))*num_features).astype(int), } ),
         ( "SVR_rbf", SVR(kernel='rbf', C=77, gamma=0.07), {'C':[20,50,100], 'gamma':[0.0001, 0.0005, 0.001,0.002]}),
         ( "SVR_linear", SVR(kernel='linear', C=1e3), {'C':[10,100,1000], }),
         #( "SVR_poly", SVR(kernel='poly', C=1e3, degree=2), {'C':[1e2,1e3,1e4], 'degree':[1,2,3]}),
         #( "LabelSpreadKNN", label_propagation.LabelSpreading(kernel='knn', alpha=1.0), {'alpha':[0.5, 0.8, 0.9, 1.0, ]}),
         #( SVR(), {'kernel':['rbf','poly'], 'degree':[1,2,3]}),
      ]

from sys import maxint
class Perfer:
    def __init__(self, feature_names, targF, algs, pca=False, boost=False, grid=False):
        self.best = maxint
        self.targF = targF
        self.algs = algs
        self.pca = pca
        self.feature_names = feature_names
        self.grid = grid

    def predict_and_err(self, clf, xs, ys):
        preds = clf.predict(xs)
        loss = self.targF(xs, ys, preds, self.feature_names)

        #worst = abs(preds - ys).argmax()
        #print '%s y:%6.0f p:%6.0f' % (xs[worst], ys[worst], preds[worst]),

        if loss < self.best:
            self.best = loss
        if (hasattr(clf, 'tree_') or (hasattr(clf, 'estimators_') and hasattr(clf.estimators_[0], 'tree_'))):
            from tree_tools import tree_str
            with open(str(self.label) + 'least_loss.txt', 'w') as f:
                tree_str(clf, feature_names=self.feature_names, outfile=f, full=True)
        return loss, median(ys), median(preds)

    def fit_and_predict(self, clf, params, trn_xs, trn_ys, tst_xs, tst_ys):
      if self.pca:
         from sklearn.decomposition import PCA
         pca = PCA() #n_components=2, whiten=True)
         pca.fit(trn_xs)
         trn_xs = pca.transform(trn_xs)
         tst_xs = pca.transform(tst_xs)
            
      if self.grid:
         clf_grid = GridSearchCV(clf, params)
         clf_grid.fit(trn_xs, trn_ys)
         fitted = clf_grid.best_estimator_
         best_params = clf_grid.best_params_
         sys.stdout.flush()
      else:
         fitted = clf.fit(trn_xs, trn_ys)
         best_params = clf.get_params()

      loss, median_ys, median_ps = self.predict_and_err(fitted, tst_xs, tst_ys)
      #print round(loss,2), round(median_ys,0), round(median_ps,0),
      return loss, best_params

    def __call__(self, tr, ts, input_features, y_field, label=''):
        self.label = label
        trn_xs = tr.ix[:,input_features].values
        tst_xs = ts.ix[:,input_features].values

        trn_ys = tr.ix[:, y_field].values
        tst_ys = ts.ix[:, y_field].values
      
        perf = {}
        best_params = {}
        for name, rgsr, params in self.algs.clfs:
            print name,
            loss, params = self.fit_and_predict(rgsr, params, trn_xs, trn_ys, tst_xs, tst_ys)
            perf[name] = loss
            best_params[name] = params
         
        return perf, best_params
