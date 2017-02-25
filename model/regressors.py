

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
          max_features=0.6
          n_estimators=96

       n_jobs = min(3,n_estimators)
       from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
       return RandomForestRegressor(bootstrap=(not self.explainable), n_estimators=n_estimators, max_features=max_features,
                                    min_samples_leaf=self.min_weight, min_samples_split=self.min_weight, max_depth=max_depth, n_jobs=n_jobs)

