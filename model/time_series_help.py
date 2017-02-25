#!/usr/bin/env python

import pandas as pd
import random
import numpy as np

def weight_decay(days):
    ''' returns a linear decay factor for a set of dates; first day -> 0.0, last day -> 1.0 '''
    mn = min(days)
    mx = max(days)
    rng = float((mx-mn).days)
    wgts = np.array([ float((d - mn).days)/rng for d in days ])
    return wgts
   
class TSCV:
    '''
    Used to perform walk-forward cross-validation training & testing datasets.
    Return lists of (train_beg, test_beg, test_end) tuples.
    '''
    def __init__(self, ts, tr_days, ts_days):
        ts = ts.order()
        n = len(ts)

        endpoints = []
        ts_end = ts[n-1] + pd.tseries.offsets.relativedelta(microseconds=1) # just after last date
        ts_beg = ts_end  + pd.tseries.offsets.relativedelta(days=-ts_days) # go back ts_days
        tr_beg = ts_beg + pd.tseries.offsets.relativedelta(days=-tr_days)

        if tr_beg >= ts[0]:  
            endpoints.append((tr_beg,ts_beg,ts_end))

            # Don't allow the training window to start before the beginning of the timeseries.
            while ts_days > 0 and tr_beg >= ts[0]:
                tr_beg = tr_beg + pd.tseries.offsets.relativedelta(days=-ts_days)
                ts_beg = tr_beg + pd.tseries.offsets.relativedelta(days= tr_days)
                ts_end = ts_beg + pd.tseries.offsets.relativedelta(days= ts_days)
                endpoints.append((tr_beg,ts_beg,ts_end))

        self.endpoints = endpoints
        self.index = ts

    def __len__(self): return len(self.endpoints)
    
    def next(self, N):
        '''
        generate cross-validation random subsets (without replacement, unique train & test row indices)
        '''
        p = (N - 1.0)/N if N > 1 else 1.0
        for tr_beg, ts_beg, ts_end in self.endpoints:
            for i in range(N):
                candidates = self.index[(self.index >= tr_beg) & (self.index < ts_beg)]
                tr_rows = list(set(random.sample(candidates, int(p*len(candidates)))))

                candidates = self.index[(self.index >= ts_beg) & (self.index < ts_end)]
                ts_rows = list(set(random.sample(candidates, int(p*len(candidates)))))

                yield tr_rows, ts_rows, tr_beg, ts_beg


if __name__ == '__main__':
   dates = ["2011030600","2011031317","2011032013","2011032701","2011040116","2011040720","2011041322","2011041923","2011042608","2011050212","2011050822","2011051611","2011052311","2011053017","2011060610","2011061411","2011062119","2011062911","2011070721","2011071420","2011072017","2011072713","2011080213","2011080816","2011081421","2011081910","2011082417","2011083016","2011090521","2011091014","2011091508","2011092016","2011092523","2011093002","2011100508","2011100917","2011101215","2011101615","2011101917","2011102320","2011102619","2011103018","2011110218","2011110714","2011111012","2011111405","2011111611","2011111821","2011112207","2011112611","2011112911","2011120116","2011120509","2011120710","2011121008","2011121313","2011121611","2011122008","2011122318","2011122720","2011123007","2012010214","2012010417","2012010622","2012010917","2012011112","2012011313","2012011615","2012011821","2012012119","2012012421","2012012717","2012013021","2012020210","2012020510","2012020802","2012021017","2012021407","2012021620","2012022014","2012022220","2012022516","2012022813","2012030204","2012030519","2012030815","2012031215","2012031516","2012031916","2012032214","2012032613","2012032912","2012040214","2012040519","2012041013","2012041410","2012041814","2012042314","2012042715","2012050211"]

   ts = pd.to_datetime(dates, format='%Y%m%d%H')
   X = pd.DataFrame(dates,index=ts)
   y = pd.DataFrame(dates,index=ts)

   cv = TSCV(X.index, ts_days=0)

   for tr_rows, ts_rows, tr_beg, ts_beg in cv.next(5):
      print '--------------------------------------------------------------------------------'
      print tr_beg, ts_beg
      

   weight_decay(tr_rows)
