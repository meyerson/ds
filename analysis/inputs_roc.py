{
    # an example inputs file for roc_curve.py
    'field_actual':'surprise_downtime',
    'field_predicted':'raw_predicted_surprise_downtime',

    'in_flav':'sqlite',
    'in_conn':'/edge/1/downtime_model/report/downtime_model.db',
    'in_query':"from training_preds where test_set=1 and model_id=49",
    
    'out_flav':'sqlite',
    'out_conn_str':'/edge/1/downtime_model/report/downtime_model.db',
    'out_table':'roc_curve',
}
