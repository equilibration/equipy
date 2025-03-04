import numpy as np
import pandas as pd
import lightgbm as lgb
import random
random.seed(2023)

def cv_early_stopping(params,
                      nfolds,
                      max_rounds,
                      early_stopping_rounds,
                      X_train,
                      y_train,
                      categorical_feats, 
                      objective='regression'):
    
    data_combined = X_train.copy()
    data_combined['target'] = y_train
    
    # Split
    fold_splits = np.array_split(data_combined, nfolds)

    # Run models
    metric_list = []
    iter_list = []

    for idx_ in range(nfolds):
        train_fold = pd.concat([x for i,x in enumerate(fold_splits) if i!=idx_])
        valid_fold = fold_splits[idx_]

        X_train = train_fold.drop(columns='target')
        y_train = train_fold.loc[:,'target']

        X_valid = valid_fold.drop(columns='target')
        y_valid = valid_fold.loc[:, 'target']


        lgb_train = lgb.Dataset(data=X_train, 
                                label=y_train, 
                                categorical_feature=categorical_feats)
        
        lgb_valid = lgb.Dataset(data=X_valid, 
                                label=y_valid)

        # train model
        model_trained = lgb.train(params=params,
                                  train_set=lgb_train, 
                                  num_boost_round=max_rounds, 
                                  valid_sets=lgb_valid,
                                  valid_names='validation',
                                  callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)])
        
        if objective == 'regression':
            metric_list.append(model_trained.best_score['validation']['l2'])
            iter_list.append(model_trained.best_iteration)
        elif objective == 'classification':
            metric_list.append(model_trained.best_score['validation']['auc'])
            iter_list.append(model_trained.best_iteration)

    metric_dict = {'metric': metric_list, 
                   'iterations': iter_list}
    
    return metric_dict
