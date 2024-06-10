import numpy as np
import pandas as pd
from typing import Optional, Annotated

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import lightgbm as lgb

def preprocess_census(data: pd.DataFrame,
                      target_feature: str, 
                      sensitive_features: list,
                      categorical_features: Optional[list]=None,
                      continuous_features: Optional[list]=None,
                      target_processing: Annotated[str, "log", "neutral"]="log", 
                      test_size: float = 0.2,
                      calib_size: float = 0.2,
                      split_seed: int = 42, 
                      objective='regression',
                      ) -> dict:
    """
    Returns: A dictionary with preprocessed data

    ToDo: Fix issues if there are no categoricals
    """
    # Out dict init
    return_dict = {}

    # Split dataset
    y = data.loc[:, target_feature]
    X = data.drop(columns=target_feature)

    # Split initial
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size = test_size,
                                                        random_state = split_seed)
    
    # Fit preprocessors
    onehot_cands = list(set(X_train.columns) -
                        set(categorical_features) -
                        set(continuous_features))
    

    if len(onehot_cands) > 0:
        one_hot = OneHotEncoder(sparse_output=False, 
                                handle_unknown='ignore', 
                                drop='first')
        
        
        one_hot.fit(X_train.loc[:, onehot_cands])
        one_hot_train = one_hot.transform(X_train.loc[:, onehot_cands])
        one_hot_test = one_hot.transform(X_test.loc[:, onehot_cands])

        return_dict['one_hot_encoder'] = one_hot   


    if len(continuous_features) > 0:
        scaler = StandardScaler()
        scaler.fit(X_train.loc[:,continuous_features])
        
        scaled_train = scaler.transform(X_train.loc[:, continuous_features])
        scaled_test = scaler.transform(X_test.loc[:, continuous_features])

        return_dict['scaler'] = scaler

    if objective == 'regression':
        if target_processing == 'log':
            y_train = np.log(y_train)
            y_test = np.log(y_test)

        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
    
    elif objective == 'classification':
        y_train = np.where(y_train > 50000,1,0)
        y_test = np.where(y_test > 50000,1,0)

    if len(categorical_features) > 0:
        categorical_train = X_train.loc[:, categorical_features]
        categorical_test = X_test.loc[:, categorical_features]

    # Recombine and resplit to calibration
    X_train = np.concatenate([one_hot_train, scaled_train, categorical_train], axis=1)
    X_test = np.concatenate([one_hot_test, scaled_test, categorical_test], axis=1)

    # Convert back to pandas for easier interpretation
    column_names = list(one_hot.get_feature_names_out()) + list(scaler.get_feature_names_out()) + categorical_features
    X_train = pd.DataFrame(X_train, columns=column_names)
    X_test = pd.DataFrame(X_test, columns=column_names)

    X_train, X_calib, y_train, _ = train_test_split(X_train,
                                                    y_train,
                                                    test_size = calib_size,
                                                    random_state = split_seed)
    
    
    # Add sensitive features
    for sens_feat in sensitive_features:
        idx_ = np.where([sens_feat in col_ for col_ in one_hot.get_feature_names_out()])[0][0]

        return_dict[f'sens_{sens_feat}_train'] = X_train.iloc[:,idx_].reset_index(drop=True)
        return_dict[f'sens_{sens_feat}_calib'] = X_calib.iloc[:,idx_].reset_index(drop=True)
        return_dict[f'sens_{sens_feat}_test'] = X_test.iloc[:,idx_].reset_index(drop=True)

    return_dict['X_train'] = X_train.reset_index(drop=True)
    return_dict['X_calib'] = X_calib.reset_index(drop=True)
    return_dict['X_test'] = X_test.reset_index(drop=True)

    if objective == 'regression':
        return_dict['y_train'] = y_train.reset_index(drop=True)
        return_dict['y_test'] = y_test.reset_index(drop=True)
    elif objective == 'classification':
        return_dict['y_train'] = y_train
        return_dict['y_test'] = y_test

    return return_dict