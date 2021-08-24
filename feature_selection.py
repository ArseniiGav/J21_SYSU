import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

path='/mnt/cephfs/ml_data/mc_2021/'
data_real = pd.read_csv('{}processed_data/ProcessedTrainReal/ProcessedTrain_1M.csv.gz'.format(path))
data_real = data_real[data_real['edepR'] < 17.2]

size = int(1e6)
n_feats = len(data_real.columns) - 5

X_val = data_real.iloc[:, :-5][size:]
y_val = data_real.iloc[:, -5][size:]
data_real = data_real[:size]

all_features_metric = np.load('feature_selection/all_features_metrics.npz', allow_pickle=True)['a'][0][0]
eps = np.load('feature_selection/all_features_metrics.npz', allow_pickle=True)['a'][0][1]
opt_features = ['AccumCharge', 'R_cht', 'jacob_cc', 'pe_std', 'nPMTs']#[]
current_metrics = [3.8076375114186476, 1.6799966127524213, 1.381430894229591, 1.2331565442044647, 1.2229199895455005]#[]
current_metric_stds = [0.004450264146659087, 0.0022321956920953352, 0.0025158649260755114, 0.0019016478876169257, 0.0022581303993379166]#[]
current_metric = 100

features = data_real.iloc[:, :-5].columns
features = features.drop(opt_features)
while abs(all_features_metric - current_metric) > eps:
    metrics = []
    metric_stds = []
    for feature in tqdm(features, "Features loop"):
        
        X = data_real.iloc[:, :-5][opt_features+[feature]]
        y = data_real.iloc[:, -5]
        
        xgbreg = XGBRegressor(
            max_depth=9,
            learning_rate=0.08,
            n_estimators=3000,
            random_state=22,
        )
        
        scores = cross_val_score(
            xgbreg,
            X,
            y,
            cv=5,
            n_jobs=5,
            verbose=False,
            fit_params={
                'eval_set': [(X_val[opt_features+[feature]], y_val)],
                'early_stopping_rounds':5
            },
            scoring='neg_mean_absolute_percentage_error'
        )
        
        metric = -100*scores.mean()
        metric_std = (100*scores).std()
        metrics.append(metric)
        metric_stds.append(metric_std)
        
        print(metrics)
        print(metric_stds)
        print(feature)
    
    best_metric_ind = np.argmin(metrics)
    current_metric = metrics[best_metric_ind]
    current_metrics.append(current_metric)

    current_metric_std = metric_stds[best_metric_ind]
    current_metric_stds.append(current_metric_std)
    
    opt_features.append(features[best_metric_ind])
    features = features.drop(features[best_metric_ind])

    print(current_metrics)
    print(current_metric_stds)
    print(opt_features)
    
    np.savez_compressed('feature_selection/opt_features.npz', a=np.array(opt_features))
    np.savez_compressed('feature_selection/current_metrics.npz', a=np.array(current_metrics))
    np.savez_compressed('feature_selection/current_metric_stds.npz', a=np.array(current_metric_stds))
