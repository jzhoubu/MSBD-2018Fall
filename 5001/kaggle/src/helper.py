import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from lightgbm.basic import Dataset
import lightgbm as lgb
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_predict,KFold
from collections import OrderedDict

def loader(type='1027'):
    local_time = pd.read_pickle("./data/LocalComputeTime{}.pkl".format(type))
    train = pd.read_csv("./data/new_train.csv")
    test=pd.read_csv("./data/test.csv")
    test['time']=np.nan
    data=pd.concat([train,test],axis=0).reset_index(drop=True)
    data=pd.concat([pd.get_dummies(data['penalty']),data], axis=1)
    data=pd.concat([data,local_time.loc[:,["timeN1","timeN2","timeN4","timeN8","timeN16"]]],axis=1)
    data.drop(['id','random_state','penalty'],axis=1,inplace=True)
    data.n_jobs=data.n_jobs.replace(-1,16)   # 假设 16 核
    for ind,row in data.iterrows():
        n=row['n_jobs']
        data.loc[ind,'computing_mean'] = row["timeN"+str(row['n_jobs'])].average
        data.loc[ind,'computing_std'] = row["timeN"+str(row['n_jobs'])].stdev
        data.loc[ind,'time_residual1'] = row["timeN1"].average - row["timeN"+str(n)].average*n
        data.loc[ind,'time_residual2'] = row["timeN2"].average - row["timeN"+str(n)].average*n/2
        data.loc[ind,'time_residual4'] = row["timeN4"].average - row["timeN"+str(n)].average*n/4
        data.loc[ind,'time_residual8'] = row["timeN8"].average - row["timeN"+str(n)].average*n/8
        data.loc[ind,'time_residual16'] = row["timeN16"].average - row["timeN"+str(n)].average*n/16
        data.loc[ind,'queue_N2'] = (row["timeN2"].average - row["timeN1"].average / 2) 
        data.loc[ind,'queue_N4'] = (row["timeN4"].average - row["timeN1"].average / 4) 
        data.loc[ind,'queue_N8'] = (row["timeN8"].average - row["timeN1"].average / 8) 
        data.loc[ind,'queue_N16'] = (row["timeN16"].average - row["timeN1"].average / 16) 
    return data

def stack(k, models, train_X, train_y, test_X):
    kf=KFold(n_splits=k,random_state=42)
    kf.get_n_splits(train_X)
    n=len(models)
    stack_train=pd.DataFrame(np.zeros([train_X.shape[0],n]),columns = models.keys())
    stack_test=pd.DataFrame(np.zeros([test_X.shape[0],n]),columns = models.keys())
    kf=KFold(n_splits=k, random_state=42)
    kf.get_n_splits(train_X)
    for ind, (train_index, valid_index) in enumerate(kf.split(train_X)):
        train_X_kf=train_X.loc[train_index,:]
        train_y_kf=train_y[train_index]
        valid_X_kf=train_X.loc[valid_index,:]
        for model_ind, (name, model) in enumerate(models.items()):
            model=model.fit(train_X_kf, train_y_kf)
            stack_train.loc[valid_index,name]=model.predict(valid_X_kf)
            stack_test.loc[:,name]+=model.predict(test_X)
    stack_test = stack_test.apply(lambda x:x/k,axis=1)   # avg on test
    return stack_train, stack_test
