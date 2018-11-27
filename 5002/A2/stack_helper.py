import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier

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
        print("{:#^40}".format(" Stack {}/{} ".format(ind+1,k)))
        for model_ind, (name, model) in enumerate(models.items()):
            print("{:#^40}".format(" Training Model {}/{}: {} ".format(model_ind+1, n, name)))
            # update stack_train
            model=model.fit(train_X_kf, train_y_kf)
            stack_train.loc[valid_index,name]=[x[1] for x in model.predict_proba(valid_X_kf)]
            # update stack_test
            print("{:#^40}".format(" Predicting Model {}/{}: {} ".format(model_ind+1, n, name)))
            stack_test.loc[:,name]+=[x[1] for x in model.predict_proba(test_X)]
    stack_test = stack_test.apply(lambda x:x/k,axis=1)   # avg on test
    return stack_train, stack_test