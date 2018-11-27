import pandas as pd
import numpy as np
import sys,os
import pickle
from data_helper import *
from stack_helper import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier

if __name__ == "__main__":
    PATH = r"B:\5002A2\data"
    train, test = load_data(PATH)
    test['label'] = np.nan
    train_X, train_y, test_X = Preprocess(train, test, ohe=True)
    models = {'rfc': RandomForestClassifier(n_estimators=2000, random_state=42, max_depth=7, min_samples_split=20),
              'gbc': GradientBoostingClassifier(n_estimators=2000, learning_rate=0.005, max_depth=4, random_state=42),
              'lgb1': LGBMClassifier(n_estimators=5000, silent=False, random_state=19, max_depth=7,
                                     num_leaves=30, objective='binary', learning_rate=0.005, colsample_bytree=0.85, subsample=0.8, verbose=-1),
              'lgb2': LGBMClassifier(n_estimators=2000, silent=False, random_state=19, max_depth=4,
                                     num_leaves=30, objective='binary', learning_rate=0.01, colsample_bytree=1, subsample=1, verbose=-1)
              }
    stack_train, stack_test = stack(k=5, models=models, train_X=train_X, train_y=train_y, test_X=test_X)
    # Main training Process
    lgb_stack = LGBMClassifier(n_estimators=2000, silent=False, random_state=19, max_depth=4,
                               num_leaves=20, objective='binary', learning_rate=0.005, colsample_bytree=1, subsample=1, verbose=-1).fit(stack_train, train.label)
    stack_pred = lgb_stack.predict(stack_test)
    pd.DataFrame({"a": stack_pred}).to_csv(PATH + "\submission.csv", header=None, index=None)  # save submission to PATH
