import numpy as np
import pandas as pd
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')



def load_data(path):
    train=pd.read_csv(path+"/trainFeatures.csv")
    label=pd.read_csv(path+"/trainLabels.csv",names=['label'])
    train=pd.concat([train, label], axis=1)
    train.label=train.label.astype(int)
    test = pd.read_csv(path + "/testFeatures.csv")
    return train, test


def categorical_process(df, feats):
    for feat in feats:
        if feat not in df:
            continue
        ohe = pd.get_dummies(df[feat])
        ohe.columns = [feat+"_"+x.strip() for x in ohe.columns.tolist()]
        df=pd.concat([df,ohe],axis=1)
    return df

def train_test_ohe(train, test, feats, label):
    assert train.shape[1]==test.shape[1]
    data = pd.concat([train,test],axis=0)
    data_ohe = categorical_process(data, feats=feats)
    for feat in feats:
        woe = WOE(train, feat, label)
        data_ohe[feat] = data_ohe[feat].apply(lambda x:woe.get(x))
    train_ohe = data_ohe.iloc[:train.shape[0], :]
    test_ohe = data_ohe.iloc[train.shape[0]:, :]
    return train_ohe, test_ohe

def train_test_woe(train, test, feats, label):
    assert train.shape[1] == test.shape[1]
    data = pd.concat([train, test], axis = 0)
    for feat in feats:
        woe = WOE(train, feat, label)
        data[feat] = data[feat].apply(lambda x:woe.get(x))
    train = data.iloc[:train.shape[0], :]
    test = data.iloc[train.shape[0]:, :]
    return train, test


def WOE(df, feat, target):
    info = df.groupby(by=[feat],as_index=False)[target].agg({"cnt": "count", "sum": "sum"})
    woe = dict(zip(info[feat].tolist(), np.log(info['sum'].div(info['cnt']))))
    woe = {k: v if v != -np.inf else -999 for k, v in woe.items()}
    woe = OrderedDict(sorted(woe.items(), key=lambda x: x[1], reverse=True))
    return woe

def ratio_df(data, L=('capital-gain','capital-loss','age','hours-per-week','fnlwgt')):
    assert all([x in data for x in L])
    df=data.copy()
    for i in range(5):
        for j in range(i+1,5):
            name=L[i]+"_div_"+L[j]
            df[name]=(df[L[i]]+0.05).div((df[L[j]]+0.05))
    return df

def Preprocess(train, test, ohe=True):
    train = ratio_df(train)
    test = ratio_df(test)
    categorical_features = ['workclass', 'education', 'Marital-status', 'race', 'sex', 'relationship', 'occupation', 'native-country']
    if ohe:
        train_ohe, test_ohe = train_test_ohe(train, test, categorical_features, 'label')
    else:
        train_ohe, test_ohe = train_test_woe(train, test, categorical_features, 'label')
    train_X = train_ohe.drop("label", axis=1).apply(lambda x: pd.to_numeric(x, errors='raise'))
    train_y = train_ohe['label'].astype(int)
    valid_X = test_ohe.drop("label", axis=1).apply(lambda x: pd.to_numeric(x, errors='raise'))
    return train_X, train_y, valid_X


if __name__  ==  "__main__":
    print(1)
