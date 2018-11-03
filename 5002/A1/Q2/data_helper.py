import os
import csv
import pandas as pd
from collections import Counter

def loader(path,verbose=True):
    """Load data and count
    """
    with open(path, newline='') as f:
        transactions=[]
        reader = csv.reader(f)
        for row in reader:
            row=[x for x in row if x]
            transactions.append(row)
    flatten = lambda L: [item for sublist in L for item in sublist]
    if verbose:
        print("There are totally %i transactions"%len(transactions))
        print("Among them, %i different types of %i items have been purchased"%(len(set(flatten(transactions))),len(flatten(transactions))))
        # print("\nData structure of transactions:")
        # print(dict(list(transactions.items())[:10]))
    return Counter(list(map(tuple,transactions)))


def saver(fpt,path):
    """   Save the result as required
    Args:
        fpt: FPTree instance with attribute frequent patterns
        path: path to generate result.csv
    """
    f=lambda s:"'"+s+"'"
    fp=list(map(lambda x:"{" + ', '.join(map(f,list(x)))+ "}",fpt.frequent_patterns.keys()))
    if not os.path.exists(path+"\\results.csv"):
        print("result.csv has been saved at " + path)
        pd.DataFrame({'patterns':fp}).to_csv(path+"\\results.csv",header=False,index=False)
    else:
        print("result.csv already exist in "+ path)


def dfs(root):
    """ depth search first to embed tree into a nested list
    """
    res=root.name+" "+str(root.cnt)
    return [res,[dfs(v) for k,v in root.children.items()]] if root.children else res