import pandas as pd
import numpy as np
import csv,os,time,argparse
from collections import OrderedDict,Counter

def get_parser():
    parser = argparse.ArgumentParser(description="Demo of FPTree")
    parser.add_argument("-v", dest="verbose", type=bool,help="verbose mode",default=True)
    parser.add_argument("-p", dest="path", type=str, help="path to input csv file",default=r"C:\Users\zjw\OneDrive\HKUST\5002\MSBD5002_Assignment_1\MSBD5002_Assignment_1\groceries.csv")
    parser.add_argument("-s", dest="support", type=int, help="minimum support thresholds",default=300)
    return parser

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
    """ depth first search to return nested list of tree as required
    """
    res=root.name+" "+str(root.cnt)
    return [res,[dfs(v) for k,v in root.children.items()]] if root.children else res

class TreeNode(object):
    """ Node in FPTree
    
    Attributes:
        name: A string of item name
        parent: TreeNode of parent
        cnt: An integer count of item occurence in this path
        children: A dict to store children TreeNode, eg.{TreeNode.name: TreeNode,}. 
    """
    def __init__(self,name='',cnt=1,parent=None):
        self.name=name
        self.cnt=cnt
        self.parent=parent
        self.children={}
    # For display
    def __str__(self):
        return "Name: '%s', Count: %s, Children: %s" %     (self.name, self.cnt, "["+",".join(list(self.children.keys()))+"]")
    __repr__=__str__


class FPTree(object):
    """ 
    Args:
        None
    Attributes:
        support: An integer indicate minimum support thresholds
        header_table: A dict store item with its occurence and Node, eg.{item: (cnt,[nodes])}
        root: A TreeNode instance of main FP tree root
        frequent_patterns: A dict to store frequent patterns with its occurence
        record: A list of TreeNode recording root of conditional FP tree with height more than 1. 
    Method:
        |--@fit
            |--@_process
            |--@_build_tree
                |--@_insert_tree
            |--@_mine_tree
                |--@_get_prefix_path
    """ 
    def fit(self,transactions,support):
        """
        ### Find frequent patterns from transactions ###
        Args:
            transactions: dict type, itemset with occurence, eg.{('a','b'):3,}
            support: Minimum support thresholds
        Return:
            None
        """
        self.support=support
        self.record=[]
        self.frequent_patterns={}
        transactions,self.header_table=self._process(transactions)
        self.root=self._build_tree(transactions,self.header_table)
        self._mine_tree(self.header_table,())
        
    def _process(self,transactions):
        """
        ### Re-rank and filter transactions ###
        Args:
            transactions: A dict to store itemset with occurence, eg.{('a','b'):3,}
        Return:
            new_transactions: A dict to store itemset with occurence which meet frequency order and support thresholds, eg.{('a','b'):3,}
            header_table: An new initialized dict with item name and its occurence, eg.{item: (cnt,[])}
        """
        # Count frequency of each item
        frequency={}
        for transaction,cnt in transactions.items():
            for item in transaction:
                frequency[item] = frequency.get(item,0) + cnt
        frequency={k:v for k, v in frequency.items() if v>=self.support}
        frequent_item=[item for item,cnt in Counter(frequency).most_common()]        
        # Generate new transactions by re-rank and filter frequent item
        rerank = lambda T:tuple(sorted([x for x in T if x in frequent_item],key=lambda x:frequent_item.index(x)))
        new_transactions={}
        for transaction, cnt in transactions.items():
            transaction = rerank(transaction)
            if transaction:
                new_transactions[transaction]=new_transactions.get(transaction,0)+cnt
        # Initialize header_table
        header_table = {k:(v,[]) for k,v in frequency.items()}
        return new_transactions,header_table 
        
    def _build_tree(self,transactions,header_table): 
        """
        ### Build FP Tree while updating header_table ### 
        Args:
            transactions: A dict to store itemset with occurence, eg.{('a','b'):3,}
            header_table: A dict indicate an new initialized header_table
        Return:
            root: TreeNode of root of FP tree
        """
        root = TreeNode('Null Set')
        for transaction,cnt in transactions.items():
            self._insert_tree(root,transaction, cnt, header_table)
        return root

    def _insert_tree(self,root,transaction,cnt, header_table):
        """
        ### Insert transaction into FP tree, more generally update the tree according to the header_table ###
        Args:
            root: A TreeNode indicate where to insert the next item
            transaction: A tuple of itemset, eg.('a','b','c')
            cnt: An integer of the occurence of the itemset
            header_table: A dict store item with its occurence and Node, eg.{item: (cnt,[nodes])}
        Return:
            None(update root)
        """
        if len(transaction)==0: 
            return
        item = transaction[0]
        if item not in root.children:
            root.children[item] = TreeNode(item, 0, root)
            header_table[item][1].append(root.children[item]) # update TreeNode
        this_node = root.children[item]
        this_node.cnt += cnt
        self._insert_tree(this_node,transaction[1:],cnt,header_table)
        
    def _get_prefix_path(self, item, header_table):
        """
        ### Find all prefix path of item in a FP tree ###
        Args:
            item: String of item name
            header_table: A dict store item with its occurence and Node, eg.{item: (cnt,[nodes])}
        Return:
            paths: A dict stored itemsets with their occurence 
        """ 
        paths={}
        for node in header_table[item][1]:
            path = ()
            cnt = node.cnt
            _node = node.parent
            while _node.parent:
                path += (_node.name,)
                _node = _node.parent
            paths[path]=cnt
        return paths
    
    
    def _mine_tree(self, header_table,condition):
        """
        ### Mine frequent pattern from conditional FP tree given condition pattern and header table ###
        Args:
            condition: tuple indicate conditional pattern
        Return:
            None(update self.frequent_patterns)
        """
        for item in header_table.keys():  # start from the least frequent item
            new_condition = condition + (item,)
            self.frequent_patterns[new_condition] = header_table[item][0] # update self.frequent_patterns
            paths = self._get_prefix_path(item,header_table) # paths here is trasactions we need to build FP-tree
            if paths:
                transactions,conditional_header_table=self._process(paths)
                # whether excatly exsit frequent pattern
                if conditional_header_table:
                    conditional_tree=self._build_tree(transactions,conditional_header_table) # build new conditional tree with new condition
                    ############################################################# 
                    # This is a special block to record root of conditional tree with height larger than 1. 
                    # Nothing to do with FP-tree Algorithm. 
                    if max(map(len,transactions.keys()))>=1:
                        self.record.append(conditional_tree)
                    ############################################################# 
                    self._mine_tree(conditional_header_table,new_condition) # mine with iteratively
                




if __name__=='__main__':
    parser=get_parser()
    args=parser.parse_args()
    print("######### Loading data #########")
    transactions=loader(args.path)
    print("######### Computing #########")
    fpt=FPTree()
    start=time.time()
    fpt.fit(transactions,args.support)
    t=time.time()-start
    if args.verbose:
        print("After %f seconds computing, we find:  "%t)
        print("--%d frequent patterns with minimum support threshold %d"%(len(fpt.frequent_patterns), fpt.support))
        print("--%d conditional trees with height more than 1"%len(fpt.record))
        print("######### Saving result #########")
    saver(fpt, "\\".join(args.path.split("\\")[:-1]))
