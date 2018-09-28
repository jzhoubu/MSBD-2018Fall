#!/usr/bin/python3
import os
import pandas as pd
import numpy as np
from collections import Counter
from data_helper import loader, saver, dfs

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
        L=[v[0] for v in sorted(header_table.items(), key=lambda kv:kv[1][0])]
        for item in L:   # start from the least frequent item
            new_condition = condition + (item,)
            self.frequent_patterns[new_condition] = header_table[item][0]    # update self.frequent_patterns
            paths = self._get_prefix_path(item,header_table)    # paths here is trasactions we need to build FP-tree
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
    CSV_PATH=os.getcwd()+"\\groceries.csv"   # set CSV_PATH to your csv file
    SUPPORT=300   # set SUPPORT threshold
    SAVE_PATH=os.getcwd()  # set path to save frequent patterns result

    transactions=loader(CSV_PATH)  
    fpt=FPTree()
    fpt.fit(transactions, SUPPORT)    
    fpt.frequent_patterns       # frequent patterns
    for i in fpt.record:   # embed each root of conditional trees we find
        dfs(i)    
    saver(fpt, SAVE_PATH) 