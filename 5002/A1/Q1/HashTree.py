
# coding: utf-8

# # Q1(a). Hash Tree (35 marks)

# **Suppose we have 35 candidate item sets of length 3** <br>
# {1 2 4}, {1 2 9}, {1 3 5}, {1 3 9}, {1 4 7}, {1 5 8}, {1 6 7}, {1 7 9}, {1 8 9},<br>
# {2 3 5}, {2 4 7}, {2 5 6}, {2 5 7}, {2 5 8}, {2 6 7}, {2 6 8}, {2 6 9}, {2 7 8},<br>
# {3 4 5}, {3 4 7}, {3 5 7}, {3 5 8}, {3 6 8}, {3 7 9}, {3 8 9},<br>
# {4 5 7}, {4 5 8}, {4 6 7}, {4 6 9}, {4 7 8},<br>
# {5 6 7}, {5 7 9}, {5 8 9}, {6 7 8}, {6 7 9}

# In[1]:


transactions="{1 2 4}, {1 2 9}, {1 3 5}, {1 3 9}, {1 4 7}, {1 5 8}, {1 6 7}, {1 7 9}, {1 8 9},             {2 3 5}, {2 4 7}, {2 5 6}, {2 5 7}, {2 5 8}, {2 6 7}, {2 6 8}, {2 6 9}, {2 7 8},              {3 4 5}, {3 4 7}, {3 5 7}, {3 5 8}, {3 6 8}, {3 7 9}, {3 8 9},              {4 5 7}, {4 5 8}, {4 6 7}, {4 6 9}, {4 7 8},              {5 6 7}, {5 7 9}, {5 8 9}, {6 7 8}, {6 7 9}"
transactions=list(map(lambda x:list(map(int,x.strip()[1:-1].split())),transactions.split(",")))


# In[2]:


def HashTree(L,max_leaf_size=3,i=0):
    """ A recursive function to construct nested list 
    Args:
        L: list-of-list input
        max_leaf_size: integer
        i: iteration time
    Return:
        A NestedList in present of HashTree 
    """
    if len(L)<=max_leaf_size or i>=len(L[0]):
        return L
    reduce = lambda L:L[0] if len(L)==1 else L
    L1=reduce([x for x in L if x[i]%3==1])
    L2=reduce([x for x in L if x[i]%3==2])
    L3=reduce([x for x in L if x[i]%3==0])
    return [HashTree(L1,max_leaf_size,i+1),HashTree(L2,max_leaf_size,i+1),HashTree(L3,max_leaf_size,i+1)]


# In[3]:


HashTree(transactions)


# In[4]:


# Draw tree

