# HashTree ![CI status](https://img.shields.io/badge/build-passing-brightgreen.svg)
This is a toy demo.

## Usage
```python
#!/usr/bin/python3
def HashTree(L,max_leaf_size=3,i=0):
    if len(L)<=max_leaf_size or i>=len(L[0]):
        return L
    reduce = lambda L:L[0] if len(L)==1 else L
    L1=reduce([x for x in L if x[i]%3==1])
    L2=reduce([x for x in L if x[i]%3==2])
    L3=reduce([x for x in L if x[i]%3==0])
    return [HashTree(L1,max_leaf_size,i+1),HashTree(L2,max_leaf_size,i+1),HashTree(L3,max_leaf_size,i+1)]

transactions="{1 2 4}, {1 2 9}, {1 3 5}, {1 3 9}, {1 4 7}, {1 5 8}, {1 6 7}, {1 7 9}, {1 8 9},\
             {2 3 5}, {2 4 7}, {2 5 6}, {2 5 7}, {2 5 8}, {2 6 7}, {2 6 8}, {2 6 9}, {2 7 8}, \
             {3 4 5}, {3 4 7}, {3 5 7}, {3 5 8}, {3 6 8}, {3 7 9}, {3 8 9}, \
             {4 5 7}, {4 5 8}, {4 6 7}, {4 6 9}, {4 7 8}, \
             {5 6 7}, {5 7 9}, {5 8 9}, {6 7 8}, {6 7 9}"
transactions=list(map(lambda x:list(map(int,x.strip()[1:-1].split())),transactions.split(",")))   # process string to list of list
HashTree(transactions)  # Embed list of list recursively
```

returns 

```
[[[[1, 4, 7], [1, 7, 9], [4, 7, 8]],
  [[[1, 2, 4], [4, 5, 7]], [[1, 5, 8], [4, 5, 8]], [[1, 2, 9], [1, 8, 9]]],
  [[[1, 6, 7], [4, 6, 7]], [1, 3, 5], [[1, 3, 9], [4, 6, 9]]]],
 [[[2, 4, 7], [2, 7, 8], [5, 7, 9]],
  [[2, 5, 7], [2, 5, 8], [[2, 5, 6], [5, 8, 9]]],
  [[[2, 6, 7], [5, 6, 7]], [[2, 3, 5], [2, 6, 8]], [2, 6, 9]]],
 [[[3, 4, 7], [[3, 4, 5], [6, 7, 8]], [[3, 7, 9], [6, 7, 9]]],
  [[3, 5, 7], [3, 5, 8], [3, 8, 9]],
  [3, 6, 8]]]
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
