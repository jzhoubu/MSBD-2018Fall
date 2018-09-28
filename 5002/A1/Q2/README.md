# FP-tree ![CI status](https://img.shields.io/badge/build-passing-brightgreen.svg)
This is a FP tree toy demo.

## Usage

```python
#!/usr/bin/python3
from FPTree import *
transactions=loader(CSV_PATH)  
fpt=FPTree()
fpt.fit(transactions, SUPPORT)    
fpt.frequent_patterns       
```



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
