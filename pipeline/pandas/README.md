\# Pandas Project - pipeline/pandas



\## General Concepts



\### What is pandas?

Pandas is a powerful Python library used for data manipulation, analysis, and cleaning. It provides fast, flexible, and easy-to-use data structures designed to work with structured data such as tables and time series.



\### What is a pd.DataFrame? How do you create one?

A `pd.DataFrame` is a 2-dimensional labeled data structure with rows and columns, similar to a spreadsheet or SQL table.  

You can create one using:

```python

import pandas as pd

data = \[\[1, 2], \[3, 4]]

df = pd.DataFrame(data, columns=\["A", "B"])



