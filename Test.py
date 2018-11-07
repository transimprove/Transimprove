import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random


df = pd.DataFrame(data={'datapoint_id': ['a','a','a','b','a','b','b'],
                        'annotation':   ['1','1','7','5','1','5','5']})

def f(group):
    calculation = group.value_counts(normalize=True)
    return pd.DataFrame(calculation).T

dk = df.groupby('datapoint_id')['annotation'].value_counts(normalize=True).unstack().fillna(0)
print(df)
print(dk)

# https://www.google.ch/search?q=groupby+apply+getting+multiindex&spell=1&sa=X&ved=0ahUKEwjqmYSuw8DeAhXE0aQKHYtnBZAQBQgqKAA&biw=1299&bih=1040
# https://groups.google.com/forum/#!topic/pydata/WRBsKOHganE
# for key, group in grouped_month:
#     print key
#    ....:
# 1
# 2


