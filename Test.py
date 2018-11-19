import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random

from statistic_analysis import rate_annotations_by_datapoint, transform_majority_label, certain_uncertain_split

df_annotations = pd.DataFrame(data={'datapoint_id': ['a', 'a', 'a', 'b', 'a', 'b', 'b', 'c', 'c'],
                        'annotation': ['1', '1', '7', '5', '1', '5', '5','1', '5']})
df_data = pd.DataFrame(index=['a', 'b', 'c'], data=['1', '2', '3'])


df = rate_annotations_by_datapoint(df_annotations)
print('--------------Input-----------------')
print(df)

certain, uncertain = certain_uncertain_split(df, 0.9)
print('--------------Certain Split-----------------')
print(certain)
print('--------------Uncertain Split-----------------')
print(uncertain)

majority_labels = transform_majority_label(certain)
print('--------------Majority Annotation-----------------')
print(majority_labels)
print('-----------Majority Annotation (as np.array)-----------')
print(np.array(majority_labels.reset_index().values))




print('-----------Uncertain Datapoints from original Data-----------')
df_for_model = df_data[df_data.index.isin(uncertain.index.values)]
print(df_for_model.reset_index().values);



# https://www.google.ch/search?q=groupby+apply+getting+multiindex&spell=1&sa=X&ved=0ahUKEwjqmYSuw8DeAhXE0aQKHYtnBZAQBQgqKAA&biw=1299&bih=1040
# https://groups.google.com/forum/#!topic/pydata/WRBsKOHganE
# for key, group in grouped_month:
#     print key
#    ....:
# 1
# 2
