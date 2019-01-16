# Transimprove
This folder contains the components of the Transimprove-Pipeline. The Pipeline can work with data points as well as associated annotations. With a consistency threshold the given data points will be marked as reliable enough or not. The data points not reaching the threshold can be passed to one to multiple prediction algorithms for relabelling. The prediction algorithm has to provide the functionality of AbstractModeladaptor.py or wrapped with an AbstractModeladaptor.py implementation.

## Inputs
Datapoints with id
```python
data_points = np.array(
    # id   vector 0         vector 1
   [['a' , 'Data for a' , 'Data2 for a' ,...],
    ['b' , 'Data for b' , 'Data2 for b' ,...],
    ['c' , 'Data for c' , 'Data2 for c' ,...]])
```

Annotations with associated id of data point
```python
annotations = np.array(
        # id   label
        [['a', 'Class1'],
         ['a', 'Class1'],
         ['a', 'Class7'],
         ['a', 'Class1'],
         ['b', 'Class2'],
         ['b', 'Class3'],
         ['c', 'Class4']])
```

Modeladaptor: Can be used for any sklearn flavored prediction algorithm. The `translation_dictionary` will map output of the `model` to one class we would like. If the output of the `model` can not be mapped with `translation_dictionary` the result will be `NaN` ant therefore ignored within the Pipeline.
```python
myAdaptor1 = AbstractModeladaptor(
    model=somePredictor(),
    translation_dictionary={'PredictedLabel': 'Class3'})
```


## Pipeline usage
#### Init
```python
pipeline = Pipeline(
                data_points,
                annotation, 
                # name for Debugging              actual adaptor/model
                [('My Adaptor predicting Class3', myAdaptor1            )
                ,....]
            )
```

#### Fit
The pipeline uses majority vote on data point with min consistency of threshold. It will consolidate myAdaptor1 for all remaining data points.

```python
pipeline = Pipeline(...)
# defining the consistency threshold
pipeline.fit(threshold)

# Use majority vote and do NOT consolidate myAdaptor1 for data points with annotations
pipeline.fit(0)

# Do NOT use majority vote and consolidate myAdaptor1 for all data points
pipeline.fit(1.1)  
```

#### Dataset & Training
The `dataset_for_training` can be used with any classifier in sklearn flavor as shown below:
```python
pipeline.fit(0.6)

# print for insight of consistencies
print(pipeline.certain_split)
print(pipeline.uncertain_split)

X, y = pipeline.full_data_set(return_X_y=True)
# [['Data for a' 'Data2 for a']       [['Class1']
#  ['Data for c' 'Data2 for c']        ['Class4']
#  ['Data for b' 'Data2 for b']]       ['Class3']]

classifier = neural_network.MLPClassifier()
classifier.fit(X, y)


```
