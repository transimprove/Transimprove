- How to do the retraining of an existing model for transfer learning (retrain last layer)
    - Use consistent data points
    - Use an expert to create a small gt dataset

- rerun experiments on CIFAR-10
- rename blue lines to baseline
- rename red to upper bound or sth
- measure plug-in model trained with 0.1 and 0.05 split
- set lr to 0.001


- Future:
    - Recolor: Orange to light-green
    - Recolor red to black
    - Red line: Re-Train dark-green model with whatever the blue gets for training
      - --load-model %PATH_TO_DARK_GREEN_TAR%
      - Rest is same as orange
    - Dark-Grey line: Train with all majority voting labelled data
    - Light-Grey: Similar as orange but use dark-grey for getting the labels

    - Active Learning with the Network Output as suggestions
      - "Semi Experts" for resolving the conflicts
      - "Real Domain Expert" for resolving conflicts of "Semi Experts"
      - Weight labels according to performance of annotator

- Bugs:
  - Search for a bug when using densenet and it starts giving suggestions

- Speedup experiments:
  - run every 10% and only 1 run at each point