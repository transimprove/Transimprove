# data-scripts

This project is managed with conda. Use the environment.yaml to automatically install all dependencies:
```bash
conda env create -f environment.yml
```
To use this environment in pycharm select the created conda environment under ```project interpreter```.

If a new dependency is added just overwrite the environment file:
```bash
source activate ENV_NAME # or use direnv see https://github.com/direnv/direnv
conda env export > environment.yaml

```

## How to start an experiment in detached mode

```bash
cd IP5_DataQuality/philipp
source activate deepdiva
nohup python Experiments/DeepDivaMnistExperiment.py &
```

## How to start tensorboard in detached mode
    nohup tensorboard --logdir /home/bonseyes/IP5_DataQuality/experiments --port 50102 &
