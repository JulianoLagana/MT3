## Introduction
This repository contains the code for the paper "Next Generation Multitarget Trackers: Random Finite Set Methods vs Transformer-based Deep Learning" (https://arxiv.org/abs/2104.00734).



## Setting up
In order to set up a conda environment with all the necessary dependencies, run the command:
  ```
   conda env create -f conda-env/environment-<gpu/cpu>.yml
  ```





## Running experiments

Run an experiment using the `train.py` script. Example usage:

```
src/training.py -tp configs/tasks/task1.yaml -mp configs/models/mt3.yaml
```

Training hyperparameters such as batch size, learning rate, checkpoint interval, etc, are found in the file `configs/models/mt3.yaml`. 



## Evaluating experiments

After an experiment has generated checkpoints, you can evaluate its average GOSPA score using the `eval.py` script. The evaluation hyperparameters can be found inside `configs/eval/default.yaml`. Example usage: 

```
src/eval.py -rp src/results/experiment_name -tp configs/tasks/task1.yaml
```

