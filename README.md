# RIS - Ranking-based Instance Selection for Pattern Recognition

RIS is a novel instance selection algorithm that attributes a score per instance that depends on its relationship with all other instances in the training set. To read more about RIS algorithm, please consider the following paper:

CAVALCANTI, George DC; SOARES, Rodolfo JO. [Ranking-based Instance Selection for Pattern Classification](https://doi.org/10.1016/j.eswa.2020.113269). Expert Systems with Applications, p. 113269, 2020.

## Getting Started

These instructions will get you replicate the experiments carry out on your local machine and reported in RIS paper.

### Prerequisites

Firstly, go to folder src/algorithm/instance_selection/ris and run the code below:

```
user@src/algorithm/instance_selection/classification/ris> python setup.py build_ext --inplace
```

This code above will compile helper functions of RIS, a cython implementation.

### Datasets

All datasets used in the experiments are available in [Knowledge Extraction based on Evolutionary Learning](https://sci2s.ugr.es/keel/datasets.php).

## Running experiments

To apply RIS method over a dataset, you should run the follow code on your machine:

```
user@ris> python run.py
```
The code above will running RIS implementation over all datasets listed in *datasets* variable inside the run.py