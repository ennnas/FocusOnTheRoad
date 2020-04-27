# Focus on the road
> A Deep Learning approach to identify distracted drivers


[![Python 3.7](https://img.shields.io/badge/Python-3.7-green.svg)](https://shields.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

We've all been there: a light turns green and the car in front of you doesn't budge. 
Or, a previously unremarkable vehicle suddenly slows and starts swerving from side-to-side.

The goal of this project is to train a deep learning model <img src="https://render.githubusercontent.com/render/math?math=f = g(h(X))">
that can identify the action that a person is carrying out while driving. The problem is formulated 
as a multi-label classification task using the dataset [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)

In order to obtain the aforementioned model we opt for an ensemble approach, building a classifier 
on top of two weak learners, namely a CNN and a Gradient Boosting classifier that relies solely on Human Pose estimation features. 
The HPE implementation uses [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) while the 
Gradient Boosting classifier is based on [CatBoost](https://catboost.ai/).

The complete pipeline is available in the following jupyter notebook [notebooks/FocusOnTheRoad - Final](notebooks/FocusOnTheRoad - OpenPose.ipynb) 

To read more about this project have a look at:
- [Project proposal](docs/proposal.pdf)
- [Report](docs/report.pdf)

## Development setup

You need to install the following software:

* Python ≥ 3.7 (older versions are not supported)
* [poetry](https://python-poetry.org/) ≥ 1.0

### Poetry installation
To install and configure `poetry` run 
```shell script
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3
poetry config virtualenvs.in-project true
```

## Installation

First, to install all the required dependencies run
```shell script
poetry install
```

Download the dataset from the [Kaggle competition](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data) and unzip it in the `data` folder

Download the pretrained weights for the OpenPose model from [here](https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABaYNMvvNVFRWqyDXl7KQUxa/body_pose_model.pth)

Follow the jupyter notebook [notebooks/FocusOnTheRoad - Final](notebooks/FocusOnTheRoad - OpenPose.ipynb) to execute the complete pipeline

## Baseline example

To train a CNN baseline use the `baseline.py` script which accepts the following arguments

```shell script
$python baseline.py -h
usage: baseline.py [-h] [--num-epochs NUM_EPOCHS] [--batch-size BATCH_SIZE]
                   [--optimizer {sgd,adagrad}] [--lr LR]
                   [--train-size TRAIN_SIZE] [--val-size VAL_SIZE]
                   [--submission]

Train a baseline model on the StateFarmDataset.

optional arguments:
  -h, --help            show this help message and exit
  --num-epochs NUM_EPOCHS
                        the number of epochs to train the model
  --batch-size BATCH_SIZE
                        the batch size for the DataLoader
  --optimizer {sgd,adagrad}
                        the optimizer to use for training
  --lr LR               the learning rate for the optimizer
  --train-size TRAIN_SIZE
                        the number of images per class to use for training
  --val-size VAL_SIZE   the number of images per class to use for validation
  --submission          Use the fitted model to compute the submission.csv
                        file

```
## Meta

Ennio Nasca – [LinkedIn](https://www.linkedin.com/in/ennio-nasca)

[https://github.com/ennnas/FocusOnTheRoad](https://github.com/ennnas/FocusOnTheRoad)

## Contributing

Tho contribute to this project follow this steps:

1. Fork it (<https://github.com/ennnas/FocusOnTheRoad/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request


