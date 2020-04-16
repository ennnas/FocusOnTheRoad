# Focus on the road
> A Deep Learning approach to identify distracted drivers


[![Python 3.7](https://img.shields.io/badge/Python-3.7-green.svg)](https://shields.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

We've all been there: a light turns green and the car in front of you doesn't budge. 
Or, a previously unremarkable vehicle suddenly slows and starts swerving from side-to-side.

The goal of this project is to train a deep learning model <img src="https://render.githubusercontent.com/render/math?math=f = g(h(X))">
that can identify the action that a person is carrying out while driving. The problem is formulated 
as a multi-label classification task using the dataset [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)

In order to obtain the aforementioned model we split this complex task into two smaller and simpler 
subproblems, namely feature extraction and classification, the former is done via a Pytorch 
implementation of [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) while the 
latter via [XGBoost](https://xgboost.readthedocs.io/en/latest/) as shown in the image below.

![](figures/workflow.png)

To read more about this project have a look at:
- [Project proposal](docs/proposal.pdf)
- [Report]() #TODO

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

You now have access to these entry points (executables) in your virtual environment:

* `demo`: description of what it does 

All these entry points accept multiple options. To obtain documentation for any one of them, run it with the `-h` flag.
Example:

```bash
$ demo -h

PUT THE SHELL OUTPUT HERE
```

## Usage example

A few motivating and useful examples of how your product can be used. Spice this up with code blocks and potentially more screenshots.


## Release History

* 0.1.0
    * Work in progress

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


