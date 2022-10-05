# tomcat-speech

This repository contains Python code to prepare multimodal (currently audio,
text, speaker) data for input into models and to instantiate and run these
models.

## Requirements

* Python 3.7 or higher.
* OpenSMILE: https://www.audeering.com/opensmile/ (the code has been tested with
  OpenSMILE v2.3.0). If you don't already have it, you can download it by
  running the following:
  ```
  ./scripts/download_opensmile
  ```
* GloVE: https://nlp.stanford.edu/projects/glove/

You will also need `ffmpeg` to use some of the modules. You can install it with
your OS package manager (e.g. MacPorts/Homebrew/apt-get/yum)

## Installation

End-users:

    pip install .

For developers, we recommend running the following invocation in a virtual
environment to install the project as an editable package.

    pip install -e .


**Note for MacPorts users only**: `pip install torchtext` is broken, so we recommend
running `sudo port install py-torchtext` to install `torchtext` system-wide,
and when you create a virtual environment, have it inherit the system-wide
packages with the `--system-site-packages` flag, like shown below:

    python -m venv --system-site-packages path/to/venv

## Training and testing multimodal, single- or multitask models

Below is information on how to train and test models within the tomcat_speech repository.

### Preparation

To make use of the data preparation modules or to run models, clone the repo `multimodal_data_preprocessing` at https://github.com/jmculnan/multimodal_data_preprocessing and place it in the same parent directory as this repo. 

```
E.g.: 
.
|__ mmml
|__ multimodal_data_preprocessing
    |__ datasets_to_use

```

### Running models 

Models are trained using the training scripts in the `tomcat_speech/train_and_test_models` subdirectory. 

The training scripts are associated with parameters files, which may be found in `tomcat_speech/models/parameters`. Alter the paths in the relevant parameters file as needed to run the training code of interest. 

You may either train a model with a single set of parameters (using `train_multitask.py` or `train_single_task.py`) or you may run a grid search by training on multiple parameter values using `grid_search_train.py`.



Multimodal Speech Encoder
-------------------------


### Installation


If you want to use the encoder as a Dockerized web service, skip to the
'With Docker' section below. Otherwise, read on.

You can install the dependencies and download the pretrained model by running

    ./scripts/mmc/install

### Usage

#### Web service

##### Without Docker

You can run the encoder as a web service by running the
following script:

    ./scripts/mmc/run_mmc_server

The app will run on `http://localhost:8001` by default. To see the
automatically generated API documentation, visit `http://localhost:8001/docs`.

##### With Docker

To run the classifier as a Dockerized service, run the following invocation:

    docker-compose up --build

This will run the service on localhost:8001 by default. You can change the port
by changing the port mapping in the `docker-compose.yml` file.

##### Testing web service

The script, `scripts/mmc/test_mmc_server` demonstrates an example HTTP GET request
to the `/encode` endpoint (the only one that is currently implemented) to get
an encoding.
