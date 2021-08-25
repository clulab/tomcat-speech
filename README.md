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

## Contents of the tomcat_speech directory.

Under the `tomcat_speech` directory, we have a few subdirectories, described
below.

### data_prep contains:
- audio_extraction.py : code to extract necessary features from audio +
  corresponding transcriptions
- data_prep.py : classes to prepare data for input into the models; mostly used
  for health outcomes data currently
- meld_input_formatting.py : Formats MELD dataset for input into models
- subset_glove.py : code to create a subset of GloVe for faster loading at test
  time


### models contains:
- parameters/ : directory of parameters files used by the models
- baselines.py : model classes to be used for baselines
- bimodal_models.py : model classes to be used with bimodal data
- input_models.py : model classes used for preparing input representations (not
  yet implemented)
- plot_training.py : initial plotting of training and dev loss/accuracy curves
- train_and_test_models.py : code for training + evaluation of models


### models_test contains:
- glove_test.py : test usage of subset_glove.py
- input_manipulation_test.py : test usage of audio_extraction.py
- meld_test.py : test usage of running a model with MELD
- model_test.py : the main file for running models with health outcomes data

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
