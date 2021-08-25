Multimodal Speech Encoder
=========================


Installation
------------

If you want to use the classifier as a Dockerized web service, skip to the
'With Docker' section below. Otherwise, read on.

You can install the dependencies and download the pretrained model by running

    ./scripts/install

Usage
-----

### Web service

#### Without Docker

You can run the dialogue act classifier as a web service by running the
following script:

    ./scripts/run_mmc_server

The app will run on `http://localhost:8000` by default. To see the
automatically generated API documentation, visit `http://localhost:8000/docs`.

#### With Docker

To run the classifier as a Dockerized service, run the following invocation:

    docker-compose up --build

This will run the service on localhost:8000 by default. You can change the port
by changing the port mapping in the `docker-compose.yml` file.

#### Testing web service

The script, `scripts/test_mmc_server` demonstrates an example HTTP GET request
to the `/encode` endpoint (the only one that is currently implemented) to get
an encoding.
