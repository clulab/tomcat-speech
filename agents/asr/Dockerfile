# Default Dockerfile for the ASR agent. The entrypoint is not specified here,
# but can be set in a docker-compose.yml file.

from python:3.7

run apt-get update

# We install the mosquitto clients in order to optionally plug into a MQTT
# message bus to communicate with other AI components.
run apt-get -y install mosquitto-clients libsndfile1-dev

# Installing requirements for working with Google Cloud Speech API and starting
# up the ASR agent in its websockets mode.
run pip install\
    websockets\
    numpy\
    google-cloud-speech\
    pysoundfile

workdir asr
copy . .
