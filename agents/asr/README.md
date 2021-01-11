ToMCAT ASR Agent
================

Agent that listens to the microphone and outputs messages corresponding to
real-time ASR transcriptions. The transcriptions are printed to standard output
by default, but can optionally be published to an MQTT message bus instead.

Example usage:

    ./tomcat_asr_agent

To see all available options, run:

    ./tomcat_asr_agent -h


Prerequisites
-------------

- You'll need Python 3.6 or newer.
- To access the microphone, you'll need the `pyaudio` package:

      pip install pyaudio

    On Ubuntu, you'll need to install portaudio before installing pyaudio. You can do so with the following command:

      sudo apt-get install portaudio19-dev

### Google Cloud engine

If you are using the Google Cloud speech recognition engine:

    pip install google-cloud-speech

To use the Google Cloud Speech Recognition engine, you will need to point the
environment variable GOOGLE_APPLICATION_CREDENTIALS to point to your Google
Cloud credentials file.

### PocketSphinx engine

If you are using the PocketSphinx engine:

    ./install_pocketsphinx_python

### Websocket server mode

To run the agent in the websocket server mode, you'll need the `websockets`
Python package (`pip install websockets`)

Docker instructions
-------------------

You can launch a containerized version of the agent that publishes to an MQTT
message bus using Docker Compose:

    docker-compose up --build
