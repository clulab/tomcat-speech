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

    sudo apt-get install portaudio19-dev
    
    pip install pyaudio

### Google Cloud engine

If you are using the Google Cloud speech recognition engine:
    
    pip install google-cloud-speech

To use the Google Cloud Speech Recognition engine, you will need to point the
environment variable GOOGLE_APPLICATION_CREDENTIALS to point to your Google
Cloud credentials file.

### PocketSphinx engine

If you are using the PocketSphinx engine:

    ./install_pocketsphinx_python
    
### MQTT message bus publishing
If you enable publishing to an MQTT message bus (with the --use_mqtt
option), you'll need to install the Eclipse Paho Python client library:

    pip install paho-mqtt
