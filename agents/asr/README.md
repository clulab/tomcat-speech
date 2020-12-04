ToMCAT ASR Agent
================

Python package requirements
---------------------------

To access the microphone, you'll need the `pyaudio` package:

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
