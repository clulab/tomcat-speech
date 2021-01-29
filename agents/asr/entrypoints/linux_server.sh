#!/bin/sh

# Entrypoint for deployment on a Linux server. 

# To make the agent accessible from a remote computer (i.e. not localhost), we
# have to enable SSL for extra security (modern web browsers will not allow
# audio capture and transmision without it).

# We add extra options for enabling encrypted websocket connections - we
# provide paths to the SSL certificate and key file using environment
# variables.

# The invocation below starts the ASR agent in its websockets mode, and
# outputs messages to two sources:
#  (i) it appends messages to a local data file 'data/asr_messages.txt', and
# (ii) it also publishes messages to a running instance of the Mosquitto
#      message broker.

# If running inside a Docker container, make sure that the broker is running on
# the same network as the container.
./tomcat_asr_agent websockets\
  --ws_host 0.0.0.0 --ws_port 8888\
  --ssl_cert_chain $SSL_CERT_CHAIN --ssl_keyfile $SSL_KEYFILE\
    | tee -a data/asr_messages.txt | mosquitto_pub -t agents/asr -l -h mosquitto
