#!/bin/sh

# Entrypoint for deployment on vanga.sista.arizona.edu. We add extra options
# for enabling https - we provide paths to the SSL certificate and key file
# using environment variables.

./tomcat_asr_agent websockets\
  --ws_host 0.0.0.0 --ws_port 8888\
  --ssl_cert_chain $SSL_CERT_CHAIN --ssl_keyfile $SSL_KEYFILE\
    | tee -a data/asr_messages.txt | mosquitto_pub -t agents/asr -l -h mosquitto
