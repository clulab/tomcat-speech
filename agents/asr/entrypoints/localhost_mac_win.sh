#!/bin/sh

# Entrypoint for working with the agent on a local computer (macOS or Windows).

# On Linux machines, host.docker.internal does not resolve to the local
# computer's IP address (as of 2020-01-26). On Linux, you would need to set
# network_mode: host in your Docker Compose file, and adjust the Mosquitto host
# appropriately.


cmd ./tomcat_asr_agent websockets --ws_host 0.0.0.0 --ws_port 8888\
    | mosquitto_pub -t agents/asr -l -h host.docker.internal
