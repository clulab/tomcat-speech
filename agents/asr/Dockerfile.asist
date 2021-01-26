# This is a Dockerfile created for the purpose of integrating with the SIM2
# testbed in DARPA's Artificial Social Intelligence for Successful Teams
# (ASIST) program.

from python:3.7

run apt update
run apt -y install mosquitto-clients
run pip install\
    websockets\
    numpy\
    google-cloud-speech
workdir asr
copy * /asr/
