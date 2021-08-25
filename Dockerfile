# Dockerfile for the Multimodal Participant State Model

from python:latest

WORKDIR /tomcat_speech

COPY mmc_server_requirements.txt .
RUN pip install -r mmc_server_requirements.txt
COPY setup.py .
COPY tomcat_speech tomcat_speech
COPY mmc_server.py .
RUN pip install -e .[mmc_server]

COPY scripts scripts
#COPY data ./data
RUN scripts/mmc/install

# Install dependencies
