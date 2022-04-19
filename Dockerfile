# Dockerfile for the Multimodal Participant State Model

from python:3.9

# Copy multimodal_data_preprocessing directory
COPY multimodal_data_preprocessing /multimodal_data_preprocessing

# Copy tomcat-speech directory
WORKDIR /tomcat_speech
COPY mmc_server_requirements.txt .
COPY setup.py .
COPY tomcat_speech tomcat_speech
COPY mmc_server.py .
COPY scripts scripts
COPY data data

# Install mmc server
RUN pip install -r mmc_server_requirements.txt
RUN pip install -e .[mmc_server]
RUN scripts/mmc/install

