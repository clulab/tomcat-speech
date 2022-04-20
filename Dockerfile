#Dockerfile for the Multimodal Participant State Model

from python:3.9

# Copy multimodal_data_preprocessing directory
COPY multimodal_data_preprocessing /multimodal_data_preprocessing

# Install mmc server requirements
WORKDIR /tomcat_speech
COPY mmc_server_requirements.txt .
RUN pip install -r mmc_server_requirements.txt

# Setup mmc server
COPY setup.py .
COPY tomcat_speech tomcat_speech
COPY mmc_server.py .
RUN pip install -e .[mmc_server]

# Install mmc server
COPY scripts scripts
COPY data data
RUN scripts/mmc/install

