# Dockerfile for the Multimodal Participant State Model

from python:latest

COPY requirements.txt .
COPY scripts ./scripts

# Install dependencies
COPY data ./data
RUN ./scripts/install
COPY *.py ./