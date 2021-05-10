FROM python:latest

RUN apt-get update
RUN apt-get install mosquitto-clients libsndfile1-dev -y

RUN pip install\
    spacy\
    argparse
    
copy . .
