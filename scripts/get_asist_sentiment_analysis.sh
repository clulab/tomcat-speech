#!/bin/bash

# takes a file with one utterance per line
# feeds file into the asist sentiment analyzer
# returns a file with a probability distribution of
# sentiment scores for the utterance

# check length of arguments
if [ "$#" -lt 2 ]; then
  echo "$#"
  echo "Input and/or output files not specified"
  exit 1
fi

# get current location
CURRENT=$PWD

# cd to the asist repo
cd ../asist

# run the analyzer
sbt "runMain org.clulab.sentiment.LexiconSentimentAnalyzer $CURRENT/$1 $CURRENT/$2"