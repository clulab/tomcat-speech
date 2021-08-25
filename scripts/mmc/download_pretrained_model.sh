#!/usr/bin/env bash

set -euo pipefail

# Script to download pretrained model

# Set the ROOT environment variable, assuming that the directory structure
# mirrors that of the git repository.
export ROOT="$(cd "$( dirname "${BASH_SOURCE[0]}" )/../../" >/dev/null 2>&1 && pwd)"

# Path on vanga for the UAZ pretrained model
MODEL_URL=http://vanga.sista.arizona.edu/tomcat/data/MC_GOLD_classwts_nogender_25to75perc_avg_IS13.pth

mkdir -p ${ROOT}/data
pushd ${ROOT}/data > /dev/null
    curl -O $MODEL_URL
popd > /dev/null

echo "Downloaded pretrained MMC model from ${MODEL_URL}"


# Path on vanga for a subsetted GloVe file
GLOVE_URL=http://vanga.sista.arizona.edu/tomcat/data/glove.short.300d.punct.txt

pushd ${ROOT}/data > /dev/null
    curl -O $GLOVE_URL
popd > /dev/null

echo "Downloaded pretrained GloVe embeddings from ${GLOVE_URL}"
