#!/usr/bin/env bash

set -euo pipefail

# Script to download pretrained model

# Set the ROOT environment variable, assuming that the directory structure
# mirrors that of the git repository.
export ROOT="$(cd "$( dirname "${BASH_SOURCE[0]}" )/../" >/dev/null 2>&1 && pwd)"

# todo: update with right URL and name
MODEL_URL=http://vanga.sista.arizona.edu/tomcat/data/baseline_model_speaker.pt

pushd ${ROOT}/data > /dev/null
    curl -O $MODEL_URL
popd > /dev/null

echo "Pretrained MMC model downloaded from ${MODEL_URL}"