#!/usr/bin/env bash

# Script to test the MMC Model server

# Set the ROOT environment variable, assuming that the directory structure
# mirrors that of the git repository.
export ROOT="$(cd "$( dirname "${BASH_SOURCE[0]}" )/../../mmc_server/" >/dev/null 2>&1 && pwd)"
echo ${ROOT}/data/test_message.json
# Test utterance encoding
curl \
    -X GET localhost:8000/encode \
    -H "Content-Type: application/json" \
    -d @${ROOT}/data/test_message.json

exit 0

# use the file from John as test.json
# todo: commit to repo