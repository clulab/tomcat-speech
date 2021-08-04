#!/usr/bin/env bash

# Script to test the MMC Model server

# Set the ROOT environment variable, assuming that the directory structure
# mirrors that of the git repository.
export ROOT="$(cd "$( dirname "${BASH_SOURCE[0]}" )/../" >/dev/null 2>&1 && pwd)"

# Test model resetting
curl localhost:8000/reset-model &> /dev/null

# Test utterance encoding
curl \
    -X GET localhost:8000/encode \  # todo: update if we change the route
    -H "Content-Type: application/json" \
    -d @${ROOT}/data/test.json

exit 0