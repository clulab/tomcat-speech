#!/usr/bin/env bash

set -euo pipefail

# Script to install dependencies and download pretrained model

# Set the ROOT environment variable, assuming that the directory structure
# mirrors that of the git repository.
export ROOT="$(cd "$( dirname "${BASH_SOURCE[0]}" )/../" >/dev/null 2>&1 && pwd)"

pushd ${ROOT} > /dev/null
    pip install wheel
    pip install -r requirements.txt
    echo "Installed dependencies."

    "$ROOT"/scripts/download_pretrained_model
popd > /dev/null

exit 0