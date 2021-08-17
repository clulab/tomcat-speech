#!/usr/bin/env bash

set -euo pipefail

# Script to install dependencies and download pretrained model

# Set the ROOT environment variable, assuming that the directory structure
# mirrors that of the git repository.
export ROOT="$(cd "$( dirname "${BASH_SOURCE[0]}" )/../../" >/dev/null 2>&1 && pwd)"

pushd ${ROOT} > /dev/null
    pip install wheel
    pip install -r mmc_server/requirements.txt
    echo "Installed dependencies."

    "$ROOT"/scripts/mmc/download_pretrained_model.sh
popd > /dev/null

exit 0