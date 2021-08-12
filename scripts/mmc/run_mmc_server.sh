#!/usr/bin/env bash

set -euo pipefail

# Script to run UAZ Multimodal Participant State Model web service

# Set the ROOT environment variable, assuming that the directory structure
# mirrors that of the git repository.
export ROOT="$(cd "$( dirname "${BASH_SOURCE[0]}" )/../../mmc_server/" >/dev/null 2>&1 && pwd)"

pushd ${ROOT} > /dev/null
    # The `--reload` flag reloads the app whenever there is a change to the program,
    # which is useful for development purposes.
    uvicorn mmc_server:app --reload
popd > /dev/null