#!/usr/bin/env bash

set -u

# Script to download OpenSmile v3.0.0

# Set the ROOTDIR environment variable, assuming that the directory structure
# mirrors that of the git repository.
ROOTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" >/dev/null 2>&1 && pwd)"
export ROOTDIR

if [[ ! $OSTYPE == "darwin"* ]]; then
    echo "This script only works on macOS. Exiting now..."
    exit 1
fi

TAR_GZ_FILE=opensmile-3.0-osx-x64.tar.gz
URL="https://github.com/audeering/opensmile/releases/download/v3.0.0/${TAR_GZ_FILE}"
mkdir -p "$ROOTDIR"/external

pushd "$ROOTDIR"/external > /dev/null
    echo "Downloading OpenSmile to $(pwd)/opensmile-3.0"
    if ! curl -LO "$URL"; then
        echo "Could not download OpenSmile."
        exit 1
    fi

    echo "Extracting OpenSmile..."
    if ! tar -xzf "$TAR_GZ_FILE"; then
        echo "Error encountered while extracting ${TAR_GZ_FILE}. This seems to
        not affect the working of the OpenSMILE SMILEExtract binary."
        mv opensmile-3.0-osx-x64 opensmile-3.0
    fi
popd > /dev/null

exit 0
