#!/bin/bash

source_path=$1
target_path=$2

mkdir -p $target_path


for file in $source_path/*.wav; do
    fbname=$(basename "$file" .wav)
    ffmpeg -i $file -vn -ar 16000 -ac 1 "$target_path/$fbname.wav"
done

