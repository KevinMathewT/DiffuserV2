#!/usr/bin/env bash
set -e
mkdir -p ./pretrained
curl -L "https://www.dropbox.com/scl/fi/y2fcr12aqzkzac3qufyje/maze2d-logs.zip?rlkey=xw6c9sno6ra8ixlt6m1eyv3p1&dl=1" -o ./pretrained/maze2d-logs.zip
unzip -o ./pretrained/maze2d-logs.zip -d ./pretrained
rm ./pretrained/maze2d-logs.zip
