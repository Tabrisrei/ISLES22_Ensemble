#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t isles_major_voting:v0.6 "$SCRIPTPATH"

# docker run -it --entrypoint=/bin/bash major_voting:v0.5
