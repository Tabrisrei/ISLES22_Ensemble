#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# docker build -t isles_major_voting:v0.6 "$SCRIPTPATH"

docker save isles_major_voting:v0.6 | gzip -c > isles_major_voting_v06.tar.gz