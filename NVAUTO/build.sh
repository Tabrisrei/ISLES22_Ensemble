#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

#sudo docker build -t threshold_model "$SCRIPTPATH"
docker build -t threshold_model .
