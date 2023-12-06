#!/usr/bin/env bash

./build.sh

docker save seals:v1.8 | gzip -c > seals_v18.tar.gz