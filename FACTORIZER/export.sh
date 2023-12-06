#!/usr/bin/env bash

./build.sh

docker save factorizer | gzip -c > factorizer.tar.gz
