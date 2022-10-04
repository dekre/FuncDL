#!/bin/bash

WORKDIR=$(pwd)
ENV_FILE="${WORKDIR}/${1}"

echo "export variables from ${ENV_FILE}"
export $(grep -v '^#' ${ENV_FILE} | xargs)
