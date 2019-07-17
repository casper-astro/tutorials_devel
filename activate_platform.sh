#!/bin/bash

SUPPORTED_PLATFORMS="roach2|snap|skarab|red_pitaya"

function help() {
  echo "Usage: $(basename $0) [${SUPPORTED_PLATFORMS}]"
  echo "Possible platforms are: ${SUPPORTED_PLATFORMS}"
  exit
}

# If no platform provided, then bail out
if (( $# < 1 )); then
  help
fi

for arg in $@; do
  PLATFORM=$arg
  if [[ "$PLATFORM" =~ ^(${SUPPORTED_PLATFORMS})$ ]]; then
    echo "Initializing libraries for $PLATFORM platform"
    git submodule init
    git submodule update $PLATFORM/mlib_devel
  else
    echo "Platform $PLATFORM is not supported"
    echo "Supported platforms are: ${SUPPORTED_PLATFORMS}"
  fi
done

