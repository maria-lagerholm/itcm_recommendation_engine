#!/usr/bin/env bash
set -euo pipefail

FLAG=/workspace/.logs/last_update_success
WORKDIR=/workspace

# only run if the updater succeeded
[ -f "$FLAG" ] || exit 0

make -C "$WORKDIR" all

# the next pipeline run will be skipped unless a fresh update recreates the flag
rm -f "$FLAG"