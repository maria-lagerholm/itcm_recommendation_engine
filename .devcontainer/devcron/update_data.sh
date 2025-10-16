#!/usr/bin/env bash
set -euo pipefail

python /workspace/scripts/update_data.py

# prevent pipeline from running again until the next update
touch /workspace/.logs/last_update_success
