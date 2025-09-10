#!/usr/bin/env bash
# safe_action.sh - example safe script executed by RPA engine
echo "Safe action executed at $(date -u) with args: $@" >> /data/rpa_script.log
