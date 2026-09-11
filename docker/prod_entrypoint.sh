#!/bin/sh

# Git-archive /app is the inbound Connect + leftover-dup source of truth.
# Wheel-overlay site-packages can lag sql.py and crash Hypercorn on import.
export PYTHONPATH="${PYTHONPATH:-/app}"

if [ "$SEPARATE_HEALTH_APP" = "1" ]; then
    export LITELLM_ARGS="$@"
    export SUPERVISORD_STOPWAITSECS="${SUPERVISORD_STOPWAITSECS:-3600}"
    exec supervisord -c /etc/supervisord.conf
fi

if [ "$USE_DDTRACE" = "true" ]; then
    export DD_TRACE_OPENAI_ENABLED="False"
    exec ddtrace-run litellm "$@"
else
    exec litellm "$@"
fi