#!/bin/sh
set -eu

app_uid="${PUID:-1000}"
app_gid="${PGID:-1000}"

case "$app_uid:$app_gid" in
  *[!0-9:]*|:*|*:)
    echo "PUID and PGID must be numeric" >&2
    exit 1
    ;;
esac

if [ "$(id -u)" = "0" ]; then
  current_gid="$(id -g analyzer)"
  current_uid="$(id -u analyzer)"
  if [ "$current_gid" != "$app_gid" ]; then
    groupmod -o -g "$app_gid" analyzer
  fi
  if [ "$current_uid" != "$app_uid" ]; then
    usermod -o -u "$app_uid" analyzer
  fi
  mkdir -p /config /exports
  chown analyzer:analyzer /config /exports
  if [ -e /config/analyzer.db ]; then
    chown analyzer:analyzer /config/analyzer.db /config/analyzer.db-wal /config/analyzer.db-shm 2>/dev/null || true
  fi
  exec gosu analyzer "$@"
fi

exec "$@"
