#!/usr/bin/env bash
# Release transaction for the watermeter service.
#
#   stage_release NEW_BIN NEW_OCR   copy into $BIN_DIR/.staged and run health
#                                   gate 1: the new binary must start and pass
#                                   --check-config against the live config.
#                                   On failure nothing live has changed.
#   activate_release                keep the running binaries as *.prev, move
#                                   the staged ones in, restart the LaunchAgent
#                                   and run health gate 2: the job must reach
#                                   "running" and keep the same pid for
#                                   HEALTH_WAIT_SEC (a crash loop changes the
#                                   pid). On failure it rolls back to *.prev.
#   rollback_release                put *.prev back (the replaced binaries are
#                                   kept as *.rolled-back) and restart.
#
# Sourced by install.sh. Run directly for a manual rollback or a health check:
#   ./release.sh rollback
#   ./release.sh status
#
# Everything host-specific is overridable through the environment, which is
# how tests/test_release.py drives it with a stub launchctl.

APP_ROOT="${APP_ROOT:-${WTM_ROOT:-$HOME/watermeter}}"
BIN_DIR="${BIN_DIR:-$APP_ROOT/bin}"
PLIST="${PLIST:-$HOME/Library/LaunchAgents/com.watermeter.ocr.plist}"
SERVICE_LABEL="${SERVICE_LABEL:-com.watermeter.ocr}"
LAUNCHCTL="${LAUNCHCTL:-launchctl}"
HEALTH_WAIT_SEC="${HEALTH_WAIT_SEC:-20}"
HEALTH_POLL_SEC="${HEALTH_POLL_SEC:-2}"
RELEASE_FILES=(watermeter ocr)

_rel_log() { echo "[watermeter] $*"; }
_rel_domain() { echo "gui/$(id -u)"; }

_rel_field() {  # <launchctl print output> <field>: top-level fields are one tab deep
  printf '%s\n' "$1" | awk -v f="$2" 'index($0, "\t" f " = ") == 1 { sub("^\t" f " = ", ""); print; exit }'
}

_restart_service() {
  "$LAUNCHCTL" bootout "$(_rel_domain)" "$PLIST" 2>/dev/null || true
  # bootstrap right after bootout can race the teardown (error 5); retry briefly.
  local i
  for i in 1 2 3 4 5; do
    if "$LAUNCHCTL" bootstrap "$(_rel_domain)" "$PLIST"; then
      return 0
    fi
    sleep 1
  done
  _rel_log "ERROR: launchctl bootstrap failed"
  return 1
}

# Health gate 2: the job is running and keeps one pid for HEALTH_WAIT_SEC.
service_healthy() {
  local start=$SECONDS first_pid="" out state pid
  while :; do
    out="$("$LAUNCHCTL" print "$(_rel_domain)/$SERVICE_LABEL" 2>/dev/null || true)"
    state="$(_rel_field "$out" state)"
    pid="$(_rel_field "$out" pid)"
    if [ "$state" = "running" ] && [ -n "$pid" ]; then
      if [ -z "$first_pid" ]; then
        first_pid="$pid"
      elif [ "$pid" != "$first_pid" ]; then
        _rel_log "health: pid changed $first_pid -> $pid (the service is restarting)"
        return 1
      fi
    elif [ -n "$first_pid" ]; then
      _rel_log "health: service stopped running (state=${state:-unknown})"
      return 1
    fi
    if [ $((SECONDS - start)) -ge "$HEALTH_WAIT_SEC" ]; then
      if [ -n "$first_pid" ]; then
        _rel_log "health: running as pid $first_pid for ${HEALTH_WAIT_SEC}s"
        return 0
      fi
      _rel_log "health: service never reached state=running (last: ${state:-unknown})"
      return 1
    fi
    sleep "$HEALTH_POLL_SEC"
  done
}

stage_release() {  # <new watermeter binary> <new ocr binary>
  local staged="$BIN_DIR/.staged"
  rm -rf "$staged"
  mkdir -p "$staged"
  install -m 0755 "$1" "$staged/watermeter"
  install -m 0755 "$2" "$staged/ocr"
  _rel_log "Health gate 1: staged binary --check-config"
  if ! "$staged/watermeter" --check-config --config "$APP_ROOT/config.yaml"; then
    _rel_log "ERROR: the new binary failed --check-config; the running service is unchanged"
    rm -rf "$staged"
    return 1
  fi
}

activate_release() {
  local staged="$BIN_DIR/.staged" f had_prev=0
  if [ ! -x "$staged/watermeter" ]; then
    _rel_log "ERROR: nothing staged (run stage_release first)"
    return 1
  fi
  for f in "${RELEASE_FILES[@]}"; do
    if [ -e "$BIN_DIR/$f" ]; then
      cp -p "$BIN_DIR/$f" "$BIN_DIR/$f.prev"
      had_prev=1
    fi
  done
  for f in "${RELEASE_FILES[@]}"; do
    mv -f "$staged/$f" "$BIN_DIR/$f"
  done
  rmdir "$staged"
  _rel_log "Restarting the service; health gate 2 (${HEALTH_WAIT_SEC}s)"
  if _restart_service && service_healthy; then
    _rel_log "Release is healthy; previous binaries kept as *.prev"
    return 0
  fi
  if [ "$had_prev" = 1 ]; then
    _rel_log "ERROR: the new release is unhealthy; rolling back"
    rollback_release || true
  else
    _rel_log "ERROR: the new release is unhealthy and there is no previous release to restore"
  fi
  return 1
}

rollback_release() {
  local f
  if [ ! -e "$BIN_DIR/watermeter.prev" ]; then
    _rel_log "ERROR: no previous release ($BIN_DIR/watermeter.prev) to roll back to"
    return 1
  fi
  for f in "${RELEASE_FILES[@]}"; do
    [ -e "$BIN_DIR/$f.prev" ] || continue
    if [ -e "$BIN_DIR/$f" ]; then
      mv -f "$BIN_DIR/$f" "$BIN_DIR/$f.rolled-back"
    fi
    cp -p "$BIN_DIR/$f.prev" "$BIN_DIR/$f"
  done
  _rel_log "Rolled back to the previous binaries; restarting"
  _restart_service && service_healthy
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -euo pipefail
  case "${1:-}" in
    rollback) rollback_release ;;
    status)   service_healthy ;;
    *) echo "usage: $0 rollback|status" >&2; exit 2 ;;
  esac
fi
