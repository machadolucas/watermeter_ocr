#!/usr/bin/env bash
# Build, sign, and install the watermeter service.
#
# Produces a single signed binary at $BIN_DIR/watermeter (no Python runtime
# dependency at deploy time), wires up the LaunchAgent, and (re)starts the
# service. The Swift OCR helper is also rebuilt fresh each run.
#
# Why a signed standalone binary instead of `venv/bin/python3 watermeter.py`:
# macOS TCC ties the Local Network privacy grant to the binary's code signing
# identity. With brew Python (ad-hoc signed, cdhash changes on every upgrade)
# the grant was lost on every `brew upgrade`, and the next external request
# popped a "allow local network" modal on the server screen until someone
# screen-shared in. By bundling with PyInstaller and signing each build with
# the same self-signed identity (created by ./setup-codesign.sh), the grant
# given once persists across every rebuild. See docs/watermeter.md in the
# macserver repo for the full rationale.
set -euo pipefail

APP_ROOT="${WTM_ROOT:-$HOME/watermeter}"
BIN_DIR="$APP_ROOT/bin"
VENV_DIR="$APP_ROOT/venv"
PLIST="$HOME/Library/LaunchAgents/com.watermeter.ocr.plist"
LOG_PATH="$APP_ROOT/watermeter.log"
SIGNING_IDENTITY="Macserver Code Signing"
BUNDLE_IDENTIFIER="me.machadolucas.watermeter"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_TMP="$REPO_DIR/build-tmp"
DEFAULT_WTM_PYTHON="$HOME/.local/share/uv/python/cpython-3.15.2-macos-aarch64-none/bin/python3.15"
WTM_PYTHON="${WTM_PYTHON:-$DEFAULT_WTM_PYTHON}"

# Validate the selected interpreter before removing build artifacts or the
# existing build venv. An explicit override is allowed, but must still match
# the Python and architecture used to validate this release.
if [ ! -x "$WTM_PYTHON" ]; then
  echo "[watermeter] ERROR: Python interpreter is not executable: $WTM_PYTHON" >&2
  exit 1
fi
if ! "$WTM_PYTHON" -c 'import platform, sys; sys.exit(0 if sys.implementation.name == "cpython" and sys.version_info[:3] == (3, 15, 2) and platform.machine() == "arm64" else 1)'; then
  echo "[watermeter] ERROR: WTM_PYTHON must be CPython 3.15.2 on arm64." >&2
  exit 1
fi

# Release transaction (stage -> health gates -> swap, keeping *.prev ->
# rollback on failure). A failed build or health gate leaves the running
# service on its previous binaries. Manual rollback: ./release.sh rollback
# shellcheck source=release.sh
source "$REPO_DIR/release.sh"

echo "[watermeter] Installing to: $APP_ROOT"
mkdir -p "$APP_ROOT" "$BIN_DIR" "$APP_ROOT/debug"
rm -rf "$BUILD_TMP"
mkdir -p "$BUILD_TMP"
trap 'rm -rf "$BUILD_TMP"' EXIT

# 0. Verify the code-signing identity exists. This is a one-time per-machine
#    setup handled by setup-codesign.sh — without it the binary would end up
#    ad-hoc signed and we'd be back to the original TCC-loses-grant problem.
if ! security find-identity -v -p codesigning ~/Library/Keychains/login.keychain-db | grep -q "\"$SIGNING_IDENTITY\""; then
  echo "[watermeter] ERROR: code-signing identity '$SIGNING_IDENTITY' not found in keychain." >&2
  echo "[watermeter] Run ./setup-codesign.sh once before install.sh." >&2
  exit 1
fi

# 1. Build Swift OCR helper. Unsigned/ad-hoc is fine here — bin/ocr is invoked
#    as a subprocess and never opens a network connection, so it doesn't
#    interact with TCC.
if ! command -v swiftc >/dev/null; then
  echo "Xcode Command Line Tools required. Installing..."
  xcode-select --install || true
fi
echo "[watermeter] Building Swift OCR helper"
swiftc -O "$REPO_DIR/ocr.swift" -o "$BUILD_TMP/ocr"

# 2. Python environment. The same venv serves build and dev. Both locks are
#    hash-checked: first install binary-only build tools, then install the
#    complete target lock using those pinned tools for any source builds.
echo "[watermeter] Preparing Python venv at $VENV_DIR"
rm -rf "$VENV_DIR"
"$WTM_PYTHON" -m venv "$VENV_DIR"
VENV_PYTHON="$VENV_DIR/bin/python3"
"$VENV_PYTHON" -m pip install -q --require-hashes --only-binary=:all: \
  -r "$REPO_DIR/requirements-build.lock"
"$VENV_PYTHON" -m pip install -q --require-hashes --no-cache-dir --no-build-isolation \
  -r "$REPO_DIR/requirements.lock"

# 3. Build the bundled binary. --onefile produces a single self-extracting
#    Mach-O; --collect-all keeps PyInstaller's hook detection from missing
#    runtime data files in cv2/numpy. distpath/workpath/specpath isolate the
#    build artifacts under build-tmp/ so the repo stays clean.
echo "[watermeter] Bundling watermeter.py with PyInstaller (this takes ~60s)"
"$VENV_DIR/bin/pyinstaller" --clean --noconfirm \
  --onefile \
  --name watermeter \
  --collect-all cv2 \
  --collect-all numpy \
  --distpath "$BUILD_TMP/dist" \
  --workpath "$BUILD_TMP/build" \
  --specpath "$BUILD_TMP" \
  "$REPO_DIR/watermeter.py" >/dev/null

# 4. Sign with our stable identity. We deliberately do NOT use --options
#    runtime: hardened runtime enforces library validation, which rejects
#    dlopen() of dylibs with a different team ID than the outer process.
#    PyInstaller --onefile extracts a bundled Python.framework to /tmp at
#    startup, and that framework's signature has a different identity than
#    our self-signed binary, so hardened runtime would block it. TCC still
#    anchors on the designated requirement (identifier + signing cert),
#    which is what we need for the Local Network grant to persist.
echo "[watermeter] Signing the binary"
codesign --force \
  --sign "$SIGNING_IDENTITY" \
  --identifier "$BUNDLE_IDENTIFIER" \
  "$BUILD_TMP/dist/watermeter"
codesign -dvv "$BUILD_TMP/dist/watermeter" 2>&1 | grep -E '(Identifier|Authority|flags)' | sed 's/^/[watermeter]   /'

# 5. Starter config, then stage the new binaries. Health gate 1 runs the
#    staged binary with --check-config: it proves the bundle starts and that
#    the live config passes this version's validation. Nothing live changes
#    if it fails.
[ -f "$APP_ROOT/config.yaml" ] || cp "$REPO_DIR/config.yaml" "$APP_ROOT/config.yaml"
stage_release "$BUILD_TMP/dist/watermeter" "$BUILD_TMP/ocr"

# 6. LaunchAgent. ProgramArguments now points at the binary directly — no
#    Python invocation at runtime, no dependency on the venv at all (the
#    venv is kept around only for dev/test convenience).
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.watermeter.ocr</string>
  <key>ProgramArguments</key>
  <array>
    <string>$BIN_DIR/watermeter</string>
    <string>--config</string><string>$APP_ROOT/config.yaml</string>
    <string>--log</string><string>$LOG_PATH</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$APP_ROOT/launch_stdout.log</string>
  <key>StandardErrorPath</key><string>$APP_ROOT/launch_stderr.log</string>
  <key>WorkingDirectory</key><string>$APP_ROOT</string>
</dict></plist>
PLIST

# 7. Swap the staged binaries in (the running ones become *.prev), restart
#    the service with bootout / bootstrap, and run health gate 2: it must stay
#    running on one pid for HEALTH_WAIT_SEC. Otherwise *.prev is restored.
activate_release

echo "[watermeter] Installed."
echo "[watermeter]   Binary:   $BIN_DIR/watermeter"
echo "[watermeter]   Config:   $APP_ROOT/config.yaml"
echo "[watermeter]   Logs:     $LOG_PATH"
echo "[watermeter]   Service:  launchctl print gui/\$(id -u)/com.watermeter.ocr"
echo "[watermeter]   Rollback: $REPO_DIR/release.sh rollback"
