#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${WTM_ROOT:-$HOME/watermeter}"
BIN_DIR="$APP_ROOT/bin"
VENV_DIR="$APP_ROOT/venv"
PLIST="$HOME/Library/LaunchAgents/com.watermeter.ocr.plist"
LOG_PATH="$APP_ROOT/watermeter.log"

echo "[watermeter] Installing to: $APP_ROOT"
mkdir -p "$APP_ROOT" "$BIN_DIR" "$APP_ROOT/debug"

# Build Swift OCR
if ! command -v swiftc >/dev/null; then
  echo "Xcode Command Line Tools required. Installing..."
  xcode-select --install || true
fi
swiftc -O ocr.swift -o "$BIN_DIR/ocr"

# Python env
if ! command -v python3 >/dev/null; then
  echo "python3 is required"; exit 1
fi
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install -U pip wheel
pip install opencv-python-headless paho-mqtt requests pyyaml

# Files
cp watermeter.py "$APP_ROOT/watermeter.py"
[ -f "$APP_ROOT/config.yaml" ] || cp config.yaml "$APP_ROOT/config.yaml"

# LaunchAgent
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.watermeter.ocr</string>
  <key>ProgramArguments</key>
  <array>
    <string>$VENV_DIR/bin/python3</string>
    <string>$APP_ROOT/watermeter.py</string>
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

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
launchctl start com.watermeter.ocr

echo "[watermeter] Installed. Edit: $APP_ROOT/config.yaml"
echo "[watermeter] Logs:   $LOG_PATH"
