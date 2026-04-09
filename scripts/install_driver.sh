#!/bin/bash
set -euo pipefail

DRIVER_NAME="EveAudio.driver"
INSTALL_DIR="/Library/Audio/Plug-Ins/HAL"
BUILD_DIR="${1:-$(dirname "$0")/../build/Debug}"

DRIVER_PATH="$BUILD_DIR/$DRIVER_NAME"

if [ ! -d "$DRIVER_PATH" ]; then
    echo "Error: Driver not found at $DRIVER_PATH"
    echo "Usage: $0 [build_dir]"
    exit 1
fi

# Re-sign the bundle (ad-hoc) so coreaudiod accepts it
echo "Signing $DRIVER_NAME..."
codesign --force --sign - --timestamp=none \
    "$DRIVER_PATH/Contents/MacOS/EveAudio"
codesign --force --sign - --timestamp=none \
    "$DRIVER_PATH"

# Create HAL directory if it doesn't exist
if [ ! -d "$INSTALL_DIR" ]; then
    echo "Creating $INSTALL_DIR..."
    sudo mkdir -p "$INSTALL_DIR"
fi

# Remove any previous version
if [ -d "$INSTALL_DIR/$DRIVER_NAME" ]; then
    echo "Removing old installation..."
    sudo rm -rf "$INSTALL_DIR/$DRIVER_NAME"
fi

echo "Installing $DRIVER_NAME to $INSTALL_DIR..."
sudo cp -R "$DRIVER_PATH" "$INSTALL_DIR/"

# Fix permissions so coreaudiod (running as _coreaudiod) can read it
sudo chmod -R 755 "$INSTALL_DIR/$DRIVER_NAME"
sudo chown -R root:wheel "$INSTALL_DIR/$DRIVER_NAME"

echo "Restarting coreaudiod..."
sudo killall coreaudiod || true   # launchd auto-restarts it

echo ""
echo "Done. Open Audio MIDI Setup to confirm 'Eve Virtual Mic' appears."
