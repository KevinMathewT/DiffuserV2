#!/bin/bash

# Define MuJoCo version and paths
MUJOCO_VERSION="210"
MUJOCO_DIR="$HOME/.mujoco"
MUJOCO_TARGET_DIR="$MUJOCO_DIR/mujoco${MUJOCO_VERSION}"
DOWNLOAD_URL="https://mujoco.org/download/mujoco${MUJOCO_VERSION}-linux-x86_64.tar.gz"
ARCHIVE_NAME="mujoco${MUJOCO_VERSION}-linux-x86_64.tar.gz"
EXTRACTED_FOLDER_NAME="mujoco${MUJOCO_VERSION}" # The name of the folder inside the archive

# --- Check if MuJoCo is already installed ---
echo "Checking for existing MuJoCo installation at ${MUJOCO_TARGET_DIR}..."
if [ -d "$MUJOCO_TARGET_DIR" ]; then
    echo "MuJoCo ${MUJOCO_VERSION} seems to be already installed in ${MUJOCO_TARGET_DIR}."
    # Optional: Add a check for a key file/binary inside to be more certain
    if [ -f "${MUJOCO_TARGET_DIR}/bin/simulate" ]; then
        echo "Found key file (${MUJOCO_TARGET_DIR}/bin/simulate). Installation looks complete."
        echo "Exiting. No action taken."
        exit 0
    else
        echo "Warning: Directory exists, but key file (${MUJOCO_TARGET_DIR}/bin/simulate) is missing. Installation might be incomplete."
        # Decide here if you want to proceed with re-installation or exit
        # For now, we'll exit to be safe. Manually remove the dir if you want to reinstall.
        echo "Please check the directory or remove it manually if you want to reinstall."
        exit 1
    fi
else
    echo "MuJoCo ${MUJOCO_VERSION} not found at the target location."
fi

# --- Check for necessary tools (wget/curl, tar) ---
echo "Checking for necessary tools..."
DOWNLOAD_CMD=""
if command -v wget >/dev/null 2>&1; then
    DOWNLOAD_CMD="wget -q --show-progress -O"
    echo "Found wget."
elif command -v curl >/dev/null 2>&1; then
    DOWNLOAD_CMD="curl -L -# -o"
    echo "Found curl."
else
    echo "Error: Neither wget nor curl found. Please install one of them (e.g., 'sudo apt update && sudo apt install wget')."
    exit 1
fi

if ! command -v tar >/dev/null 2>&1; then
    echo "Error: 'tar' command not found. Please install it (e.g., 'sudo apt update && sudo apt install tar')."
    exit 1
fi
echo "Tools check passed."

# --- Create necessary directories ---
echo "Creating parent directory ${MUJOCO_DIR} if it doesn't exist..."
mkdir -p "$MUJOCO_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Failed to create directory ${MUJOCO_DIR}."
    exit 1
fi

# --- Create a temporary directory for download and extraction ---
# Using mktemp for safety and avoiding conflicts
TMP_DIR=$(mktemp -d -t mujoco_download_XXXXXX)
if [ ! -d "$TMP_DIR" ]; then
    echo "Error: Failed to create temporary directory."
    exit 1
fi
echo "Created temporary directory: ${TMP_DIR}"

# --- Download MuJoCo ---
echo "Navigating to temporary directory..."
cd "$TMP_DIR" || exit 1 # Exit if cd fails

echo "Downloading MuJoCo ${MUJOCO_VERSION} from ${DOWNLOAD_URL}..."
# Use the selected download command
$DOWNLOAD_CMD "$ARCHIVE_NAME" "$DOWNLOAD_URL"

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to download MuJoCo archive from ${DOWNLOAD_URL}."
    echo "Cleaning up temporary directory: ${TMP_DIR}"
    rm -rf "$TMP_DIR"
    exit 1
fi

if [ ! -f "$ARCHIVE_NAME" ]; then
    echo "Error: Download command succeeded but the archive file (${ARCHIVE_NAME}) was not found."
    echo "Cleaning up temporary directory: ${TMP_DIR}"
    rm -rf "$TMP_DIR"
    exit 1
fi
echo "Download complete: ${ARCHIVE_NAME}"

# --- Extract MuJoCo ---
echo "Extracting ${ARCHIVE_NAME}..."
tar -xzf "$ARCHIVE_NAME"
if [ $? -ne 0 ]; then
    echo "Error: Failed to extract MuJoCo archive ${ARCHIVE_NAME}."
    echo "Cleaning up temporary directory: ${TMP_DIR}"
    rm -rf "$TMP_DIR"
    exit 1
fi
echo "Extraction complete."

# --- Verify extracted folder ---
if [ ! -d "$EXTRACTED_FOLDER_NAME" ]; then
    echo "Error: Extracted folder named '${EXTRACTED_FOLDER_NAME}' not found inside the archive."
    echo "Contents of temporary directory:"
    ls -la "$TMP_DIR"
    echo "Cleaning up temporary directory: ${TMP_DIR}"
    rm -rf "$TMP_DIR"
    exit 1
fi
echo "Verified extracted folder: ${EXTRACTED_FOLDER_NAME}"

# --- Move to target location ---
echo "Moving extracted folder '${EXTRACTED_FOLDER_NAME}' to ${MUJOCO_DIR}..."
# The target should be the parent directory, as mv will place the folder inside it.
mv "$EXTRACTED_FOLDER_NAME" "$MUJOCO_DIR/"
if [ $? -ne 0 ]; then
    echo "Error: Failed to move '${EXTRACTED_FOLDER_NAME}' to '${MUJOCO_DIR}/'."
    echo "Make sure you have write permissions for ${MUJOCO_DIR}."
    echo "Cleaning up temporary directory: ${TMP_DIR}"
    # Don't remove the extracted folder if move failed, user might want it
    rm -f "$TMP_DIR/$ARCHIVE_NAME"
    exit 1
fi

# Verify the final location
if [ -d "$MUJOCO_TARGET_DIR" ]; then
    echo "Successfully moved MuJoCo to ${MUJOCO_TARGET_DIR}."
else
    echo "Error: Move command seemed successful, but the target directory ${MUJOCO_TARGET_DIR} was not found afterwards."
    echo "Cleaning up temporary directory: ${TMP_DIR}"
    rm -f "$TMP_DIR/$ARCHIVE_NAME" # Only remove archive if move seemed ok but dir not found
    exit 1
fi

# --- Clean up ---
echo "Cleaning up temporary directory: ${TMP_DIR}"
cd "$HOME" # Navigate out of the temp dir before removing it
rm -rf "$TMP_DIR"

echo "---"
echo "MuJoCo ${MUJOCO_VERSION} installation completed successfully!"
echo "It is located at: ${MUJOCO_TARGET_DIR}"
echo "You might need to ensure the mujoco-py library bindings are installed in your python environment (e.g., 'pip install mujoco-py==2.1.2.14')."
echo "You may also need to set environment variables like LD_LIBRARY_PATH, although mujoco-py often handles this."
echo "---"

exit 0