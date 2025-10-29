#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# --- 1. Get This Script's Directory ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# --- 2. Define Path to the "Worker" Script ---
# (Assumes it's in the same directory as this master script)
WORKER_SCRIPT_PATH="$SCRIPT_DIR/grandtour_bag2frames.sh"

if [ ! -f "$WORKER_SCRIPT_PATH" ]; then
    echo "Error: Worker script not found at $WORKER_SCRIPT_PATH"
    echo "Make sure 'grandtour_bag2frames.sh' is in the same directory."
    exit 1
fi

# --- 3. Define Directories to Process ---
GRAND_TOUR_BASE_DIR="/media/slsecret/T7/grand_tour"

DIRS_TO_PROCESS=(
    "100111"
    "100111b"
    "100112"
    "110217"
    "110217c"
    "110221"
    "110308"
    "111116"
    "111511"
)

echo "Starting batch extraction..."

# --- 4. Loop Through Directories and Call Worker Script ---
for DIR_NAME in "${DIRS_TO_PROCESS[@]}"; do
    FULL_TARGET_DIR="$GRAND_TOUR_BASE_DIR/$DIR_NAME"
    
    # Check if the target directory actually exists
    if [ ! -d "$FULL_TARGET_DIR" ]; then
        echo -e "\n--- WARNING: Directory not found, skipping: $FULL_TARGET_DIR ---"
        continue
    fi

    echo -e "\n====================================================="
    echo "=== PROCESSING DIRECTORY: $FULL_TARGET_DIR"
    echo "====================================================="
    
    # Call the worker script and pass the full path as an argument
    # The worker script will source ROS, create the /frames dir, etc.
    bash "$WORKER_SCRIPT_PATH" "$FULL_TARGET_DIR"
    
    echo "=== FINISHED DIRECTORY: $FULL_TARGET_DIR"
done

echo -e "\n====================================================="
echo "All directories processed."
echo "====================================================="
