#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# --- 2. Define Path to the "Worker" Script ---
# (Assumes it's in the same directory as this master script)
WORKER_SCRIPT_PATH="$SCRIPT_DIR/grandtour_bag2frames.sh"

if [ ! -f "$WORKER_SCRIPT_PATH" ]; then
    echo "Error: Worker script not found at $WORKER_SCRIPT_PATH"
    echo "Make sure 'grandtour_bag2frames.sh' is in the same directory."
    exit 1
fi

# --- 3. Define Base Directory ---
GRAND_TOUR_BASE_DIR="/media/slsecret/T7/grand_tour"

echo "Starting batch extraction..."
echo "Looking for directories in: $GRAND_TOUR_BASE_DIR"

# --- 4. Loop Through Found Directories and Call Worker Script ---
#
#    This command finds all directories (-type d) that are exactly
#    one level deep (-maxdepth 1) in the base directory.
#    The 'while read' loop processes each found directory.
#
find "$GRAND_TOUR_BASE_DIR" -mindepth 1 -maxdepth 1 -type d | while read FULL_TARGET_DIR; do
    
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

