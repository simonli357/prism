#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# --- 1. Get This Script's Directory ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PYTHON_SCRIPT_NAME="bag2frames.py"
PYTHON_SCRIPT_PATH="$SCRIPT_DIR/$PYTHON_SCRIPT_NAME"

# Check if the python script exists
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT_PATH"
    exit 1
fi

# --- 2. Setup ROS1 Environment ---
echo "Sourcing ROS1 environment..."
unset ROS_DISTRO
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH
# --- End ROS1 Setup ---


# --- 3. Configuration ---
if [ -z "$1" ]; then
    echo "Error: No bag directory provided."
    echo "Usage: ./grandtour_bag2frames.sh /path/to/bag/directory"
    exit 1
fi
BAG_DIR="$1" # Get the bag directory from the first argument
FRAME_DIR="$BAG_DIR/frames"

SAVE_EVERY_N_FRAMES=4

echo "Base Bag Directory: $BAG_DIR"
echo "Base Output Directory: $FRAME_DIR"
mkdir -p "$FRAME_DIR" # Create the main 'frames' directory

# Loop over every .bag file ONCE
for BAG_FILE in "$BAG_DIR"/*.bag; do
    
    # Skip the 'frames' directory if it's somehow matched
    if [ -d "$BAG_FILE" ]; then
        continue
    fi

    echo -e "\n-----------------------------------------------------"
    echo "--- Checking Bag: $(basename "$BAG_FILE")"
    echo "-----------------------------------------------------"

    # This array will hold the topics to check for *this bag only*
    declare -A TOPICS_TO_CHECK
    
    # Use a case statement to find the bag we want and assign its topics
    case "$(basename "$BAG_FILE")" in
        # *alphasense.bag)
        #     echo "-> Found Alphasense bag. Checking Alphasense topics..."
        #     TOPICS_TO_CHECK=(
        #         ["/boxi/alphasense/front_center/image_raw/compressed"]="alphasense_front_center"
        #         ["/boxi/alphasense/front_left/image_raw/compressed"]="alphasense_front_left"
        #         ["/boxi/alphasense/front_right/image_raw/compressed"]="alphasense_front_right"
        #         ["/boxi/alphasense/left/image_raw/compressed"]="alphasense_left"
        #         ["/boxi/alphasense/right/image_raw/compressed"]="alphasense_right"
        #     )
        #     ;;

        *hdr_front.bag)
            echo "-> Found HDR Front bag. Checking HDR Front topic..."
            TOPICS_TO_CHECK=(
                ["/boxi/hdr/front/image_raw/compressed"]="hdr_front"
            )
            ;;

        # *hdr_left.bag)
        #     echo "-> Found HDR Left bag. Checking HDR Left topic..."
        #     TOPICS_TO_CHECK=(
        #         ["/boxi/hdr/left/image_raw/compressed"]="hdr_left"
        #     )
        #     ;;
            
        # *hdr_right.bag)
        #     echo "-> Found HDR Right bag. Checking HDR Right topic..."
        #     TOPICS_TO_CHECK=(
        #         ["/boxi/hdr/right/image_raw/compressed"]="hdr_right"
        #     )
        #     ;;
            
        # *zed2i_images.bag)
        #     echo "-> Found ZED2i Images bag. Checking ZED2i topics..."
        #     TOPICS_TO_CHECK=(
        #         ["/boxi/zed2i/left/image_raw/compressed"]="zed2i_left"
        #         ["/boxi/zed2i/right/image_raw/compressed"]="zed2i_right"
        #     )
        #     ;;

        # *zed2i_depth.bag)
        #     echo "-> Found ZED2i Depth bag. Checking ZED2i topics..."
        #     TOPICS_TO_CHECK=(
        #         ["/boxi/zed2i/depth/image_raw/compressed"]="zed2i_depth"
        #         ["/boxi/zed2i/confidence/image_raw/compressed"]="zed2i_confidence"
        #     )
        #     ;;

        # *anymal_depth_cameras.bag)
        #     echo "-> Found Anymal Depth bag. Checking Anymal topics..."
        #     TOPICS_TO_CHECK=(
        #         ["/anymal/depth_camera/left/depth/image_rect_raw"]="anymal_depth_left"
        #         ["/anymal/depth_camera/right/depth/image_rect_raw"]="anymal_depth_right"
        #         ["/anymal/depth_camera/front_lower/depth/image_rect_raw"]="anymal_depth_front_lower"
        #         ["/anymal/depth_camera/front_upper/depth/image_rect_raw"]="anymal_depth_front_upper"
        #         ["/anymal/depth_camera/rear_lower/depth/image_rect_raw"]="anymal_depth_rear_lower"
        #         ["/anymal/depth_camera/rear_upper/depth/image_rect_raw"]="anymal_depth_rear_upper"
        #     )
        #     ;;
            
        *)
            echo "-> Skipping bag (not a known image bag)."
            continue # Skip to the next bag file
            ;;
    esac

    # Now, for this specific bag, loop over the topics we just selected
    for TOPIC in "${!TOPICS_TO_CHECK[@]}"; do
        FOLDER_NAME=${TOPICS_TO_CHECK[$TOPIC]}
        
        echo "  -> Processing Topic: $TOPIC (Saving 1 every $SAVE_EVERY_N_FRAMES frames)"
        
        # Run the Python script
        python3 "$PYTHON_SCRIPT_PATH" \
            --bagfile "$BAG_FILE" \
            --topic "$TOPIC" \
            --output_dir "$FRAME_DIR" \
            --subfolder "$FOLDER_NAME" \
            --every_n_frames "$SAVE_EVERY_N_FRAMES" # <-- MODIFIED: Added new argument
    done
done

echo -e "\n-----------------------------------------------------"
echo "All processing complete."
echo "-----------------------------------------------------"