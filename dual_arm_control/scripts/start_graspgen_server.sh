#!/bin/bash

# This script starts the GraspGen ZMQ servers for the dual arm setup.
# It pulls configuration from ROS Parameters.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# IMPORTANT: This must match the 'name' attribute in your launch file
NODE_NAME="start_graspgen_servers"

echo "--- Fetching ROS Parameters for $NODE_NAME ---"

# Helper to fetch param or exit if missing
get_param() {
    local param_key="/$NODE_NAME/$1"
    local value
    # rosparam get returns the value. If it fails, the set -e above will stop the script.
    value=$(rosparam get "$param_key")
    
    if [ -z "$value" ] || [ "$value" == "null" ]; then
        echo "Error: Could not retrieve parameter '$param_key'"
        exit 1
    fi
    echo "$value"
}

# Fetch values from the parameter server
CONDA_ENV_NAME=$(get_param "conda_env")
LEFT_GRIPPER_TYPE=$(get_param "left_gripper")
RIGHT_GRIPPER_TYPE=$(get_param "right_gripper")
LEFT_PORT=$(get_param "left_port")
RIGHT_PORT=$(get_param "right_port")

echo "Configuration Loaded:"
echo "  Env: $CONDA_ENV_NAME"
echo "  Left: $LEFT_GRIPPER_TYPE ($LEFT_PORT)"
echo "  Right: $RIGHT_GRIPPER_TYPE ($RIGHT_PORT)"

# --- Helper Functions ---
activate_conda_env() {
    local env_name=$1
    local conda_sh_path
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        conda_sh_path="$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        conda_sh_path="$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        echo "Could not find conda.sh. Please edit this script to point to your conda.sh."
        exit 1
    fi
    
    echo "Activating conda environment '$env_name'..."
    source "$conda_sh_path"
    conda activate "$env_name"
}

# --- Main Script ---
activate_conda_env "$CONDA_ENV_NAME"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
GRASP_SERVER_SCRIPT="$SCRIPT_DIR/Perception/Grasp/grasp_server.py"

if [ ! -f "$GRASP_SERVER_SCRIPT" ]; then
    echo "Grasp server script not found at: $GRASP_SERVER_SCRIPT"
    exit 1
fi

echo "--- Starting GraspGen ZMQ Servers ---"

echo "Starting server for LEFT arm ($LEFT_GRIPPER_TYPE gripper) on port $LEFT_PORT..."
python "$GRASP_SERVER_SCRIPT" "$LEFT_GRIPPER_TYPE" --port "$LEFT_PORT" &
LEFT_PID=$!

echo "Starting server for RIGHT arm ($RIGHT_GRIPPER_TYPE gripper) on port $RIGHT_PORT..."
python "$GRASP_SERVER_SCRIPT" "$RIGHT_GRIPPER_TYPE" --port "$RIGHT_PORT" &
RIGHT_PID=$!

echo "--- Servers Started ---"
echo "Left arm ($LEFT_GRIPPER_TYPE) server PID: $LEFT_PID"
echo "Right arm ($RIGHT_GRIPPER_TYPE) server PID: $RIGHT_PID"
echo ""
echo "Press Ctrl+C to stop all servers."

# Define a cleanup function
cleanup() {
    echo ""
    echo "--- Shutting down GraspGen servers ---"
    kill $LEFT_PID $RIGHT_PID
    echo "Servers stopped."
}

# Trap SIGINT (Ctrl+C) and call the cleanup function
trap cleanup SIGINT SIGTERM

# Wait for both background processes to finish
wait $LEFT_PID
wait $RIGHT_PID