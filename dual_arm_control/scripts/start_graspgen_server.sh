#!/bin/bash

# This script starts the GraspGen ZMQ servers for the dual arm setup.
# It launches two instances of the GraspGen server, one for each gripper type.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
CONDA_ENV_NAME="GraspGen"
LEFT_GRIPPER_TYPE="finger"
RIGHT_GRIPPER_TYPE="suction"
LEFT_PORT=5557
RIGHT_PORT=5558

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
