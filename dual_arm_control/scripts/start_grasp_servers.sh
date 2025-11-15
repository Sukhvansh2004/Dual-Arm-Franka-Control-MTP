#!/bin/bash

# This script starts the GraspGen ZMQ servers for the dual arm setup.
# It assumes the left arm uses a finger gripper and the right arm uses a suction gripper.

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to find the conda installation and activate the environment
activate_conda_env() {
    local env_name=$1
    # Try to find conda.sh in common locations
    local conda_sh_path
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        conda_sh_path="$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        conda_sh_path="$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        conda_sh_path="/opt/conda/etc/profile.d/conda.sh"
    else
        echo "Could not find conda.sh. Please edit this script to point to your conda.sh."
        exit 1
    fi
    
    echo "Activating conda environment '$env_name'..."
    source "$conda_sh_path"
    conda activate "$env_name"
}

# Activate conda environment
# Replace 'GraspGen' with your actual conda environment name if different.
activate_conda_env "GraspGen"

# Get the directory of this script to reliably locate the server script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SERVER_SCRIPT="$SCRIPT_DIR/Perception/combined_server.py"

if [ ! -f "$SERVER_SCRIPT" ]; then
    echo "Server script not found at: $SERVER_SCRIPT"
    exit 1
fi

LEFT_ARM_PORT=5555
RIGHT_ARM_PORT=5556

echo "Starting server for LEFT arm (finger gripper) on port $LEFT_ARM_PORT..."
python "$SERVER_SCRIPT" finger --port "$LEFT_ARM_PORT" &
LEFT_PID=$!

echo "Starting server for RIGHT arm (suction gripper) on port $RIGHT_ARM_PORT..."
python "$SERVER_SCRIPT" suction --port "$RIGHT_ARM_PORT" &
RIGHT_PID=$!

echo "Grasp servers started."
echo "Left arm (finger) server PID: $LEFT_PID"
echo "Right arm (suction) server PID: $RIGHT_PID"
echo ""
echo "Use 'kill $LEFT_PID $RIGHT_PID' or 'pkill -f combined_server.py' to stop them."

# Define a cleanup function
cleanup() {
    echo "Shutting down grasp servers..."
    kill $LEFT_PID $RIGHT_PID
    echo "Servers stopped."
}

# Trap SIGINT (Ctrl+C) and call the cleanup function
trap cleanup SIGINT SIGTERM

# Wait for both background processes to finish
wait $LEFT_PID
wait $RIGHT_PID
