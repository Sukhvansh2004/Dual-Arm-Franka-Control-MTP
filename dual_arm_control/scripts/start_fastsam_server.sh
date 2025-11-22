#!/bin/bash

# This script starts the FastSAM ZMQ server and pulls configuration 
# (Conda environment name and port) from ROS Parameters.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# IMPORTANT: This must match the 'name' attribute in your launch file
NODE_NAME="start_fastsam_servers" 

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
PORT=$(get_param "port")

echo "Configuration Loaded:"
echo "  Env: $CONDA_ENV_NAME"
echo "  Port: $PORT"

# --- Helper Functions (Ensures 'python' points to the correct Conda environment) ---
activate_conda_env() {
    local env_name=$1
    local conda_sh_path
    
    # Try finding common conda installation paths
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        conda_sh_path="$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        conda_sh_path="$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        echo "Could not find conda.sh. Please ensure Conda is initialized in your shell."
        exit 1
    fi
    
    echo "Activating conda environment '$env_name'..."
    # Source the shell initialization script for conda
    source "$conda_sh_path"
    # Activate the environment
    conda activate "$env_name"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to activate conda environment '$env_name'. Check its existence."
        exit 1
    fi
}

# --- Main Script ---
activate_conda_env "$CONDA_ENV_NAME"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
FASTSAM_SERVER_SCRIPT="$SCRIPT_DIR/Perception/FastSAM/fastsam_server.py"

# --- Validation ---
if [ ! -f "$FASTSAM_SERVER_SCRIPT" ]; then
    echo "Error: FastSAM server script not found at $FASTSAM_SERVER_SCRIPT"
    exit 1
fi

# --- Execution ---
echo "====================================================="
echo "Starting FastSAM ZMQ Server..."
echo "  Port: $PORT"
echo "  Script: $FASTSAM_SERVER_SCRIPT"
echo "====================================================="

# Execute the python script using 'python' now that the correct environment is active
set -f
python "$FASTSAM_SERVER_SCRIPT" --port "$PORT"
set +f

echo "FastSAM ZMQ Server has terminated."