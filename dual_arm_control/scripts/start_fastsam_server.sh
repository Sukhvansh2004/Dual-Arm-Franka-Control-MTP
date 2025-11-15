#!/bin/bash

# Script to launch the FastSAM ZMQ server.

# --- Configuration ---
# Path to the GraspGen python executable in your conda environment
# IMPORTANT: Make sure this path is correct for your system.
CONDA_ENV_PYTHON="/home/sukhvansh/anaconda3/envs/GraspGen/bin/python"

# Path to the fastsam server script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
FASTSAM_SERVER_SCRIPT="$SCRIPT_DIR/Perception/FastSAM/fastsam_server.py"


# --- Argument Parsing ---
PORT=$1

PORT_ARG=""
if [[ -n "$PORT" ]]; then
    PORT_ARG="--port $PORT"
    echo "Using specified port: $PORT"
else
    echo "Using default port (5556) from server script."
fi

# --- Validation ---
if [ ! -f "$CONDA_ENV_PYTHON" ]; then
    echo "Error: Python executable not found at $CONDA_ENV_PYTHON"
    echo "Please update the CONDA_ENV_PYTHON variable in this script."
    exit 1
fi

if [ ! -f "$FASTSAM_SERVER_SCRIPT" ]; then
    echo "Error: FastSAM server script not found at $FASTSAM_SERVER_SCRIPT"
    exit 1
fi


# --- Execution ---
echo "====================================================="
echo "Starting FastSAM ZMQ Server..."
echo "  Port Args: $PORT_ARG"
echo "  Script: $FASTSAM_SERVER_SCRIPT"
echo "====================================================="

# Execute the python script, passing the arguments
set -f
"$CONDA_ENV_PYTHON" "$FASTSAM_SERVER_SCRIPT" $PORT_ARG
set +f

echo "FastSAM ZMQ Server has terminated."
