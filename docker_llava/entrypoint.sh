#!/bin/bash

# Create the symbolic link
ln -s /workspace/home/google-research /workspace/home01/google-research-docker

# Execute the original command
exec "$@"
