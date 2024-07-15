#!/bin/bash

# Create the symbolic link
ln -s /workspace/home/google-research /workspace/home01/google-research-docker

# setup dito
cd home01/google-research-docker/fvlm
pip install -r requirements.txt
pip install -e .
cd ./dito/checkpoints
chmod +x download.sh
./download.sh
cd ../embeddings
chmod +x download.sh
./download.sh

# Execute the original command
exec "$@"
