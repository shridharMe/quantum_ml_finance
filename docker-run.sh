#!/bin/bash

# Script to build and run the Quantum ML Finance application using Docker

# Build the Docker image
echo "Building Docker image..."
podman build -t quantum_ml_finance .

# Run the application
echo "Running Quantum ML Finance application..."
podman run -it --rm \
  -v $(pwd)/output:/tmp/matplotlib \
  quantum_ml_finance

# For interactive shell, uncomment the following line:
# docker run -it --rm quantum_ml_finance /bin/bash