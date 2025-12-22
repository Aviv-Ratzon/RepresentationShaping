#!/bin/bash

# Batch script to run MNIST experiments with different regularization types
# This script runs the same experiment with different regularization methods

echo "Starting regularization experiments..."
echo "======================================"

# Base command parameters
BASE_CMD="python small_MNIST.py --A 1 --samples_M 1024 --N 100 --epochs 20 --checkpoint_interval 1 --lr 0.0001"

# Define regularization types
REG_TYPES=("none" "l1" "l2" "dropout" "l1_l2" "all")

# Run experiments for each regularization type
for reg_type in "${REG_TYPES[@]}"; do
    echo ""
    echo "Running experiment with regularization: $reg_type"
    echo "----------------------------------------"
    
    # Set run directory name with regularization type
    RUN_DIR="A_1_${reg_type}"
    
    # Run the experiment
    $BASE_CMD --run_directory "$RUN_DIR" --regularization_type "$reg_type"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✅ Experiment with $reg_type completed successfully!"
    else
        echo "❌ Experiment with $reg_type failed!"
    fi
    
    echo "Results saved to: MNIST_small/$RUN_DIR/"
    echo ""
done

echo "======================================"
echo "All regularization experiments completed!"
echo ""
echo "Summary of experiments:"
echo "- none: No regularization"
echo "- l1: L1 regularization (λ=0.001)"
echo "- l2: L2 regularization (λ=0.001)"
echo "- dropout: Dropout regularization (rate=0.3)"
echo "- l1_l2: Combined L1+L2 regularization (λ=0.0005 each)"
echo "- all: All regularization types (L1=0.0003, L2=0.0003, Dropout=0.2)"
echo ""
echo "Check the results in the MNIST_small/ directory for each experiment."

