#!/bin/bash

# Detailed batch script to run MNIST experiments with different regularization types
# This script runs comprehensive experiments with various regularization methods

echo "Starting detailed regularization experiments..."
echo "=============================================="

# Base command parameters
BASE_CMD="python small_MNIST.py --A 1 --samples_M 1024 --N 100 --epochs 20 --checkpoint_interval 1 --lr 0.0001"

# Create results summary file
SUMMARY_FILE="regularization_experiments_summary.txt"
echo "Regularization Experiments Summary" > $SUMMARY_FILE
echo "Generated on: $(date)" >> $SUMMARY_FILE
echo "=================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Define regularization types with descriptions
declare -A REG_TYPES
REG_TYPES["none"]="No regularization (baseline)"
REG_TYPES["l1"]="L1 regularization (λ=0.001)"
REG_TYPES["l2"]="L2 regularization (λ=0.001)"
REG_TYPES["dropout"]="Dropout regularization (rate=0.3)"
REG_TYPES["l1_l2"]="Combined L1+L2 regularization (λ=0.0005 each)"
REG_TYPES["all"]="All regularization types (L1=0.0003, L2=0.0003, Dropout=0.2)"

# Additional regularization experiments with different strengths
declare -A ADDITIONAL_REG
ADDITIONAL_REG["l1_strong"]="Strong L1 regularization (λ=0.01)"
ADDITIONAL_REG["l2_strong"]="Strong L2 regularization (λ=0.01)"
ADDITIONAL_REG["dropout_strong"]="Strong Dropout regularization (rate=0.5)"
ADDITIONAL_REG["l1_weak"]="Weak L1 regularization (λ=0.0001)"
ADDITIONAL_REG["l2_weak"]="Weak L2 regularization (λ=0.0001)"

# Function to run experiment
run_experiment() {
    local reg_type=$1
    local description=$2
    local run_dir="A_1_${reg_type}"
    
    echo ""
    echo "Running experiment: $reg_type"
    echo "Description: $description"
    echo "Run directory: $run_dir"
    echo "----------------------------------------"
    
    # Record start time
    start_time=$(date +%s)
    
    # Run the experiment
    $BASE_CMD --run_directory "$run_dir" --regularization_type "$reg_type" 2>&1 | tee "${run_dir}_output.log"
    
    # Check if the command was successful
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Experiment with $reg_type completed successfully! (Duration: ${duration}s)"
        echo "$reg_type: SUCCESS (${duration}s) - $description" >> $SUMMARY_FILE
    else
        echo "❌ Experiment with $reg_type failed! (Duration: ${duration}s)"
        echo "$reg_type: FAILED (${duration}s) - $description" >> $SUMMARY_FILE
    fi
    
    echo "Results saved to: MNIST_small/$run_dir/"
    echo "Log saved to: ${run_dir}_output.log"
    echo ""
}

# Run basic regularization experiments
echo "Running basic regularization experiments..."
for reg_type in "${!REG_TYPES[@]}"; do
    run_experiment "$reg_type" "${REG_TYPES[$reg_type]}"
done

# Run additional regularization experiments
echo "Running additional regularization experiments..."
for reg_type in "${!ADDITIONAL_REG[@]}"; do
    run_experiment "$reg_type" "${ADDITIONAL_REG[$reg_type]}"
done

echo "=============================================="
echo "All regularization experiments completed!"
echo ""
echo "Summary of experiments:"
echo "======================"
cat $SUMMARY_FILE
echo ""
echo "Check individual experiment logs for detailed output."
echo "Results are organized in MNIST_small/ directory with subdirectories for each experiment."

