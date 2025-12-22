#!/usr/bin/env python3
"""
Parameter sweep script for A=9 training.

This script runs multiple training jobs with different parameter combinations
and saves results in subfolders under A_9_sweeps/.
"""

import subprocess
import os
import itertools
from datetime import datetime

def run_training(config):
    """Run a single training job with given configuration."""
    # Build command
    cmd = [
        'python', 'small_MNIST.py',
        '--A', '9',
        '--run_directory', config['run_directory'],
        '--N', str(config['N']),
        '--batch_size', str(config['batch_size']),
        '--lr', str(config['lr']),
        '--epochs', str(config['epochs']),
        '--lambda_recon', str(config['lambda_recon']),
        '--d_train_ratio', str(config['d_train_ratio']),
        '--adversarial_weight', str(config['adversarial_weight']),
    ]
    
    # Add optional parameters if specified
    if config.get('d_lr') is not None:
        cmd.extend(['--d_lr', str(config['d_lr'])])
    if config.get('g_lr') is not None:
        cmd.extend(['--g_lr', str(config['g_lr'])])
    if config.get('use_feature_matching', False):
        cmd.append('--use_feature_matching')
    if config.get('use_gradient_penalty', False):
        cmd.append('--use_gradient_penalty')
    if config.get('use_lr_scheduler', False):
        cmd.append('--use_lr_scheduler')
    if config.get('gradient_clip', 0) > 0:
        cmd.extend(['--gradient_clip', str(config['gradient_clip'])])
    
    print(f"\n{'='*80}")
    print(f"Running: {config['run_directory']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Run training
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"\n✓ Successfully completed: {config['run_directory']}\n")
    else:
        print(f"\n✗ Failed: {config['run_directory']} (exit code: {result.returncode})\n")
    
    return result.returncode == 0


def main():
    """Main function to run parameter sweep."""
    
    # Base configuration
    base_config = {
        'A': 9,
        'N': 100,
        'epochs': 100,
        'checkpoint_interval': 10,
        'seed': 42,
    }
    
    # Parameter sweep configurations - 50 total configurations
    # Define ranges for parameters you want to sweep
    sweep_configs = [
        # === BASELINE AND VARIATIONS ===
        # 1. Baseline configuration
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        
        # === RECONSTRUCTION WEIGHT VARIATIONS ===
        # 2-6. Different lambda_recon values
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.01,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.05,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.2,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.3,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.5,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 1.0,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        
        # === LEARNING RATE VARIATIONS ===
        # 8-12. Different learning rates
        {
            'batch_size': 640,
            'lr': 0.00005,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0001,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0003,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0005,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.001,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        
        # === ADVERSARIAL WEIGHT VARIATIONS ===
        # 13-17. Different adversarial weights
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.05,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.2,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.3,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.5,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 1.0,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        
        # === D_TRAIN_RATIO VARIATIONS ===
        # 18-22. Different discriminator training ratios
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 2,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 3,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 4,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 5,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        
        # === GRADIENT CLIPPING VARIATIONS ===
        # 23-26. Different gradient clipping values
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.5,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 1.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 2.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 5.0,
        },
        
        # === DIFFERENT D/G LEARNING RATES ===
        # 27-31. Different D/G learning rate combinations
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': 0.00005,
            'g_lr': 0.0004,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': 0.0001,
            'g_lr': 0.0004,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': 0.0001,
            'g_lr': 0.0003,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': 0.0003,
            'g_lr': 0.0001,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': 0.0004,
            'g_lr': 0.0001,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        
        # === FEATURE MATCHING VARIATIONS ===
        # 32-36. Feature matching with different parameters
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': True,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.2,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': True,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0003,
            'lambda_recon': 0.1,
            'd_train_ratio': 2,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': True,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        
        # === GRADIENT PENALTY VARIATIONS ===
        # 35-38. Gradient penalty with different parameters
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': True,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.2,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': True,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0003,
            'lambda_recon': 0.1,
            'd_train_ratio': 2,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': True,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        
        # === COMBINED FEATURES ===
        # 39-45. Combinations of features
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': True,
            'use_gradient_penalty': True,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.2,
            'd_train_ratio': 2,
            'adversarial_weight': 0.2,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': True,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 1.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.2,
            'd_train_ratio': 2,
            'adversarial_weight': 0.2,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': True,
            'use_lr_scheduler': False,
            'gradient_clip': 1.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.2,
            'd_train_ratio': 2,
            'adversarial_weight': 0.2,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': True,
            'use_gradient_penalty': True,
            'use_lr_scheduler': False,
            'gradient_clip': 1.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0003,
            'lambda_recon': 0.2,
            'd_train_ratio': 2,
            'adversarial_weight': 0.2,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': True,
            'use_gradient_penalty': True,
            'use_lr_scheduler': False,
            'gradient_clip': 1.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.3,
            'd_train_ratio': 3,
            'adversarial_weight': 0.3,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': True,
            'use_gradient_penalty': True,
            'use_lr_scheduler': False,
            'gradient_clip': 2.0,
        },
        
        # === LEARNING RATE SCHEDULER ===
        # 46-48. With learning rate scheduler
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': True,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.2,
            'd_train_ratio': 2,
            'adversarial_weight': 0.2,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': True,
            'use_gradient_penalty': False,
            'use_lr_scheduler': True,
            'gradient_clip': 1.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0002,
            'lambda_recon': 0.2,
            'd_train_ratio': 2,
            'adversarial_weight': 0.2,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': True,
            'use_gradient_penalty': True,
            'use_lr_scheduler': True,
            'gradient_clip': 1.0,
        },
        
        # === BATCH SIZE VARIATIONS ===
        # 49-50. Different batch sizes
        {
            'batch_size': 320,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        {
            'batch_size': 1280,
            'lr': 0.0002,
            'lambda_recon': 0.1,
            'd_train_ratio': 1,
            'adversarial_weight': 0.1,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': False,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 0.0,
        },
        
        # === ADDITIONAL COMBINATIONS ===
        # 48-50. Additional interesting combinations
        {
            'batch_size': 640,
            'lr': 0.0001,
            'lambda_recon': 0.2,
            'd_train_ratio': 3,
            'adversarial_weight': 0.3,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': True,
            'use_gradient_penalty': True,
            'use_lr_scheduler': True,
            'gradient_clip': 1.0,
        },
        {
            'batch_size': 640,
            'lr': 0.0003,
            'lambda_recon': 0.3,
            'd_train_ratio': 2,
            'adversarial_weight': 0.2,
            'd_lr': 0.0002,
            'g_lr': 0.0004,
            'use_feature_matching': True,
            'use_gradient_penalty': False,
            'use_lr_scheduler': False,
            'gradient_clip': 2.0,
        },
        {
            'batch_size': 1280,
            'lr': 0.0002,
            'lambda_recon': 0.2,
            'd_train_ratio': 2,
            'adversarial_weight': 0.2,
            'd_lr': None,
            'g_lr': None,
            'use_feature_matching': True,
            'use_gradient_penalty': True,
            'use_lr_scheduler': False,
            'gradient_clip': 1.0,
        },
    ]
    
    # Generate run directory names and merge configs
    configs = []
    for i, sweep_config in enumerate(sweep_configs):
        config = {**base_config, **sweep_config}
        
        # Create descriptive run directory name
        run_dir_parts = ['A_9_sweeps']
        
        # Add parameter identifiers
        if sweep_config['batch_size'] != 640:
            run_dir_parts.append(f'bs{sweep_config["batch_size"]}')
        if sweep_config['lambda_recon'] != 0.1:
            run_dir_parts.append(f'lambda{sweep_config["lambda_recon"]}')
        if sweep_config['lr'] != 0.0002:
            run_dir_parts.append(f'lr{sweep_config["lr"]}')
        if sweep_config['adversarial_weight'] != 0.1:
            run_dir_parts.append(f'adv{sweep_config["adversarial_weight"]}')
        if sweep_config['d_train_ratio'] != 1:
            run_dir_parts.append(f'dtr{sweep_config["d_train_ratio"]}')
        if sweep_config.get('d_lr') is not None:
            run_dir_parts.append(f'dlr{sweep_config["d_lr"]}_glr{sweep_config["g_lr"]}')
        if sweep_config.get('use_feature_matching', False):
            run_dir_parts.append('fm')
        if sweep_config.get('use_gradient_penalty', False):
            run_dir_parts.append('gp')
        if sweep_config.get('gradient_clip', 0) > 0:
            run_dir_parts.append(f'clip{sweep_config["gradient_clip"]}')
        if sweep_config.get('use_lr_scheduler', False):
            run_dir_parts.append('sched')
        
        # If no special params, use baseline
        if len(run_dir_parts) == 1:
            run_dir_parts.append('baseline')
        
        config['run_directory'] = '_'.join(run_dir_parts)
        configs.append(config)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Parameter Sweep for A=9")
    print(f"Total configurations: {len(configs)}")
    print(f"{'='*80}\n")
    
    for i, config in enumerate(configs, 1):
        print(f"{i:2d}. {config['run_directory']}")
        print(f"    batch_size={config['batch_size']}, lr={config['lr']}, "
              f"lambda_recon={config['lambda_recon']}, "
              f"d_train_ratio={config['d_train_ratio']}, "
              f"adversarial_weight={config['adversarial_weight']}")
        if config.get('d_lr') is not None:
            print(f"    d_lr={config['d_lr']}, g_lr={config['g_lr']}")
        if config.get('use_feature_matching'):
            print(f"    feature_matching=True")
        if config.get('use_gradient_penalty'):
            print(f"    gradient_penalty=True")
        if config.get('gradient_clip', 0) > 0:
            print(f"    gradient_clip={config['gradient_clip']}")
        if config.get('use_lr_scheduler'):
            print(f"    lr_scheduler=True")
    
    print(f"\n{'='*80}\n")
    
    # Ask for confirmation
    response = input("Proceed with training? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Aborted.")
        return
    
    # Run all configurations
    results = []
    start_time = datetime.now()
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Starting training for: {config['run_directory']}")
        success = run_training(config)
        results.append({
            'config': config,
            'success': success
        })
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"Parameter Sweep Complete")
    print(f"Duration: {duration}")
    print(f"{'='*80}\n")
    
    print("Results Summary:")
    print("-" * 80)
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    for i, result in enumerate(results, 1):
        status = "✓" if result['success'] else "✗"
        print(f"{status} {i:2d}. {result['config']['run_directory']}")
    
    print(f"\nSuccessful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

