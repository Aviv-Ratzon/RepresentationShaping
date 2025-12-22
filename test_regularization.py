#!/usr/bin/env python3
"""
Test script to verify regularization functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from small_MNIST import MNISTActionGAN
import argparse

def test_regularization():
    """Test different regularization types."""
    
    # Test configurations
    test_configs = [
        {'regularization_type': 'none', 'epochs': 1, 'N': 10},
        {'regularization_type': 'l1', 'epochs': 1, 'N': 10},
        {'regularization_type': 'l2', 'epochs': 1, 'N': 10},
        {'regularization_type': 'dropout', 'epochs': 1, 'N': 10},
        {'regularization_type': 'l1_l2', 'epochs': 1, 'N': 10},
        {'regularization_type': 'all', 'epochs': 1, 'N': 10},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n{'='*50}")
        print(f"Testing regularization: {config['regularization_type']}")
        print(f"{'='*50}")
        
        # Create args
        args = argparse.Namespace(
            A=1,
            run_directory=f"test_{config['regularization_type']}",
            samples_M=64,
            N=config['N'],
            epochs=config['epochs'],
            checkpoint_interval=1,
            lr=0.0001,
            latent_dim=64,
            batch_size=32,
            n_layers=2,
            seed=42,
            # Regularization parameters
            l1_reg=0.0,
            l2_reg=0.0,
            dropout_rate=0.0,
            regularization_type=config['regularization_type'],
            # Other required parameters
            cyclic=False,
            lambda_recon=0.0,
            samples_gen_pc=5,
            pc_sampling_interval=10,
            plot_only=False,
            checkpoint_path=None,
            use_lr_scheduler=False,
            gradient_clip=0.0,
            d_loss_weight=1.0,
            g_loss_weight=1.0,
            use_feature_matching=False,
            use_gradient_penalty=False,
            gp_weight=10.0,
            d_train_ratio=1,
            adversarial_weight=0.1,
            feature_matching_weight=1.0,
            d_lr=None,
            g_lr=None,
            adaptive_d_train_ratio=False,
            target_d_g_ratio=1.0
        )
        
        try:
            # Create model
            model = MNISTActionGAN(args)
            
            # Test regularization loss computation
            reg_loss = model._compute_regularization_loss()
            print(f"Regularization loss: {reg_loss.item():.6f}")
            
            # Test one training step
            print("Testing one training step...")
            batch = next(iter(model.dataloader))
            input_images, input_labels, actions, target_labels = batch
            input_images = input_images.to(model.device)
            input_labels = input_labels.to(model.device)
            actions = actions.to(model.device)
            target_labels = target_labels.to(model.device)
            
            # Train discriminator
            with model.encoder.no_grad():
                z = model.encoder(input_images, actions)
                fake_images = model.generator(z)
            
            d_loss, real_loss, fake_loss = model.train_discriminator(
                input_images, input_labels, fake_images, target_labels
            )
            
            # Train generator
            g_loss, gan_loss, recon_loss = model.train_generator_encoder(
                input_images, input_labels, actions, target_labels
            )
            
            print(f"Training step completed successfully!")
            print(f"  D_loss: {d_loss:.4f}")
            print(f"  G_loss: {g_loss:.4f}")
            print(f"  Real_loss: {real_loss:.4f}")
            print(f"  Fake_loss: {fake_loss:.4f}")
            
        except Exception as e:
            print(f"❌ Error testing {config['regularization_type']}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print("Regularization testing completed!")
    print(f"{'='*50}")

if __name__ == "__main__":
    test_regularization()

