"""
Quick test to verify the model dimensions are correct.
"""

import torch
import torch.nn as nn
from mnist_action_transformer import MNISTActionTransformer

def test_model_dimensions():
    """Test that the model can handle the correct input/output dimensions."""
    print("Testing model dimensions...")
    
    # Create model
    max_move = 2
    model = MNISTActionTransformer(max_move=max_move, hidden_dim=256, latent_dim=128)
    
    # Test with batch size 1
    batch_size = 1
    input_img = torch.randn(batch_size, 1, 28, 28)
    action = torch.zeros(batch_size, 2*max_move+1)
    action[:, max_move] = 1.0  # Action = 0
    
    print(f"Input image shape: {input_img.shape}")
    print(f"Action shape: {action.shape}")
    
    # Test encoder
    with torch.no_grad():
        # Test each part of the encoder
        x = input_img
        print(f"Initial input: {x.shape}")
        
        # First conv layer: 28x28 -> 14x14
        x = model.image_encoder[0](x)  # Conv2d
        x = model.image_encoder[1](x)  # BatchNorm
        x = model.image_encoder[2](x)  # ReLU
        print(f"After first conv: {x.shape}")
        
        # Second conv layer: 14x14 -> 7x7
        x = model.image_encoder[3](x)  # Conv2d
        x = model.image_encoder[4](x)  # BatchNorm
        x = model.image_encoder[5](x)  # ReLU
        print(f"After second conv: {x.shape}")
        
        # Third conv layer: 7x7 -> 4x4
        x = model.image_encoder[6](x)  # Conv2d
        x = model.image_encoder[7](x)  # BatchNorm
        x = model.image_encoder[8](x)  # ReLU
        print(f"After third conv: {x.shape}")
        
        # Flatten
        x = model.image_encoder[9](x)  # Flatten
        print(f"After flatten: {x.shape}")
        
        # Linear layer
        x = model.image_encoder[10](x)  # Linear
        x = model.image_encoder[11](x)  # ReLU
        print(f"After linear: {x.shape}")
        
        # Test full forward pass
        output = model(input_img, action)
        print(f"Final output shape: {output.shape}")
        
        assert output.shape == input_img.shape, f"Output shape {output.shape} doesn't match input shape {input_img.shape}"
        print("✓ Model dimensions test passed!")

if __name__ == '__main__':
    test_model_dimensions()
