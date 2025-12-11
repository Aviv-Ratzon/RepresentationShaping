#!/usr/bin/env python3
"""
Quick test script to verify convergence improvements
"""

import subprocess
import sys

def test_convergence():
    """Test training with improved hyperparameters"""
    
    print("Testing convergence with improved hyperparameters...")
    print("=" * 60)
    
    # Test configurations
    configs = [
        {
            "name": "Conservative (Small Batch, High LR)",
            "args": ["--epochs", "2", "--batch_size", "64", "--lr", "1e-3", "--grad_clip", "1.0", "--skip_analysis"]
        },
        {
            "name": "Balanced (Default)",
            "args": ["--epochs", "2", "--batch_size", "128", "--lr", "1e-3", "--grad_clip", "1.0", "--skip_analysis"]
        },
        {
            "name": "With LR Scheduler",
            "args": ["--epochs", "2", "--batch_size", "128", "--lr", "1e-3", "--lr_scheduler", "--grad_clip", "1.0", "--skip_analysis"]
        }
    ]
    
    for config in configs:
        print(f"\n🧪 Testing: {config['name']}")
        print("-" * 40)
        
        cmd = ["python", "cifar100_conditional_generator.py"] + config["args"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
            
            if result.returncode == 0:
                print("✅ SUCCESS: Training completed without errors")
                
                # Extract loss information
                lines = result.stdout.split('\n')
                train_losses = []
                val_losses = []
                
                for line in lines:
                    if "Train Loss:" in line:
                        try:
                            loss = float(line.split("Train Loss:")[1].split()[0])
                            train_losses.append(loss)
                        except:
                            pass
                    elif "Val Loss:" in line:
                        try:
                            loss = float(line.split("Val Loss:")[1].split()[0])
                            val_losses.append(loss)
                        except:
                            pass
                
                if len(train_losses) >= 2:
                    loss_trend = "📈 Increasing" if train_losses[-1] > train_losses[0] else "📉 Decreasing"
                    print(f"   Loss trend: {loss_trend}")
                    print(f"   Final train loss: {train_losses[-1]:.4f}")
                    if val_losses:
                        print(f"   Final val loss: {val_losses[-1]:.4f}")
                else:
                    print("   ⚠️  Could not extract loss information")
                    
            else:
                print("❌ FAILED: Training failed")
                print(f"   Error: {result.stderr[:200]}...")
                
        except subprocess.TimeoutExpired:
            print("⏰ TIMEOUT: Training took too long")
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("Convergence test completed!")

if __name__ == "__main__":
    test_convergence()
