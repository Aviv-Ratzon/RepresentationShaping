"""
Helper script to run PredNet training on remote GPU server.
This script can be run on your local machine to set up and execute training remotely.
"""

import subprocess
import sys
import os

def run_command(cmd, check=True):
    """Run a shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_on_remote.py username@remote-server [remote-directory]")
        print("Example: python run_on_remote.py user@server.com ~/prednet_training")
        sys.exit(1)
    
    remote = sys.argv[1]
    remote_dir = sys.argv[2] if len(sys.argv) > 2 else "~/prednet_training"
    
    script_file = "train_prednet_moving_mnist.py"
    if not os.path.exists(script_file):
        print(f"Error: {script_file} not found in current directory")
        sys.exit(1)
    
    print(f"Transferring {script_file} to {remote}:{remote_dir}/")
    run_command(f'scp {script_file} {remote}:{remote_dir}/')
    
    print("\nSetting up environment and running on remote server...")
    
    # Create setup and run script
    setup_script = f"""
cd {remote_dir}

# Check Python
python3 --version || python --version

# Install dependencies if needed
pip install torch torchvision numpy matplotlib tqdm --quiet

# Check GPU
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

# Run training
echo "Starting training..."
python3 train_prednet_moving_mnist.py
"""
    
    # Save setup script temporarily
    with open("remote_setup.sh", "w") as f:
        f.write(setup_script)
    
    # Transfer and run
    run_command(f'scp remote_setup.sh {remote}:{remote_dir}/setup.sh')
    run_command(f'ssh {remote} "bash {remote_dir}/setup.sh"', check=False)
    
    # Cleanup
    os.remove("remote_setup.sh")
    
    print("\nTraining started!")
    print(f"To monitor: ssh {remote} && tail -f {remote_dir}/training.log")
    print(f"To check GPU: ssh {remote} && nvidia-smi")

if __name__ == "__main__":
    main()






