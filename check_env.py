import os
import sys

# Print Python executable path
print(f"Python executable: {sys.executable}")

# Print virtual environment path if active
print(f"Virtual env: {os.environ.get('VIRTUAL_ENV', 'Not in a virtual environment')}")

# Print all environment variables
print("\nAll environment variables:")
for key, value in os.environ.items():
    print(f"{key}: {value}") 