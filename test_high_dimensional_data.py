import numpy as np
import time

def benchmark_matrix_multiplication(n=1000, repetitions=5):
    times = []

    for i in range(repetitions):
        # Create two random matrices
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)

        # Time the matrix multiplication
        start = time.time()
        C = np.dot(A, B)  # or A @ B
        end = time.time()

        elapsed = end - start
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.4f} seconds")

    avg_time = sum(times) / repetitions
    print(f"\nAverage time over {repetitions} runs: {avg_time:.4f} seconds")

if __name__ == "__main__":
    benchmark_matrix_multiplication(n=10000, repetitions=10)
