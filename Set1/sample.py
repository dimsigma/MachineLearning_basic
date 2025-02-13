import numpy as np
import json

size_b, size_c = 1000000, 200

# Sample for f1(x)
def sample_f1(size):
    choice = np.random.normal(0, 1, size)
    samples1 = np.random.normal(-1, 1, size)
    samples2 = np.random.normal(1, 1, size)
    samples = np.where(choice < 0, samples1, samples2)
    return samples

if __name__ == "__main__":
    # Create 10e6 samples to compute total probability error
    f0_x0_samples_b = np.random.normal(0, 1, size_b)
    f0_x1_samples_b = np.random.normal(0, 1, size_b)
    f1_x0_samples_b = sample_f1(size_b)
    f1_x1_samples_b = sample_f1(size_b)

    # Create 200 samples to train neural networks
    f0_x0_samples_c = np.random.normal(0, 1, size_c)
    f0_x1_samples_c = np.random.normal(0, 1, size_c)
    f1_x0_samples_c = sample_f1(size_c)
    f1_x1_samples_c = sample_f1(size_c)

    # Clear json files
    open('f0_x0_samples_b.json', 'w').close()
    open('f0_x1_samples_b.json', 'w').close()
    open('f1_x0_samples_b.json', 'w').close()
    open('f1_x1_samples_b.json', 'w').close()
    open('f0_x0_samples_c.json', 'w').close()
    open('f0_x1_samples_c.json', 'w').close()
    open('f1_x0_samples_c.json', 'w').close()
    open('f1_x1_samples_c.json', 'w').close()

    # Fill json files with samples
    with open('f0_x0_samples_b.json', 'a') as f: json.dump(f0_x0_samples_b.tolist(), f)
    with open('f0_x1_samples_b.json', 'a') as f: json.dump(f0_x1_samples_b.tolist(), f)
    with open('f1_x0_samples_b.json', 'a') as f: json.dump(f1_x0_samples_b.tolist(), f)
    with open('f1_x1_samples_b.json', 'a') as f: json.dump(f1_x1_samples_b.tolist(), f)
    with open('f0_x0_samples_c.json', 'a') as f: json.dump(f0_x0_samples_c.tolist(), f)
    with open('f0_x1_samples_c.json', 'a') as f: json.dump(f0_x1_samples_c.tolist(), f)
    with open('f1_x0_samples_c.json', 'a') as f: json.dump(f1_x0_samples_c.tolist(), f)
    with open('f1_x1_samples_c.json', 'a') as f: json.dump(f1_x1_samples_c.tolist(), f)