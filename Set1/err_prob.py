import numpy as np
import json

from sample import size_b

factor = 1 / np.sqrt(2*np.pi)

# Calculate f0(x)
def f0(x):
    return factor * np.exp(-0.5 * x**2)

# Calculate f1(x)
def f1(x):
    return 0.5 * factor * (np.exp(-0.5 * (x + 1)**2) + np.exp(-0.5 * (x - 1)**2))

# Calculate f0(x1, x2)
def f_0(x1, x2):
    return f0(x1) * f0(x2)

# Calulate f1(x1, x2)
def f_1(x1, x2):
    return f1(x1) * f1(x2)

# Calculate error for dataset from f0
def h0_error(samples1, samples2, size):
    count = 0
    for i in range(size):
        if f_0(samples1[i], samples2[i]) < f_1(samples1[i], samples2[i]): count = count + 1
    return count / size

# Calculate error for dataset from f1
def h1_error(samples1, samples2, size):
    count = 0
    for i in range(size):
        if f_0(samples1[i], samples2[i]) > f_1(samples1[i], samples2[i]): count = count + 1
    return count / size

with open('f0_x0_samples_b.json', 'r') as f: f0_x0_samples = json.load(f)
with open('f0_x1_samples_b.json', 'r') as f: f0_x1_samples = json.load(f)
with open('f1_x0_samples_b.json', 'r') as f: f1_x0_samples = json.load(f)
with open('f1_x1_samples_b.json', 'r') as f: f1_x1_samples = json.load(f)

err_0 = h0_error(f0_x0_samples, f0_x1_samples, size_b)
err_1 = h1_error(f1_x0_samples, f1_x1_samples, size_b)
total_err = (err_0 + err_1) / 2

print("The percentage of wrong decisions for the dataset with PDF f0 is: ", np.round(err_0 * 100, 4), "%")
print("The percentage of wrong decisions for the dataset with PDF f1 is: ", np.round(err_1 * 100, 4), "%")
print("The total probability of error is: ", np.round(total_err * 100, 4), "%")