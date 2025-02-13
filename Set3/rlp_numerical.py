import numpy as np
import random
from scipy.stats import norm
import json
from matplotlib import pyplot as plt

N = 1100
N_actual = 1000
iterations = 100

def getf_matrix(y: np.ndarray, x: np.ndarray, N: int, a: int) -> np.ndarray:

    if a == 1:
        f_matrix = np.zeros(N**2).reshape(N, N)
        f_matrix[:, 0] = 0.5 * (norm.cdf(y[1] - 0.8 * x[:] - 1) - norm.cdf(y[0] - 0.8 * x[:] - 1))
        f_matrix[:, N-1] = 0.5 * (norm.cdf(y[N-1] - 0.8 * x[:] - 1) - norm.cdf(y[N-2] - 0.8 * x[:] - 1))
        for i in range(N):
            f_matrix[i, 1:N-2] = 0.5 * (norm.cdf(y[2:N-1] - 0.8 * x[i] - 1) - norm.cdf(y[0:N-3] - 0.8 * x[i] - 1))
        return f_matrix
    
    elif a == 2:
        f_matrix = np.zeros(N**2).reshape(N, N)
        f_matrix[:, 0] = 0.5 * (norm.cdf(y[1] + 2) - norm.cdf(y[0] + 2))
        f_matrix[:, N-1] = 0.5 * (norm.cdf(y[N-1] + 2) - norm.cdf(y[N-2] + 2))
        for i in range(N):
            f_matrix[i, 1:N-2] = 0.5 * (norm.cdf(y[2:N-1] + 2) - norm.cdf(y[0:N-3] + 2))
        return f_matrix

# Generate samples to use for numerical and training
a_samples = np.random.randint(1, 3, N)
W_samples = np.random.normal(0, 1, N)
S_samples = np.zeros(N + 1)
S_samples[0] = random.gauss(0, 1)

for i in range(N):
    S_samples[i+1] = np.where(a_samples[i] == 1,
                                0.8 * S_samples[i] + W_samples[i] + 1,
                                -2 + W_samples[i])

R_samples = np.minimum(2, np.square(S_samples))
R_sample_first = np.minimum(2, np.square(S_samples[:N]))
R_samples_short = np.minimum(2, np.square(S_samples[1:]))
R_samples_inf = np.minimum(2, np.square(S_samples[1:]))

# Compute using short sighted numerical method
set1_short = np.array([]).reshape(0, 3)
set2_short = np.array([]).reshape(0, 3)

for i in range(N):
    if a_samples[i] == 1:
        set1_short = np.vstack((set1_short, [S_samples[i], S_samples[i+1], R_samples[i+1]]))
    elif a_samples[i] == 2:
        set2_short = np.vstack((set2_short, [S_samples[i], S_samples[i+1], R_samples[i+1]]))

set1_short = set1_short[:500]
set2_short = set2_short[:500]

f1_matrix = getf_matrix(set1_short[:, 1], set1_short[:, 0], len(set1_short[:, 0]), 1)
f2_matrix = getf_matrix(set2_short[:, 1], set2_short[:, 0], len(set2_short[:, 0]), 2)

v1_vector_short = np.array(np.dot(f1_matrix, set1_short[:, 2].reshape(len(set1_short[:, 2]), 1)))
v2_vector_short = np.array(np.dot(f2_matrix, set2_short[:, 2].reshape(len(set2_short[:, 2]), 1)))

# Store samples
open("2_set1.json", 'w').close()
open("2_set2.json", 'w').close()
open("2_numerical1.json", 'w').close()
open("2_numerical2.json", 'w').close()

with open("2_set1.json", 'w') as f: json.dump(set1_short.tolist(), f)
with open("2_set2.json", 'w') as f: json.dump(set2_short.tolist(), f)
with open("2_numerical1.json", 'w') as f: json.dump(v1_vector_short.tolist(), f)
with open("2_numerical2.json", 'w') as f: json.dump(v2_vector_short.tolist(), f)

# Compute using infinte future reward numerical method
S1_possible = 0.8 * S_samples[:N_actual] + W_samples[:N_actual] + 1
S2_possible = -2 + W_samples[:N_actual]

f1_matrix_inf = getf_matrix(S1_possible, S_samples[:N_actual], N_actual, 1)
f2_matrix_inf = getf_matrix(S2_possible, S_samples[:N_actual], N_actual, 2)

v1_vector_inf = np.zeros(N_actual)
v2_vector_inf = np.zeros(N_actual)
convergance1 = np.array([])
convergance2 = np.array([])
for i in range(iterations):
    v1_vector_inf = np.array(np.dot(f1_matrix_inf, R_samples[:N_actual] + 0.8 * np.maximum(v1_vector_inf, v2_vector_inf)))
    v2_vector_inf = np.array(np.dot(f2_matrix_inf, R_samples[:N_actual] + 0.8 * np.maximum(v1_vector_inf, v2_vector_inf)))
    convergance1 = np.append(convergance1, np.max(v1_vector_inf))
    convergance2 = np.append(convergance2, np.max(v2_vector_inf))

# Store results
open("3_numerical_x.json", 'w').close()
open("3_numerical_v1.json", 'w').close()
open("3_numerical_v2.json", 'w').close()

with open("3_numerical_x.json", 'w') as f: json.dump(S_samples[:N_actual].tolist(), f)
with open("3_numerical_v1.json", 'w') as f: json.dump(v1_vector_inf.tolist(), f)
with open("3_numerical_v2.json", 'w') as f: json.dump(v2_vector_inf.tolist(), f)

# Plot convergance results
x = np.linspace(0, iterations, iterations)
plt.subplot(3, 1, 1)
plt.scatter(S_samples[:N_actual], v1_vector_inf, label="v1", color='b')
plt.scatter(S_samples[:N_actual], v2_vector_inf, label="v2", color='r')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(x, convergance1)
plt.title("Conv 1")

plt.subplot(3, 1, 3)
plt.plot(x, convergance2)
plt.title("Conv 2")

plt.tight_layout(pad=0, h_pad=0, w_pad=0)
plt.show()