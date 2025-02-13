import numpy as np
from scipy.stats import norm
import json

N = 500
xmin = -100
xmax = 100

def getf_matrix(y: np.ndarray, x: np.ndarray, N: int) -> np.ndarray:
    f_matrix = np.zeros(N**2).reshape(N, N)
    f_matrix[:, 0] = 0.5 * (norm.cdf(y[1] - 0.8 * x[:]) - norm.cdf(y[0] - 0.8 * x[:]))
    f_matrix[:, N-1] = 0.5 * (norm.cdf(y[N-1] - 0.8 * x[:]) - norm.cdf(y[N-2] - 0.8 * x[:]))
    for i in range(N):
        f_matrix[i, 1:N-2] = 0.5 * (norm.cdf(y[2:N-1] - 0.8 * x[i]) - norm.cdf(y[0:N-3] - 0.8 * x[i]))
    return f_matrix

x_samples = np.linspace(xmin, xmax, N)
w_samples = np.random.normal(0, 1, N)
y_samples = 0.8 * x_samples + w_samples

g1_vector = y_samples.reshape(N, 1)
g2_vector = np.minimum(1, np.maximum(-1, y_samples)).reshape(N, 1)
f_matrix = getf_matrix(y_samples, x_samples, N)
v1_vector = np.array(np.dot(f_matrix, g1_vector)).reshape(N)
v2_vector = np.array(np.dot(f_matrix, g2_vector)).reshape(N)

open("1_numerical_x.json", 'w').close()
open("1_numerical_1.json", 'w').close()
open("1_numerical_2.json", 'w').close()

with open("1_numerical_x.json", 'w') as f: json.dump(x_samples.tolist(), f)
with open("1_numerical_1.json", 'w') as f: json.dump(v1_vector.tolist(), f)
with open("1_numerical_2.json", 'w') as f: json.dump(v2_vector.tolist(), f)