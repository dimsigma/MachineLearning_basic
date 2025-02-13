from neural_net3 import NeuralNetwork
import numpy as np
import json

N = 500
n = 1
m = 50
k = 1
learning_rate = 0.001 # 0.001
epochs = 100 #100

# Generate samples
x_samples = np.random.normal(0, 20, N)
w_samples = np.random.normal(0, 1, N)
y_samples = 0.8 * x_samples + w_samples
g1_samples = y_samples
g2_samples = np.minimum(1, np.maximum(-1, y_samples))

# Create nns
nn1_A1 = NeuralNetwork(n, m, k, "A1")
nn1_A2 = NeuralNetwork(n, m, k, "A2")
nn2_A1 = NeuralNetwork(n, m, k, "A1")
nn2_C1 = NeuralNetwork(n, m, k, "C1_1")

# Training
output1_1 = nn1_A1.cond_exp_compute_gd(x_samples, g1_samples, learning_rate, epochs)
output1_2 = nn1_A2.cond_exp_compute_gd(x_samples, g1_samples, learning_rate, epochs)
output2_1 = nn2_A1.cond_exp_compute_gd(x_samples, g2_samples, learning_rate * 10, epochs) # *10
output2_2 = nn2_C1.cond_exp_compute_gd(x_samples, g2_samples, learning_rate * 50, epochs * 2) # *50, *2
results = np.array([output1_1[1], output1_2[1], output2_1[1], output2_2[1]])

# Store results
open("1_data_x.json", 'w').close()
open("1_data_1A1.json", 'w').close()
open("1_data_1A2.json", 'w').close()
open("1_data_2A1.json", 'w').close()
open("1_data_2C1.json", 'w').close()
open("1_data_res.json", 'w').close()

with open("1_data_x.json", 'w') as f: json.dump(x_samples.tolist(), f)
with open("1_data_1A1.json", 'w') as f: json.dump(output1_1[0].tolist(), f)
with open("1_data_1A2.json", 'w') as f: json.dump(output1_2[0].tolist(), f)
with open("1_data_2A1.json", 'w') as f: json.dump(output2_1[0].tolist(), f)
with open("1_data_2C1.json", 'w') as f: json.dump(output2_2[0].tolist(), f)
with open("1_data_res.json", 'w') as f: json.dump(results.tolist(), f)