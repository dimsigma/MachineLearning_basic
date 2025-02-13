from neural_net3 import NeuralNetwork
import numpy as np
import json

n = 1
m = 100
k = 1
learning_rate = 0.01
epochs = 100

# Load sample sets
with open("2_set1.json", 'r') as f: set1 = np.array(json.load(f))
with open("2_set2.json", 'r') as f: set2 = np.array(json.load(f))

# Create nns
nn1 = NeuralNetwork(n, m, k, "A1", SGD=True)
nn2 = NeuralNetwork(n, m, k, "C1_2", SGD=True)
nn3 = NeuralNetwork(n, m, k, "A1", SGD=True)
nn4 = NeuralNetwork(n, m, k, "C1_2", SGD=True)

# Training
output1 = nn1.cond_exp_compute_sgd(set1[:, 0], set1[:, 2], learning_rate, epochs)
output2 = nn2.cond_exp_compute_sgd(set1[:, 0], set1[:, 2], learning_rate, epochs)
output3 = nn3.cond_exp_compute_sgd(set2[:, 0], set2[:, 2], learning_rate, epochs)
output4 = nn4.cond_exp_compute_sgd(set2[:, 0], set2[:, 2], learning_rate, epochs)

# Store results
open("2_data_res_1A1.json", 'w').close()
open("2_data_res_1C1.json", 'w').close()
open("2_data_res_2A1.json", 'w').close()
open("2_data_res_2C1.json", 'w').close()
open("2_data_conv_1A1.json", 'w').close()
open("2_data_conv_1C1.json", 'w').close()
open("2_data_conv_2A1.json", 'w').close()
open("2_data_conv_2C1.json", 'w').close()

with open("2_data_res_1A1.json", 'w') as f: json.dump(output1[1].tolist(), f)
with open("2_data_res_1C1.json", 'w') as f: json.dump(output2[1].tolist(), f)
with open("2_data_res_2A1.json", 'w') as f: json.dump(output3[1].tolist(), f)
with open("2_data_res_2C1.json", 'w') as f: json.dump(output4[1].tolist(), f)
with open("2_data_conv_1A1.json", 'w') as f: json.dump(output1[0].tolist(), f)
with open("2_data_conv_1C1.json", 'w') as f: json.dump(output2[0].tolist(), f)
with open("2_data_conv_2A1.json", 'w') as f: json.dump(output3[0].tolist(), f)
with open("2_data_conv_2C1.json", 'w') as f: json.dump(output4[0].tolist(), f)