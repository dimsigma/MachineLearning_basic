from neural_net3 import NeuralNetwork, omega
import numpy as np
import json

n = 1
m = 100
k = 1
learning_rate = 0.01
epochs = 100

def compute_data_SGD_inf(nn1: NeuralNetwork, nn2: NeuralNetwork, X_dataset1: np.ndarray, X_dataset2: np.ndarray, Y_dataset1: np.ndarray, Y_dataset2: np.ndarray, DataModel_output1: np.ndarray, DataModel_output2: np.ndarray, learning_rate: float, epochs: int) -> list[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    convergance1 = np.array([])
    convergance2 = np.array([])
    output1 = np.array([])
    output2 = np.array([])

    for _ in range(epochs):
        for i in range(len(X_dataset1)):
            nn1.forward(X_dataset1[i])
            nn2.forward(X_dataset2[i])
            z1 = nn1.Z
            z2 = nn2.Z

            nn1.calc_cost(DataModel_output1[i])
            nn2.calc_cost(DataModel_output2[i])

            nn1.backward(X_dataset1[i])
            nn2.backward(X_dataset2[i])

            nn1.forward(Y_dataset1[i])
            nn2.forward(Y_dataset1[i])
            nn1.updateParams_sgd_inf(DataModel_output1[i], z1, nn2.Z, learning_rate)

            nn1.forward(Y_dataset2[i])
            nn2.forward(Y_dataset2[i])
            nn2.updateParams_sgd_inf(DataModel_output2[i], z2, nn1.Z, learning_rate)

        convergance1 = np.append(convergance1, np.average(nn1.cost))
        convergance2 = np.append(convergance2, np.average(nn2.cost))

    for i in range(len(X_dataset1)):
        nn1.forward(X_dataset1[i])
        nn2.forward(X_dataset2[i])

        output1 = np.append(output1, nn1.Z)
        output2 = np.append(output2, nn2.Z)

    return [convergance1, convergance2, omega(output1, nn1.method), omega(output2, nn2.method)]

# Load sample sets
with open("2_set1.json", 'r') as f: set1 = np.array(json.load(f))
with open("2_set2.json", 'r') as f: set2 = np.array(json.load(f))

# Create nns
nn1 = NeuralNetwork(n, m, k, "A1", SGD=True)
nn2 = NeuralNetwork(n, m, k, "C1_3", SGD=True)
nn3 = NeuralNetwork(n, m, k, "A1", SGD=True)
nn4 = NeuralNetwork(n, m, k, "C1_3", SGD=True)

output_A1 = compute_data_SGD_inf(nn1, nn3, set1[:, 0], set2[:, 0], set1[:, 1], set2[:, 1], set1[:, 2], set2[:, 2], learning_rate, epochs)
output_C1 = compute_data_SGD_inf(nn2, nn4, set1[:, 0], set2[:, 0], set1[:, 1], set2[:, 1], set1[:, 2], set2[:, 2], 5 * learning_rate, 2 * epochs)

# Store results
open("3_data_res_1A1.json", 'w').close()
open("3_data_res_1C1.json", 'w').close()
open("3_data_res_2A1.json", 'w').close()
open("3_data_res_2C1.json", 'w').close()

open("3_data_conv_1A1.json", 'w').close()
open("3_data_conv_1C1.json", 'w').close()
open("3_data_conv_2A1.json", 'w').close()
open("3_data_conv_2C1.json", 'w').close()

with open("3_data_res_1A1.json", 'w') as f: json.dump(output_A1[2].tolist(), f)
with open("3_data_res_1C1.json", 'w') as f: json.dump(output_C1[2].tolist(), f)
with open("3_data_res_2A1.json", 'w') as f: json.dump(output_A1[3].tolist(), f)
with open("3_data_res_2C1.json", 'w') as f: json.dump(output_C1[3].tolist(), f)

with open("3_data_conv_1A1.json", 'w') as f: json.dump(output_A1[0].tolist(), f)
with open("3_data_conv_1C1.json", 'w') as f: json.dump(output_C1[0].tolist(), f)
with open("3_data_conv_2A1.json", 'w') as f: json.dump(output_A1[1].tolist(), f)
with open("3_data_conv_2C1.json", 'w') as f: json.dump(output_C1[1].tolist(), f)
