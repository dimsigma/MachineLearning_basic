import numpy as np
from matplotlib import pyplot as plt
import json

# Load results from numerical method
with open("2_set1.json", 'r') as f: x1_samples = np.array(json.load(f))[:, 0]
with open("2_set2.json", 'r') as f: x2_samples = np.array(json.load(f))[:, 0]
with open("2_numerical1.json", 'r') as f: num_v1 = np.array(json.load(f))[:, 0]
with open("2_numerical2.json", 'r') as f: num_v2 = np.array(json.load(f))[:, 0]

# Load results from data method
with open("2_data_res_1A1.json", 'r') as f: data_res1A1 = np.array(json.load(f))
with open("2_data_res_1C1.json", 'r') as f: data_res1C1 = np.array(json.load(f))
with open("2_data_res_2A1.json", 'r') as f: data_res2A1 = np.array(json.load(f))
with open("2_data_res_2C1.json", 'r') as f: data_res2C1 = np.array(json.load(f))
with open("2_data_conv_1A1.json", 'r') as f: data_conv1A1 = np.array(json.load(f))
with open("2_data_conv_1C1.json", 'r') as f: data_conv1C1 = np.array(json.load(f))
with open("2_data_conv_2A1.json", 'r') as f: data_conv2A1 = np.array(json.load(f))
with open("2_data_conv_2C1.json", 'r') as f: data_conv2C1 = np.array(json.load(f))


# Plot the results
figure, axis = plt.subplots(4)

axis[0].scatter(x1_samples, num_v1, label="V1", color='b', s=2, linewidths=2)
axis[0].scatter(x2_samples, num_v2, label="V2", color='r', s=2, linewidths=2)
axis[0].set_title("Numerical approach")
axis[0].legend()

axis[1].scatter(x1_samples, data_res1A1, label="V1", color='b', s=2, linewidths=2)
axis[1].scatter(x2_samples, data_res2A1, label="V2", color='r', s=2, linewidths=2)
axis[1].set_title("Data approach (A1)")
axis[1].legend()

axis[2].scatter(x1_samples, data_res1C1, label="V1", color='b', s=2, linewidths=2)
axis[2].scatter(x2_samples, data_res2C1, label="V2", color='r', s=2, linewidths=2)
axis[2].set_title("Data approach (C1)")
axis[2].legend()

s = np.linspace(-6, 6, 1000)
axis[3].plot(s, np.minimum(2, np.square(s)), color="grey")
axis[3].set_title("R(S)")

plt.tight_layout(pad=0, h_pad=0, w_pad=0)
plt.show()

x1 = np.linspace(0, len(data_conv1A1), len(data_conv1A1))
x2 = np.linspace(0, len(data_conv1C1), len(data_conv1C1))
x3 = np.linspace(0, len(data_conv2A1), len(data_conv2A1))
x4 = np.linspace(0, len(data_conv2C1), len(data_conv2C1))
figure, axis = plt.subplots(2, 2)

axis[0, 0].plot(x1, data_conv1A1, label="V1 (A1)")
axis[0, 0].set_title("Convergance 1 (A1)")
axis[0, 1].plot(x2, data_conv1C1, label="V1 (C1)")
axis[0, 1].set_title("Convergance 1 (C1)")
axis[1, 0].plot(x3, data_conv2A1, label="V2 (A1)")
axis[1, 0].set_title("Convergance 2 (A1)")
axis[1, 1].plot(x4, data_conv2C1, label="V2 (C1)")
axis[1, 1].set_title("Convergance 2 (C1)")

plt.tight_layout(pad=0, h_pad=0, w_pad=0)
plt.show()
