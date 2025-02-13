import numpy as np
from matplotlib import pyplot as plt
import json

# Load results from numerical method
with open("1_numerical_x.json", 'r') as f: num_x = np.array(json.load(f))
with open("1_numerical_1.json", 'r') as f: num_1 = np.array(json.load(f))
with open("1_numerical_2.json", 'r') as f: num_2 = np.array(json.load(f))

# Load results from data method
with open("1_data_x.json", 'r') as f: data_x = np.array(json.load(f))
with open("1_data_res.json", 'r') as f: data_approach = np.array(json.load(f))
data_1A1 = data_approach[0]
data_1A2 = data_approach[1]
data_2A1 = data_approach[2]
data_2C1 = data_approach[3]
with open("1_data_1A1.json", 'r') as f: data_conv_1A1 = np.array(json.load(f))
with open("1_data_1A2.json", 'r') as f: data_conv_1A2 = np.array(json.load(f))
with open("1_data_2A1.json", 'r') as f: data_conv_2A1 = np.array(json.load(f))
with open("1_data_2C1.json", 'r') as f: data_conv_2C1 = np.array(json.load(f))

# Present the slope results
slope_num1 = np.average(num_1 / num_x)
slope_num2 = np.average(num_2 / num_x)
print("\nNumerical approach:\nSlope 1 =", slope_num1)
slope_data1_A1 = np.average(data_1A1 / data_x)
slope_data1_A2 = np.average(data_1A2 / data_x)
slope_data2_A1 = np.average(data_2A1 / data_x)
slope_data2_C1 = np.average(data_2C1 / data_x)
print("\nData approach:\nSlope 1 (A1) =", slope_data1_A1, "\nSlope 1 (A2) =", slope_data1_A2)

# Plot the results
x1 = np.linspace(0, len(data_conv_1A1), len(data_conv_1A1))
x2 = np.linspace(0, len(data_conv_2C1), len(data_conv_2C1))
figure, axis = plt.subplots(3, 2)

axis[0, 0].plot(num_x, num_1)
axis[0, 0].set_title("G1 = Y - Numerical approach")

axis[0, 1].plot(num_x, num_2)
axis[0, 1].set_title("G2 = min(1, max(-1, Y)) - Numerical approach")

axis[1, 0].scatter(data_x, data_1A1, s=2, linewidths=2)
axis[1, 0].set_title("G1 - Data approach (A1)")

axis[2, 0].scatter(data_x, data_1A2, s=2, linewidths=2)
axis[2, 0].set_title("G1 - Data approach (A2)")

axis[1, 1].scatter(data_x, data_2A1, s=2, linewidths=2)
axis[1, 1].set_title("G2 - Data approach (A1)")

axis[2, 1].scatter(data_x, data_2C1, s=2, linewidths=2)
axis[2, 1].set_title("G2 - Data approach (C1)")

plt.tight_layout(pad=0, h_pad=0, w_pad=0)
plt.show()

figure, axis = plt.subplots(2, 2)

axis[0, 0].plot(x1, data_conv_1A1)
axis[0, 0].set_title("G1 - Data approach (A1) - convergance")

axis[1, 0].plot(x1, data_conv_1A2)
axis[1, 0].set_title("G1 - Data approach (A2) - convergance")

axis[0, 1].plot(x1, data_conv_2A1)
axis[0, 1].set_title("G2 - Data approach (A1) - convergance")

axis[1, 1].plot(x2, data_conv_2C1)
axis[1, 1].set_title("G2 - Data approach (C1) - convergance")

plt.tight_layout(pad=0, h_pad=0, w_pad=0)
plt.show()
