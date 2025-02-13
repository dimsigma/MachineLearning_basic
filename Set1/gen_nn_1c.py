from neural_net1 import NeuralNetwork
import numpy as np
import json
import time

# Get training samples
with open("f0_x0_samples_c.json", "r") as f: samples0x0 = np.array(json.load(f))
with open("f0_x1_samples_c.json", "r") as f: samples0x1 = np.array(json.load(f))
with open("f1_x0_samples_c.json", "r") as f: samples1x0 = np.array(json.load(f))
with open("f1_x1_samples_c.json", "r") as f: samples1x1 = np.array(json.load(f))

# Get testing samples
with open("f0_x0_samples_b.json", "r") as f: samples0x0test = np.array(json.load(f))
with open("f0_x1_samples_b.json", "r") as f: samples0x1test = np.array(json.load(f))
with open("f1_x0_samples_b.json", "r") as f: samples1x0test = np.array(json.load(f))
with open("f1_x1_samples_b.json", "r") as f: samples1x1test = np.array(json.load(f))

# Process training samples
samples_x0 = np.concatenate([samples0x0, samples1x0])
samples_x1 = np.concatenate([samples0x1, samples1x1])
outputs = np.concatenate([np.zeros(len(samples0x0)), np.ones(len(samples1x0))])

# Process testing samples
samples_H0_test = np.vstack([samples0x0test, samples0x1test])
samples_H1_test = np.vstack([samples1x0test, samples1x1test])

# Create neural network (2x20x1)
nn_cem = NeuralNetwork(2, 20, "CEM")
nn_exp = NeuralNetwork(2, 20, "EXP")
ep = 400

# Best m:
# CEM --> 0.0001, EXP --> 0.0001

t0 = time.time()
nn_cem.train1(samples_x0, samples_x1, outputs, m=0.0001, epochs=ep)
nn_cem.test1(samples_H0_test, samples_H1_test)
t1 = time.time()
print("Time (CEM): ", t1 - t0, " seconds")

t0 = time.time()
nn_exp.train1(samples_x0, samples_x1, outputs, m=0.0001, epochs=ep)
nn_exp.test1(samples_H0_test, samples_H1_test)
t1 = time.time()
print("Time (EXP): ", t1 - t0, " seconds")