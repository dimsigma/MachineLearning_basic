from neural_net2 import NeuralNetwork
import numpy as np
import json
from matplotlib import pyplot as plt

k = 784
N = 300
m = 0.005
images = 4
pix_ver = 28
pix_hor = 28
iterations = 2500

# Load the generative model's neural network parameters
with open("A1.json", "r") as f: a1 = np.array(json.load(f))
with open("A2.json", "r") as f: a2 = np.array(json.load(f))
with open("B1.json", "r") as f: b1 = np.array(json.load(f))
with open("B2.json", "r") as f: b2 = np.array(json.load(f))

# Load the images
with open("Xi2.json", 'r') as f: X_ideal = np.array(json.load(f))
with open("Xn2.json", 'r') as f: X_noisy = np.array(json.load(f))

im_ideal = []
im_noisy = []
im_rec = []
cost_trend = []

# Create the original image, corrupted image and generated image from the neural network
for i in range(images):
    nn = NeuralNetwork(a1, a2, b1, b2)
    im_ideal.append(X_ideal[:, i].reshape(pix_ver, pix_hor).T)
    im_noisy.append(np.concatenate((X_noisy[:N, i], np.zeros(k - N))).reshape(pix_ver, pix_hor).T)
    im_rec.append(nn.image_proc(X_noisy[:, i].reshape(k, 1), N, iterations, m, "2.2").reshape(pix_ver, pix_hor).T)
    cost_trend.append(nn.cost)

# Display the images and the cost convergance
figure , axis = plt.subplots(images, 4)
for i in range(images):
    axis[i, 0].imshow(im_ideal[i], cmap="gray")
    axis[i, 1].imshow(im_noisy[i], cmap="gray")
    axis[i, 2].imshow(im_rec[i], cmap="gray")
    for j in range(3): axis[i, j].axis("off")
    axis[i, 3].plot(np.linspace(0, len(cost_trend[i]), len(cost_trend[i])), cost_trend[i])
plt.show()