from neural_net2 import NeuralNetwork
import numpy as np
import json
from matplotlib import pyplot as plt

k = 784
pixels_hor = 28
pixels_ver = 28
z = np.array([])
image = np.array([])

# Load the generative model's neural network parameters
with open("A1.json", "r") as f: a1 = np.array(json.load(f))
with open("A2.json", "r") as f: a2 = np.array(json.load(f))
with open("B1.json", "r") as f: b1 = np.array(json.load(f))
with open("B2.json", "r") as f: b2 = np.array(json.load(f))

# Load the generative model's pdf samples
for i in range(100): z = np.append(z, np.random.normal(0, 1, 10))
z = z.reshape(100, 10, 1)

# Create the neural network and apply it to the samples
nn = NeuralNetwork(a1, a2, b1, b2)
nn.prop_vec_bunch(z)
array = nn.out.reshape(10, 10, k)

# Process the results to display images correctly (10 x 10)
for i in range(10):
    for j in range(int(k/pixels_hor)):
        for l in range(10):
            image = np.append(image, array[i, l, j*pixels_hor:(j+1)*pixels_hor])

# Display the images
image = image.reshape(10 * pixels_ver, 10 * pixels_hor).T
plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()