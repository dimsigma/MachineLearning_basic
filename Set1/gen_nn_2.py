from neural_net1 import NeuralNetwork
from sklearn import datasets
import time


mnist = datasets.fetch_openml('mnist_784')
x_train = mnist.data[:60000]
x_test = mnist.data[60000:]
y_train = mnist.target.astype(int)[:60000]
y_test = mnist.target.astype(int)[60000:]

mask_train = (y_train == 0) | (y_train == 8)
train = x_train[mask_train]
vals_train = y_train[mask_train]

mask_test0 = (y_test == 0)
mask_test8 = (y_test == 8)
test0 = x_test[mask_test0]
test8 = x_test[mask_test8]
vals_test0 = y_test[mask_test0]
vals_test8 = y_test[mask_test8]

print("MNIST loaded")

# Create neural network (784x20x1)
nn_cem = NeuralNetwork(784, 300, "CEM")
nn_exp = NeuralNetwork(784, 300, "EXP")
ep = 1

# Best m:
# CEM --> 0.0001, EXP --> 0.0001

t0 = time.time()
print("Traing using CEM method...")
nn_cem.train2(train, vals_train, m=0.0001, epochs=ep)
print("Testing the neural network...")
nn_cem.test2(test0, test8, vals_test0, vals_test8)
t1 = time.time()
print("Time (CEM): ", t1 - t0, " seconds")

t0 = time.time()
print("Traing using EXP method...")
nn_exp.train2(train, vals_train, m=0.0001, epochs=ep)
print("Testing the neural network...")
nn_exp.test2(test0, test8, vals_test0, vals_test8)
t1 = time.time()
print("Time (EXP): ", t1 - t0, " seconds")