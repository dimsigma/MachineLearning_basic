import numpy as np
import json

def ReLU(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def der_ReLU(x: np.ndarray) -> int:
    return np.where(x > 0, 1, 0)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(x))

def der_sigmoid(x: np.ndarray) -> np.ndarray:
    return - sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    def __init__(self, A1: np.ndarray, A2: np.ndarray, B1: np.ndarray, B2: np.ndarray, in_size = 10,  hid_size = 128, out_size = 784):
        """A Neural Network consisting of three layers to be used as a generative model.\n
        Since this is an already trained generative model the parameters (weights, offsets) must be given.
        in_size, hid_size, out_size are the sizes of each of the three layers (n, m, k).\n
        It is advised that n, m, k remain at their default values since this is a generative model and the
        paramteters of the neural network which represent the generative function shouldn't change."""

        # Layer sizes
        self.in_size = in_size      # n
        self.hid_size = hid_size    # m
        self.out_size = out_size    # k

        # Weights and offsets
        self.A1 = A1    # m x n
        self.A2 = A2    # k x m
        self.B1 = B1    # m x 1
        self.B2 = B2    # k x 1

        # Power estimation and constants
        self.P_dZ = 0
        self.lamda = 0.000001 # 0.000001
        self.c = 0.00000001
        self.N = 0

        self.cost = np.array([])

    def forward(self, Z: np.ndarray):
        """Propagate forward through the neural network. This generative model uses as input a normal
        distribution with mean 0 and covariance 1.\n
        Uses a ReLU activation function for the hidden layer and a sigmoid activation
        function for the output layer.\n
        Returns the output layer of the network. Z must be an array of shape (n, 1)."""
        
        # W1 = A1 x Z + B1
        self.W1 = np.dot(self.A1, Z) + self.B1          # m x 1
        # Z1 = ReLU(W1)
        self.Z1 = ReLU(self.W1)                         # m x 1
        # W2 = A2 x Z1 + B2
        self.W2 = np.dot(self.A2, self.Z1) + self.B2    # k x 1
        # X = sig(W2)
        self.X = sigmoid(self.W2)                       # k x 1

    def T_matrix_create(self, problem: str):
        if problem == "2.2":
            self.T_mat = np.eye(self.N, self.out_size) # N x k
        elif problem == "2.3": # Note that this only works for 784 pixels to 49 pixels
            self.T_mat = np.array([])
            for i in range(7):
                for j in range(7):
                    self.T_mat = np.append(self.T_mat, np.zeros(112 * i))
                    for l in range(4):
                        self.T_mat = np.append(self.T_mat, np.zeros(4 * j))
                        self.T_mat = np.append(self.T_mat, [0.0625, 0.0625, 0.0625, 0.0625])
                        self.T_mat = np.append(self.T_mat, np.zeros(24 - 4 * j))
                    self.T_mat = np.append(self.T_mat, np.zeros(672 - 112 * i))
            self.T_mat = self.T_mat.reshape(49, 784)

    def calc_cost(self, Xn: np.ndarray, Z: np.ndarray):
        """Calculate the cost value of the output based on the correct value.
        Cost function is:\n
        J(Z) = N*log(||T*X - Xn||^2) + ||Z||^2"""

        self.TX_gen = np.dot(self.T_mat, self.X) # N x 1
        self.J = self.N * np.log(np.square(np.linalg.vector_norm(self.TX_gen - Xn))) + np.square(np.linalg.vector_norm(Z))
        self.cost = np.append(self.cost, self.J)

    def backward(self, Xn: np.ndarray, Z: np.ndarray):
        """Propagate backwards through the neural network calculating the gradient
        of the input vector for the cost function."""
        
        # u2 = 2T^T * (TX - Xn) / ||TX - Xn||^2
        u2 = 2 * np.dot(self.T_mat.T, (self.TX_gen - Xn)) / np.square(np.linalg.vector_norm(self.TX_gen - Xn)) # k x 1
        v2 = u2 * der_sigmoid(self.W2)      # k x 1
        u1 = np.dot(self.A2.T, v2)          # m x 1
        v1 = u1 * der_ReLU(self.W1)         # m x 1
        u0 = np.dot(self.A1.T, v1)          # n x 1
        self.dJ_dZ = self.N * u0 + 2 * Z    # n x 1

    def power_init(self):
        """Compute the power estimate of the Z gradient for the first iteration."""

        self.P_dZ = np.square(self.dJ_dZ)

    def power_get(self):
        """Compute the power estimate of the Z gradient after the first iteration."""

        self.P_dZ = (1 - self.lamda) * self.P_dZ + self.lamda * np.square(self.dJ_dZ)

    def updated_input(self, Z: np.ndarray, m: float) -> np.ndarray:
        """Return the updated input vector to minimize the cost function"""

        return Z - (m * self.dJ_dZ) / (self.c + np.sqrt(self.P_dZ))

    def prop_vec_bunch(self, Z: np.ndarray):
        """Apply the neural network as a vector tranformation G(Z) of a generative model
        to the vectors of list Z and present the results."""

        results = np.array([])

        for array in Z:
            self.forward(array)
            results = np.append(results, np.array(self.X))
        self.out = results.reshape(len(Z), self.out_size, 1)
    
    def image_proc(self, Xn: np.ndarray, N: int, reps: int, m: float, problem: str) -> np.ndarray:
        """Restore a noisy image that has either pixels missing or resolution lowered
        \nreps is the number of repetitions of the SGD algorithm.
        \nm is the SGD algorithm learning rate.
        \nproblem is the exercise problem number"""

        self.N = N
        self.T_matrix_create(problem)
        input_Z = np.random.normal(0, 1, self.in_size).reshape(self.in_size, 1)

        if problem == "2.2":
            Xn = Xn[:N]

        for i in range(reps - 1):
            self.forward(input_Z)
            self.calc_cost(Xn, input_Z)
            self.backward(Xn, input_Z)
            if i == 0: 
                self.power_init()
            else: 
                self.power_get()
            input_Z = self.updated_input(input_Z, m)

        self.forward(input_Z)
        self.calc_cost(Xn, input_Z)
        return self.X