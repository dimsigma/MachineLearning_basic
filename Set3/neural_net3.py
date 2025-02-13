import numpy as np

nn_method = enumerate(("A1", "A2", "C1_1", "C1_2", None))

def ReLU(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def der_ReLU(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)

def rho(z: np.ndarray, method: enumerate) -> np.ndarray:
    match method:
        case "A1":
            return -np.ones(len(z))
        case "A2":
            return -np.exp(-0.5 * np.abs(z))
        case "C1_1" | "C1_2" | "C1_3":
            return -np.exp(z) / (1 + np.exp(z))

def omega(z: np.ndarray, method: enumerate) -> np.ndarray:
    match method:
        case "A1":
            return z
        case "A2":
            return np.sinh(z)
        case "C1_1":
            return (np.exp(z) - 1) / (1 + np.exp(z))
        case "C1_2":
            return 2 * np.exp(z) / (1 + np.exp(z))
        case "C1_3":
            return 10 * np.exp(z) / (1 + np.exp(z))

def phi(z: np.ndarray, method: enumerate) -> np.ndarray:
    match method:
        case "A1":
            return 0.5 * np.square(z)
        case "A2":
            return np.exp(0.5 * np.abs(z)) + np.exp(-1.5 * np.abs(z)) / 3 - 4 / 3
        case "C1_1" | "C1_2":
            return 2 / (1 + np.exp(z)) + np.log(1 + np.exp(z))
        case "C1_3":
            return 10 / (1 + np.exp(z)) + 10 * np.log(1 + np.exp(z))

def psi(z: np.ndarray, method: enumerate) -> np.ndarray:
    match method:
        case "A1":
            return -z
        case "A2":
            return 2 * np.sign(z) * (np.exp(-0.5 * np.abs(z)) - 1)
        case "C1_1" | "C1_2" | "C1_3":
            return -np.log(1 + np.exp(z))

class NeuralNetwork:
    def __init__(self, in_size: int, hid_size: int, out_size: int, method: enumerate, SGD=False):
        self.n = in_size
        self.m = hid_size
        self.k = out_size
        self.method = method
        self.SGD = SGD

        # Weights and offsets
        self.A1 = np.random.normal(0, 1/(self.n + self.m), (self.n, self.m)) # n x m
        self.A2 = np.random.normal(0, 1/(self.m + self.k), (self.m, self.k)) # m x k
        self.B1 = np.zeros((1, self.m)) # 1 x m
        self.B2 = np.zeros((1, self.k)) # 1 x k

        if not self.SGD:
            self.sdA1 = np.array([])
            self.sdB1 = np.array([])
            self.sdA2 = np.array([])
            self.sdB2 = np.array([])
            self.Z_storage = np.array([])
        
        self.cost = np.array([])

    def forward(self, X: np.ndarray):
        # W1 = A1 x X + B1 -- 1 x m
        self.W1 = np.array(np.dot(X, self.A1) + self.B1)
        # Z1 = ReLU(W1) -- 1 x m
        self.Z1 = ReLU(self.W1)
        # Z = W2 = A2 x Z1 + B2 -- 1 x k
        self.Z = np.array(np.dot(self.Z1, self.A2) + self.B2)

        if not self.SGD: self.Z_storage = np.append(self.Z_storage, self.Z)
    
    def calc_cost(self, DataModel_output: float):
        g = DataModel_output
        z = self.Z
        self.J = phi(z, self.method) + g * psi(z, self.method)

        self.cost = np.append(self.cost, self.J)

    def backward(self, X: np.ndarray):
        u2 = 1
        v2 = u2
        u1 = v2 * self.A2.T
        v1 = u1 * der_ReLU(self.W1)

        self.dA2 = np.dot(self.Z1.T, v2)
        self.dB2 = v2
        self.dA1 = np.dot(X.T, v1)
        self.dB1 = v1

        if not self.SGD:
            self.sdA1 = np.append(self.sdA1, self.dA1)
            self.sdB1 = np.append(self.sdB1, self.dB1)
            self.sdA2 = np.append(self.sdA2, self.dA2)
            self.sdB2 = np.append(self.sdB2, self.dB2)
    
    def updateParams_gd(self, g_data: np.ndarray, learning_rate: float):
        z = self.Z_storage

        self.sdA1 = self.sdA1.reshape(len(z), self.n, self.m)
        self.sdB1 = self.sdB1.reshape(len(z), 1, self.m)
        self.sdA2 = self.sdA2.reshape(len(z), self.m, self.k)
        self.sdB2 = self.sdB2.reshape(len(z), 1, self.k)

        grad_delta = (g_data - omega(z, self.method)) * rho(z, self.method)
        gradA1 = np.zeros_like(self.sdA1)
        gradB1 = np.zeros_like(self.sdB1)
        gradA2 = np.zeros_like(self.sdA2)
        gradB2 = np.zeros_like(self.sdB2)

        gradA1 = grad_delta[:, np.newaxis, np.newaxis] * self.sdA1
        gradB1 = grad_delta[:, np.newaxis, np.newaxis] * self.sdB1
        gradA2 = grad_delta[:, np.newaxis, np.newaxis] * self.sdA2
        gradB2 = grad_delta[:, np.newaxis, np.newaxis] * self.sdB2

        self.A1 = self.A1 - learning_rate * np.average(gradA1, axis=0)
        self.B1 = self.B1 - learning_rate * np.average(gradB1, axis=0)
        self.A2 = self.A2 - learning_rate * np.average(gradA2, axis=0)
        self.B2 = self.B2 - learning_rate * np.average(gradB2, axis=0)

    def updateParams_sgd(self, reward: float, learning_rate: float):
        z = self.Z

        grad_delta = (reward - omega(z, self.method)) * rho(z, self.method)
        self.A1 = self.A1 - learning_rate * grad_delta * self.dA1
        self.B1 = self.B1 - learning_rate * grad_delta * self.dB1
        self.A2 = self.A2 - learning_rate * grad_delta * self.dA2
        self.B2 = self.B2 - learning_rate * grad_delta * self.dB2

    def updateParams_sgd_inf(self, reward: float, z_curr: float, z_foreign: float, learning_rate: float):
        z_next = self.Z

        grad_delta = (reward + 0.8 * np.max(omega(np.array([z_next, z_foreign]), self.method)) - omega(z_curr, self.method)) * rho(z_curr, self.method)
        self.A1 = self.A1 - learning_rate * grad_delta * self.dA1
        self.B1 = self.B1 - learning_rate * grad_delta * self.dB1
        self.A2 = self.A2 - learning_rate * grad_delta * self.dA2
        self.B2 = self.B2 - learning_rate * grad_delta * self.dB2

    def empty_arrays(self):
        self.sdA1 = np.array([])
        self.sdB1 = np.array([])
        self.sdA2 = np.array([])
        self.sdB2 = np.array([])
        self.Z_storage = np.array([])
        self.cost = np.array([])
    
    def cond_exp_compute_gd(self, X_dataset: np.ndarray, DataModel_output: np.ndarray, learning_rate: float, epochs: int) -> list[np.ndarray, np.ndarray]:
        convergance = np.array([])
    
        for _ in range(epochs):
            for j in range(len(X_dataset)):
                self.forward(X_dataset[j])
                self.calc_cost(DataModel_output[j])
                self.backward(X_dataset[j])
            
            self.updateParams_gd(DataModel_output, learning_rate)
            convergance = np.append(convergance, np.average(self.cost))
            self.empty_arrays()

        for i in range(len(X_dataset)):
            self.forward(X_dataset[i])

        return [convergance, omega(self.Z_storage, self.method)]
    
    def cond_exp_compute_sgd(self, X_dataset: np.ndarray, DataModel_output: np.ndarray, learning_rate: float, epochs: int) -> list[np.ndarray, np.ndarray]:
        output = np.array([])
        convergance = np.array([])

        for _ in range(epochs):
            for j in range(len(X_dataset)):
                self.forward(X_dataset[j])
                self.calc_cost(DataModel_output[j])
                self.backward(X_dataset[j])
                self.updateParams_sgd(DataModel_output[j], learning_rate)

            convergance = np.append(convergance, np.average(self.cost))
            self.empty_arrays()

        for i in range(len(X_dataset)):
            self.forward(X_dataset[i])
            output = np.append(output, self.Z)
        
        return [convergance, omega(output, self.method)]