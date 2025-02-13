import numpy as np
from sklearn import datasets

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def ReLU(X):
    return np.maximum(0, X)

class NeuralNetwork:
    def __init__(self, in_size: int, hid_size: int, method: str):
        self.in_size = in_size      #n
        self.hid_size = hid_size    #m
        self.out_size = 1           #k
        self.method = method
        self.cost = np.array([])
        self.converge = np.array([])
        self.error = np.array([])
        self.it = 0

        # Weights (A, B --> matrix) and offsets (a, b --> vector) of each layer
        self.A1 = np.matrix(np.random.normal(0, 1/(self.in_size + self.hid_size), (self.in_size, self.hid_size))) # n x m
        self.A2 = np.matrix(np.random.normal(0, 1/(self.hid_size + self.out_size), (self.hid_size, self.out_size))) # m x k
        self.B1 = np.matrix(np.zeros(self.hid_size)) # 1 x m
        self.B2 = np.matrix(np.array([0])) # 1 x k

        # Weight gradients
        self.dA1 = 0
        self.dA2 = 0
        self.dB1 = 0
        self.dB2 = 0

        # Weight gradient powers
        self.P_dA1 = 0
        self.P_dA2 = 0
        self.P_dB1 = 0
        self.P_dB2 = 0
        self.lamda = 0.000001
        self.c = 0.00000001

    # Propagate forward through the neural network
    def forward(self, X: np.ndarray) -> np.matrix:
        """Propagate forward through the neural network.\n
        Uses a ReLU activation function for the hidden layer and a sigmoid activation
        function for the output layer if self.method == "CEM".\n
        Returns the output layer of the network."""

        self.Z0 = np.matrix(X)
        #W1 = A1*Z0 + B1 --> Hidden layer output before non-linear function 1 x m
        self.W1 = np.dot(self.Z0, self.A1) + self.B1
        #Z1 = ReLU(W1) --> Hidden layer output after non-linear function 1 x m
        self.Z1 = ReLU(self.W1)
        #W2 = A2*Z1 + B2 --> Output layer output before non-linear function 1 x k
        self.W2 = np.dot(self.Z1, self.A2) + self.B2
        #Z2 = σ(W2) or W2 --> Output layer output after non-linear function 1 x k
        if self.method == "CEM":
            self.Z2 = sigmoid(self.W2)
        elif self.method == "EXP":
            self.Z2 = self.W2
        self.it += 1
        return self.Z2
    
    # Calculate the cost value
    def calc_cost(self, y: float):
        """Calculates the cost value of the output based on the correct value 
        and the method.\n
        For the CEM method φ(z) =  -log(1 - z) and ψ(z) =  -log(z)\n
        For the EXP method φ(z) = e^(0.5z) and ψ(z) = e^(-0.5z)"""

        if self.method == "CEM":
            if y == 0:
                # φ(z) =  -log(1 - z)
                c = - np.log10(1 - self.Z2[0, 0])
            else:
                # ψ(z) =  -log(z)
                c = - np.log10(self.Z2[0, 0]) 
        elif self.method == "EXP":
            if y == 0:
                # φ(z) = e^(0.5z)
                c = np.exp(0.5* self.Z2[0, 0])
            else:
                # ψ(z) = e^(-0.5z)
                c = np.exp(-0.5* self.Z2[0, 0])
        self.cost = np.append(self.cost, c)
    
    # Compute the gradients
    def backward(self, y: float):
        """Propagate backwards through the neueral network calculating the gradients
        of the weights and the offsets with respect to the cost function (L(z)).\n
        self.dW1 --> dL / dW2, self.dW2 --> dL / dW1, self.db1 --> dL / db1, 
        self.db1 --> dL / db2"""

        if self.method == "CEM":
            if y == 0:
                dZ2 = 1 / (np.log(10) * (1 - self.Z2[0, 0]))
            else:
                dZ2 = - 1 / (np.log(10) * self.Z2[0, 0])
            dW2 = np.matrix(dZ2 * sigmoid(self.W2) * (1 - sigmoid(self.W2)))
        elif self.method == "EXP":
            if y == 0:
                dZ2 = 0.5 * np.exp(0.5* self.Z2[0, 0])
            else:
                dZ2 = -0.5 * np.exp(-0.5 * self.Z2[0, 0])
            dW2 = np.matrix(dZ2)
        self.dA2 = np.dot(self.Z1.T, dW2)
        self.dB2 = dW2
        dZ1 = np.dot(dW2, self.A2.T)
        dW1 = np.multiply(dZ1, np.where(self.W1 > 0, 1, 0))
        self.dA1 = np.dot(self.Z0.T, dW1)
        self.dB1 = dW1

    # Compute the power estimate of each gradient for the first iteration
    def power_init(self):
        """Compute the power estimate of each gradient for the first iteration."""

        self.P_dA1 = np.square(self.dA1)
        self.P_dA2 = np.square(self.dA2)
        self.P_dB1 = np.square(self.dB1)
        self.P_dB2 = np.square(self.dB2)

    # Compute the power estimate of each gradient
    def power_get(self):
        """Compute the power estimate of each gradient after the first iteration."""

        self.P_dA1 = (1 - self.lamda) * self.P_dA1 + self.lamda * np.square(self.dA1)
        self.P_dA2 = (1 - self.lamda) * self.P_dA2 + self.lamda * np.square(self.dA2)
        self.P_dB1 = (1 - self.lamda) * self.P_dB1 + self.lamda * np.square(self.dB1)
        self.P_dB2 = (1 - self.lamda) * self.P_dB2 + self.lamda * np.square(self.dB2)

    # Update the weights and the offsets
    def update(self, m: float):
        """Update the weights and the offsets using the calculated gradients 
        and power estimations.\n
        m = learning rate"""

        self.A1 = self.A1 - np.divide(np.multiply(m, self.dA1), np.add(self.c, np.sqrt(self.P_dA1)))
        self.A2 = self.A2 - np.divide(np.multiply(m, self.dA2), np.add(self.c, np.sqrt(self.P_dA2)))
        self.B1 = self.B1 - np.divide(np.multiply(m, self.dB1), np.add(self.c, np.sqrt(self.P_dB1)))
        self.B2 = self.B2 - np.divide(np.multiply(m, self.dB2), np.add(self.c, np.sqrt(self.P_dB2)))

    # Train the neural network for problem 1
    def train1(self, samples0: np.ndarray, samples1: np.ndarray, outputs: np.ndarray, m: float, epochs=100):
        """Train the neural network for problem 1."""

        for ep in range(epochs):
            r = np.arange(len(outputs))
            np.random.shuffle(r)
            samples0 = samples0[r]
            samples1 = samples1[r]
            outputs = outputs[r]
            cost20 = 0
            counter = 0

            for sample in range(len(outputs)):
                X = np.array([samples0[sample], samples1[sample]])
                self.forward(X)
                self.calc_cost(outputs[sample])
                self.backward(outputs[sample])
                if ep == 0 and sample == 0:
                    self.power_init()
                else:
                    self.power_get()
                self.update(m)
                if counter == 19:
                    cost20 = np.mean(self.cost)
                    self.cost = np.array([])
                    counter = 0
                    self.converge = np.append(self.converge, cost20)
                counter += 1

    # Train the neural network for problem 2
    def train2(self, samples, vals, m, epochs):
        """Train the neural network for problem 2 --> MNIST dataset\n
        This time X is 784"""

        for ep in range(epochs):
            r = np.arange(len(samples))
            np.random.shuffle(r)
            samples_rand = samples.iloc[r]
            vals_rand = vals.iloc[r]
            cost20 = 0
            counter = 0

            for sample in range(len(samples_rand)):
                x = np.divide(np.array([samples_rand.iloc[sample]])[0], 255)
                if vals_rand.iloc[sample] == 8:
                    y = 1
                else:
                    y = 0
                self.forward(x)
                self.calc_cost(y)
                self.backward(y)
                if ep == 0 and sample == 0:
                    self.power_init()
                else:
                    self.power_get()
                self.update(m)
                if counter == 19:
                    cost20 = np.mean(self.cost)
                    self.cost = np.array([])
                    counter = 0
                    self.converge = np.append(self.converge, cost20)
                counter += 1

    # Apply the neural network to generated samples for problem 1
    def test1(self, samplesH0: np.ndarray, samplesH1: np.ndarray):
        """Apply the neural network to generated samples for problem 1 and print 
        the error percentages.\n
        samplesH0: 2xn array with n samples from hypothesis H0\n
        samplesH1: 2xn array with n samples from hypothesis H1"""

        errH0 = 0
        errH1 = 0

        if self.method == "CEM":
            for sample in range(len(samplesH0[0])):
                z = self.forward(np.array([samplesH0[0][sample], samplesH0[1][sample]]))
                if z > 0.5: errH0 += 1
            
            for sample in range(len(samplesH1[0])):
                z = self.forward(np.array([samplesH1[0][sample], samplesH1[1][sample]]))
                if z < 0.5: errH1 += 1
        
        elif self.method == "EXP":
            for sample in range(len(samplesH0[0])):
                z = self.forward(np.array([samplesH0[0][sample], samplesH0[1][sample]]))
                if z > 0: errH0 += 1
            
            for sample in range(len(samplesH1[0])):
                z = self.forward(np.array([samplesH1[0][sample], samplesH1[1][sample]]))
                if z < 0: errH1 += 1
        

        errorH0 = errH0 / len(samplesH0[0])
        errorH1 = errH1 / len(samplesH1[0])
        total_error = (errorH0 + errorH1) / 2

        print("H0 error: ", np.round(errorH0 * 100, 4), "%")
        print("H1 error: ", np.round(errorH1 * 100, 4), "%")
        if self.method == "CEM":
            print("Total error using the cross-entropy method: ", np.round(total_error * 100, 4), "%")
        elif self.method == "EXP":
            print("Total error using the exponential method: ", np.round(total_error * 100, 4), "%")

    def test2(self, samples0, samples8, vals0, vals8):
        """Apply the neural network to generated samples for problem 1 and print 
        the error percentages.\n
        samples: mnist database testing samples\n
        vals: mnist database corresponding values"""

        err0 = 0
        err8 = 0

        if self.method == "CEM":
            for sample in range(len(samples0)):
                x = np.divide(np.array([samples0.iloc[sample]])[0], 255)
                z = self.forward(x)
                if z > 0.5 and vals0.iloc[sample] == 0: err0 += 1
            
            for sample in range(len(samples8)):
                x = np.divide(np.array([samples8.iloc[sample]])[0], 255)
                z = self.forward(x)
                if z < 0.5 and vals8.iloc[sample] == 0: err8 += 1

        elif self.method == "EXP":
            for sample in range(len(samples0)):
                x = np.divide(np.array([samples0.iloc[sample]])[0], 255)
                z = self.forward(x)
                if z > 0 and vals0.iloc[sample] == 0: err0 += 1

            for sample in range(len(samples8)):
                x = np.divide(np.array([samples8.iloc[sample]])[0], 255)
                z = self.forward(x)
                if z < 0 and vals8.iloc[sample] == 8: err8 += 1
        
        error0 = err0 / len(samples0)
        error8 = err8 / len(samples8)
        total_error = (error0 * len(samples0) + error8 * len(samples8)) / (len(samples0) + len(samples8))

        print("Error rate for numeral 0: ", np.round(error0 * 100, 4), "%")
        print("Error rate for numeral 8: ", np.round(error8 * 100, 4), "%")
        if self.method == "CEM":
            print("Total error using the cross-entropy method: ", np.round(total_error * 100, 4), "%")
        elif self.method == "EXP":
            print("Total error using the exponential method: ", np.round(total_error * 100, 4), "%")