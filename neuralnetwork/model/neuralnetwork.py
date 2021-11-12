import numpy as np

from ..activationfunctions.sigmoid import Sigmoid


class NeuralNetwork():

    def __init__(self, lbfgs_or_md, regulariz_lambda, loss_type):
        self.weights = []
        self.optimization_algorithm = lbfgs_or_md
        self.regulariz_lambda = regulariz_lambda  # lambda è già built-in variable
        self.activations = []
        self.n_layers = 0
        self.loss = []
        self.loss_validation = []
        self.accuracy = []
        self.accuracy_validation = []
        self.time = []
        self.loss_type = loss_type
        self.norm_of_gradient = []

    def learn(self, X_train, Y_train, iterations=0, X_valid=None, Y_valid=None):
        # Just a wrapper to pass the proper parameters to the real optimizer (lbfgs or momentum descent)
        self.optimization_algorithm.learn(model=self, iterations=iterations, X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid)

    def predict(self, inputs):
        data = inputs.copy()
        for layer in range(0, self.n_layers):
            data = self.activations[layer].evaluate(np.dot(np.insert(data, 0, 1, 1), self.weights[layer]))
        return data

    def add_layer(self, n_of_neurons, features=None, activation=Sigmoid()):
        dim_input_layer = 0
        if self.n_layers < 1:
            dim_input_layer = features + 1
        elif features is None:
            dim_input_layer = self.weights[-1].shape[1] + 1

        self.weights.append(np.random.uniform(-0.1, 0.1, size=(dim_input_layer, n_of_neurons)))
        self.activations.append(activation)
        self.n_layers += 1

    def get_weights(self):
        # Restituiamo una versione "flattened" dei pesi
        flattened = []

        for w in self.weights:
            flattened.append(w.flatten())

        flattened = np.concatenate(flattened).reshape(-1, 1)
        return flattened

    def set_weights(self, w):
        # Qui riportiamo i pesi passati come argomento che corrispondono ad una versione flattened per i calcoli
        # in forma prettamente matriciale come dovrebbe essere
        weights = [0] * self.n_layers

        pos = 0
        for i in range(self.n_layers):
            n_rows = self.weights[i].shape[0]
            n_cols = self.weights[i].shape[1]
            end = n_rows * n_cols
            weights[i] = w[pos:pos + end].reshape(n_rows, n_cols)
            pos = pos + end

        self.weights = weights
