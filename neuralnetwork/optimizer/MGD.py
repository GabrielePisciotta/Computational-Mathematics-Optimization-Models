import time as t

import numpy as np

from neuralnetwork.model import metrics
import datetime

class MGD():
    def __init__(self, loss="loss_mse", learning_rate=0.1, momentum=0, threshold_gradient_norm=None, threshold_loss=None):
        self.loss = loss
        self.iteration = 0
        self.norm_of_gradient_threshold = threshold_gradient_norm
        self.threshold_loss = threshold_loss
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.stop_because_of = "max it. reached"


    def stop_conditions(self, model, norm_of_gradient):
        #print("||g||: ", normalized_gradient, "\theta:", self.norm_of_gradient_threshold)
        #print("loss: ", model.loss[-1], "\theta: ", self.threshold_loss)
        # Check if the loss is below a chertain threshold (if it's been setted outside)
        if self.threshold_loss:
            if model.loss[-1] < self.threshold_loss:
                print("[STOP COND.] Loss {} < {}".format(model.loss[-1], self.threshold_loss))
                self.stop_because_of = "f threshold reached"
                return True

        # Check if the norm of the gradient is below a certain threshold (if it's been setted outside)
        if self.norm_of_gradient_threshold:
            if norm_of_gradient < self.norm_of_gradient_threshold:
                print("[STOP COND.] Norm of the gradient: {} < {}".format(norm_of_gradient, self.norm_of_gradient_threshold))
                self.stop_because_of = "||g|| threshold reached"
                return True
        return False

    def compute_metrics(self, model, X, Y, kind):
        out = model.predict(X)

        if kind == 'train':
            # Compute metrics and save it as model variables
            self.model.loss.append(metrics.regularized_MSE(Y, out, model, model.weights))
            try:
                self.model.accuracy.append(metrics.accuracy(Y, out))
            except:
                None
        else:
            # Compute metrics and save it as model variables
            self.model.loss_validation.append(metrics.regularized_MSE(Y, out, model, model.weights))
            try:
                self.model.accuracy_validation.append(metrics.accuracy(Y, out))
            except:
                None

    def learn(self, model, iterations, X_train, Y_train, X_valid=None, Y_valid=None):
        self.model = model
        self.direction = [0] * len(self.model.weights)
        self.total_time = datetime.datetime.now()

        for it in range(iterations):
            norm_of_gradient, time = self.iteration_pass(model, X_train, Y_train)
            self.model.norm_of_gradient.append(norm_of_gradient)

            self.compute_metrics(model,X_train, Y_train, kind="train")
            if X_valid is not None and Y_valid is not None:
                self.compute_metrics(model, X_valid, Y_valid, kind="valid")

            self.model.time.append(time)

            # If stop conditions are meet, stop
            if self.stop_conditions(model, norm_of_gradient):
                break
            self.iteration += 1

        self.total_time = (datetime.datetime.now() - self.total_time).total_seconds()

    def forward(self, weights, X):
        # copiamo i dati in input per evitare riscritture ed effetti collaterali sulla
        # variabile inizialmente passata. Essendo una procedura iterativa, inseriamo il
        # tutto dentro una variabile che verrà riscritta con l'output del layer che diventerà
        # l'input del layer successivo
        data = X.copy()
        for idx, layer_weights in enumerate(weights):
            net = np.dot(np.insert(data, 0, 1, 1), layer_weights)
            data = self.model.activations[idx].evaluate(net)
        return data

    def backpropagation(self, model, weights, X, Y):
        gradients = [0] * model.n_layers  # list of gradient for each layer (hidden to output)
        outputs_before_activ_f = [0] * model.n_layers  # outputs before the activation functions of all layers (hidden layers to output)
        outputs_with_activ_f = [0] * (model.n_layers + 1)  # outputs after the activation functions of all layers (input to output)
        outputs_with_activ_f[0] = np.insert(X, 0, 1, 1)

        # classic feed forward phase in which the input flows through all the layers
        # being non-linearly transformed. Each output of a layer flows to the next
        # as input. In the end, we will have the the output result
        for layer in range(0, model.n_layers):
            outputs_before_activ_f[layer] = np.dot(outputs_with_activ_f[layer], weights[layer])
            output = model.activations[layer].evaluate(outputs_before_activ_f[layer])
            outputs_with_activ_f[layer + 1] = np.insert(output, 0, 1, 1)
        Y_pred = outputs_with_activ_f[-1][:, 1:]

        # we divide the cases between the output layer and the hidden layer
        # in the output layer we have access to the desired response of the model
        # so we can compute the error and start accumulating deltas
        error_output = Y - Y_pred
        derivates = model.activations[-1].derivative(outputs_before_activ_f[-1])
        accumulated_gradient = -error_output * derivates
        gradients[model.n_layers - 1] = np.dot(outputs_with_activ_f[-2].T, accumulated_gradient)

        # following the backpropagation algorithm, we compute the hidden derivatives propagating back the
        # information needed for computing the same in the previous layer
        for layer in range(2, model.n_layers + 1):
            grad_weights = np.dot(accumulated_gradient, weights[-layer + 1].T)[:, 1:]
            # take into account here the fact that the activation function has been putted in reverse order so
            # we use the "reverse indexing" python functionality with -layer
            hidden_derivatives = model.activations[-layer].derivative(outputs_before_activ_f[-layer])
            accumulated_gradient = grad_weights * hidden_derivatives
            gradients[model.n_layers - layer] = np.dot(outputs_with_activ_f[-layer - 1].T, accumulated_gradient)

        for i, g in enumerate(gradients):
            gradients[i] = g + (2 * model.regulariz_lambda) * weights[i]

        return gradients

    def iteration_pass(self, model, X, Y):
        start_time = datetime.datetime.now()

        weights = model.weights
        gradient = self.backpropagation(model, weights, X, Y)
        norm_of_gradient = np.sqrt(np.sum([np.sum(np.square(g)) for g in gradient]))

        learning_rate = self.learning_rate / X.shape[0]

        for i, w in enumerate(model.weights):
            
            #self.direction[i] = -learning_rate*gradient[i] + self.momentum*self.direction[i]
            #w += self.direction[i] - w

            self.direction[i] = -gradient[i] + self.momentum * self.direction[i]
            w += learning_rate * self.direction[i]

        time = (datetime.datetime.now() - start_time).total_seconds()
        return norm_of_gradient, time
