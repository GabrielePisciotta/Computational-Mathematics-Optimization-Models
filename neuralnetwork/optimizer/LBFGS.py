import copy
# import time <- ci da qualche problema quindi continuiamo con datetime
import datetime
from os import sys

import numpy as np

import neuralnetwork.model.metrics as metrics
from neuralnetwork.optimizer.linesearch import LineSearch

class LBFGS:

    def __init__(self, m=3, c1=1e-4, c2=.9, ls_maxiter=10, threshold_gradient_norm=None, threshold_loss=None):
        self.iteration = 0
        self.norm_of_gradient_threshold = threshold_gradient_norm
        self.threshold_loss = threshold_loss

        self.c1 = c1
        self.c2 = c2

        self.m = m # number of curvatures to be accounted

        self.LS_maxiter = ls_maxiter

        self.stop_because_of = "max it. reached"
        self.gradient = None
        self.previous_phi0 = None

        # S and Y related to the LBFGS algorithm
        self.s = []
        self.y = []



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
        self.dataset_size = X_train.shape[0]
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
            gradients[i] = (2 / X.shape[0]) * g + (2 * model.regulariz_lambda) * weights[i]

        return gradients

    def vectorize_gradient(self, gradient):
        vectorized = []

        for g in gradient:
            vectorized.append(g.flatten())

        vectorized = np.concatenate(vectorized).reshape(-1, 1)
        return vectorized

    def iteration_pass(self, model, X, Y):
        start_time = datetime.datetime.now()

        # All'inizio non abbiamo abbastanza info, quindi è analogo al gradient descent
        if self.iteration == 0:
            self.gradient = self.vectorize_gradient(self.backpropagation(self.model, self.model.weights, X, Y))

        # Copiamo il gradiente per evitare effetti collaterali sulla variabile iniziale (usiamo la notazione del libro, con "q")
        q = copy.deepcopy(self.gradient)

        # Computazione direzione
        direction = -self.compute_direction(q)

        # Alla prima iterazione, phi0 dobbiamo calcolarlo qui
        if self.iteration == 0:
            phi0 = metrics.regularized_MSE(Y, model.predict(X), model, model.weights)
        # Altrimenti possiamo recuperarlo
        else:
            phi0 = model.loss[-1]

            # Check dell secant equation seguendo pag 137 di Numerical Optimization book
            if np.dot(self.s[-1].T, self.y[-1]) < 0:
                print("curvature condition: {}".format(np.dot(self.s[-1].T, self.y[-1])))
                sys.exit(1)

        normalized_gradient = np.linalg.norm(self.gradient)

        w = model.get_weights()  # appiattisco pesi

        LS = LineSearch(model, self, w, X, Y, direction, phi0, c1=self.c1, c2=self.c2)
        alpha = LS.start()

        self.previous_phi0 = phi0
        gradient_new = LS.last_gradient

        # delta_w = alpha*direction
        stepped_direction = alpha * direction
        # aggiorno i pesi
        w += stepped_direction
        # li settiamo nel modello
        self.model.set_weights(w)

        # Aggiungiamo
        if np.dot(stepped_direction.T, (gradient_new-self.gradient)) > np.finfo(np.float64).eps:
            self.s.append(stepped_direction)
            self.y.append(gradient_new - self.gradient)

        # Eliminiamo info delle curvature in eccesso
        if len(self.s) > self.m:
            self.s.pop(0)
        if len(self.y) > self.m:
            self.y.pop(0)

        self.gradient = gradient_new
        end_time = (datetime.datetime.now() - start_time).total_seconds()
        return normalized_gradient, end_time

    def compute_direction(self, q):
        # This method returns the direction -r = - H_{k} ∇ f_{k}

        # This condition is to skip the computation of the complete direction (due to the fact that we don't have sufficient
        # information from the past curvatures)
        if self.iteration > 0 and len(self.s) > 0 and len(self.y) > 0:

            # Lavoriamo su delle copie
            s = copy.deepcopy(self.s)
            y = copy.deepcopy(self.y)

            # Gamma equation from Numerical Optimization book
            H_0 = np.dot(s[-1].T, y[-1]) / np.dot(y[-1].T, y[-1])

            # There's no need to recompute it in the 2nd cycle, saved here for reuse
            alphas = []
            ros = []

            # Following the Numerical Optimization book, this is the first cycle from latest curvature to the first
            for s_k, y_k in zip(reversed(s), reversed(y)):
                ros.append(1 / (np.dot(y_k.T, s_k)))
                alphas.append(ros[-1] * np.dot(s_k.T, q))
                q -= alphas[-1] * y_k

            r = H_0 * q

            # This is the second cycle from the first curvature to the last
            # we have previously computed the alphas in reversed order, so we need to reverse it
            # the last element is to get the alphas and ros from the arrays saved in the previous cycle
            # we need to use it in reverse order
            for s_k, y_k, counter in zip(s, y, reversed(range(len(alphas)))):
                a_i = alphas[counter]
                ro_i = ros[counter]
                b = ro_i * np.dot(y_k.T, r)
                r += s_k * (a_i - b)
            return r
        else:
            return q
