import random

import numpy as np

import neuralnetwork.model.metrics as metrics


class LineSearch:

    # Using default c1 and c2 suggested by Numerical Optimization book
    def __init__(self, model, optimizer, w, X, Y, direction, phi0=None, deriv_phi0=None, c1=1e-4, c2=0.9, max_iter=10):
        self.optimizer = optimizer
        self.model = model

        self.w = self.vectorize_gradient(w)
        self.X = X
        self.Y = Y
        self.direction = direction
        self.last_gradient = None
        self.max_iter = max_iter

        self.c1 = c1
        self.c2 = c2

        if phi0 is None:
            self.phi0 = self.phi(0.)
        else:
            self.phi0 = phi0
        if deriv_phi0 is None:
            self.deriv_phi0 = self.deriv_phi(0.)
        else:
            self.deriv_phi0 = deriv_phi0

    def phi(self, alpha):

        val = self.w + alpha * self.direction
        w = [0] * self.model.n_layers
        start = 0
        for i in range(self.model.n_layers):
            n_rows = self.model.weights[i].shape[0]
            n_cols = self.model.weights[i].shape[1]
            end = n_rows * n_cols
            w[i] = val[start:start + end].reshape(n_rows, n_cols)
            start = start + end
            
        return metrics.regularized_MSE(self.Y, self.optimizer.forward(w, self.X), self.model, w)


    def deriv_phi(self, a):

        w = self.w + (a * self.direction)

        # convertiamo
        start = 0
        l_w = [0] * self.model.n_layers
        for i in range(self.model.n_layers):
            n_rows = self.model.weights[i].shape[0]
            n_cols = self.model.weights[i].shape[1]
            end = n_rows * n_cols
            l_w[i] = w[start:start + end].reshape(n_rows, n_cols)
            start = start + end

        gradient = self.vectorize_gradient(self.optimizer.backpropagation(self.model, l_w, self.X, self.Y))
        self.last_gradient = gradient

        return np.dot(gradient.T, self.direction)[0]
    
    def vectorize_gradient(self, gradient):
        vectorized = []

        for g in gradient:
            vectorized.append(g.flatten())

        vectorized = np.concatenate(vectorized).reshape(-1, 1)
        return vectorized


    def interpolation(self, alpha_high, alpha_low, phi_low, deriv_phi_low, phi_high, a_rec, phi_rec):

        # cubic
        alpha_j = self.cubicInterpolation(alpha_low, phi_low, deriv_phi_low, alpha_high, phi_high, phi_rec)

        # quadratic il risultato della cubic è fuori dal range (o se c'è stato un errore numerico nella cubic interp)
        if  alpha_j == 0 or alpha_j > alpha_high or alpha_j < alpha_low:
            alpha_j = self.quadraticInterpolation(phi_low, deriv_phi_low, alpha_high, phi_high)

        # bisection se c'è stato un errore numerico nella cubic interp
        if  alpha_j == 0 or alpha_j > alpha_high or alpha_j < alpha_low:
            alpha_j = (alpha_low + alpha_high)/ 2
        return alpha_j

    def zoom(self, alpha_low, alpha_high, phi_low, phi_high, deriv_phi_low):

        phi = self.phi0
        alpha = 0

        for i in range(10):  # non alziamolo troppo perchè impatta la velocità

            # Following Numerical Optimization book, first we interpolate to find the alpha
            alpha_j = self.interpolation(alpha_high, alpha_low, phi_low, deriv_phi_low, phi_high, alpha, phi)

            # Now that alpha_j is found, check conditions
            phi_alpha_j = self.phi(alpha_j)
            if (phi_alpha_j > self.phi0 + self.c1 * alpha_j * self.deriv_phi0) or (phi_alpha_j >= phi_low):
                phi = phi_high
                alpha = alpha_high
                alpha_high = alpha_j
                phi_high = phi_alpha_j

            else:
                deriv_phi_alpha_j = self.deriv_phi(alpha_j)

                if abs(deriv_phi_alpha_j) <= -self.c2 * self.deriv_phi0:
                    return alpha_j

                if (alpha_high - alpha_low) * deriv_phi_alpha_j >= 0:
                    phi = phi_high
                    alpha = alpha_high
                    alpha_high = alpha_low
                    phi_high = phi_low
                else:
                    phi = phi_low
                    alpha = alpha_low

                alpha_low = alpha_j
                phi_low = phi_alpha_j
                deriv_phi_low = deriv_phi_alpha_j

        return alpha_j

    def start(self):

        alpha0 = 0
        alpha1 = 1

        phi_alpha1 = self.phi(alpha1)
        phi_alpha0 = self.phi0
        deriv_phi_alpha0 = self.deriv_phi0

        """
        From the "Numerical Optimization" book:
             Therefore, the line search must include a stopping test if it cannot attain a lower function
            value after a certain number (typically, ten) of trial iteration_pass lengths. Some procedures also
            stop if the relative change in x is close to machine precision, or to some user-speciﬁed
            threshold

        Hence, we set it to 10
        """
        for i in range(self.max_iter):
            deriv_phi_alpha1 = self.deriv_phi(alpha1)

            # Zooming
            if (phi_alpha1 > self.phi0 + self.c1 * alpha1 * self.deriv_phi0):
                return self.zoom(alpha0, alpha1, phi_alpha0, phi_alpha1, deriv_phi_alpha0)
            if (abs(deriv_phi_alpha1) <= -self.c2 * self.deriv_phi0):
                return alpha1
            if (deriv_phi_alpha1 >= 0):
                return self.zoom(alpha1, alpha0, phi_alpha1, phi_alpha0, deriv_phi_alpha1)

            # Altrimenti troviamo un valore di alpha random tra l'attuale e 1, e procediamo
            alpha0 = alpha1
            alpha1 = random.uniform(alpha1, 1)
            phi_alpha0 = phi_alpha1
            phi_alpha1 = self.phi(alpha1)
            deriv_phi_alpha0 = deriv_phi_alpha1

        # Se siam qui le procedure di zooming non saranno andate a buon fine e restituiremo
        # l'ultimo alpha
        return alpha1

    # Following Numerical Optimization book
    def cubicInterpolation(self, alpha_low, phi_low, deriv_phi_low, alpha_high, phi_high, phi):
        np.seterr('raise')
        try:
            d1 = deriv_phi_low + phi - \
                 3 * (phi_low - phi_high) / (alpha_low - alpha_high)
            d2 = (1 if np.signbit(alpha_high - alpha_low) else -1) * np.sqrt(
                d1 ** 2 - deriv_phi_low * phi)
            toreturn = alpha_high - (alpha_high - alpha_low) * ((phi + d2 - d1) / (phi - deriv_phi_low + 2 * d2))
        except:
            toreturn = 0
        np.seterr('warn')
        return toreturn

    def quadraticInterpolation(self, phi_low, deriv_phi_low, alpha_high, phi_high):
        np.seterr('raise')
        try:
            toreturn =  -(deriv_phi_low * alpha_high ** 2) / (2 * (phi_high - phi_low - deriv_phi_low * alpha_high))
        except:
            toreturn = 0
        np.seterr('warn')
        return toreturn