import numpy as np


def MSE(y_true, y_pred):
    delta = y_true - y_pred
    return np.mean(np.square(delta))

def regularized_MSE(y_true, y_pred, model, weights):
    # mse con la regolarizzazione: lambda ora se lo prende direttamente dal modello così
    # ci evitiamo di ripassarlo qua e la
    norma_pesi = np.sqrt(np.sum([np.sum(np.square(weights[i])) for i in range(0, len(weights))]))
    return MSE(y_true, y_pred) + model.regulariz_lambda * np.square(norma_pesi)

def accuracy(y_true, y_pred):
    correctly_classified = 0
    for y1, y2 in zip(y_true, y_pred):
        if y2 >= 0.5:
            label = 1
        else:
            label = 0

        if y1 >= 0.5:
            prediction = 1
        else:
            prediction = 0

        if (label == prediction):
            correctly_classified += 1

    return correctly_classified / len(y_true) * 100


## giusto per snellire un po' il codice altrove :)
def evaluate_MSE_reg(y_pred, y_true, model, weights):
    regmse = regularized_MSE(y_true, y_pred, model, weights)
    print("Regularized MSE: {} ".format(regmse))

def evaluate_accuracy(y_pred, y_true):
    acc_value = accuracy(y_pred, y_true)
    print("Accuracy: {} ".format(acc_value))
