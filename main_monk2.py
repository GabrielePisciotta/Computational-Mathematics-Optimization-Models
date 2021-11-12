import sys

sys.path.insert(0, "../../")
sys.path.insert(0, "./")
from os import makedirs
from neuralnetwork.model.neuralnetwork import NeuralNetwork
from neuralnetwork.optimizer import MGD, LBFGS
from neuralnetwork.datasets.dataset_utils import load_monk_dataset
from neuralnetwork.utils.utils import horizontal_plots_loss, print_results_to_table, plot_single_loss, \
    plot_convergence_rates, plot_single_convergence_rate, plot_losses
from neuralnetwork.model.metrics import evaluate_accuracy, evaluate_MSE_reg
import numpy as np


def main():
    np.random.seed(seed=123)
    ds = "results/Monk2" #<--- cambiamo qui così propaghiamo il dataset
    makedirs(ds, exist_ok=True)

    print("Loading datasets...")
    X_train, Y_train, X_test, Y_test = load_monk_dataset(2)

    print("Building LBFGS model ... ")

    model = NeuralNetwork(LBFGS(m=15, c1=1e-4, c2=0.9, threshold_gradient_norm=0.00001, threshold_loss=0.01),
                          regulariz_lambda=1e-5, loss_type="regularized_mse")
    model.add_layer(4, features = 17)
    model.add_layer(1)
    model.learn(X_train, Y_train, iterations=500)
    predicted_value = model.predict(X_test)

    print("Building Momentum Descent model ... ")
    model2 = NeuralNetwork(MGD(learning_rate=0.9, momentum=0.9, threshold_gradient_norm=0.00001, threshold_loss=0.01),
                           regulariz_lambda=1e-5,
                           loss_type="regularized_mse")
    model2.add_layer(4, features= 17)
    model2.add_layer(1)
    model2.learn(X_train, Y_train, iterations=500)
    predicted_value = model.predict(X_test)

    # Print plots
    plot_losses(model.loss, model2.loss,ds)
    horizontal_plots_loss(model.loss, model2.loss, ds, algo1="LBFGS", algo2="Momentum Descent")
    plot_single_loss(model.loss, ds, "LBFGS")
    plot_single_loss(model2.loss, ds, "Momentum Descent")
    plot_convergence_rates(model.loss, model2.loss,ds)
    plot_single_convergence_rate(model.loss, ds, "LBFGS")
    plot_single_convergence_rate(model2.loss, ds,"Momentum Descent")
    print_results_to_table(model, model2, ds)

if __name__ == '__main__':
    main()
