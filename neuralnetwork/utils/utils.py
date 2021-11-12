import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def horizontal_plots_loss(loss1, loss2, ds_path, algo1="", algo2=""):
    max_y = max([max(loss1), max(loss2)])
    max_y += max_y * 0.05

    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    alg1 = sns.lineplot(ax=ax[0], data=loss1)
    alg1.set_ylim([0, max_y])
    alg1.set(title=algo1, xlabel='Iteration', ylabel='Loss')

    alg2 = sns.lineplot(ax=ax[1], data=loss2)
    alg2.set_ylim([0, max_y])
    alg2.set(title=algo2, xlabel='Iteration', ylabel='Loss')

    plt.show()

    fig.savefig("{}/horizontal_losses.png".format(ds_path))


def plot_single_loss(loss, ds_path, algo):
    sns.set_theme(style="darkgrid")

    ax = sns.lineplot(data=loss, dashes=False, legend=False)
    ax.set(title=algo, xlabel='Iteration', ylabel='Loss')

    plt.show()
    ax.get_figure().savefig("{}/loss_{}.png".format( ds_path, algo))


def plot_losses(loss1, loss2, ds_path, algo1="", algo2=""):
    sns.set_theme(style="darkgrid")
    np.seterr(all='raise')

    ax = sns.lineplot(data=loss1, dashes=False)
    ax = sns.lineplot(data=loss2, dashes=False)
    ax.set(xlabel='Iteration', ylabel="Loss")

    plt.legend(loc='upper right', labels=['LBFGS', 'Momentum Descent'])
    plt.show()
    ax.get_figure().savefig("{}/losses_in_single_plot.png".format(ds_path))


def plot_convergence_rates(loss1, loss2, ds_path, algo1="", algo2=""):
    sns.set_theme(style="darkgrid")
    np.seterr(all='raise')
    rates1 = []
    d = np.abs(loss1 - loss1[-1])
    for i in range(len(d) - 1):
        try:
            rates1.append(np.log(d[i + 1]) / np.log(d[i]))
        except:
            rates1.append(1)
    rates2 = []
    d = np.abs(loss2 - loss2[-1])
    for i in range(len(d) - 1):
        try:
            rates2.append(np.log(d[i + 1]) / np.log(d[i]))
        except:
            rates2.append(1)
    ax = sns.lineplot(data=rates1, dashes=False)
    ax = sns.lineplot(data=rates2, dashes=False)
    ax.set(title="Convergence rate", xlabel='Iteration', ylabel="Convergence rate")

    plt.legend(loc='upper right', labels=['LBFGS', 'Momentum Descent'])
    plt.show()
    ax.get_figure().savefig("{}/convergence_rate.png".format(ds_path))

def plot_single_convergence_rate(loss, ds_path, algo="" ):
    sns.set_theme(style="darkgrid")
    np.seterr(all='raise')
    rates = []
    d = np.abs(loss - loss[-1])
    for i in range(len(d) - 1):
        try:
            rates.append(np.log(d[i + 1]) / np.log(d[i]))
        except:
            rates.append(1)

    ax = sns.lineplot(data=rates, dashes=False)
    ax.set(title="Convergence rate of {}".format(algo), xlabel='Iteration', ylabel="Convergence rate")
    plt.show()
    ax.get_figure().savefig("{}/convergence_rate_{}.png".format(ds_path, algo))


def print_results_to_table(model1, model2, ds_path):
    f_stars = [
        model1.loss[-1],
        model2.loss[-1]
    ]
    norms_of_gradient = [
        model1.norm_of_gradient[-1],
        model2.norm_of_gradient[-1]
    ]
    n_of_iterations = [
        model1.optimization_algorithm.iteration,
        model2.optimization_algorithm.iteration
    ]

    total_times = [
        model1.optimization_algorithm.total_time,
        model2.optimization_algorithm.total_time
    ]

    iteration_time_mean = [
        np.mean(model1.time),
        np.mean(model2.time)
    ]

    optimizers = [
        "L-BFGS",
        "MGD"
    ]

    stop_reasons = [
        model1.optimization_algorithm.stop_because_of,
        model2.optimization_algorithm.stop_because_of,
    ]
    df = pd.DataFrame({'opt':optimizers,
                       'f *':f_stars,
                       '||g_k||': norms_of_gradient,
                       '# iterations': n_of_iterations,
                       'total time (s)': total_times,
                       'mean time per it. (s)':iteration_time_mean,
                       'stop reason': stop_reasons})
    df = df.set_index('opt')

    print(df.to_markdown())

    df.to_csv('{}/results_comparison.csv'.format(ds_path))

    # Write also directly latex table to file
    with open('{}/results_comparison_latex.txt'.format(ds_path), 'w') as f:
        f.write(str(df.to_latex()))


def print_results_to_table_array_of_model(models, ds_path, optimizers):
    f_stars = []
    norms_of_gradient = []
    n_of_iterations = []
    total_times = []
    iteration_time_mean = []
    stop_reasons = []

    for m in models:
        f_stars.append(m.loss[-1])
        norms_of_gradient.append(m.norm_of_gradient[-1])
        n_of_iterations.append(m.optimization_algorithm.iteration)
        total_times.append(m.optimization_algorithm.total_time)
        iteration_time_mean.append(np.mean(m.time))
        stop_reasons.append(m.optimization_algorithm.stop_because_of)

    df = pd.DataFrame({'opt': optimizers,
                       'f *': f_stars,
                       '||g_k||': norms_of_gradient,
                       '# iterations': n_of_iterations,
                       'total time (s)': total_times,
                       'mean time per it. (s)': iteration_time_mean,
                       'stop reason': stop_reasons})
    df = df.set_index('opt')

    df.to_csv('{}/results_comparison.csv'.format(ds_path))
    # Write also directly latex table to file
    with open('{}/results_comparison_latex.txt'.format(ds_path), 'w') as f:
        f.write(str(df.to_latex()))

    df.sort_values(['f *', '# iterations'], ascending=[True, True], inplace=True )

    df.to_csv('{}/results_comparison_sorted.csv'.format(ds_path))
    # Write also directly latex table to file
    with open('{}/results_comparison_sorted_latex.txt'.format(ds_path), 'w') as f:
        f.write(str(df.to_latex()))
