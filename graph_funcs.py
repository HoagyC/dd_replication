import torch
from matplotlib import pyplot as plt

def plot_results(results_list, hidden_dim, datapoint_sizes):
    plot_size = (10, 10)
    x_range = (3, 100_000)
    fig, ax = plt.subplots(figsize=plot_size)
    for i, results in enumerate(results_list):
        ax.scatter(
            results.n_datapoints * torch.ones_like(results.hidden_capacities), results.hidden_capacities, color="r"
        )
        ax.scatter(
            results.n_datapoints * torch.ones_like(results.weight_capacities), results.weight_capacities, color="b"
        )

    # log scale, x_ticks as the datapoint sizes
    ax.set_xscale("log")
    ax.set_xticks(datapoint_sizes)
    ax.set_xticklabels(datapoint_sizes)
    ax.set_xlabel("Number of Datapoints")
    ax.set_ylabel("Capacity")

    # add a dotted line representing hidden_dim / n_datapoints
    ax.plot(x_range, [hidden_dim / x for x in x_range], "k--", alpha=0.5)
    
    plt.show()
