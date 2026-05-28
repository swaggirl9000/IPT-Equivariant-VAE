import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from torch import Tensor
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F

pv.set_jupyter_backend("static")

DEVICE = "cuda:0"
ECT_PLOT_CONFIG = {"cmap": "bone", "vmin": -0.5, "vmax": 1.5}
PC_PLOT_CONFIG = {"s": 5, "c": ".5"}
LIGHTRED = [255, 100, 100]

def plot_graph(x, edge_index, edge_weigths=None, ax=None):

    nodes = [i for i in range(len(x))]
    pos_dict = {i: p for i, p in zip(nodes, x)}

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_index)
    nx.draw_networkx_nodes(nodes, pos=pos_dict, node_size=100, ax=ax)
    for idx, edge in enumerate(edge_index):
        if edge_weigths is not None:
            if edge_weigths[idx] > 0.01:
                nx.draw_networkx_edges(
                    G,
                    pos_dict,
                    [edge],
                    alpha=edge_weigths[idx],
                    width=2,
                    edge_color="b",
                    ax=ax,
                )
        else:
            nx.draw_networkx_edges(
                G,
                pos_dict,
                [edge],
                width=2,
                edge_color="b",
                ax=ax,
            )
    nx.draw_networkx_labels(G, pos_dict, ax=ax)
    ax.set_aspect(1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    return ax


def plot_ect(ect_gt, ect_pred, num_ects=5, filename=None):

    fig, axes = plt.subplots(nrows=2, ncols=num_ects, figsize=(3 * num_ects, 6))
    for ax, gt, pred in zip(axes.T, ect_gt, ect_pred):

        ax[0].imshow(pred.cpu().detach().squeeze().numpy())
        ax[0].axis("off")

        ax[1].imshow(gt.cpu().squeeze().numpy())
        ax[1].axis("off")

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
        