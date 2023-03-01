import typing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats, spatial
from .dataset_builder import Dataset


def repr_scatterplot(reprs: np.ndarray, title=None, alpha=1, marker='o', figsize=(8, 6), savepath=None):
    """
    Create a simple scatterplot of sequence representations.
    Suggested alpha for SILVA dataset representations: .05
    @param reprs: array of sequence representations
    @param title: plot title
    @param alpha: alpha value for plot points
    @param marker: marker icon
    @param figsize: size of figure to generate
    @param savepath: path to save plot to.
    """
    x, y = np.array(reprs).T
    plt.figure(figsize=figsize)
    plt.scatter(x, y, alpha=alpha, marker=marker)
    if title:
        plt.title(title)
    if savepath:
        plt.savefig(savepath)
    plt.show()


def comparative_scatter(data: list[tuple], title=None, marker='o', figsize=(8, 6),
                        sample_size=None, savepath=None):
    rng = np.random.default_rng()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    for group in data:
        ingroup = group[1][rng.integers(0, len(group[1]), sample_size)] if sample_size else data[1]
        outgroup = np.concatenate([i[1] for i in data if i[0] != group[0]])
        outgroup = outgroup[rng.integers(0, len(outgroup), sample_size)] if sample_size else outgroup
        distances_from_outgroup = spatial.distance_matrix(ingroup, outgroup)
        avg_from_outgroup = distances_from_outgroup.mean(axis=0)
        distances_from_ingroup = spatial.distance_matrix(ingroup, ingroup)
        avg_from_ingroup = distances_from_ingroup.mean(axis=0)

        outgroup_zscores = -1 * stats.zscore(avg_from_outgroup)
        outgroup_norm = outgroup_zscores / 3 + .5
        outgroup_norm = np.vectorize(lambda i: min(max(i, 0), 1))(outgroup_norm)

        ingroup_zscores = stats.zscore(avg_from_ingroup)
        ingroup_norm = ingroup_zscores / 3 + .5
        ingroup_norm = np.vectorize(lambda i: min(max(i, 0), 1))(ingroup_norm)

        alphas = (4 * outgroup_norm + ingroup_norm) / 5
        alphas = alphas / (sample_size ** (1 / 4))

        x = ingroup[:, 0]
        y = ingroup[:, 1]
        ax.scatter(x, y, alpha=alphas, marker=marker)

    if title:
        ax.set_title(title)
    leg = plt.legend([i[0] for i in data], markerscale=1, borderpad=1)
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    if savepath:
        plt.savefig(savepath)
    plt.show()
