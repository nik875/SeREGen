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

    
def reprs_by_taxa(reprs: np.ndarray, ds: Dataset, level: typing.Union[str, int], title: str,
                  alpha=.3, filter=None, savepath=None, mask=None):
    """
    Scatterplot of representations colored by taxa at a particular level. Precondition: bad headers
    have been dropped.
    @param reprs: sequence representations
    @param ds: dataset object with header data
    @param level: taxonomic level to plot down to, passed as either an int index or string name
    @param title: plot title
    @param alpha: alpha value for plotted points
    @param filter: minimum number of sequences for a category to be plotted
    @param savepath: path to save to
    @param mask: boolean mask to apply to all arrays
    """
    if mask is not None:
        reprs, ds = reprs[mask], ds.loc[mask]
    tax = np.stack(ds['tax'], axis=0)

    level_idx = level if isinstance(level, int) else ds['tax'].index_of_level(level)
    assert level_idx != -1

    unique_taxa, counts = np.unique(tax[:, level_idx], return_counts=True)
    plottable_taxa = unique_taxa if not filter else unique_taxa[counts > filter]
    plottable = np.isin(tax[:, level_idx], plottable_taxa)
    to_plot = np.zeros((np.nonzero(plottable)[0].shape[0], plottable_taxa.shape[0]))
    for i in range(len(plottable_taxa)):
        to_plot[tax[plottable][:, level_idx] == plottable_taxa[i], i] = 1
    plottable_seqs = reprs[plottable]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    rng = np.random.default_rng()
    for i in to_plot.T:
        pop = plottable_seqs[i.astype(bool)]
        samp = rng.integers(0, len(pop), 1000)
        x, y = zip(*pop[samp])
        ax.scatter(x, y, alpha=alpha, marker='o')
    ax.set_title(title)
    leg = plt.legend(plottable_taxa, markerscale=1, borderpad=1)

    for lh in leg.legendHandles:
        lh.set_alpha(1)
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
