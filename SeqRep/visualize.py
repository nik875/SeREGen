import typing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats, spatial
from .dataset_builder import Dataset


def repr_scatterplot(reprs: np.ndarray, title=None, alpha=.1, marker='o', figsize=(8, 6), savepath=None):
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


def reprs_by_label(reprs: np.ndarray, ds: Dataset, label: typing.Union[str, int], title: str,
                  alpha=.3, marker='o', filter=None, savepath=None, mask=None, **kwargs):
    """
    Scatterplot of representations colored by the values of a given label. Precondition: bad headers
    have been dropped.
    @param reprs: sequence representations
    @param ds: dataset object with header data
    @param label: label to plot, passed as either an int index or string name
    @param title: plot title
    @param alpha: alpha value for plotted points
    @param marker: marker symbol
    @param filter: minimum number of sequences for a category to be plotted
    @param savepath: path to save to
    @param mask: boolean mask to apply to all arrays
    """
    if mask is not None:
        reprs, ds = reprs[mask], ds.loc[mask]
    tax = np.stack(ds['labels'], axis=0)

    label_idx = label if isinstance(label, int) else ds['labels'].index_of_label(label)
    if label_idx == -1:
        raise ValueError('Given label not in dataset!')

    unique_taxa, counts = np.unique(tax[:, label_idx], return_counts=True)
    plottable_taxa = unique_taxa if not filter else unique_taxa[counts > filter]
    plottable = np.isin(tax[:, label_idx], plottable_taxa)
    to_plot = np.zeros((np.nonzero(plottable)[0].shape[0], plottable_taxa.shape[0]))
    for i in range(len(plottable_taxa)):
        to_plot[tax[plottable][:, label_idx] == plottable_taxa[i], i] = 1
    plottable_seqs = reprs[plottable]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    rng = np.random.default_rng()
    for i in to_plot.T:
        pop = plottable_seqs[i.astype(bool)]
        samp = rng.integers(0, len(pop), 1000)
        x, y = zip(*pop[samp])
        ax.scatter(x, y, alpha=alpha, marker=marker, **kwargs)
    ax.set_title(title)
    leg = plt.legend(plottable_taxa, markerscale=1, borderpad=1)

    for lh in leg.legendHandles:
        lh.set_alpha(1)
    if savepath:
        plt.savefig(savepath)
    plt.show()

