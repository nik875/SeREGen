import typing
import matplotlib.pyplot as plt
import numpy as np
from .dataset_builder import Dataset


def repr_scatterplot(reprs: np.ndarray, title: str, savepath=None):
    """
    Create a simple scatterplot of sequence representations.
    @param reprs: array of sequence representations
    @param title: plot title
    @param savepath: path to save plot to.
    """
    x, y = np.array(reprs).T
    f = plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=.05, marker='o')
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
