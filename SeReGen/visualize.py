"""
Visualization of sequence representations.
"""
import typing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .dataset_builder import Dataset


def repr_scatterplot(reprs: np.ndarray, title=None, alpha=.1, marker='.', figsize=(8, 6),
                     savepath=None):
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


def reprs_by_ds_label(reprs: np.ndarray, ds: Dataset, label: typing.Union[str, int], *args, **kwargs):
    """
    Scatterplot of representations colored by the values of a given label. Precondition: bad headers
    have been dropped. Needs args and kwargs for reprs_by_label.
    @param reprs: same as for reprs_by_label
    @param ds: dataset object with header data
    @param label: label to plot, passed as either an int index or string name
    @param *args: other necessary arguments for reprs_by_label
    @param **kwargs: optional kwargs for reprs_by_label
    """
    if mask is not None:
        reprs, ds = reprs[mask], ds.loc[mask]
    tax = np.stack(ds['labels'])

    label_idx = label if isinstance(label, int) else ds['labels'].index_of_label(label)
    if label_idx == -1:
        raise ValueError('Given label not in dataset!')
    labels = tax[:, label_idx]
    reprs_by_label(reprs, labels, *args, **kwargs)


def reprs_by_label(reprs: np.ndarray, lbls: np.ndarray, title: str,
                  alpha=.1, marker='.', filter=None, savepath=None, mask=None, **kwargs):
    """
    Scatterplot of representations colored based on an array of categorical labels. Lower-level
    function.
    @param reprs: sequence representations
    @param lbls: np.ndarray object with label data
    @param title: plot title
    @param alpha: alpha value for plotted points
    @param marker: marker symbol
    @param filter: minimum number of sequences for a category to be plotted
    @param savepath: path to save to
    @param mask: boolean mask to apply to all arrays
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    rng = np.random.default_rng()
    for i in (unique_labels := np.unique(lbls)):  # For each unique value
        pop = reprs[lbls == i]  # All labels matching it
        if filter and len(pop) > filter:  # If we are filtering and have a sufficient count
            # Randomly sample down to the filter size (balances the plot)
            samp = rng.choice(len(pop), size=filter, replace=False)
            x, y = zip(*pop[samp])  # Prepare x, y for plotting
        elif not filter:  # If we aren't filtering, just prepare to plot
            x, y = zip(*pop)
        else:
            continue
        # Plot x, y with all given plot arguments
        ax.scatter(x, y, alpha=alpha, marker=marker, **kwargs)

    # Other plot stuff
    ax.set_title(title)
    leg = plt.legend(unique_labels, markerscale=1, borderpad=1)
    for lh in leg.legendHandles:  # Set alpha of legend so that we can see it
        lh.set_alpha(1)
    if savepath:  # Save file if requested
        plt.savefig(savepath)
    plt.show()  # Show plot