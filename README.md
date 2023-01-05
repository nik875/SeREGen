# SeqRep

**This tool is still in active development.** Install all requirements with `pip install -r requirements.txt`. Requirements file is not clean and may contain unnecessary or outdated packages.

SeqRep is a DNA dataset visualization and sequence representation tool available as a Python package. This is a computationally intensive analysis methodology, and best results are achieved when running on a modern computer with multiple CPU cores and a powerful GPU. The central idea is to train a machine learning model to convert individual DNA sequences into n-dimensional points, such that the distances between any two points in space is correlated with the true dissimilarity of those points' parent DNA sequences.

Currently, a preliminary library implementation of this methodology is offered, with goals of being easy-to-use, efficient, and highly extensible. Some knowledge of machine learning and TensorFlow is helpful, but not required to use this library. A copy of SILVA's 16S database is included, along with a basic Jupyter Notebook that highlights this tool's visualization applications. A full analysis pipeline is also planned for the future to make this methodology more accessible to less technical researchers.

Inside the SeqRep library are six modules:

- dataset_builder.py
    * DatasetBuilder
        + Imports FASTA data into a custom Dataset object and parses out taxonomic information from FASTA headers automatically. Header parsing is designed to be extensible to additional formats.
    * Dataset
        + Builds on top of the pandas DataFrame to allow easy importing of FASTA data, parsing out of taxonomic information, and dataset filtering.
        + Taxonomic information can be added after object creation if the source is something other than the FASTA headers.
        + Integrates with visualization module to simplify plot generation.
- output_mgmt.py
    * An internal utils module that helps manage output.
- comparative_encoder.py
    * ComparativeEncoder
        + Converts a TensorFlow encoder model into a comparative encoder model. Takes a Distance object as an argument.
        + Designed to be as generic and extensible as possible, and can function on any input shape and with any output size. Encoder model can either be built using included utilities or programmed from scratch and passed as an argument.
- encoders.py
    * ModelBuilder
        + Helpful class that can reduce the difficulty of designing an encoder model.
- distance.py
    * EuclideanDistance
        + Currently the only available distance metric, implements a simple euclidean distance measure between the inputs and normalizes that distance as a z-score.
        + Works best when the distribution of distances between randomly sampled points in the dataset is approximately normal (as in the SILVA dataset).
- visualize.py
    * repr_scatterplot
        + The most basic scatterplot function that wraps matplotlib and plots a scatterplot of sequence representations.
    * reprs_by_taxa
        + An incredibly useful function that can filter down input arrays with a boolean mask and plot all points in each value of a given taxonomic level, colored by taxonomic classification.

Many of these files have reasonable internal documentation, so it's worth looking at that for assistance as well.
