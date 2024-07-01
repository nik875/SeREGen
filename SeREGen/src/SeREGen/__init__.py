# pylint: disable=wrong-import-position
import os  # noqa
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'  # noqa


import comparative_encoder
import pipeline
import distance
import dataset_builder
import encoders
import kmers
import kmer_compression
import visualize


DNA = ['A', 'C', 'G', 'T']
RNA = ['A', 'C', 'G', 'U']
AminoAcids = list("ACDEFGHIKLMNPQRSTVWY")
