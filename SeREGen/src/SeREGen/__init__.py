# pylint: disable=wrong-import-position
import os  # noqa
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'  # noqa


DNA = ['A', 'C', 'G', 'T']
RNA = ['A', 'C', 'G', 'U']
AminoAcids = list("ACDEFGHIKLMNPQRSTVWY")
