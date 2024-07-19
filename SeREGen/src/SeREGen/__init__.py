import os
import pandas as pd
from tqdm import tqdm


os.environ["GEOMSTATS_BACKEND"] = "pytorch"
tqdm.pandas()


DNA = ["A", "C", "G", "T"]
RNA = ["A", "C", "G", "U"]
AminoAcids = list("ACDEFGHIKLMNPQRSTVWY")
