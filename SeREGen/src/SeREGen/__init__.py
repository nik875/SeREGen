import matplotlib.pyplot as plt
import tensorflow as tf


plt.style.use('dark_background')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


DNA = ['A', 'C', 'G', 'T']
RNA = ['A', 'C', 'G', 'U']
AminoAcids = list("ACDEFGHIKLMNPQRSTVWY")
