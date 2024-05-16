import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf


# One of these should disable annoying warnings
logging.disable(logging.WARNING)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.FATAL)
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(1)


DNA = ['A', 'C', 'G', 'T']
RNA = ['A', 'C', 'G', 'U']
AminoAcids = list("ACDEFGHIKLMNPQRSTVWY")
