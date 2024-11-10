import torch

EMBEDDING_SIZE = 128
DTYPE = torch.float64
DEVICE = 'cpu'
torch.manual_seed(2021)

import pickle
with open('edit_qiita_large_np.pkl', 'rb') as f:
    sequences, distances = pickle.load(f)
from SeREGen.encoders import ModelBuilder
builder = ModelBuilder((sequences['train'].shape[-1],))
builder.one_hot_encoding(4)

from SeREGen.comparative_encoder import ComparativeEncoder
from SeREGen.distance import EditDistance
dist = EditDistance(silence=True)
seregen_obj = ComparativeEncoder.from_model_builder(builder, repr_size=EMBEDDING_SIZE, embed_dist='hyperbolic',
                                            norm_type='l2', dist=dist, loss='mse', device=DEVICE, dtype=DTYPE)
print(seregen_obj.summary())
model = seregen_obj.model

import numpy as np
seqs = sequences['train']
distance_on = np.array(['A', 'C', 'G', 'T'])[seqs]
distance_on = np.fromiter((''.join(i) for i in distance_on), dtype=object).astype(str)
seregen_obj.fit(seqs, distance_on=distance_on, epoch_factor=4, epochs=20, clip_grad=False)