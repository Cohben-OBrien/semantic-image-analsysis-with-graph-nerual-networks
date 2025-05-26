#%%
from curses.ascii import isspace
%load_ext autoreload
%autoreload 2


import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import keras
import numpy as np
from graph import *
from encoder import create_encoder
import pandas as pd

from data_processing import processer
from semantic_clustering import *

from RepresentationLearner import RepresentationLearner
from compute import *
from GNN import *
import matplotlib.pyplot as plt
#%%


num_classes = 10
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

#%%
x_data = np.concatenate([x_train, x_test])
y_data = np.concatenate([y_train, y_test])

classes = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]
#%%
target_size = 32
representation_dims = 512
projection_units = 128
num_clusters = 28
kn = 5
tune_encoder_during_clustering = False
#%%
x_data = x_data.astype('float32')
processer.layers[-1].adapt(x_data)
#%%
encoder = create_encoder(
    representation_dims
)
#%%
representation_learner = RepresentationLearner(encoder, projection_units, num_augmentations=4)

lr_scheduler = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=500,
    alpha=0.1
)
optimizer = keras.optimizers.AdamW(learning_rate=lr_scheduler, weight_decay=0.0001)

representation_learner.compile(
    optimizer=optimizer,
    jit_compile=False,
)

history = representation_learner.fit(
    x=x_data,
    batch_size=512,
    epochs=50
)


#%%
batch_size = 500
feature_vectors = encoder.predict(x_data, batch_size=batch_size, verbose=1)
feature_vectors = keras.utils.normalize(feature_vectors)

#%%
knns = compute_knn(feature_vectors, batch_size=batch_size, kn=kn)

#%%
for layer in encoder.layers:
    layer.trainable = tune_encoder_during_clustering

clustering_model = create_clustering_model(encoder, num_clusters, input_shape, name='clustering')
clustering_learner = create_clustering_learner(clustering_model, input_shape)

losses = [ClustersEntropyLoss(), ClustersEntropyLoss(entropy_loss_weight=5)]

inputs = {"anchor": x_data, 'neighbours': tf.gather(x_data, knns)}
labels = [np.ones(shape=(x_data.shape[0], kn)), np.ones(shape=(x_data.shape[0], kn))]

clustering_learner.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=0.0001),
    loss=losses,
    jit_compile=False,
)

clustering_learner.fit(
    x=inputs,
    y=labels,
    batch_size=512,
    epochs=50
)


#%%
clustering_probabilitiys = clustering_model.predict(x_data, batch_size=batch_size)
cluster_assigment = keras.ops.argmax(clustering_probabilitiys, axis=-1).numpy()

cluster_confidence = keras.ops.max(clustering_probabilitiys, axis=-1).numpy()
#%%
from collections import defaultdict

clusters = defaultdict(list)

for idx, c in enumerate(cluster_assigment):
    clusters[c].append((idx, cluster_confidence[idx]))

non_empty_clusters = defaultdict(list)

for c in clusters.keys():
    if clusters[c]:
        non_empty_clusters[c] = clusters[c]

for c in range(num_clusters):
    print(f"Cluster {c}:  {len(clusters[c])}")
#%%
from sklearn.neighbors import NearestNeighbors

k = 5
cluster_knns = {}

for c, members in non_empty_clusters.items():
    indices = [idx for idx, _ in members]
    if len(indices) < k + 1:

        continue

    clutser_features = feature_vectors[indices]
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(clutser_features)

    distances, n_indices = nbrs.kneighbors(clutser_features)

    knns = [ [indices[n] for n in n[1:]] for n in n_indices]

    cluster_knns[c] = dict(zip(indices, knns))


#%%
from scipy.sparse import lil_matrix, issparse, csr_matrix

num_nodes = feature_vectors.shape[0]
adjencency = lil_matrix((num_nodes, num_nodes))

for cluster in cluster_knns.values():
    for node_idx, knn_lisy in cluster.items():
        for neighbor in knn_lisy:
            adjencency[node_idx, neighbor] = 1
            adjencency[neighbor, node_idx] = 2
#%%
from scipy.sparse import csr_matrix, issparse

def graph_augmented(features, adjencency):
    if not issparse(adjencency):
        adj_aug = csr_matrix(adjencency)
    else:
        adj_aug = adjencency.copy()

    mask = np.random.rand(*adj_aug.shape) < 0.1
    mask_sparse = csr_matrix(mask)

    adj_aug = adj_aug.multiply(mask_sparse)

    feat_aug = features.copy()
    mask_features = np.random.rand(*feat_aug.shape) < 0.1
    feat_aug[~mask_features] = 0

    return adj_aug, feat_aug
#%%
generator = GraphDataGenerator(feature_vectors, adjencency, augmented_fn=graph_augmented)
#%%
feature_dims = feature_vectors.shape[0]
proj_dims = 64
output_dims = 64
hidden_dims = 128

encoder = create_gnn_encoder(feature_dims, hidden_dims, proj_dims)
projector = create_projector(proj_dims, output_dims)

model = SimSiamGNN(encoder, projector)

#%%
model.compile(
    optimizer=keras.optimizers.Adam(1e-3)
)
#%%
model.fit(generator, epochs=50)
#%%
model.save('first.keras')