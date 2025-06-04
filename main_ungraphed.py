#%%
from curses.ascii import isspace



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


strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

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
strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

encoder = create_encoder(representation_dims)
    
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



losses = [ClustersEntropyLoss(), ClustersEntropyLoss(entropy_loss_weight=5)]

inputs = {"anchor": x_data, 'neighbours': tf.gather(x_data, knns)}
labels = [np.ones(shape=(x_data.shape[0], kn)), np.ones(shape=(x_data.shape[0], kn))]



clustering_model = create_clustering_model(encoder, num_clusters, input_shape, name='clustering')
clustering_learner = create_clustering_learner(clustering_model, input_shape)

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
#%
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
feature_vectors = keras.utils.normalize(feature_vectors)

#%%
edge_list = []
for cluster in cluster_knns.values():
    for node, neighbors in cluster.items():
        for neighbor in neighbors:
            edge_list.append([node, neighbor])

edge_list = np.array(edge_list)
edge_list = tf.cast(edge_list, dtype=tf.int32)
#%%
hidden_units = 128
num_heads = 8
num_layers = 3
output_dim = 10

num_epochs = 100
batch_size = 256
validation_split = 0.1
learning_rate = 3e-1
momentum = 0.9



#%%
#%%
with strategy.scope():

    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.SGD(learning_rate, momentum=momentum)
    accuracy = keras.metrics.CategoricalAccuracy()
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=2e-3,
        patience=6,
        restore_best_weights=True
    )

    # %%
    feature_vectors = tf.convert_to_tensor(feature_vectors, dtype=tf.float32)
    edge_list = tf.convert_to_tensor(edge_list, dtype=tf.int32)
    model = GraphAttentionNetwork(
        node_states=feature_vectors,
        edges=edge_list,
        hidden_units=hidden_units,
        num_heads=num_heads,
        num_layers=num_layers,
        output_dim=output_dim,
    )
#%%
    model.compile(
        loss=loss_fn,
        optimizer=optimizer,
        metrics=[accuracy],
    )
#%%
    train_indices = np.arange(len(y_data))
    train_labels = y_data.flatten().astype(np.int32)
#%%
    train_labels = keras.utils.to_categorical(train_labels, num_classes=num_classes)
#%%
    train_indices = np.array(train_indices)
    train_labels = np.array(train_labels)
#%%
    train_indices = tf.convert_to_tensor(train_indices, dtype=tf.int32)
    train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
#%%

model.fit(
    x=train_indices,
    y=train_labels,
    batch_size=32,
    epochs=num_epochs,
    callbacks=[early_stopping],
    verbose=2,
)
#%%
model.save('completed_model.h5')