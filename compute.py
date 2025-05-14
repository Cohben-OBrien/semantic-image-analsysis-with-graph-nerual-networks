import numpy as np
import tensorflow as tf
from tqdm import tqdm
import keras


def compute_knn(feature_vector, batch_size, kn):
    neighbours = []
    num_batches = feature_vector.shape[0] // batch_size

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        current_batch = feature_vector[start_idx:end_idx]

        similarities = tf.linalg.matmul(current_batch, feature_vector, transpose_b=True)
        _, indices = keras.ops.top_k(similarities, k=kn + 1, sorted=True)

        neighbours.append(indices[..., 1:])

    return np.reshape(np.array(neighbours), (-1, kn))
