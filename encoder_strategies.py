import numpy as np
from sklearn.cluster import KMeans


def random_encoder(num_meanings, lexicon_size, **kwargs):
    """
    Strategy 0: Completely random encoder. Each row is a random probability distribution over words.
    """
    encoder = np.random.rand(num_meanings, lexicon_size)
    row_sums = encoder.sum(axis=1, keepdims=True)
    encoder = encoder / row_sums
    return encoder


def clustered_encoder(num_meanings, lexicon_size, meaning_space=None, sharpness=100.0, random_state=None, **kwargs):
    """
    Strategy 1: Clustered encoder using k-means on meaning_space.
    - meaning_space: (num_meanings, D) array of meaning coordinates (required)
    - sharpness: higher = harder assignment (softmax temperature)
    - random_state: for reproducibility
    """
    if meaning_space is None:
        raise ValueError("meaning_space must be provided for clustered_encoder.")
    kmeans = KMeans(n_clusters=lexicon_size, n_init=10, random_state=random_state)
    cluster_labels = kmeans.fit_predict(meaning_space)
    cluster_centers = kmeans.cluster_centers_
    dists = np.linalg.norm(meaning_space[:, None, :] - cluster_centers[None, :, :], axis=2)  # (num_meanings, lexicon_size)
    logits = -sharpness * dists
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    encoder = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return encoder


STRATEGY_FUNCTIONS = {
    0: random_encoder,
    1: clustered_encoder,
}

AVAILABLE_STRATEGIES = list(STRATEGY_FUNCTIONS.keys())


def get_encoder_by_strategy(strategy, num_meanings, lexicon_size, **kwargs):
    if strategy in STRATEGY_FUNCTIONS:
        return STRATEGY_FUNCTIONS[strategy](num_meanings, lexicon_size, **kwargs)
    else:
        raise ValueError(f"Unknown encoder strategy: {strategy}") 