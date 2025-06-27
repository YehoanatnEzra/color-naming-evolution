# language_evolution/utils.py

"""
Utility functions for information-theoretic calculations based on Zaslavsky et al.
"""

import numpy as np


def calculate_complexity(encoder_q_w_m, cognitive_source_p_m):
    """
    Calculates the complexity of a lexicon, I(M;W).

    Args:
        encoder_q_w_m (np.ndarray): The language encoder q(w|m), shape (num_meanings, num_words).
        cognitive_source_p_m (np.ndarray): The cognitive source p(m), shape (num_meanings,).

    Returns:
        float: The complexity in bits.
    """
    # Ensure no division by zero or log(0)
    epsilon = 1e-10

    # p(m,w) = p(m) * q(w|m)
    p_m_w = cognitive_source_p_m[:, np.newaxis] * encoder_q_w_m

    # q(w) = sum over m of p(m,w)
    q_w = np.sum(p_m_w, axis=0)

    # I(M;W) = sum_{m,w} p(m,w) * log2( p(m,w) / (p(m)q(w)) )
    # This simplifies to: sum_{m,w} p(m,w) * log2( q(w|m) / q(w) )
    log_term = np.log2(encoder_q_w_m / (q_w + epsilon) + epsilon)

    complexity = np.sum(p_m_w * log_term)

    return complexity


def compute_I_MU(cognitive_source_p_m, meaning_space, sigma):
    """
    Computes I(M;U) for the environment (independent of the lexicon),
    as the expected squared distance from each meaning to the centroid,
    divided by (2 * sigma^2 * ln 2).
    """
    centroid = np.sum(cognitive_source_p_m[:, None] * meaning_space, axis=0)
    dist_sq = np.sum((meaning_space - centroid) ** 2, axis=1)
    expected_distortion = np.sum(cognitive_source_p_m * dist_sq)
    I_MU = expected_distortion / (2 * sigma**2 * np.log(2))
    return I_MU


def calculate_accuracy(encoder_q_w_m: np.ndarray,
                       cognitive_source_p_m: np.ndarray,
                       meaning_space: np.ndarray,
                       sigma: float,
                       I_MU: float = None) -> tuple:
    """
    Compute informativeness (accuracy) as defined in Zaslavsky et al.:
    I_q(W;U) = I(M;U) - expected distortion / (2 sigma^2 ln 2)
    Returns (informativeness, expected_distortion)
    """
    eps = 1e-12  # to avoid div-by-zero

    # p(m,w) = p(m) q(w|m)
    p_m_w = cognitive_source_p_m[:, None] * encoder_q_w_m
    # q(w) = Σ_m p(m,w)
    q_w = p_m_w.sum(axis=0)
    # p(m|w) = p(m,w) / q(w)
    p_m_given_w = p_m_w / (q_w + eps)
    # m̂_w  = Σ_m p(m|w) m   (reconstructed colour for each word)
    m_hat_w = p_m_given_w.T @ meaning_space                      # shape (K, D)
    # Squared Euclidean distance ‖m – m̂_w‖²  for every (m,w)
    dist_sq = ((meaning_space[:, None, :] - m_hat_w[None, :, :]) ** 2).sum(axis=2)
    # E[‖m – m̂_w‖²] = Σ_{m,w} p(m,w) ‖m – m̂_w‖²
    expected_distortion = (p_m_w * dist_sq).sum()
    # If I_MU is not provided, compute it
    if I_MU is None:
        I_MU = compute_I_MU(cognitive_source_p_m, meaning_space, sigma)
    informativeness = I_MU - (expected_distortion / (2 * sigma**2 * np.log(2)))
    return informativeness, expected_distortion


def calculate_fitness(encoder, cognitive_source, meaning_space, sigma, beta):
    """
    Calculates the fitness of a language using the IB objective function.
    fitness = beta * Accuracy - Complexity

    Args:
        encoder (np.ndarray or Language): The language encoder q(w|m), shape (num_meanings, lexicon_size).
        Can be a Language object or a raw encoder matrix.
        cognitive_source (np.ndarray): The cognitive source p(m).
        meaning_space (np.ndarray): The coordinates of meanings.
        sigma (float): The std deviation of Gaussian meanings.
        beta (float): The IB trade-off parameter.

    Returns:
        float: The fitness score for the language.
    """
    # If a Language object is passed, extract the encoder
    if hasattr(encoder, 'encoder'):
        encoder = encoder.encoder
    complexity = calculate_complexity(encoder, cognitive_source)
    accuracy_proxy = calculate_accuracy(encoder, cognitive_source, meaning_space, sigma)
    fitness = beta * accuracy_proxy[0] - complexity
    return fitness