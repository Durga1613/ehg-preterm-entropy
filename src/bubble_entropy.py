import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _swap_counts(signal, m):
    """
    Count the number of inversions/swaps needed to sort
    every embedded vector of length m.

    Bubble-sort swap count = inversion count.
    """

    signal = np.asarray(signal, dtype=float)

    vectors = sliding_window_view(
        signal,
        window_shape=m
    )

    counts = np.zeros(
        vectors.shape[0],
        dtype=np.int32
    )

    for i in range(m - 1):
        for j in range(i + 1, m):

            counts += (
                vectors[:, i] > vectors[:, j]
            )

    return counts


def _renyi_swap_entropy(signal, m):
    """
    Second-order Renyi entropy of the distribution
    of bubble-sort swap counts.
    """

    counts = _swap_counts(signal, m)

    max_swaps = m * (m - 1) // 2

    histogram = np.bincount(
        counts,
        minlength=max_swaps + 1
    )

    probabilities = (
        histogram / histogram.sum()
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.log(
        np.sum(probabilities ** 2)
    )

    return entropy


def bubble_entropy(signal, m):
    """
    Compute Bubble Entropy.

    Parameters
    ----------
    signal : array-like
        One-dimensional signal.

    m : int
        Embedding dimension.

    Returns
    -------
    float
        Bubble Entropy.
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    signal = signal[
        np.isfinite(signal)
    ]

    if m <= 1:
        raise ValueError(
            "m must be greater than 1"
        )

    if len(signal) <= m + 1:
        raise ValueError(
            "Signal is too short for selected m"
        )

    H_m = _renyi_swap_entropy(
        signal,
        m
    )

    H_m1 = _renyi_swap_entropy(
        signal,
        m + 1
    )

    normalization = np.log(
        (m + 1) / (m - 1)
    )

    return (
        H_m1 - H_m
    ) / normalization