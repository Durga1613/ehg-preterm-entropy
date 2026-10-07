import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(
    signal,
    fs,
    lowcut=0.1,
    highcut=4.0,
    order=5
):
    """
    Apply a zero-phase Butterworth band-pass filter.
    """

    signal = np.asarray(signal, dtype=float)

    sos = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    filtered = sosfiltfilt(
        sos,
        signal
    )

    return filtered


def create_windows(
    signal,
    fs,
    window_seconds=120,
    overlap=0.5
):
    """
    Divide a signal into overlapping windows.
    """

    signal = np.asarray(signal)

    window_size = int(window_seconds * fs)

    step_size = int(
        window_size * (1 - overlap)
    )

    windows = []

    for start in range(
        0,
        len(signal) - window_size + 1,
        step_size
    ):
        end = start + window_size

        windows.append(
            signal[start:end]
        )

    return np.asarray(windows)