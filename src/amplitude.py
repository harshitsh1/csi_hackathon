import numpy as np


def csi_to_amplitude(csi, fftshift=True):
    """Complex CSI (N,S) -> float amplitude (N,S). fftshift puts DC at the array centre so the
    subcarrier axis is in ascending-frequency order."""
    amp = np.abs(csi).astype(np.float64)
    return np.fft.fftshift(amp, axes=1) if fftshift else amp
