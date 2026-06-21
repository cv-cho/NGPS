from __future__ import annotations

import numpy as np
import scipy.ndimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(ref: np.ndarray, pred: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(ref, pred, data_range=1.0))


def ssim(ref: np.ndarray, pred: np.ndarray) -> float:
    return float(structural_similarity(ref, pred, data_range=1.0))


def hfen(ref: np.ndarray, pred: np.ndarray, sigma: float = 1.5) -> float:
    ref_log = scipy.ndimage.gaussian_laplace(ref, sigma=sigma)
    pred_log = scipy.ndimage.gaussian_laplace(pred, sigma=sigma)
    return float(np.linalg.norm(ref_log - pred_log) / (np.linalg.norm(ref_log) + 1e-8))


def gmsd(ref: np.ndarray, pred: np.ndarray) -> float:
    hx = np.array([[1 / 3, 0, -1 / 3]] * 3)
    hy = hx.T
    ref_d = scipy.ndimage.uniform_filter(ref, size=2)[::2, ::2]
    pred_d = scipy.ndimage.uniform_filter(pred, size=2)[::2, ::2]
    grad_ref = np.sqrt(scipy.ndimage.convolve(ref_d, hx) ** 2 + scipy.ndimage.convolve(ref_d, hy) ** 2)
    grad_pred = np.sqrt(scipy.ndimage.convolve(pred_d, hx) ** 2 + scipy.ndimage.convolve(pred_d, hy) ** 2)
    c = 0.0026
    return float(np.std((2 * grad_ref * grad_pred + c) / (grad_ref**2 + grad_pred**2 + c)))


def fsim(ref: np.ndarray, pred: np.ndarray) -> float:
    """A compact feature-similarity proxy based on gradient magnitude agreement.

    This keeps evaluation lightweight for the release package. Projects that
    require exact FSIM variants can replace this function while keeping the same
    evaluation script interface.
    """

    gx_ref = scipy.ndimage.sobel(ref, axis=1)
    gy_ref = scipy.ndimage.sobel(ref, axis=0)
    gx_pred = scipy.ndimage.sobel(pred, axis=1)
    gy_pred = scipy.ndimage.sobel(pred, axis=0)
    mag_ref = np.sqrt(gx_ref**2 + gy_ref**2)
    mag_pred = np.sqrt(gx_pred**2 + gy_pred**2)
    c = 1e-4
    sim = (2 * mag_ref * mag_pred + c) / (mag_ref**2 + mag_pred**2 + c)
    weight = np.maximum(mag_ref, mag_pred)
    return float((sim * weight).sum() / (weight.sum() + 1e-8))
