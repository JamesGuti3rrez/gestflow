# =============================================================
#  gestflow — training/augmentation.py
#  6 tecnicas de data augmentation sobre frames crudos
#  Entrada:  tensor (FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH, 3)
#  Salida:   lista de tensores aumentados incluyendo el original
# =============================================================

import os
import sys
import numpy as np
import cv2
from scipy import interpolate

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    FRAME_COUNT,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    AUGMENTATION_FACTOR,
    GAUSSIAN_SIGMA,
    SCALE_MIN,
    SCALE_MAX,
    TIME_WARP_MIN,
    TIME_WARP_MAX,
    BRIGHTNESS_MIN,
    BRIGHTNESS_MAX,
)


# -------------------------------------------------------------
#  Tecnica 1 — Original
# -------------------------------------------------------------

def augment_original(frames):
    return frames.copy()


# -------------------------------------------------------------
#  Tecnica 2 — Gaussian Jitter
#  Agrega ruido gaussiano sobre los pixeles del frame
# -------------------------------------------------------------

def augment_gaussian_jitter(frames):
    noise = np.random.normal(
        loc=0.0,
        scale=GAUSSIAN_SIGMA,
        size=frames.shape
    ).astype(np.float32)
    jittered = frames + noise
    return np.clip(jittered, 0.0, 1.0)


# -------------------------------------------------------------
#  Tecnica 3 — Spatial Scaling (zoom)
#  Escala la mano hacia adentro o afuera del centro del frame
# -------------------------------------------------------------

def augment_spatial_scaling(frames):
    scale = np.random.uniform(SCALE_MIN, SCALE_MAX)
    scaled_frames = np.zeros_like(frames)

    for i, frame in enumerate(frames):
        img = (frame * 255).astype(np.uint8)
        h, w = img.shape[:2]

        center_x, center_y = w // 2, h // 2
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(img, (new_w, new_h))

        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        x_start = max(0, center_x - new_w // 2)
        y_start = max(0, center_y - new_h // 2)
        x_end   = min(w, x_start + new_w)
        y_end   = min(h, y_start + new_h)

        rx_start = 0
        ry_start = 0
        rx_end   = x_end - x_start
        ry_end   = y_end - y_start

        canvas[y_start:y_end, x_start:x_end] = resized[ry_start:ry_end, rx_start:rx_end]
        scaled_frames[i] = canvas.astype(np.float32) / 255.0

    return scaled_frames


# -------------------------------------------------------------
#  Tecnica 4 — Horizontal Mirroring
#  Espeja todos los frames horizontalmente
# -------------------------------------------------------------

def augment_horizontal_mirroring(frames):
    mirrored = np.zeros_like(frames)
    for i, frame in enumerate(frames):
        img = (frame * 255).astype(np.uint8)
        flipped = cv2.flip(img, 1)
        mirrored[i] = flipped.astype(np.float32) / 255.0
    return mirrored


# -------------------------------------------------------------
#  Tecnica 5 — Time Warping
#  Remuestrea los frames en el eje temporal
# -------------------------------------------------------------

def augment_time_warping(frames):
    n_frames = frames.shape[0]
    warp_factor = np.random.uniform(TIME_WARP_MIN, TIME_WARP_MAX)

    original_indices = np.linspace(0, n_frames - 1, n_frames)
    warped_length    = max(2, int(n_frames * warp_factor))
    warped_indices   = np.linspace(0, n_frames - 1, warped_length)

    warped_frames = np.zeros((warped_length,) + frames.shape[1:], dtype=np.float32)
    for c in range(frames.shape[-1]):
        for h in range(frames.shape[1]):
            for w in range(frames.shape[2]):
                signal = frames[:, h, w, c]
                f      = interpolate.interp1d(original_indices, signal, kind="linear")
                warped_frames[:, h, w, c] = f(warped_indices)

    resampled = np.zeros_like(frames)
    for i in range(n_frames):
        src_idx = int(i * warped_length / n_frames)
        src_idx = min(src_idx, warped_length - 1)
        resampled[i] = warped_frames[src_idx]

    return resampled


# -------------------------------------------------------------
#  Tecnica 6 — Brightness Shift
#  Varia el brillo global del video
# -------------------------------------------------------------

def augment_brightness_shift(frames):
    factor = np.random.uniform(BRIGHTNESS_MIN, BRIGHTNESS_MAX)
    brightened = frames * factor
    return np.clip(brightened, 0.0, 1.0)


# -------------------------------------------------------------
#  Aplicador principal
# -------------------------------------------------------------

TECHNIQUES = [
    augment_original,
    augment_gaussian_jitter,
    augment_spatial_scaling,
    augment_horizontal_mirroring,
    augment_time_warping,
    augment_brightness_shift,
]


def apply_augmentation(frames):
    if len(TECHNIQUES) != AUGMENTATION_FACTOR:
        raise ValueError(
            f"AUGMENTATION_FACTOR ({AUGMENTATION_FACTOR}) debe coincidir "
            f"con el numero de tecnicas ({len(TECHNIQUES)})"
        )

    augmented = []
    for technique in TECHNIQUES:
        result = technique(frames)
        augmented.append(result.astype(np.float32))

    return augmented


# -------------------------------------------------------------
#  Aplicador sobre dataset completo
# -------------------------------------------------------------

def augment_dataset(X, y):
    X_aug = []
    y_aug = []

    for i in range(len(X)):
        augmented_samples = apply_augmentation(X[i])
        for sample in augmented_samples:
            X_aug.append(sample)
            y_aug.append(y[i])

    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug, dtype=np.int32)

    return X_aug, y_aug