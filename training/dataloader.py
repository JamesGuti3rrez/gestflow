# =============================================================
#  gestflow — training/dataloader.py
#  Carga videos de raw_videos/, extrae frames uniformes,
#  normaliza y devuelve tensores X e y listos para entrenar
# =============================================================

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATASET_DIR,
    TRAIN_CLASSES,
    NUM_CLASSES,
    FRAME_COUNT,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    TEST_SPLIT,
    VALIDATION_SPLIT,
)


# -------------------------------------------------------------
#  Extraccion de frames de un video
# -------------------------------------------------------------

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < FRAME_COUNT:
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 1, FRAME_COUNT, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            cap.release()
            return None

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) != FRAME_COUNT:
        return None

    return np.array(frames, dtype=np.float32)


# -------------------------------------------------------------
#  Verificacion del dataset
# -------------------------------------------------------------

def verify_dataset():
    errores = []

    if not os.path.exists(DATASET_DIR):
        errores.append(f"No se encontro el directorio del dataset: {DATASET_DIR}")
        return False, errores

    for gesto in TRAIN_CLASSES:
        gesto_dir = os.path.join(DATASET_DIR, gesto)
        if not os.path.exists(gesto_dir):
            errores.append(f"Falta la carpeta: {gesto}")
            continue

        videos = [
            f for f in os.listdir(gesto_dir)
            if f.lower().endswith(".mp4")
        ]

        if len(videos) == 0:
            errores.append(f"Sin videos en: {gesto}")

    if errores:
        return False, errores

    return True, []


# -------------------------------------------------------------
#  Carga del dataset completo
# -------------------------------------------------------------

def load_dataset():
    valido, errores = verify_dataset()

    if not valido:
        print("\nErrores en el dataset:")
        for error in errores:
            print(f"  {error}")
        sys.exit(1)

    X = []
    y = []
    descartados = 0

    print("\nCargando dataset...")

    for clase_idx, gesto in enumerate(TRAIN_CLASSES):
        gesto_dir = os.path.join(DATASET_DIR, gesto)
        videos    = sorted([
            f for f in os.listdir(gesto_dir)
            if f.lower().endswith(".mp4")
        ])

        print(f"\n  {gesto} ({len(videos)} videos)")

        for video_name in tqdm(videos, ncols=70, leave=False):
            video_path = os.path.join(gesto_dir, video_name)
            frames     = extract_frames(video_path)

            if frames is None:
                descartados += 1
                continue

            X.append(frames)
            y.append(clase_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"\nDataset cargado:")
    print(f"  Total de muestras  : {len(X)}")
    print(f"  Descartados        : {descartados}")
    print(f"  Shape X            : {X.shape}")
    print(f"  Clases             : {NUM_CLASSES}")

    return X, y


# -------------------------------------------------------------
#  Resumen del dataset por clase
# -------------------------------------------------------------

def dataset_summary():
    valido, errores = verify_dataset()

    if not valido:
        print("\nErrores encontrados:")
        for error in errores:
            print(f"  {error}")
        return

    print("\nResumen del dataset:")
    print(f"  {'Gesto':<20} {'Videos':>8}")
    print(f"  {'-'*20} {'-'*8}")

    total = 0
    for gesto in TRAIN_CLASSES:
        gesto_dir = os.path.join(DATASET_DIR, gesto)
        videos    = [
            f for f in os.listdir(gesto_dir)
            if f.lower().endswith(".mp4")
        ]
        count  = len(videos)
        total += count
        print(f"  {gesto:<20} {count:>8}")

    print(f"  {'-'*20} {'-'*8}")
    print(f"  {'TOTAL':<20} {total:>8}")


# -------------------------------------------------------------
#  Split estratificado del dataset
# -------------------------------------------------------------

def split_dataset(X, y):
    from sklearn.model_selection import train_test_split

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT,
        stratify=y,
        random_state=42
    )

    val_size_adjusted = VALIDATION_SPLIT / (1.0 - TEST_SPLIT)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size_adjusted,
        stratify=y_train_val,
        random_state=42
    )

    print(f"\nSplit del dataset:")
    print(f"  Entrenamiento : {len(X_train)} muestras")
    print(f"  Validacion    : {len(X_val)} muestras")
    print(f"  Test          : {len(X_test)} muestras")

    return X_train, X_val, X_test, y_train, y_val, y_test