# =============================================================
#  gestflow — training/callbacks.py
#  EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
#  y registros de entrenamiento por fold
# =============================================================

import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL_PATH,
    MODEL_DIR,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    REDUCE_LR_MIN,
    CHECKPOINT_MONITOR,
)


# -------------------------------------------------------------
#  EarlyStopping
#  Detiene el entrenamiento si val_loss no mejora
#  en EARLY_STOPPING_PATIENCE epocas consecutivas
#  Restaura los mejores pesos automaticamente
# -------------------------------------------------------------

def get_early_stopping():
    return EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
        mode="min",
    )


# -------------------------------------------------------------
#  ReduceLROnPlateau
#  Reduce el learning rate cuando val_loss se estanca
#  Factor y paciencia definidos en config.py
# -------------------------------------------------------------

def get_reduce_lr():
    return ReduceLROnPlateau(
        monitor="val_loss",
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=REDUCE_LR_MIN,
        verbose=1,
        mode="min",
    )


# -------------------------------------------------------------
#  ModelCheckpoint
#  Guarda el mejor modelo segun val_accuracy
#  Solo guarda cuando hay mejora real
# -------------------------------------------------------------

def get_model_checkpoint(fold=None):
    os.makedirs(MODEL_DIR, exist_ok=True)

    if fold is not None:
        filename = f"gestflow_fold_{fold:02d}.keras"
    else:
        filename = "gesture_model.keras"

    path = os.path.join(MODEL_DIR, filename)

    return ModelCheckpoint(
        filepath=path,
        monitor=CHECKPOINT_MONITOR,
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        mode="max",
    )


# -------------------------------------------------------------
#  CSVLogger
#  Guarda el historial de entrenamiento por fold
#  en un archivo CSV para analisis posterior
# -------------------------------------------------------------

def get_csv_logger(fold=None):
    os.makedirs(MODEL_DIR, exist_ok=True)

    if fold is not None:
        filename = f"historial_fold_{fold:02d}.csv"
    else:
        filename = "historial_entrenamiento.csv"

    path = os.path.join(MODEL_DIR, filename)

    return CSVLogger(
        filename=path,
        separator=",",
        append=False,
    )


# -------------------------------------------------------------
#  Callback personalizado — reporte por epoca
#  Imprime loss, accuracy, val_loss y val_accuracy
#  de forma limpia al final de cada epoca
# -------------------------------------------------------------

class ReporteEpoca(tf.keras.callbacks.Callback):

    def __init__(self, fold=None):
        super().__init__()
        self.fold = fold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        fold_str = f"Fold {self.fold:02d} | " if self.fold is not None else ""

        loss     = logs.get("loss",         0.0)
        acc      = logs.get("accuracy",     0.0)
        val_loss = logs.get("val_loss",     0.0)
        val_acc  = logs.get("val_accuracy", 0.0)
        lr       = float(
            tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        )

        print(
            f"  {fold_str}Epoca {epoch + 1:03d} | "
            f"loss: {loss:.4f} | acc: {acc:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | "
            f"lr: {lr:.6f}"
        )


# -------------------------------------------------------------
#  Callback personalizado — reporte de fine-tuning
#  Avisa cuando empieza la fase de fine-tuning
# -------------------------------------------------------------

class AvisoFineTuning(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        trainable = sum(
            tf.size(w).numpy()
            for w in self.model.trainable_weights
        )
        print(f"\n  Parametros entrenables en esta fase: {trainable:,}")


# -------------------------------------------------------------
#  Conjunto completo de callbacks para entrenamiento inicial
#  Fase 1 — MobileNetV2 congelada
# -------------------------------------------------------------

def get_callbacks_fase1(fold=None):
    return [
        get_early_stopping(),
        get_reduce_lr(),
        get_model_checkpoint(fold=fold),
        get_csv_logger(fold=fold),
        ReporteEpoca(fold=fold),
    ]


# -------------------------------------------------------------
#  Conjunto completo de callbacks para fine-tuning
#  Fase 2 — ultimas capas de MobileNetV2 descongeladas
#  Paciencia mas alta porque el fine-tuning es mas lento
# -------------------------------------------------------------

def get_callbacks_fase2(fold=None):
    early_stopping_ft = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE + 5,
        restore_best_weights=True,
        verbose=1,
        mode="min",
    )

    if fold is not None:
        filename = f"gestflow_fold_{fold:02d}_finetuned.keras"
    else:
        filename = "gesture_model_finetuned.keras"

    checkpoint_ft = ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, filename),
        monitor=CHECKPOINT_MONITOR,
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        mode="max",
    )

    if fold is not None:
        log_filename = f"historial_fold_{fold:02d}_finetuned.csv"
    else:
        log_filename = "historial_finetuned.csv"

    csv_ft = CSVLogger(
        filename=os.path.join(MODEL_DIR, log_filename),
        separator=",",
        append=False,
    )

    return [
        early_stopping_ft,
        get_reduce_lr(),
        checkpoint_ft,
        csv_ft,
        ReporteEpoca(fold=fold),
        AvisoFineTuning(),
    ]