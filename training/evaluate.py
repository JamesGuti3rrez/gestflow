# =============================================================
#  gestflow — training/evaluate.py
#  Validacion cruzada 5-fold estratificada
#  Metricas por clase: Precision, Recall, F1
#  Matriz de confusion y curvas de entrenamiento
# =============================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    GESTURES,
    NUM_CLASSES,
    KFOLD_SPLITS,
    EPOCHS,
    BATCH_SIZE,
    MODEL_DIR,
    MODEL_PATH,
)


# -------------------------------------------------------------
#  Validacion cruzada 5-fold estratificada
#  Entrena y evalua el modelo en cada fold
#  Devuelve historial completo y metricas por fold
# -------------------------------------------------------------

def cross_validate(X, y, build_model_fn, get_callbacks_fn):
    from augmentation import augment_dataset

    skf     = StratifiedKFold(
        n_splits=KFOLD_SPLITS,
        shuffle=True,
        random_state=42,
    )

    resultados = []

    print(f"\nValidacion cruzada {KFOLD_SPLITS}-Fold estratificada")
    print(f"  Total de muestras : {len(X)}")
    print(f"  Clases            : {NUM_CLASSES}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold} / {KFOLD_SPLITS}")
        print(f"{'='*60}")

        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold   = X[val_idx]
        y_val_fold   = y[val_idx]

        print(f"\n  Aplicando data augmentation al fold {fold}...")
        X_train_aug, y_train_aug = augment_dataset(X_train_fold, y_train_fold)

        print(f"  Muestras entrenamiento (aumentadas) : {len(X_train_aug)}")
        print(f"  Muestras validacion                 : {len(X_val_fold)}")

        model = build_model_fn()
        callbacks = get_callbacks_fn(fold=fold)

        print(f"\n  Fase 1 — Entrenamiento con MobileNetV2 congelada")
        history = model.fit(
            X_train_aug, y_train_aug,
            validation_data=(X_val_fold, y_val_fold),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=0,
        )

        from model import unfreeze_top_layers
        from callbacks import get_callbacks_fase2

        print(f"\n  Fase 2 — Fine-tuning")
        model = unfreeze_top_layers(model)
        callbacks_ft = get_callbacks_fase2(fold=fold)

        history_ft = model.fit(
            X_train_aug, y_train_aug,
            validation_data=(X_val_fold, y_val_fold),
            epochs=EPOCHS // 2,
            batch_size=BATCH_SIZE,
            callbacks=callbacks_ft,
            verbose=0,
        )

        y_pred_proba = model.predict(X_val_fold, verbose=0)
        y_pred       = np.argmax(y_pred_proba, axis=1)
        acc          = accuracy_score(y_val_fold, y_pred)

        print(f"\n  Accuracy fold {fold}: {acc:.4f}")

        resultados.append({
            "fold"       : fold,
            "accuracy"   : acc,
            "history"    : history,
            "history_ft" : history_ft,
            "y_true"     : y_val_fold,
            "y_pred"     : y_pred,
        })

    return resultados


# -------------------------------------------------------------
#  Reporte de metricas por fold y promedio final
#  Precision, Recall, F1 por clase
#  Accuracy promedio con desviacion estandar
# -------------------------------------------------------------

def report_metrics(resultados):
    print(f"\n{'='*60}")
    print(f"  RESULTADOS FINALES — VALIDACION CRUZADA")
    print(f"{'='*60}")

    accuracies = []

    for r in resultados:
        fold = r["fold"]
        acc  = r["accuracy"]
        accuracies.append(acc)

        print(f"\n  Fold {fold} — Accuracy: {acc:.4f}")
        print(
            classification_report(
                r["y_true"],
                r["y_pred"],
                target_names=GESTURES,
                digits=4,
            )
        )

    mean_acc = np.mean(accuracies)
    std_acc  = np.std(accuracies)

    print(f"\n{'='*60}")
    print(f"  Accuracy promedio : {mean_acc:.4f}")
    print(f"  Desviacion estandar : {std_acc:.4f}")
    print(f"{'='*60}")

    return mean_acc, std_acc


# -------------------------------------------------------------
#  Matriz de confusion agregada
#  Suma las predicciones de todos los folds
#  Guarda la imagen en MODEL_DIR
# -------------------------------------------------------------

def plot_confusion_matrix(resultados):
    os.makedirs(MODEL_DIR, exist_ok=True)

    cm_total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    for r in resultados:
        cm = confusion_matrix(r["y_true"], r["y_pred"])
        cm_total += cm

    cm_norm = cm_total.astype(float) / cm_total.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=GESTURES,
        yticklabels=GESTURES,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title("Matriz de confusion — gestflow (5-Fold agregado)", fontsize=14, pad=16)
    ax.set_xlabel("Prediccion", fontsize=12)
    ax.set_ylabel("Clase real", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"\n  Matriz de confusion guardada en: {path}")


# -------------------------------------------------------------
#  Curvas de entrenamiento por fold
#  Loss y accuracy — fase 1 y fase 2 (fine-tuning)
#  Guarda una imagen por fold en MODEL_DIR
# -------------------------------------------------------------

def plot_training_curves(resultados):
    os.makedirs(MODEL_DIR, exist_ok=True)

    for r in resultados:
        fold       = r["fold"]
        history    = r["history"].history
        history_ft = r["history_ft"].history

        loss_total     = history["loss"]     + history_ft["loss"]
        val_loss_total = history["val_loss"] + history_ft["val_loss"]
        acc_total      = history["accuracy"] + history_ft["accuracy"]
        val_acc_total  = history["val_accuracy"] + history_ft["val_accuracy"]

        epocas         = range(1, len(loss_total) + 1)
        ft_inicio      = len(history["loss"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epocas, loss_total,     label="Train loss",      color="steelblue")
        ax1.plot(epocas, val_loss_total, label="Val loss",        color="coral")
        ax1.axvline(x=ft_inicio, color="gray", linestyle="--", linewidth=1, label="Fine-tuning")
        ax1.set_title(f"Loss — Fold {fold}", fontsize=12)
        ax1.set_xlabel("Epoca")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epocas, acc_total,     label="Train accuracy",   color="steelblue")
        ax2.plot(epocas, val_acc_total, label="Val accuracy",     color="coral")
        ax2.axvline(x=ft_inicio, color="gray", linestyle="--", linewidth=1, label="Fine-tuning")
        ax2.set_title(f"Accuracy — Fold {fold}", fontsize=12)
        ax2.set_xlabel("Epoca")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f"Curvas de entrenamiento — Fold {fold}", fontsize=13)
        plt.tight_layout()

        path = os.path.join(MODEL_DIR, f"curvas_fold_{fold:02d}.png")
        plt.savefig(path, dpi=150)
        plt.close()

        print(f"  Curvas fold {fold} guardadas en: {path}")


# -------------------------------------------------------------
#  Evaluacion final sobre el test set
#  Se usa el modelo final entrenado con todos los datos
# -------------------------------------------------------------

def evaluate_test_set(model, X_test, y_test):
    print(f"\n{'='*60}")
    print(f"  EVALUACION FINAL — TEST SET")
    print(f"{'='*60}")

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred       = np.argmax(y_pred_proba, axis=1)
    acc          = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy en test set: {acc:.4f}")
    print(f"\n  Reporte por clase:\n")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=GESTURES,
            digits=4,
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=GESTURES,
        yticklabels=GESTURES,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title("Matriz de confusion — Test set final", fontsize=14, pad=16)
    ax.set_xlabel("Prediccion", fontsize=12)
    ax.set_ylabel("Clase real", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    path = os.path.join(MODEL_DIR, "confusion_matrix_test.png")
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"\n  Matriz de confusion test guardada en: {path}")

    return acc