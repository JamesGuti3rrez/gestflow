# =============================================================
#  gestflow — training/02_train.py
#  Orquestador principal del entrenamiento
#  Llama a dataloader, augmentation, model,
#  callbacks y evaluate en el orden correcto
#
#  Uso:
#      python training/02_train.py
# =============================================================

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))

from config import (
    MODEL_PATH,
    MODEL_DIR,
    GESTURES,
    NUM_CLASSES,
    KFOLD_SPLITS,
    EPOCHS,
    BATCH_SIZE,
)

from dataloader   import load_dataset, split_dataset, dataset_summary
from augmentation import augment_dataset
from model        import build_model, unfreeze_top_layers, model_summary
from callbacks    import get_callbacks_fase1, get_callbacks_fase2
from evaluate     import (
    cross_validate,
    report_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    evaluate_test_set,
)


# -------------------------------------------------------------
#  Separador visual para consola
# -------------------------------------------------------------

def separador(titulo=""):
    linea = "=" * 60
    if titulo:
        print(f"\n{linea}")
        print(f"  {titulo}")
        print(f"{linea}")
    else:
        print(f"\n{linea}")


# -------------------------------------------------------------
#  Fase 0 — Verificacion y resumen del dataset
# -------------------------------------------------------------

def fase_0_verificacion():
    separador("FASE 0 — VERIFICACION DEL DATASET")
    dataset_summary()


# -------------------------------------------------------------
#  Fase 1 — Carga y split del dataset
# -------------------------------------------------------------

def fase_1_carga():
    separador("FASE 1 — CARGA DEL DATASET")
    X, y = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    return X, y, X_train, X_val, X_test, y_train, y_val, y_test


# -------------------------------------------------------------
#  Fase 2 — Validacion cruzada
# -------------------------------------------------------------

def fase_2_cross_validation(X, y):
    separador(f"FASE 2 — VALIDACION CRUZADA {KFOLD_SPLITS}-FOLD")

    resultados = cross_validate(
        X=X,
        y=y,
        build_model_fn=build_model,
        get_callbacks_fn=get_callbacks_fase1,
    )

    return resultados


# -------------------------------------------------------------
#  Fase 3 — Reporte de metricas y visualizaciones
# -------------------------------------------------------------

def fase_3_reporte(resultados):
    separador("FASE 3 — METRICAS Y VISUALIZACIONES")

    mean_acc, std_acc = report_metrics(resultados)

    print("\n  Generando visualizaciones...")
    plot_confusion_matrix(resultados)
    plot_training_curves(resultados)

    return mean_acc, std_acc


# -------------------------------------------------------------
#  Fase 4 — Entrenamiento final con todos los datos
# -------------------------------------------------------------

def fase_4_entrenamiento_final(X_train, X_val, X_test,
                               y_train, y_val, y_test):
    separador("FASE 4 — ENTRENAMIENTO FINAL")

    print("\n  Aplicando data augmentation al set de entrenamiento...")
    X_train_aug, y_train_aug = augment_dataset(X_train, y_train)

    print(f"  Muestras entrenamiento (aumentadas) : {len(X_train_aug)}")
    print(f"  Muestras validacion                 : {len(X_val)}")
    print(f"  Muestras test                       : {len(X_test)}")

    model = build_model()
    model_summary(model)

    print(f"\n  Fase 1 — MobileNetV2 congelada")
    callbacks_fase1 = get_callbacks_fase1(fold=None)

    model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_fase1,
        verbose=0,
    )

    print(f"\n  Fase 2 — Fine-tuning")
    model = unfreeze_top_layers(model)
    callbacks_fase2 = get_callbacks_fase2(fold=None)

    model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=EPOCHS // 2,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_fase2,
        verbose=0,
    )

    return model


# -------------------------------------------------------------
#  Fase 5 — Evaluacion final en test set
# -------------------------------------------------------------

def fase_5_evaluacion(model, X_test, y_test):
    separador("FASE 5 — EVALUACION EN TEST SET")
    acc = evaluate_test_set(model, X_test, y_test)
    return acc


# -------------------------------------------------------------
#  Guardado del modelo final
# -------------------------------------------------------------

def guardar_modelo(model):
    separador("GUARDADO DEL MODELO")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"\n  Modelo guardado en : {MODEL_PATH}")
    print(f"  Tamanio            : {size_mb:.2f} MB")


# -------------------------------------------------------------
#  Main
# -------------------------------------------------------------

def main():
    inicio = time.time()

    separador("gestflow — ENTRENAMIENTO")
    print(f"\n  Clases    : {NUM_CLASSES}")
    print(f"  Gestos    : {', '.join(GESTURES)}")
    print(f"  Folds     : {KFOLD_SPLITS}")
    print(f"  Epocas    : {EPOCHS}")
    print(f"  Batch     : {BATCH_SIZE}")

    fase_0_verificacion()

    X, y, X_train, X_val, X_test, y_train, y_val, y_test = fase_1_carga()

    resultados = fase_2_cross_validation(X, y)

    mean_acc, std_acc = fase_3_reporte(resultados)

    model = fase_4_entrenamiento_final(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
    )

    acc_test = fase_5_evaluacion(model, X_test, y_test)

    guardar_modelo(model)

    fin      = time.time()
    elapsed  = fin - inicio
    horas    = int(elapsed // 3600)
    minutos  = int((elapsed % 3600) // 60)
    segundos = int(elapsed % 60)

    separador("ENTRENAMIENTO COMPLETADO")
    print(f"\n  Accuracy validacion cruzada : {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"  Accuracy test set           : {acc_test:.4f}")
    print(f"  Tiempo total                : {horas:02d}h {minutos:02d}m {segundos:02d}s")
    print(f"  Modelo guardado en          : {MODEL_PATH}")
    separador()


if __name__ == "__main__":
    main()