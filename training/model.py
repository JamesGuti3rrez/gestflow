# =============================================================
#  gestflow — training/model.py
#  Arquitectura CNN + LSTM
#  TimeDistributed MobileNetV2 + LSTM apiladas
#  + Dropout + BatchNormalization + Softmax
# =============================================================

import os
import sys
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    TimeDistributed,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Masking,
    GlobalAveragePooling2D,
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    FRAME_COUNT,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    FRAME_CHANNELS,
    NUM_CLASSES,
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    DENSE_UNITS,
    DROPOUT_RATE,
    MOBILENET_ALPHA,
    LEARNING_RATE,
    FINE_TUNING_LAYERS,
)


# -------------------------------------------------------------
#  Construccion del extractor CNN
# -------------------------------------------------------------

def build_cnn_extractor():
    base = MobileNetV2(
        input_shape=(FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS),
        alpha=MOBILENET_ALPHA,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs  = Input(shape=(FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS))
    x       = base(inputs, training=False)
    outputs = GlobalAveragePooling2D()(x)

    return Model(inputs, outputs, name="cnn_extractor")


# -------------------------------------------------------------
#  Construccion del modelo completo
# -------------------------------------------------------------

def build_model():
    cnn_extractor = build_cnn_extractor()

    inputs = Input(
        shape=(FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS),
        name="input_frames"
    )

    x = TimeDistributed(cnn_extractor, name="feature_extractor")(inputs)

    x = Masking(mask_value=0.0, name="masking")(x)

    x = LSTM(
        LSTM_UNITS_1,
        return_sequences=True,
        name="lstm_1"
    )(x)
    x = Dropout(DROPOUT_RATE, name="dropout_1")(x)
    x = BatchNormalization(name="bn_1")(x)

    x = LSTM(
        LSTM_UNITS_2,
        return_sequences=False,
        name="lstm_2"
    )(x)
    x = Dropout(DROPOUT_RATE, name="dropout_2")(x)
    x = BatchNormalization(name="bn_2")(x)

    x = Dense(DENSE_UNITS, activation="relu", name="dense_1")(x)
    x = Dropout(DROPOUT_RATE, name="dropout_3")(x)

    outputs = Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="gestflow_model")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# -------------------------------------------------------------
#  Fine-tuning
#  Detecta la capa MobileNetV2 dinamicamente por nombre
#  Funciona con cualquier FRAME_WIDTH configurado
# -------------------------------------------------------------

def unfreeze_top_layers(model, n_layers=None):
    if n_layers is None:
        n_layers = FINE_TUNING_LAYERS

    cnn_extractor = model.get_layer("feature_extractor").layer

    base_model = None
    for layer in cnn_extractor.layers:
        if "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError(
            "No se encontro la capa MobileNetV2 dentro del extractor"
        )

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-n_layers:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\nFine-tuning activado:")
    print(f"  Capa CNN detectada         : {base_model.name}")
    print(f"  Ultimas {n_layers} capas descongeladas")
    print(f"  Learning rate reducido a   : {LEARNING_RATE / 10:.6f}")

    return model


# -------------------------------------------------------------
#  Resumen del modelo
# -------------------------------------------------------------

def model_summary(model):
    print("\nResumen del modelo gestflow:")
    model.summary()

    trainable    = sum(tf.size(w).numpy() for w in model.trainable_weights)
    no_trainable = sum(tf.size(w).numpy() for w in model.non_trainable_weights)

    print(f"\n  Parametros entrenables     : {trainable:,}")
    print(f"  Parametros no entrenables  : {no_trainable:,}")
    print(f"  Total                      : {trainable + no_trainable:,}")


# -------------------------------------------------------------
#  Carga del modelo guardado
# -------------------------------------------------------------

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontro el modelo en: {model_path}\n"
            f"Ejecuta training/02_train.py primero."
        )

    model = tf.keras.models.load_model(model_path)
    print(f"\nModelo cargado desde: {model_path}")
    return model