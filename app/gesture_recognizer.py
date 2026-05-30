# =============================================================
#  gestflow — app/gesture_recognizer.py
#  Captura la camara en vivo, acumula FRAME_COUNT frames,
#  los preprocesa igual que en entrenamiento y llama
#  al modelo para obtener el gesto y la precision
# =============================================================

import os
import sys
import cv2
import numpy as np
import time
from collections import deque

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CAMERA_INDEX,
    FRAME_COUNT,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GESTURES,
    NUM_CLASSES,
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    ACTION_COOLDOWN,
)


# -------------------------------------------------------------
#  Clase principal de reconocimiento de gestos
#  Mantiene un buffer circular de frames
#  Cuando el buffer esta lleno infiere el gesto
# -------------------------------------------------------------

class GestureRecognizer:

    def __init__(self, model):
        self.model          = model
        self.frame_buffer   = deque(maxlen=FRAME_COUNT)
        self.ultimo_gesto   = None
        self.ultima_precision = 0.0
        self.ultimo_tiempo  = 0.0
        self.listo          = False

    # ---------------------------------------------------------
    #  Preprocesa un frame crudo de la camara
    #  Igual que en entrenamiento: resize + RGB + normalize
    # ---------------------------------------------------------

    def _preprocesar_frame(self, frame):
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        return frame

    # ---------------------------------------------------------
    #  Agrega un frame al buffer
    #  Devuelve True cuando el buffer esta lleno
    # ---------------------------------------------------------

    def agregar_frame(self, frame):
        procesado = self._preprocesar_frame(frame)
        self.frame_buffer.append(procesado)
        self.listo = len(self.frame_buffer) == FRAME_COUNT
        return self.listo

    # ---------------------------------------------------------
    #  Infiere el gesto actual con el buffer lleno
    #  Devuelve (nombre_gesto, precision) o (None, 0.0)
    #  si la precision no supera CONFIDENCE_THRESHOLD
    #  o si no ha pasado ACTION_COOLDOWN desde la ultima accion
    # ---------------------------------------------------------

    def inferir(self):
        if not self.listo:
            return None, 0.0

        tensor = np.array(self.frame_buffer, dtype=np.float32)
        tensor = np.expand_dims(tensor, axis=0)

        predicciones = self.model.predict(tensor, verbose=0)[0]
        clase_idx    = int(np.argmax(predicciones))
        precision    = float(predicciones[clase_idx])
        gesto        = GESTURES[clase_idx]

        self.ultimo_gesto     = gesto
        self.ultima_precision = precision

        ahora = time.time()

        if precision < CONFIDENCE_THRESHOLD:
            return None, precision

        if ahora - self.ultimo_tiempo < ACTION_COOLDOWN:
            return None, precision

        self.ultimo_tiempo = ahora
        return gesto, precision

    # ---------------------------------------------------------
    #  Devuelve el ultimo gesto y precision detectados
    #  independientemente del cooldown y threshold
    #  Se usa para mostrar en el overlay
    # ---------------------------------------------------------

    def estado_actual(self):
        return self.ultimo_gesto, self.ultima_precision

    # ---------------------------------------------------------
    #  Limpia el buffer de frames
    #  Se llama cuando se pausa el sistema
    # ---------------------------------------------------------

    def limpiar_buffer(self):
        self.frame_buffer.clear()
        self.listo          = False
        self.ultimo_gesto   = None
        self.ultima_precision = 0.0


# -------------------------------------------------------------
#  Clase de captura de camara
#  Maneja la inicializacion, lectura y liberacion
# -------------------------------------------------------------

class CameraCapture:

    def __init__(self):
        self.cap    = None
        self.activa = False

    # ---------------------------------------------------------
    #  Inicializa la camara con los parametros del proyecto
    # ---------------------------------------------------------

    def iniciar(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la camara (indice {CAMERA_INDEX}). "
                f"Verifica CAMERA_INDEX en config.py"
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS,          30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        self.activa = True
        print("  Camara inicializada correctamente.")

    # ---------------------------------------------------------
    #  Lee el siguiente frame de la camara
    #  Devuelve (True, frame) o (False, None)
    # ---------------------------------------------------------

    def leer(self):
        if not self.activa or self.cap is None:
            return False, None

        ret, frame = self.cap.read()

        if not ret:
            return False, None

        frame = cv2.flip(frame, 1)
        return True, frame

    # ---------------------------------------------------------
    #  Libera la camara
    # ---------------------------------------------------------

    def liberar(self):
        if self.cap is not None:
            self.cap.release()
        self.activa = False
        print("  Camara liberada.")

    # ---------------------------------------------------------
    #  Verifica si la camara esta activa
    # ---------------------------------------------------------

    def esta_activa(self):
        return self.activa and self.cap is not None and self.cap.isOpened()


# -------------------------------------------------------------
#  Carga el modelo desde MODEL_PATH
#  Verifica que el archivo exista antes de intentar cargarlo
# -------------------------------------------------------------

def cargar_modelo():
    import tensorflow as tf

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontro el modelo en: {MODEL_PATH}\n"
            f"Ejecuta training/02_train.py primero."
        )

    print(f"  Cargando modelo desde: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("  Modelo cargado correctamente.")

    return model


# -------------------------------------------------------------
#  Calentamiento del modelo
#  Pasa un tensor de ceros para inicializar los pesos
#  Evita la demora en la primera inferencia real
# -------------------------------------------------------------

def calentar_modelo(model):
    dummy = np.zeros(
        (1, FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH, 3),
        dtype=np.float32
    )
    model.predict(dummy, verbose=0)
    print("  Modelo calentado y listo para inferencia.")