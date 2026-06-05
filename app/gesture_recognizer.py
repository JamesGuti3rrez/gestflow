# =============================================================
#  gestflow — app/gesture_recognizer.py
#  Captura la camara, recorta la zona ROI igual que en
#  grabacion, acumula frames, infiere cada INFERENCE_INTERVAL
#  frames y aplica buffer de votacion para estabilidad
# =============================================================

import os
import sys
import cv2
import numpy as np
import time
from collections import deque, Counter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CAMERA_INDEX,
    FRAME_COUNT,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    RECORD_WIDTH,
    RECORD_HEIGHT,
    VIDEO_FPS,
    GESTURES,
    NEUTRAL_CLASS,
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    DETECTION_FLOOR,
    ACTION_COOLDOWN,
    INFERENCE_INTERVAL,
    VOTING_WINDOW,
    VOTING_THRESHOLD,
    ROI_X1,
    ROI_Y1,
    ROI_X2,
    ROI_Y2,
)


# -------------------------------------------------------------
#  Clase principal de reconocimiento de gestos
# -------------------------------------------------------------

class GestureRecognizer:

    def __init__(self, model):
        self.model            = model
        self.frame_buffer     = deque(maxlen=FRAME_COUNT)
        self.raw_frame_buffer = deque(maxlen=FRAME_COUNT)
        self.votos            = deque(maxlen=VOTING_WINDOW)
        self.ultimo_gesto     = None
        self.ultima_precision = 0.0
        self.ultimo_tiempo    = 0.0
        self.listo            = False
        self.contador_frames  = 0
        self.gesto_confirmado = None

    # ---------------------------------------------------------
    #  Recorta la zona ROI del frame igual que en grabacion
    # ---------------------------------------------------------

    def _recortar_roi(self, frame):
        recorte = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
        if recorte.size == 0:
            return cv2.resize(frame, (RECORD_WIDTH, RECORD_HEIGHT))
        return cv2.resize(recorte, (RECORD_WIDTH, RECORD_HEIGHT))

    # ---------------------------------------------------------
    #  Preprocesa un frame para el modelo
    # ---------------------------------------------------------

    def _preprocesar_frame(self, frame):
        f = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = f.astype(np.float32) / 255.0
        return f

    # ---------------------------------------------------------
    #  Agrega un frame al buffer
    #  Recorta el ROI antes de preprocesar
    # ---------------------------------------------------------

    def agregar_frame(self, frame):
        roi = self._recortar_roi(frame)
        self.frame_buffer.append(self._preprocesar_frame(roi))
        self.raw_frame_buffer.append(roi.copy())
        self.listo = len(self.frame_buffer) == FRAME_COUNT
        self.contador_frames += 1
        return self.listo

    # ---------------------------------------------------------
    #  Calcula la direccion del movimiento dentro del ROI
    # ---------------------------------------------------------

    def calcular_direccion(self):
        frames = list(self.raw_frame_buffer)
        if len(frames) < FRAME_COUNT:
            return 0.0, 0.0

        def centroide(f_ant, f_act):
            g1 = cv2.cvtColor(f_ant, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(f_act, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(g1, g2)
            _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            th = cv2.dilate(th, None, iterations=2)
            contornos, _ = cv2.findContours(
                th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contornos:
                return None
            mayor = max(contornos, key=cv2.contourArea)
            if cv2.contourArea(mayor) < 200:
                return None
            M = cv2.moments(mayor)
            if M["m00"] == 0:
                return None
            return M["m10"] / M["m00"], M["m01"] / M["m00"]

        tercio   = max(1, len(frames) // 3)
        c_inicio = centroide(frames[0], frames[tercio])
        c_final  = centroide(frames[-tercio], frames[-1])

        if c_inicio is None or c_final is None:
            return 0.0, 0.0

        h, w = frames[0].shape[:2]
        dx = max(-1.0, min(1.0, (c_final[0] - c_inicio[0]) / (w / 2)))
        dy = max(-1.0, min(1.0, (c_final[1] - c_inicio[1]) / (h / 2)))
        return dx, dy

    # ---------------------------------------------------------
    #  Inferencia con buffer de votacion
    # ---------------------------------------------------------

    def inferir(self):
        if not self.listo:
            return None, 0.0

        if self.contador_frames % INFERENCE_INTERVAL != 0:
            return self.gesto_confirmado, self.ultima_precision

        tensor = np.array(self.frame_buffer, dtype=np.float32)
        tensor = np.expand_dims(tensor, axis=0)

        predicciones = self.model.predict(tensor, verbose=0)[0]
        clase_idx    = int(np.argmax(predicciones))
        precision    = float(predicciones[clase_idx])
        gesto        = GESTURES[clase_idx]

        self.ultima_precision = precision

        if gesto == NEUTRAL_CLASS or precision < DETECTION_FLOOR:
            self.votos.append(None)
        elif precision >= CONFIDENCE_THRESHOLD:
            self.votos.append(gesto)
        else:
            self.votos.append(None)

        conteo = Counter(v for v in self.votos if v is not None)

        if not conteo:
            self.gesto_confirmado = None
            self.ultimo_gesto     = None
            return None, precision

        gesto_ganador, num_votos = conteo.most_common(1)[0]

        if num_votos < VOTING_THRESHOLD:
            self.gesto_confirmado = None
            self.ultimo_gesto     = None
            return None, precision

        self.gesto_confirmado = gesto_ganador
        self.ultimo_gesto     = gesto_ganador
        return gesto_ganador, precision

    # ---------------------------------------------------------
    #  Indica si un gesto discreto puede ejecutarse
    # ---------------------------------------------------------

    def puede_ejecutar_discreto(self):
        ahora = time.time()
        if ahora - self.ultimo_tiempo < ACTION_COOLDOWN:
            return False
        self.ultimo_tiempo = ahora
        return True

    # ---------------------------------------------------------
    #  Estado actual para el overlay
    # ---------------------------------------------------------

    def estado_actual(self):
        hay_deteccion = self.gesto_confirmado is not None
        return self.ultimo_gesto, self.ultima_precision, hay_deteccion

    # ---------------------------------------------------------
    #  Limpia los buffers
    # ---------------------------------------------------------

    def limpiar_buffer(self):
        self.frame_buffer.clear()
        self.raw_frame_buffer.clear()
        self.votos.clear()
        self.listo            = False
        self.ultimo_gesto     = None
        self.ultima_precision = 0.0
        self.gesto_confirmado = None


# -------------------------------------------------------------
#  Clase de captura de camara
# -------------------------------------------------------------

class CameraCapture:

    def __init__(self):
        self.cap    = None
        self.activa = False

    def iniciar(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la camara (indice {CAMERA_INDEX}). "
                f"Verifica CAMERA_INDEX en config.py"
            )

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RECORD_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RECORD_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          VIDEO_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        ret, _ = self.cap.read()
        if not ret:
            self.cap.release()
            raise RuntimeError("La camara se abrio pero no devuelve frames.")

        self.activa = True
        print("  Camara inicializada correctamente.")

    def leer(self):
        if not self.activa or self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, cv2.flip(frame, 1)

    def liberar(self):
        if self.cap is not None:
            self.cap.release()
        self.activa = False
        print("  Camara liberada.")

    def esta_activa(self):
        return self.activa and self.cap is not None and self.cap.isOpened()


# -------------------------------------------------------------
#  Carga el modelo desde MODEL_PATH
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
# -------------------------------------------------------------

def calentar_modelo(model):
    dummy = np.zeros(
        (1, FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH, 3),
        dtype=np.float32
    )
    model.predict(dummy, verbose=0)
    print("  Modelo calentado y listo para inferencia.")