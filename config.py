# =============================================================
#  gestflow — config.py
#  Parametros globales del proyecto
#  Ningun otro archivo tiene valores hardcodeados
# =============================================================

import os

# -------------------------------------------------------------
#  Rutas del proyecto
# -------------------------------------------------------------

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR     = os.path.join(BASE_DIR, "dataset", "raw_videos")
MODEL_DIR       = os.path.join(BASE_DIR, "training", "model")
MODEL_PATH      = os.path.join(MODEL_DIR, "gesture_model.keras")

# -------------------------------------------------------------
#  Clases de gestos
# -------------------------------------------------------------

GESTURES = [
    "MOVE_CURSOR",
    "LEFT_CLICK",
    "RIGHT_CLICK",
    "SCROLL",
    "DRAG",
    "ALT_TAB",
    "PAUSE",
    "SAFE_MODE",
    "OPEN_APP",
]

NUM_CLASSES = len(GESTURES)

# -------------------------------------------------------------
#  Parametros de video y frames
#  FRAME_WIDTH y FRAME_HEIGHT son el tamaño que entra al modelo
#  RECORD_WIDTH y RECORD_HEIGHT son el tamaño de grabacion
# -------------------------------------------------------------

FRAME_COUNT        = 15        # Frames extraidos por video
FRAME_WIDTH        = 112       # Ancho del frame para el modelo (px)
FRAME_HEIGHT       = 112       # Alto del frame para el modelo (px)
FRAME_CHANNELS     = 3         # RGB
VIDEO_FPS          = 30        # FPS de grabacion
VIDEO_DURATION     = 2.5       # Duracion en segundos por video
VIDEOS_PER_GESTURE = 60        # Videos a grabar por gesto
RECORD_WIDTH       = 640       # Ancho de grabacion y preview (px)
RECORD_HEIGHT      = 480       # Alto de grabacion y preview (px)

# -------------------------------------------------------------
#  Parametros del modelo
# -------------------------------------------------------------

LSTM_UNITS_1    = 128       # Unidades primera capa LSTM
LSTM_UNITS_2    = 64        # Unidades segunda capa LSTM
DENSE_UNITS     = 64        # Unidades capa Dense antes de Softmax
DROPOUT_RATE    = 0.4       # Tasa de Dropout entre capas
MOBILENET_ALPHA = 1.0       # Factor de ancho MobileNetV2

# -------------------------------------------------------------
#  Parametros de entrenamiento
# -------------------------------------------------------------

BATCH_SIZE          = 8
EPOCHS              = 100
LEARNING_RATE       = 0.001
VALIDATION_SPLIT    = 0.1
TEST_SPLIT          = 0.1
KFOLD_SPLITS        = 5

# -------------------------------------------------------------
#  Parametros de Data Augmentation
# -------------------------------------------------------------

AUGMENTATION_FACTOR     = 6        # Multiplicador del dataset original
GAUSSIAN_SIGMA          = 0.01     # Desviacion estandar ruido gaussiano
SCALE_MIN               = 0.85     # Escala minima para zoom
SCALE_MAX               = 1.15     # Escala maxima para zoom
TIME_WARP_MIN           = 0.75     # Factor minimo de time warping
TIME_WARP_MAX           = 1.25     # Factor maximo de time warping
BRIGHTNESS_MIN          = 0.7      # Factor minimo de brillo
BRIGHTNESS_MAX          = 1.3      # Factor maximo de brillo

# -------------------------------------------------------------
#  Parametros de callbacks
# -------------------------------------------------------------

EARLY_STOPPING_PATIENCE     = 10
REDUCE_LR_PATIENCE          = 5
REDUCE_LR_FACTOR            = 0.5
REDUCE_LR_MIN               = 1e-6
CHECKPOINT_MONITOR          = "val_accuracy"

# -------------------------------------------------------------
#  Parametros de grabacion
# -------------------------------------------------------------

CAMERA_INDEX        = 0
COUNTDOWN_SECONDS   = 3               # Cuenta regresiva antes de grabar
RECORDING_COLOR     = (0, 0, 255)     # BGR rojo — grabando
READY_COLOR         = (0, 255, 0)     # BGR verde — listo
FONT_SCALE          = 1.0
FONT_THICKNESS      = 2

# -------------------------------------------------------------
#  Parametros de inferencia en tiempo real
# -------------------------------------------------------------

CONFIDENCE_THRESHOLD    = 0.75     # Precision minima para ejecutar accion
ACTION_COOLDOWN         = 0.5      # Segundos entre acciones consecutivas
SCROLL_SPEED            = 3        # Unidades de scroll por deteccion
CURSOR_SMOOTHING        = 7        # Factor de suavizado del cursor
DRAG_THRESHOLD          = 0.80     # Precision minima especifica para DRAG

# -------------------------------------------------------------
#  Parametros de la ventana flotante (overlay)
# -------------------------------------------------------------

OVERLAY_WIDTH           = 320
OVERLAY_HEIGHT          = 280
OVERLAY_WINDOW_NAME     = "gestflow"
OVERLAY_FPS             = 30
BAR_COLOR_HIGH          = (100, 200, 100)   # BGR verde — alta precision
BAR_COLOR_MID           = (100, 200, 255)   # BGR amarillo — precision media
BAR_COLOR_LOW           = (80, 80, 200)     # BGR rojo — baja precision
BAR_THRESHOLD_HIGH      = 0.85
BAR_THRESHOLD_MID       = 0.65