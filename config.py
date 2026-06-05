# =============================================================
#  gestflow — config.py
#  Parametros globales del proyecto
# =============================================================

import os

# -------------------------------------------------------------
#  Rutas del proyecto
# -------------------------------------------------------------

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "raw_videos")
MODEL_DIR   = os.path.join(BASE_DIR, "training", "model")
MODEL_PATH  = os.path.join(MODEL_DIR, "gesture_model.keras")

# -------------------------------------------------------------
#  Identificacion del integrante que graba
# -------------------------------------------------------------

RECORDER_NAME = "user"

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

NUM_CLASSES   = len(GESTURES)
NEUTRAL_CLASS = "NEUTRAL"

# -------------------------------------------------------------
#  Parametros de video y frames
# -------------------------------------------------------------

FRAME_COUNT        = 15
FRAME_WIDTH        = 112
FRAME_HEIGHT       = 112
FRAME_CHANNELS     = 3
VIDEO_FPS          = 30
VIDEO_DURATION     = 2.5
VIDEOS_PER_GESTURE = 60
RECORD_WIDTH       = 640
RECORD_HEIGHT      = 480

# -------------------------------------------------------------
#  Zona de grabacion (ROI — Region of Interest)
#  Recuadro fijo donde el usuario pone la mano
#  Solo esta zona se graba en el dataset
#  Coordenadas sobre el frame de 640x480
#  Ajusta segun donde pones tu mano naturalmente
#
#  Para encontrar los valores correctos:
#  corre 01_record.py y observa donde queda el recuadro
#  Si esta muy a la izquierda: sube ROI_X1 y ROI_X2
#  Si esta muy arriba:         sube ROI_Y1 y ROI_Y2
#  Si es muy pequeno:          baja ROI_X1 o sube ROI_X2
# -------------------------------------------------------------

ROI_X1 = 320      # Borde izquierdo del recuadro
ROI_Y1 = 80       # Borde superior del recuadro
ROI_X2 = 600      # Borde derecho del recuadro
ROI_Y2 = 420      # Borde inferior del recuadro

# -------------------------------------------------------------
#  Parametros del modelo
# -------------------------------------------------------------

LSTM_UNITS_1    = 128
LSTM_UNITS_2    = 64
DENSE_UNITS     = 64
DROPOUT_RATE    = 0.4
MOBILENET_ALPHA = 1.0

# -------------------------------------------------------------
#  Parametros de entrenamiento
# -------------------------------------------------------------

BATCH_SIZE       = 8
EPOCHS           = 100
LEARNING_RATE    = 0.001
VALIDATION_SPLIT = 0.1
TEST_SPLIT       = 0.1
KFOLD_SPLITS     = 5

# -------------------------------------------------------------
#  Parametros de Data Augmentation
# -------------------------------------------------------------

AUGMENTATION_FACTOR = 6
GAUSSIAN_SIGMA      = 0.01
SCALE_MIN           = 0.85
SCALE_MAX           = 1.15
TIME_WARP_MIN       = 0.75
TIME_WARP_MAX       = 1.25
BRIGHTNESS_MIN      = 0.7
BRIGHTNESS_MAX      = 1.3

# -------------------------------------------------------------
#  Parametros de callbacks
# -------------------------------------------------------------

EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE      = 5
REDUCE_LR_FACTOR        = 0.5
REDUCE_LR_MIN           = 1e-6
CHECKPOINT_MONITOR      = "val_accuracy"
FINE_TUNING_LAYERS      = 30

# -------------------------------------------------------------
#  Parametros de grabacion
# -------------------------------------------------------------

CAMERA_INDEX      = 0
COUNTDOWN_SECONDS = 3
RECORDING_COLOR   = (0, 0, 255)
READY_COLOR       = (0, 255, 0)
FONT_SCALE        = 1.0
FONT_THICKNESS    = 2

# -------------------------------------------------------------
#  Parametros de inferencia en tiempo real
# -------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.75
DETECTION_FLOOR      = 0.55
ACTION_COOLDOWN      = 0.4
INFERENCE_INTERVAL   = 3

# -------------------------------------------------------------
#  Buffer de votacion
# -------------------------------------------------------------

VOTING_WINDOW    = 5
VOTING_THRESHOLD = 3

# -------------------------------------------------------------
#  Movimiento del cursor
# -------------------------------------------------------------

SCROLL_SPEED     = 3
CURSOR_SMOOTHING = 4
CURSOR_SPEED     = 90
CURSOR_DEAD_ZONE = 0.08
DRAG_THRESHOLD   = 0.80

# -------------------------------------------------------------
#  Parametros de la ventana flotante (overlay)
# -------------------------------------------------------------

OVERLAY_WIDTH       = 640
OVERLAY_HEIGHT      = 500
OVERLAY_WINDOW_NAME = "gestflow"
OVERLAY_FPS         = 60
BAR_COLOR_HIGH      = (100, 200, 100)
BAR_COLOR_MID       = (100, 200, 255)
BAR_COLOR_LOW       = (80,  80,  200)
BAR_THRESHOLD_HIGH  = 0.85
BAR_THRESHOLD_MID   = 0.65