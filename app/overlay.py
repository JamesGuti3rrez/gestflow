# =============================================================
#  gestflow — app/overlay.py
#  Ventana flotante con feed de camara, recuadro ROI,
#  gesto detectado, precision y barra de progreso
# =============================================================

import os
import sys
import cv2
import numpy as np
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    OVERLAY_WIDTH,
    OVERLAY_HEIGHT,
    OVERLAY_WINDOW_NAME,
    BAR_COLOR_HIGH,
    BAR_COLOR_MID,
    BAR_COLOR_LOW,
    BAR_THRESHOLD_HIGH,
    BAR_THRESHOLD_MID,
    CONFIDENCE_THRESHOLD,
    ROI_X1,
    ROI_Y1,
    ROI_X2,
    ROI_Y2,
    RECORD_WIDTH,
    RECORD_HEIGHT,
)


# -------------------------------------------------------------
#  Constantes visuales
# -------------------------------------------------------------

FONT         = cv2.FONT_HERSHEY_SIMPLEX
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0,   0,   0  )
COLOR_GRAY   = (80,  80,  80 )
COLOR_DARK   = (20,  20,  20 )
COLOR_PANEL  = (30,  30,  30 )
COLOR_ACCENT = (180, 140, 80 )
COLOR_YELLOW = (0,   255, 255)

FEED_HEIGHT     = 380
PANEL_Y         = FEED_HEIGHT
BAR_HEIGHT      = 10
BAR_Y           = PANEL_Y + 88
BAR_X_START     = 16
BAR_X_END       = OVERLAY_WIDTH - 16
BAR_WIDTH_TOTAL = BAR_X_END - BAR_X_START

# Escala del ROI para dibujarlo sobre el feed del overlay
# El feed se redimensiona de RECORD_WIDTH x RECORD_HEIGHT
# a OVERLAY_WIDTH x FEED_HEIGHT
ESCALA_X = OVERLAY_WIDTH  / RECORD_WIDTH
ESCALA_Y = FEED_HEIGHT    / RECORD_HEIGHT

ROI_OX1  = int(ROI_X1 * ESCALA_X)
ROI_OY1  = int(ROI_Y1 * ESCALA_Y)
ROI_OX2  = int(ROI_X2 * ESCALA_X)
ROI_OY2  = int(ROI_Y2 * ESCALA_Y)


# -------------------------------------------------------------
#  Utilidades
# -------------------------------------------------------------

def _draw_text(frame, texto, pos, color=COLOR_WHITE,
               scale=0.55, thickness=1):
    (tw, th), _ = cv2.getTextSize(texto, FONT, scale, thickness)
    x, y        = pos
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 3, y - th - 3),
                  (x + tw + 3, y + 4), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, texto, (x, y), FONT, scale,
                color, thickness, cv2.LINE_AA)


def _color_barra(precision):
    if precision >= BAR_THRESHOLD_HIGH:
        return BAR_COLOR_HIGH
    elif precision >= BAR_THRESHOLD_MID:
        return BAR_COLOR_MID
    else:
        return BAR_COLOR_LOW


# -------------------------------------------------------------
#  Clase principal del overlay
# -------------------------------------------------------------

class Overlay:

    def __init__(self):
        self.activo           = False
        self.ultimo_gesto     = None
        self.ultima_precision = 0.0
        self.hay_deteccion    = False
        self.sistema_pausado  = False
        self.safe_mode        = False
        self.fps_contador     = 0
        self.fps_tiempo       = time.time()
        self.fps_actual       = 0.0

    # ---------------------------------------------------------
    #  Inicializa la ventana flotante
    # ---------------------------------------------------------

    def iniciar(self):
        cv2.namedWindow(OVERLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(OVERLAY_WINDOW_NAME, OVERLAY_WIDTH, OVERLAY_HEIGHT)
        cv2.setWindowProperty(
            OVERLAY_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1
        )
        self.activo = True
        print("  Overlay iniciado.")

    # ---------------------------------------------------------
    #  Actualiza el estado interno
    # ---------------------------------------------------------

    def actualizar_estado(self, gesto, precision, hay_deteccion,
                          estado_controller):
        self.ultimo_gesto     = gesto
        self.ultima_precision = precision
        self.hay_deteccion    = hay_deteccion
        self.sistema_pausado  = estado_controller.get("pausado",   False)
        self.safe_mode        = estado_controller.get("safe_mode", False)

    # ---------------------------------------------------------
    #  Renderiza el frame completo
    # ---------------------------------------------------------

    def renderizar(self, frame_camara):
        if not self.activo:
            return

        canvas = np.zeros((OVERLAY_HEIGHT, OVERLAY_WIDTH, 3), dtype=np.uint8)
        self._render_feed(canvas, frame_camara)
        self._render_roi(canvas)
        self._render_panel(canvas)
        self._render_estado_sistema(canvas)
        self._calcular_fps()
        cv2.imshow(OVERLAY_WINDOW_NAME, canvas)

    # ---------------------------------------------------------
    #  Renderiza el feed de la camara completo
    # ---------------------------------------------------------

    def _render_feed(self, canvas, frame_camara):
        feed = cv2.resize(frame_camara, (OVERLAY_WIDTH, FEED_HEIGHT))
        canvas[0:FEED_HEIGHT, 0:OVERLAY_WIDTH] = feed

        cv2.rectangle(canvas, (0, 0),
                      (OVERLAY_WIDTH - 1, FEED_HEIGHT - 1),
                      COLOR_ACCENT, 1)

        en_vivo         = not self.sistema_pausado
        indicador_color = (0, 200, 80) if en_vivo else (80, 80, 200)
        cv2.circle(canvas, (OVERLAY_WIDTH - 14, 14), 5,
                   indicador_color, -1)
        _draw_text(canvas, "EN VIVO" if en_vivo else "PAUSADO",
                   (12, 20), indicador_color, scale=0.45)

        cv2.putText(canvas, f"{int(self.fps_actual)} fps",
                    (OVERLAY_WIDTH - 70, FEED_HEIGHT - 10),
                    FONT, 0.42, COLOR_WHITE, 1, cv2.LINE_AA)

    # ---------------------------------------------------------
    #  Dibuja el recuadro ROI sobre el feed escalado
    #  Misma zona que en la grabacion
    # ---------------------------------------------------------

    def _render_roi(self, canvas):
        color  = (0, 200, 80) if self.hay_deteccion else COLOR_YELLOW
        grosor = 2

        cv2.rectangle(canvas, (ROI_OX1, ROI_OY1),
                      (ROI_OX2, ROI_OY2), color, grosor)

        esquina = 12
        cv2.line(canvas, (ROI_OX1, ROI_OY1),
                 (ROI_OX1 + esquina, ROI_OY1), color, grosor + 1)
        cv2.line(canvas, (ROI_OX1, ROI_OY1),
                 (ROI_OX1, ROI_OY1 + esquina), color, grosor + 1)
        cv2.line(canvas, (ROI_OX2, ROI_OY1),
                 (ROI_OX2 - esquina, ROI_OY1), color, grosor + 1)
        cv2.line(canvas, (ROI_OX2, ROI_OY1),
                 (ROI_OX2, ROI_OY1 + esquina), color, grosor + 1)
        cv2.line(canvas, (ROI_OX1, ROI_OY2),
                 (ROI_OX1 + esquina, ROI_OY2), color, grosor + 1)
        cv2.line(canvas, (ROI_OX1, ROI_OY2),
                 (ROI_OX1, ROI_OY2 - esquina), color, grosor + 1)
        cv2.line(canvas, (ROI_OX2, ROI_OY2),
                 (ROI_OX2 - esquina, ROI_OY2), color, grosor + 1)
        cv2.line(canvas, (ROI_OX2, ROI_OY2),
                 (ROI_OX2, ROI_OY2 - esquina), color, grosor + 1)

        if not self.hay_deteccion:
            _draw_text(canvas, "PON TU MANO AQUI",
                       (ROI_OX1 + 8, ROI_OY1 + 24),
                       COLOR_YELLOW, scale=0.45)

    # ---------------------------------------------------------
    #  Renderiza el panel inferior
    # ---------------------------------------------------------

    def _render_panel(self, canvas):
        cv2.rectangle(canvas, (0, PANEL_Y),
                      (OVERLAY_WIDTH, OVERLAY_HEIGHT),
                      COLOR_PANEL, -1)
        cv2.line(canvas, (0, PANEL_Y),
                 (OVERLAY_WIDTH, PANEL_Y), COLOR_ACCENT, 1)
        self._render_gesto(canvas)
        self._render_barra_precision(canvas)

    # ---------------------------------------------------------
    #  Renderiza el nombre del gesto y la precision
    # ---------------------------------------------------------

    def _render_gesto(self, canvas):
        if not self.hay_deteccion:
            cv2.putText(canvas, "Sin deteccion",
                        (BAR_X_START, PANEL_Y + 36),
                        FONT, 0.66, COLOR_GRAY, 2, cv2.LINE_AA)
            cv2.putText(canvas, "Muestra tu mano en el recuadro",
                        (BAR_X_START, PANEL_Y + 64),
                        FONT, 0.42, COLOR_GRAY, 1, cv2.LINE_AA)
            return

        gesto       = self.ultimo_gesto or "Detectando..."
        precision   = self.ultima_precision
        label_color = _color_barra(precision) if self.ultimo_gesto else COLOR_GRAY

        cv2.putText(canvas, gesto,
                    (BAR_X_START, PANEL_Y + 36),
                    FONT, 0.72, label_color, 2, cv2.LINE_AA)

        cv2.putText(canvas, f"{precision * 100:.1f}%",
                    (BAR_X_START, PANEL_Y + 64),
                    FONT, 0.55, label_color, 1, cv2.LINE_AA)

        cv2.putText(canvas,
                    f"Umbral: {int(CONFIDENCE_THRESHOLD * 100)}%",
                    (OVERLAY_WIDTH - 110, PANEL_Y + 64),
                    FONT, 0.40, COLOR_GRAY, 1, cv2.LINE_AA)

    # ---------------------------------------------------------
    #  Renderiza la barra de precision
    # ---------------------------------------------------------

    def _render_barra_precision(self, canvas):
        cv2.rectangle(canvas, (BAR_X_START, BAR_Y),
                      (BAR_X_END, BAR_Y + BAR_HEIGHT),
                      COLOR_DARK, -1)

        precision  = self.ultima_precision if self.hay_deteccion else 0.0
        fill_width = max(0, min(int(BAR_WIDTH_TOTAL * precision),
                                BAR_WIDTH_TOTAL))
        color      = _color_barra(precision)

        if fill_width > 0:
            cv2.rectangle(canvas,
                          (BAR_X_START, BAR_Y),
                          (BAR_X_START + fill_width, BAR_Y + BAR_HEIGHT),
                          color, -1)

        umbral_x = BAR_X_START + int(BAR_WIDTH_TOTAL * CONFIDENCE_THRESHOLD)
        cv2.line(canvas, (umbral_x, BAR_Y - 3),
                 (umbral_x, BAR_Y + BAR_HEIGHT + 3),
                 COLOR_WHITE, 1)

        cv2.rectangle(canvas, (BAR_X_START, BAR_Y),
                      (BAR_X_END, BAR_Y + BAR_HEIGHT),
                      COLOR_GRAY, 1)

    # ---------------------------------------------------------
    #  Renderiza indicadores de estado del sistema
    # ---------------------------------------------------------

    def _render_estado_sistema(self, canvas):
        y_base = OVERLAY_HEIGHT - 14
        if self.safe_mode:
            cv2.putText(canvas, "SAFE MODE",
                        (BAR_X_START, y_base),
                        FONT, 0.40, BAR_COLOR_LOW, 1, cv2.LINE_AA)
        cv2.putText(canvas, "ESC para salir",
                    (OVERLAY_WIDTH - 110, y_base),
                    FONT, 0.38, COLOR_GRAY, 1, cv2.LINE_AA)

    # ---------------------------------------------------------
    #  Calcula los FPS del overlay
    # ---------------------------------------------------------

    def _calcular_fps(self):
        self.fps_contador += 1
        ahora = time.time()
        if ahora - self.fps_tiempo >= 1.0:
            self.fps_actual   = self.fps_contador
            self.fps_contador = 0
            self.fps_tiempo   = ahora

    # ---------------------------------------------------------
    #  Verifica si se presiono ESC
    # ---------------------------------------------------------

    def verificar_salida(self):
        return (cv2.waitKey(1) & 0xFF) == 27

    # ---------------------------------------------------------
    #  Cierra la ventana
    # ---------------------------------------------------------

    def cerrar(self):
        try:
            cv2.destroyWindow(OVERLAY_WINDOW_NAME)
        except cv2.error:
            pass
        self.activo = False
        print("  Overlay cerrado.")