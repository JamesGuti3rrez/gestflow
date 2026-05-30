# =============================================================
#  gestflow — app/overlay.py
#  Ventana flotante siempre encima del escritorio
#  Muestra el feed de la camara, el gesto detectado
#  y la precision con su barra de progreso
# =============================================================

import os
import sys
import cv2
import numpy as np
import ctypes
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    OVERLAY_WIDTH,
    OVERLAY_HEIGHT,
    OVERLAY_WINDOW_NAME,
    OVERLAY_FPS,
    BAR_COLOR_HIGH,
    BAR_COLOR_MID,
    BAR_COLOR_LOW,
    BAR_THRESHOLD_HIGH,
    BAR_THRESHOLD_MID,
    CONFIDENCE_THRESHOLD,
)


# -------------------------------------------------------------
#  Constantes visuales internas del overlay
# -------------------------------------------------------------

FONT              = cv2.FONT_HERSHEY_SIMPLEX
COLOR_WHITE       = (255, 255, 255)
COLOR_BLACK       = (0,   0,   0  )
COLOR_GRAY        = (80,  80,  80 )
COLOR_DARK        = (20,  20,  20 )
COLOR_PANEL       = (30,  30,  30 )
COLOR_ACCENT      = (180, 140, 80 )

FEED_HEIGHT       = 180
PANEL_HEIGHT      = OVERLAY_HEIGHT - FEED_HEIGHT
PANEL_Y           = FEED_HEIGHT
BAR_HEIGHT        = 10
BAR_Y             = PANEL_Y + 88
BAR_X_START       = 16
BAR_X_END         = OVERLAY_WIDTH - 16
BAR_WIDTH_TOTAL   = BAR_X_END - BAR_X_START


# -------------------------------------------------------------
#  Dibuja texto con fondo semitransparente
# -------------------------------------------------------------

def _draw_text(frame, texto, pos, color=COLOR_WHITE,
               scale=0.55, thickness=1):
    font = FONT
    (tw, th), _ = cv2.getTextSize(texto, font, scale, thickness)
    x, y        = pos

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - 3, y - th - 3),
        (x + tw + 3, y + 4),
        COLOR_BLACK,
        -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(
        frame, texto, (x, y),
        font, scale, color, thickness,
        cv2.LINE_AA,
    )


# -------------------------------------------------------------
#  Selecciona el color de la barra segun la precision
# -------------------------------------------------------------

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
        self.sistema_pausado  = False
        self.safe_mode        = False
        self.fps_contador     = 0
        self.fps_tiempo       = time.time()
        self.fps_actual       = 0.0

    # ---------------------------------------------------------
    #  Inicializa la ventana flotante
    #  La mantiene siempre encima con WND_PROP_TOPMOST
    # ---------------------------------------------------------

    def iniciar(self):
        cv2.namedWindow(OVERLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(OVERLAY_WINDOW_NAME, OVERLAY_WIDTH, OVERLAY_HEIGHT)
        cv2.setWindowProperty(
            OVERLAY_WINDOW_NAME,
            cv2.WND_PROP_TOPMOST,
            1,
        )
        self.activo = True
        print("  Overlay iniciado.")

    # ---------------------------------------------------------
    #  Actualiza el estado interno del overlay
    #  Se llama desde main.py en cada iteracion del loop
    # ---------------------------------------------------------

    def actualizar_estado(self, gesto, precision, estado_controller):
        self.ultimo_gesto     = gesto
        self.ultima_precision = precision
        self.sistema_pausado  = estado_controller.get("pausado",   False)
        self.safe_mode        = estado_controller.get("safe_mode", False)

    # ---------------------------------------------------------
    #  Renderiza el frame completo del overlay
    #  Compone el feed de camara + panel inferior
    # ---------------------------------------------------------

    def renderizar(self, frame_camara):
        if not self.activo:
            return

        canvas = np.zeros((OVERLAY_HEIGHT, OVERLAY_WIDTH, 3), dtype=np.uint8)

        self._render_feed(canvas, frame_camara)
        self._render_panel(canvas)
        self._render_estado_sistema(canvas)
        self._calcular_fps()

        cv2.imshow(OVERLAY_WINDOW_NAME, canvas)

    # ---------------------------------------------------------
    #  Renderiza el feed de la camara en la parte superior
    # ---------------------------------------------------------

    def _render_feed(self, canvas, frame_camara):
        feed = cv2.resize(frame_camara, (OVERLAY_WIDTH, FEED_HEIGHT))
        canvas[0:FEED_HEIGHT, 0:OVERLAY_WIDTH] = feed

        cv2.rectangle(
            canvas,
            (0, 0),
            (OVERLAY_WIDTH - 1, FEED_HEIGHT - 1),
            COLOR_ACCENT,
            1,
        )

        indicador_color = (0, 200, 80) if not self.sistema_pausado else (80, 80, 200)
        cv2.circle(canvas, (OVERLAY_WIDTH - 14, 14), 5, indicador_color, -1)

        _draw_text(
            canvas,
            "EN VIVO" if not self.sistema_pausado else "PAUSADO",
            (12, 20),
            indicador_color,
            scale=0.45,
        )

    # ---------------------------------------------------------
    #  Renderiza el panel inferior con gesto y precision
    # ---------------------------------------------------------

    def _render_panel(self, canvas):
        cv2.rectangle(
            canvas,
            (0, PANEL_Y),
            (OVERLAY_WIDTH, OVERLAY_HEIGHT),
            COLOR_PANEL,
            -1,
        )

        cv2.line(
            canvas,
            (0, PANEL_Y),
            (OVERLAY_WIDTH, PANEL_Y),
            COLOR_ACCENT,
            1,
        )

        self._render_gesto(canvas)
        self._render_barra_precision(canvas)
        self._render_fps(canvas)

    # ---------------------------------------------------------
    #  Renderiza el nombre del gesto y la precision
    # ---------------------------------------------------------

    def _render_gesto(self, canvas):
        gesto     = self.ultimo_gesto     or "Esperando..."
        precision = self.ultima_precision

        label_color = _color_barra(precision) if self.ultimo_gesto else COLOR_GRAY

        cv2.putText(
            canvas,
            gesto,
            (BAR_X_START, PANEL_Y + 36),
            FONT,
            0.72,
            label_color,
            2,
            cv2.LINE_AA,
        )

        porcentaje = f"{precision * 100:.1f}%"
        cv2.putText(
            canvas,
            porcentaje,
            (BAR_X_START, PANEL_Y + 68),
            FONT,
            0.55,
            label_color,
            1,
            cv2.LINE_AA,
        )

        umbral_texto = f"Umbral: {int(CONFIDENCE_THRESHOLD * 100)}%"
        cv2.putText(
            canvas,
            umbral_texto,
            (OVERLAY_WIDTH - 100, PANEL_Y + 68),
            FONT,
            0.40,
            COLOR_GRAY,
            1,
            cv2.LINE_AA,
        )

    # ---------------------------------------------------------
    #  Renderiza la barra de precision
    # ---------------------------------------------------------

    def _render_barra_precision(self, canvas):
        cv2.rectangle(
            canvas,
            (BAR_X_START, BAR_Y),
            (BAR_X_END,   BAR_Y + BAR_HEIGHT),
            COLOR_DARK,
            -1,
        )

        precision     = self.ultima_precision
        fill_width    = int(BAR_WIDTH_TOTAL * precision)
        fill_width    = max(0, min(fill_width, BAR_WIDTH_TOTAL))
        color_barra   = _color_barra(precision)

        if fill_width > 0:
            cv2.rectangle(
                canvas,
                (BAR_X_START,              BAR_Y),
                (BAR_X_START + fill_width, BAR_Y + BAR_HEIGHT),
                color_barra,
                -1,
            )

        umbral_x = BAR_X_START + int(BAR_WIDTH_TOTAL * CONFIDENCE_THRESHOLD)
        cv2.line(
            canvas,
            (umbral_x, BAR_Y - 3),
            (umbral_x, BAR_Y + BAR_HEIGHT + 3),
            COLOR_WHITE,
            1,
        )

        cv2.rectangle(
            canvas,
            (BAR_X_START, BAR_Y),
            (BAR_X_END,   BAR_Y + BAR_HEIGHT),
            COLOR_GRAY,
            1,
        )

    # ---------------------------------------------------------
    #  Renderiza indicadores de estado del sistema
    #  Safe mode y sistema pausado
    # ---------------------------------------------------------

    def _render_estado_sistema(self, canvas):
        y_base = OVERLAY_HEIGHT - 14

        if self.safe_mode:
            cv2.putText(
                canvas,
                "SAFE MODE",
                (BAR_X_START, y_base),
                FONT,
                0.40,
                BAR_COLOR_LOW,
                1,
                cv2.LINE_AA,
            )

        texto_esc = "ESC para salir"
        cv2.putText(
            canvas,
            texto_esc,
            (OVERLAY_WIDTH - 110, y_base),
            FONT,
            0.38,
            COLOR_GRAY,
            1,
            cv2.LINE_AA,
        )

    # ---------------------------------------------------------
    #  Calcula y renderiza los FPS del overlay
    # ---------------------------------------------------------

    def _calcular_fps(self):
        self.fps_contador += 1
        ahora = time.time()

        if ahora - self.fps_tiempo >= 1.0:
            self.fps_actual  = self.fps_contador
            self.fps_contador = 0
            self.fps_tiempo  = ahora

    def _render_fps(self, canvas):
        cv2.putText(
            canvas,
            f"{int(self.fps_actual)} fps",
            (OVERLAY_WIDTH - 60, PANEL_Y + 20),
            FONT,
            0.38,
            COLOR_GRAY,
            1,
            cv2.LINE_AA,
        )

    # ---------------------------------------------------------
    #  Verifica si se presiono ESC para cerrar
    # ---------------------------------------------------------

    def verificar_salida(self):
        key = cv2.waitKey(1) & 0xFF
        return key == 27

    # ---------------------------------------------------------
    #  Cierra la ventana y libera recursos
    # ---------------------------------------------------------

    def cerrar(self):
        cv2.destroyWindow(OVERLAY_WINDOW_NAME)
        self.activo = False
        print("  Overlay cerrado.")