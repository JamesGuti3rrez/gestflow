# =============================================================
#  gestflow — app/mouse_controller.py
#  Ejecuta la accion del gesto confirmado por votacion.
#  MOVE_CURSOR funciona como joystick direccional usando
#  la direccion del movimiento de la mano.
# =============================================================

import os
import sys
import pyautogui

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    SCROLL_SPEED,
    CURSOR_SMOOTHING,
    CURSOR_SPEED,
    CURSOR_DEAD_ZONE,
    DRAG_THRESHOLD,
)


pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0.0


class MouseController:

    def __init__(self):
        self.dx_anterior      = 0.0
        self.dy_anterior      = 0.0
        self.en_drag          = False
        self.sistema_pausado  = False
        self.safe_mode_activo = False
        self.ultimo_gesto     = None

        ancho, alto          = pyautogui.size()
        self.pantalla_ancho  = ancho
        self.pantalla_alto   = alto

    # ---------------------------------------------------------
    #  Ejecuta la accion del gesto confirmado
    #  Algunos gestos son continuos (cursor, drag, scroll) y
    #  otros discretos (clicks, alt+tab). Los discretos solo
    #  se ejecutan si el recognizer da permiso (cooldown).
    # ---------------------------------------------------------

    def ejecutar(self, gesto, precision, direccion, puede_discreto):
        if gesto is None:
            return

        if self.safe_mode_activo and gesto != "SAFE_MODE":
            return

        if self.sistema_pausado and gesto != "PAUSE":
            return

        gestos_continuos = {"MOVE_CURSOR", "DRAG", "SCROLL"}
        gestos_discretos = {
            "LEFT_CLICK", "RIGHT_CLICK", "ALT_TAB",
            "PAUSE", "SAFE_MODE", "OPEN_APP"
        }

        if gesto in gestos_continuos:
            self._ejecutar_continuo(gesto, precision, direccion)
        elif gesto in gestos_discretos:
            if puede_discreto:
                self._ejecutar_discreto(gesto, precision)

        self.ultimo_gesto = gesto

    # ---------------------------------------------------------
    #  Gestos continuos
    # ---------------------------------------------------------

    def _ejecutar_continuo(self, gesto, precision, direccion):
        if gesto == "MOVE_CURSOR":
            self._move_cursor(direccion)
        elif gesto == "DRAG":
            self._drag(precision, direccion)
        elif gesto == "SCROLL":
            self._scroll(direccion)

    # ---------------------------------------------------------
    #  Gestos discretos
    # ---------------------------------------------------------

    def _ejecutar_discreto(self, gesto, precision):
        if gesto == "LEFT_CLICK":
            self._left_click()
        elif gesto == "RIGHT_CLICK":
            self._right_click()
        elif gesto == "ALT_TAB":
            self._alt_tab()
        elif gesto == "PAUSE":
            self._pause()
        elif gesto == "SAFE_MODE":
            self._safe_mode()
        elif gesto == "OPEN_APP":
            self._open_app()

    # ---------------------------------------------------------
    #  MOVE_CURSOR — joystick direccional
    #  Mueve el cursor en la direccion del movimiento de la
    #  mano. Zona muerta para evitar movimiento por ruido.
    # ---------------------------------------------------------

    def _move_cursor(self, direccion):
        dx, dy = direccion

        if abs(dx) < CURSOR_DEAD_ZONE and abs(dy) < CURSOR_DEAD_ZONE:
            return

        alpha    = 1.0 / CURSOR_SMOOTHING
        dx_suave = alpha * dx + (1.0 - alpha) * self.dx_anterior
        dy_suave = alpha * dy + (1.0 - alpha) * self.dy_anterior
        self.dx_anterior = dx_suave
        self.dy_anterior = dy_suave

        mov_x = int(dx_suave * CURSOR_SPEED)
        mov_y = int(dy_suave * CURSOR_SPEED)

        if mov_x == 0 and mov_y == 0:
            return

        pyautogui.moveRel(mov_x, mov_y, duration=0.0)

    # ---------------------------------------------------------
    #  DRAG — arrastra mientras mantiene el pellizco
    # ---------------------------------------------------------

    def _drag(self, precision, direccion):
        if precision < DRAG_THRESHOLD:
            return

        if not self.en_drag:
            pyautogui.mouseDown(button="left")
            self.en_drag = True

        dx, dy = direccion
        if abs(dx) < CURSOR_DEAD_ZONE and abs(dy) < CURSOR_DEAD_ZONE:
            return

        alpha    = 1.0 / CURSOR_SMOOTHING
        dx_suave = alpha * dx + (1.0 - alpha) * self.dx_anterior
        dy_suave = alpha * dy + (1.0 - alpha) * self.dy_anterior
        self.dx_anterior = dx_suave
        self.dy_anterior = dy_suave

        mov_x = int(dx_suave * CURSOR_SPEED)
        mov_y = int(dy_suave * CURSOR_SPEED)
        if mov_x != 0 or mov_y != 0:
            pyautogui.moveRel(mov_x, mov_y, duration=0.0)

    # ---------------------------------------------------------
    #  SCROLL — direccion vertical decide arriba o abajo
    # ---------------------------------------------------------

    def _scroll(self, direccion):
        _, dy = direccion
        if dy < -0.1:
            pyautogui.scroll(SCROLL_SPEED)
        elif dy > 0.1:
            pyautogui.scroll(-SCROLL_SPEED)

    # ---------------------------------------------------------
    #  LEFT_CLICK
    # ---------------------------------------------------------

    def _left_click(self):
        if self.en_drag:
            pyautogui.mouseUp(button="left")
            self.en_drag = False
        pyautogui.click(button="left")
        self._resetear_direccion()

    # ---------------------------------------------------------
    #  RIGHT_CLICK
    # ---------------------------------------------------------

    def _right_click(self):
        pyautogui.click(button="right")
        self._resetear_direccion()

    # ---------------------------------------------------------
    #  ALT_TAB
    # ---------------------------------------------------------

    def _alt_tab(self):
        if self.en_drag:
            pyautogui.mouseUp(button="left")
            self.en_drag = False
        pyautogui.hotkey("alt", "tab")
        self._resetear_direccion()

    # ---------------------------------------------------------
    #  PAUSE
    # ---------------------------------------------------------

    def _pause(self):
        if self.en_drag:
            pyautogui.mouseUp(button="left")
            self.en_drag = False
        self.sistema_pausado = not self.sistema_pausado
        self._resetear_direccion()
        estado = "PAUSADO" if self.sistema_pausado else "ACTIVO"
        print(f"  Sistema {estado}")

    # ---------------------------------------------------------
    #  SAFE_MODE
    # ---------------------------------------------------------

    def _safe_mode(self):
        if self.en_drag:
            pyautogui.mouseUp(button="left")
            self.en_drag = False
        self.safe_mode_activo = not self.safe_mode_activo
        self._resetear_direccion()
        estado = "ACTIVADO" if self.safe_mode_activo else "DESACTIVADO"
        print(f"  Safe mode {estado}")

    # ---------------------------------------------------------
    #  OPEN_APP — doble click
    # ---------------------------------------------------------

    def _open_app(self):
        if self.en_drag:
            pyautogui.mouseUp(button="left")
            self.en_drag = False
        pyautogui.doubleClick()
        self._resetear_direccion()

    # ---------------------------------------------------------
    #  Utilidades
    # ---------------------------------------------------------

    def _resetear_direccion(self):
        self.dx_anterior = 0.0
        self.dy_anterior = 0.0

    def liberar(self):
        if self.en_drag:
            pyautogui.mouseUp(button="left")
            self.en_drag = False
        self.sistema_pausado  = False
        self.safe_mode_activo = False
        print("  MouseController liberado.")

    def estado(self):
        return {
            "pausado"      : self.sistema_pausado,
            "safe_mode"    : self.safe_mode_activo,
            "en_drag"      : self.en_drag,
            "ultimo_gesto" : self.ultimo_gesto,
        }