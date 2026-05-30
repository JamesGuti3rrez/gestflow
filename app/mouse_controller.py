# =============================================================
#  gestflow — app/mouse_controller.py
#  Recibe el nombre del gesto detectado y ejecuta
#  la accion correspondiente en el escritorio Windows
#  usando PyAutoGUI y pywin32
# =============================================================

import os
import sys
import time
import pyautogui
import pygetwindow as gw

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    GESTURES,
    CONFIDENCE_THRESHOLD,
    ACTION_COOLDOWN,
    SCROLL_SPEED,
    CURSOR_SMOOTHING,
    DRAG_THRESHOLD,
)


# -------------------------------------------------------------
#  Configuracion de PyAutoGUI
#  Desactiva el failsafe por esquina
#  Reduce el pause entre acciones al minimo
# -------------------------------------------------------------

pyautogui.FAILSAFE   = False
pyautogui.PAUSE      = 0.0


# -------------------------------------------------------------
#  Clase principal de control del raton
#  Mantiene estado interno para suavizado del cursor
#  y deteccion de arrastre
# -------------------------------------------------------------

class MouseController:

    def __init__(self):
        self.posicion_anterior_x  = None
        self.posicion_anterior_y  = None
        self.en_drag              = False
        self.sistema_pausado      = False
        self.safe_mode_activo     = False
        self.ultimo_gesto         = None
        self.ultima_accion_tiempo = 0.0

        ancho, alto               = pyautogui.size()
        self.pantalla_ancho       = ancho
        self.pantalla_alto        = alto

    # ---------------------------------------------------------
    #  Ejecuta la accion correspondiente al gesto recibido
    #  Verifica safe_mode y sistema_pausado antes de actuar
    # ---------------------------------------------------------

    def ejecutar(self, gesto, precision):
        if gesto is None:
            return

        if self.safe_mode_activo and gesto != "SAFE_MODE":
            return

        if self.sistema_pausado and gesto != "PAUSE":
            return

        accion = self._mapear_gesto(gesto)

        if accion is not None:
            accion(precision)
            self.ultimo_gesto = gesto

    # ---------------------------------------------------------
    #  Mapea el nombre del gesto a su metodo correspondiente
    # ---------------------------------------------------------

    def _mapear_gesto(self, gesto):
        mapa = {
            "MOVE_CURSOR" : self._move_cursor,
            "LEFT_CLICK"  : self._left_click,
            "RIGHT_CLICK" : self._right_click,
            "SCROLL"      : self._scroll,
            "DRAG"        : self._drag,
            "ALT_TAB"     : self._alt_tab,
            "PAUSE"       : self._pause,
            "SAFE_MODE"   : self._safe_mode,
            "OPEN_APP"    : self._open_app,
        }
        return mapa.get(gesto, None)

    # ---------------------------------------------------------
    #  MOVE_CURSOR
    #  Mueve el cursor con suavizado exponencial
    #  La posicion se calcula relativa al centro del frame
    # ---------------------------------------------------------

    def _move_cursor(self, precision):
        pos_actual_x, pos_actual_y = pyautogui.position()

        if self.posicion_anterior_x is None:
            self.posicion_anterior_x = pos_actual_x
            self.posicion_anterior_y = pos_actual_y
            return

        alpha    = 1.0 / CURSOR_SMOOTHING
        nuevo_x  = int(
            alpha * pos_actual_x +
            (1.0 - alpha) * self.posicion_anterior_x
        )
        nuevo_y  = int(
            alpha * pos_actual_y +
            (1.0 - alpha) * self.posicion_anterior_y
        )

        nuevo_x = max(0, min(nuevo_x, self.pantalla_ancho - 1))
        nuevo_y = max(0, min(nuevo_y, self.pantalla_alto  - 1))

        pyautogui.moveTo(nuevo_x, nuevo_y, duration=0.0)

        self.posicion_anterior_x = nuevo_x
        self.posicion_anterior_y = nuevo_y

    # ---------------------------------------------------------
    #  LEFT_CLICK
    #  Clic izquierdo en la posicion actual del cursor
    # ---------------------------------------------------------

    def _left_click(self, precision):
        if self.en_drag:
            pyautogui.mouseUp(button="left")
            self.en_drag = False

        pyautogui.click(button="left")
        self._resetear_posicion()

    # ---------------------------------------------------------
    #  RIGHT_CLICK
    #  Clic derecho en la posicion actual del cursor
    # ---------------------------------------------------------

    def _right_click(self, precision):
        pyautogui.click(button="right")
        self._resetear_posicion()

    # ---------------------------------------------------------
    #  SCROLL
    #  Scroll vertical hacia arriba
    #  La direccion se puede extender en futuras versiones
    # ---------------------------------------------------------

    def _scroll(self, precision):
        pyautogui.scroll(SCROLL_SPEED)
        self._resetear_posicion()

    # ---------------------------------------------------------
    #  DRAG
    #  Inicia o mantiene el arrastre del raton
    #  Solo actua si la precision supera DRAG_THRESHOLD
    # ---------------------------------------------------------

    def _drag(self, precision):
        if precision < DRAG_THRESHOLD:
            return

        if not self.en_drag:
            pyautogui.mouseDown(button="left")
            self.en_drag = True

    # ---------------------------------------------------------
    #  ALT_TAB
    #  Cambia a la ventana anterior con Alt+Tab
    # ---------------------------------------------------------

    def _alt_tab(self, precision):
        if self.en_drag:
            pyautogui.mouseUp(button="left")
            self.en_drag = False

        pyautogui.hotkey("alt", "tab")
        self._resetear_posicion()

    # ---------------------------------------------------------
    #  PAUSE
    #  Pausa o reanuda el sistema de gestos
    #  Libera el boton del raton si estaba en drag
    # ---------------------------------------------------------

    def _pause(self, precision):
        if self.en_drag:
            pyautogui.mouseUp(button="left")
            self.en_drag = False

        self.sistema_pausado = not self.sistema_pausado
        self._resetear_posicion()

        estado = "PAUSADO" if self.sistema_pausado else "ACTIVO"
        print(f"  Sistema {estado}")

    # ---------------------------------------------------------
    #  SAFE_MODE
    #  Activa o desactiva el modo seguro
    #  En modo seguro solo se detecta SAFE_MODE
    #  para desactivarlo
    # ---------------------------------------------------------

    def _safe_mode(self, precision):
        if self.en_drag:
            pyautogui.mouseUp(button="left")
            self.en_drag = False

        self.safe_mode_activo = not self.safe_mode_activo
        self._resetear_posicion()

        estado = "ACTIVADO" if self.safe_mode_activo else "DESACTIVADO"
        print(f"  Safe mode {estado}")

    # ---------------------------------------------------------
    #  OPEN_APP
    #  Abre el menu de inicio de Windows
    #  Se puede extender en config.py para abrir
    #  una aplicacion especifica
    # ---------------------------------------------------------

    def _open_app(self, precision):
        if self.en_drag:
            pyautogui.mouseUp(button="left")
            self.en_drag = False

        pyautogui.hotkey("win")
        self._resetear_posicion()

    # ---------------------------------------------------------
    #  Resetea la posicion anterior del cursor
    #  Se llama despues de acciones que no mueven el cursor
    # ---------------------------------------------------------

    def _resetear_posicion(self):
        self.posicion_anterior_x = None
        self.posicion_anterior_y = None

    # ---------------------------------------------------------
    #  Liberacion de recursos
    #  Asegura que el boton del raton quede libre
    # ---------------------------------------------------------

    def liberar(self):
        if self.en_drag:
            pyautogui.mouseUp(button="left")
            self.en_drag = False

        self.sistema_pausado  = False
        self.safe_mode_activo = False
        print("  MouseController liberado.")

    # ---------------------------------------------------------
    #  Estado actual del controlador
    #  Devuelve un diccionario con el estado interno
    #  Se usa para mostrarlo en el overlay
    # ---------------------------------------------------------

    def estado(self):
        return {
            "pausado"    : self.sistema_pausado,
            "safe_mode"  : self.safe_mode_activo,
            "en_drag"    : self.en_drag,
            "ultimo_gesto": self.ultimo_gesto,
        }