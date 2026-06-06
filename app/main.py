# =============================================================
#  gestflow — app/main.py
#  Punto de entrada principal del sistema
# =============================================================

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))

from config import (
    MODEL_PATH,
    OVERLAY_FPS,
    GESTURES,
    NUM_CLASSES,
    CONFIDENCE_THRESHOLD,
    PANIC_HOTKEY,
    KILL_HOTKEY,
)

from gesture_recognizer import (
    GestureRecognizer,
    CameraCapture,
    cargar_modelo,
    calentar_modelo,
)
from mouse_controller import MouseController
from overlay          import Overlay


# Bandera global de apagado de emergencia, activada por el hotkey KILL.
solicitud_apagado = {"activo": False}


def registrar_hotkeys_globales(controller):
    # Hotkeys de seguridad que funcionan aunque el overlay no tenga foco.
    # Si la captura global falla (p. ej. sin permisos), el sistema sigue
    # operando con ESC sobre la ventana como salida.
    try:
        import keyboard
    except Exception as e:
        print(f"  Aviso: hotkeys globales no disponibles ({e}).")
        print("  Usa ESC sobre la ventana o el gesto PAUSE.")
        return

    def _panico():
        controller.parada_emergencia()

    def _kill():
        controller.parada_emergencia()
        solicitud_apagado["activo"] = True

    try:
        keyboard.add_hotkey(PANIC_HOTKEY, _panico)
        keyboard.add_hotkey(KILL_HOTKEY,  _kill)
        print(f"  Hotkey de panico  : {PANIC_HOTKEY} (pausa + suelta raton)")
        print(f"  Hotkey de apagado : {KILL_HOTKEY}")
    except Exception as e:
        print(f"  Aviso: no se pudieron registrar hotkeys globales ({e}).")


def liberar_hotkeys_globales():
    try:
        import keyboard
        keyboard.unhook_all_hotkeys()
    except Exception:
        pass


def separador(titulo=""):
    linea = "=" * 60
    if titulo:
        print(f"\n{linea}")
        print(f"  {titulo}")
        print(f"{linea}")
    else:
        print(f"\n{linea}")


def inicializar():
    separador("gestflow — iniciando sistema")
    print(f"\n  Clases     : {NUM_CLASSES}")
    print(f"  Gestos     : {', '.join(GESTURES)}")
    print(f"  Modelo     : {MODEL_PATH}")
    print(f"  Umbral     : {int(CONFIDENCE_THRESHOLD * 100)}%")

    separador("Cargando componentes")

    print("\n  Cargando modelo...")
    model = cargar_modelo()
    calentar_modelo(model)

    print("\n  Iniciando camara...")
    camara = CameraCapture()
    camara.iniciar()

    print("\n  Iniciando reconocedor de gestos...")
    recognizer = GestureRecognizer(model)

    print("\n  Iniciando controlador del raton...")
    controller = MouseController()

    print("\n  Iniciando overlay...")
    overlay = Overlay()
    overlay.iniciar()

    print("\n  Registrando hotkeys de seguridad...")
    registrar_hotkeys_globales(controller)

    separador("Sistema listo")
    print("\n  Realiza un gesto frente a la camara.")
    print("  Presiona ESC para salir.")
    print("  Gesto PAUSE para pausar el sistema.")
    print("  Gesto SAFE_MODE para activar modo seguro.")
    separador()

    return model, camara, recognizer, controller, overlay


def loop_principal(camara, recognizer, controller, overlay):
    intervalo        = 1.0 / OVERLAY_FPS
    tiempo_inicio    = time.time()
    frames_totales   = 0
    acciones_totales = 0

    overlay.actualizar_estado(None, 0.0, False, controller.estado())

    while True:
        t_inicio_frame = time.time()

        ret, frame = camara.leer()
        if not ret:
            print("\n  Error: No se pudo leer el frame de la camara.")
            break

        buffer_listo = recognizer.agregar_frame(frame)

        if buffer_listo:
            gesto, precision = recognizer.inferir()

            if gesto is not None:
                direccion      = recognizer.calcular_direccion()
                puede_discreto = recognizer.puede_ejecutar_discreto()
                controller.ejecutar(gesto, precision, direccion, puede_discreto)
                acciones_totales += 1

        gesto_display, precision_display, hay_deteccion = recognizer.estado_actual()
        estado_controller = controller.estado()

        overlay.actualizar_estado(
            gesto_display,
            precision_display,
            hay_deteccion,
            estado_controller,
        )
        overlay.renderizar(frame)

        if overlay.verificar_salida():
            print("\n  ESC presionado. Cerrando sistema...")
            break

        if solicitud_apagado["activo"]:
            print("\n  Apagado de emergencia solicitado. Cerrando sistema...")
            break

        frames_totales += 1

        t_espera = intervalo - (time.time() - t_inicio_frame)
        if t_espera > 0:
            time.sleep(t_espera)

    tiempo_total = time.time() - tiempo_inicio
    return frames_totales, acciones_totales, tiempo_total


def liberar_recursos(camara, controller, overlay):
    separador("Cerrando sistema")
    print("\n  Liberando recursos...")

    liberar_hotkeys_globales()

    if controller is not None:
        try:
            controller.liberar()
        except Exception as e:
            print(f"  Error al liberar controller: {e}")

    if camara is not None:
        try:
            camara.liberar()
        except Exception as e:
            print(f"  Error al liberar camara: {e}")

    if overlay is not None:
        try:
            overlay.cerrar()
        except Exception as e:
            print(f"  Error al cerrar overlay: {e}")

    print("  Recursos liberados correctamente.")


def reporte_sesion(frames_totales, acciones_totales, tiempo_total):
    separador("Reporte de sesion")
    minutos  = int(tiempo_total // 60)
    segundos = int(tiempo_total  % 60)
    fps_prom = frames_totales / tiempo_total if tiempo_total > 0 else 0
    print(f"\n  Duracion de la sesion : {minutos:02d}m {segundos:02d}s")
    print(f"  Frames procesados     : {frames_totales:,}")
    print(f"  FPS promedio          : {fps_prom:.1f}")
    print(f"  Acciones ejecutadas   : {acciones_totales:,}")
    separador()


def verificar_requisitos():
    if not os.path.exists(MODEL_PATH):
        separador("Errores encontrados")
        print(f"\n  Modelo no encontrado en: {MODEL_PATH}")
        print(f"  Ejecuta training/02_train.py primero.")
        separador()
        sys.exit(1)


def main():
    verificar_requisitos()

    camara           = None
    controller       = None
    overlay          = None
    frames_totales   = 0
    acciones_totales = 0
    tiempo_total     = 0.0

    try:
        model, camara, recognizer, controller, overlay = inicializar()
        frames_totales, acciones_totales, tiempo_total = loop_principal(
            camara, recognizer, controller, overlay
        )
    except KeyboardInterrupt:
        print("\n\n  Interrupcion manual detectada.")
    except Exception as e:
        print(f"\n  Error inesperado: {e}")
        raise
    finally:
        liberar_recursos(camara, controller, overlay)
        if frames_totales > 0:
            reporte_sesion(frames_totales, acciones_totales, tiempo_total)


if __name__ == "__main__":
    main()