# =============================================================
#  gestflow — app/main.py
#  Punto de entrada principal del sistema
#  Inicializa la camara, carga el modelo, instancia
#  gesture_recognizer, mouse_controller y overlay
#  y corre el loop principal
#
#  Uso:
#      python app/main.py
# =============================================================

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL_PATH,
    OVERLAY_FPS,
    GESTURES,
    NUM_CLASSES,
    CONFIDENCE_THRESHOLD,
)

from gesture_recognizer import (
    GestureRecognizer,
    CameraCapture,
    cargar_modelo,
    calentar_modelo,
)
from mouse_controller import MouseController
from overlay          import Overlay


# -------------------------------------------------------------
#  Separador visual para consola
# -------------------------------------------------------------

def separador(titulo=""):
    linea = "=" * 60
    if titulo:
        print(f"\n{linea}")
        print(f"  {titulo}")
        print(f"{linea}")
    else:
        print(f"\n{linea}")


# -------------------------------------------------------------
#  Inicializacion de todos los componentes
#  Se ejecuta una sola vez al arrancar
# -------------------------------------------------------------

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

    separador("Sistema listo")
    print("\n  Realiza un gesto frente a la camara.")
    print("  Presiona ESC para salir.")
    print("  Gesto PAUSE para pausar el sistema.")
    print("  Gesto SAFE_MODE para activar modo seguro.")
    separador()

    return model, camara, recognizer, controller, overlay


# -------------------------------------------------------------
#  Loop principal
#  Captura frames, infiere gestos, ejecuta acciones
#  y actualiza el overlay en cada iteracion
# -------------------------------------------------------------

def loop_principal(camara, recognizer, controller, overlay):
    intervalo     = 1.0 / OVERLAY_FPS
    tiempo_inicio = time.time()
    frames_totales = 0
    acciones_totales = 0

    while True:
        t_inicio_frame = time.time()

        ret, frame = camara.leer()

        if not ret:
            print("\n  Error: No se pudo leer el frame de la camara.")
            break

        buffer_listo = recognizer.agregar_frame(frame)

        gesto     = None
        precision = 0.0

        if buffer_listo:
            gesto, precision = recognizer.inferir()

            if gesto is not None:
                controller.ejecutar(gesto, precision)
                acciones_totales += 1

        gesto_display, precision_display = recognizer.estado_actual()

        estado_controller = controller.estado()

        overlay.actualizar_estado(
            gesto_display,
            precision_display,
            estado_controller,
        )

        overlay.renderizar(frame)

        if overlay.verificar_salida():
            print("\n  ESC presionado. Cerrando sistema...")
            break

        frames_totales += 1

        t_fin_frame  = time.time()
        t_transcurrido = t_fin_frame - t_inicio_frame
        t_espera     = intervalo - t_transcurrido

        if t_espera > 0:
            time.sleep(t_espera)

    tiempo_total = time.time() - tiempo_inicio

    return frames_totales, acciones_totales, tiempo_total


# -------------------------------------------------------------
#  Liberacion de todos los recursos
#  Se ejecuta siempre al cerrar, incluso si hay error
# -------------------------------------------------------------

def liberar_recursos(camara, controller, overlay):
    separador("Cerrando sistema")

    print("\n  Liberando recursos...")

    try:
        controller.liberar()
    except Exception as e:
        print(f"  Error al liberar controller: {e}")

    try:
        camara.liberar()
    except Exception as e:
        print(f"  Error al liberar camara: {e}")

    try:
        overlay.cerrar()
    except Exception as e:
        print(f"  Error al cerrar overlay: {e}")

    print("  Recursos liberados correctamente.")


# -------------------------------------------------------------
#  Reporte final de sesion
#  Muestra estadisticas de uso al cerrar
# -------------------------------------------------------------

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


# -------------------------------------------------------------
#  Verificacion previa al inicio
#  Comprueba que el modelo existe antes de arrancar
# -------------------------------------------------------------

def verificar_requisitos():
    errores = []

    if not os.path.exists(MODEL_PATH):
        errores.append(
            f"Modelo no encontrado en: {MODEL_PATH}\n"
            f"  Ejecuta training/02_train.py primero."
        )

    if errores:
        separador("Errores encontrados")
        for error in errores:
            print(f"\n  {error}")
        separador()
        sys.exit(1)


# -------------------------------------------------------------
#  Main
# -------------------------------------------------------------

def main():
    verificar_requisitos()

    camara     = None
    controller = None
    overlay    = None

    try:
        model, camara, recognizer, controller, overlay = inicializar()

        frames_totales, acciones_totales, tiempo_total = loop_principal(
            camara,
            recognizer,
            controller,
            overlay,
        )

    except KeyboardInterrupt:
        print("\n\n  Interrupcion manual detectada.")

    except Exception as e:
        print(f"\n  Error inesperado: {e}")
        raise

    finally:
        if camara or controller or overlay:
            liberar_recursos(camara, controller, overlay)

        if "frames_totales" in dir():
            reporte_sesion(frames_totales, acciones_totales, tiempo_total)


if __name__ == "__main__":
    main()