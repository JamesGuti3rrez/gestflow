# =============================================================
#  gestflow — training/01_record.py
#  Script de grabacion del dataset
#  Muestra un recuadro fijo donde el usuario pone la mano
#  Solo graba esa zona, ignorando cara y cuerpo
#
#  Uso:
#      python training/01_record.py
# =============================================================

import os
import sys
import cv2
import numpy as np
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATASET_DIR,
    GESTURES,
    CAMERA_INDEX,
    VIDEO_FPS,
    VIDEO_DURATION,
    VIDEOS_PER_GESTURE,
    RECORD_WIDTH,
    RECORD_HEIGHT,
    COUNTDOWN_SECONDS,
    RECORDING_COLOR,
    READY_COLOR,
    FONT_SCALE,
    FONT_THICKNESS,
    RECORDER_NAME,
    ROI_X1,
    ROI_Y1,
    ROI_X2,
    ROI_Y2,
)


# -------------------------------------------------------------
#  Constantes visuales
# -------------------------------------------------------------

FONT         = cv2.FONT_HERSHEY_SIMPLEX
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0,   0,   0  )
COLOR_YELLOW = (0,   255, 255)
COLOR_GREEN  = (0,   255, 0  )
COLOR_RED    = (0,   0,   255)
FRAME_TOTAL  = int(VIDEO_FPS * VIDEO_DURATION)


# -------------------------------------------------------------
#  Dibuja texto con fondo negro
# -------------------------------------------------------------

def draw_text(frame, texto, pos, color=COLOR_WHITE,
              scale=None, thickness=None):
    scale     = scale     or FONT_SCALE
    thickness = thickness or FONT_THICKNESS

    (tw, th), _ = cv2.getTextSize(texto, FONT, scale, thickness)
    x, y        = pos

    cv2.rectangle(frame, (x - 4, y - th - 4),
                  (x + tw + 4, y + 4), COLOR_BLACK, -1)
    cv2.putText(frame, texto, (x, y), FONT, scale,
                color, thickness, cv2.LINE_AA)


# -------------------------------------------------------------
#  Dibuja el recuadro de la zona de grabacion
# -------------------------------------------------------------

def dibujar_roi(frame, grabando=False):
    color = RECORDING_COLOR if grabando else COLOR_YELLOW
    grosor = 3

    cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), color, grosor)

    esquina = 20
    cv2.line(frame, (ROI_X1, ROI_Y1), (ROI_X1 + esquina, ROI_Y1), color, grosor + 1)
    cv2.line(frame, (ROI_X1, ROI_Y1), (ROI_X1, ROI_Y1 + esquina), color, grosor + 1)
    cv2.line(frame, (ROI_X2, ROI_Y1), (ROI_X2 - esquina, ROI_Y1), color, grosor + 1)
    cv2.line(frame, (ROI_X2, ROI_Y1), (ROI_X2, ROI_Y1 + esquina), color, grosor + 1)
    cv2.line(frame, (ROI_X1, ROI_Y2), (ROI_X1 + esquina, ROI_Y2), color, grosor + 1)
    cv2.line(frame, (ROI_X1, ROI_Y2), (ROI_X1, ROI_Y2 - esquina), color, grosor + 1)
    cv2.line(frame, (ROI_X2, ROI_Y2), (ROI_X2 - esquina, ROI_Y2), color, grosor + 1)
    cv2.line(frame, (ROI_X2, ROI_Y2), (ROI_X2, ROI_Y2 - esquina), color, grosor + 1)

    if not grabando:
        draw_text(frame, "PON TU MANO AQUI",
                  (ROI_X1 + 10, ROI_Y1 + 30),
                  COLOR_YELLOW, scale=0.6)


# -------------------------------------------------------------
#  Recorta la zona de la mano del frame completo
#  y la redimensiona al tamano de grabacion
# -------------------------------------------------------------

def recortar_roi(frame):
    recorte = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
    if recorte.size == 0:
        return cv2.resize(frame, (RECORD_WIDTH, RECORD_HEIGHT))
    return cv2.resize(recorte, (RECORD_WIDTH, RECORD_HEIGHT))


# -------------------------------------------------------------
#  Creacion de carpetas del dataset
# -------------------------------------------------------------

def crear_carpetas():
    for gesto in GESTURES:
        path = os.path.join(DATASET_DIR, gesto)
        os.makedirs(path, exist_ok=True)
    print("  Carpetas del dataset verificadas.")


# -------------------------------------------------------------
#  Conteo de videos existentes por gesto
# -------------------------------------------------------------

def contar_videos(gesto):
    path = os.path.join(DATASET_DIR, gesto)
    if not os.path.exists(path):
        return 0
    return len([f for f in os.listdir(path) if f.lower().endswith(".mp4")])


# -------------------------------------------------------------
#  Siguiente numero de video disponible
# -------------------------------------------------------------

def siguiente_numero(gesto):
    path    = os.path.join(DATASET_DIR, gesto)
    prefijo = f"{RECORDER_NAME}_{gesto.lower()}_"
    numeros = []

    if os.path.exists(path):
        for f in os.listdir(path):
            if f.lower().endswith(".mp4") and f.startswith(prefijo):
                try:
                    numeros.append(
                        int(f.replace(prefijo, "").replace(".mp4", ""))
                    )
                except ValueError:
                    continue

    return max(numeros) + 1 if numeros else 1


# -------------------------------------------------------------
#  Pantalla de bienvenida
# -------------------------------------------------------------

def pantalla_bienvenida(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        overlay_h = min(60 + len(GESTURES) * 36 + 60, h)
        panel     = frame.copy()
        cv2.rectangle(panel, (0, 0), (w, overlay_h), COLOR_BLACK, -1)
        cv2.addWeighted(panel, 0.6, frame, 0.4, 0, frame)

        draw_text(frame, f"gestflow — Grabacion ({RECORDER_NAME})",
                  (20, 36), COLOR_YELLOW, scale=0.8)

        for i, gesto in enumerate(GESTURES):
            count  = contar_videos(gesto)
            estado = f"{count:3d} / {VIDEOS_PER_GESTURE}"
            color  = READY_COLOR if count >= VIDEOS_PER_GESTURE else COLOR_WHITE
            draw_text(frame, f"  {gesto:<20} {estado}",
                      (20, 72 + i * 36), color, scale=0.65)

        draw_text(frame,
                  "ESPACIO para empezar  |  ESC para salir",
                  (20, overlay_h - 16), COLOR_YELLOW, scale=0.6)

        dibujar_roi(frame)

        cv2.imshow("gestflow — grabacion", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            return False
        if key == 32:
            return True


# -------------------------------------------------------------
#  Pantalla de instrucciones por gesto
# -------------------------------------------------------------

def pantalla_gesto(cap, gesto, numero, total):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        dibujar_roi(frame)

        panel = frame.copy()
        cv2.rectangle(panel, (0, 0), (w, 150), COLOR_BLACK, -1)
        cv2.addWeighted(panel, 0.65, frame, 0.35, 0, frame)

        draw_text(frame, f"Gesto : {gesto}",
                  (20, 36), COLOR_YELLOW, scale=0.9)
        draw_text(frame, f"Video : {numero} de {total}",
                  (20, 72), COLOR_WHITE, scale=0.7)
        draw_text(frame,
                  "ESPACIO grabar  |  S saltar  |  ESC salir",
                  (20, 108), COLOR_YELLOW, scale=0.55)
        draw_text(frame,
                  "Manten la mano dentro del recuadro amarillo",
                  (20, h - 20), COLOR_YELLOW, scale=0.55)

        cv2.imshow("gestflow — grabacion", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            return "salir"
        if key == ord("s"):
            return "saltar"
        if key == 32:
            return "grabar"


# -------------------------------------------------------------
#  Cuenta regresiva antes de grabar
# -------------------------------------------------------------

def cuenta_regresiva(cap, gesto):
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        t_inicio = time.time()
        while time.time() - t_inicio < 1.0:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            dibujar_roi(frame)

            draw_text(frame, f"Gesto : {gesto}",
                      (20, 40), COLOR_YELLOW, scale=0.9)
            draw_text(frame, f"Grabando en : {i}",
                      (w // 2 - 120, h // 2),
                      COLOR_YELLOW, scale=1.2, thickness=3)
            draw_text(frame,
                      "Manten el gesto formado dentro del recuadro",
                      (20, h - 20), COLOR_WHITE, scale=0.55)

            cv2.imshow("gestflow — grabacion", frame)
            cv2.waitKey(1)


# -------------------------------------------------------------
#  Grabacion de un video
#  Recorta solo la zona del recuadro y la guarda
# -------------------------------------------------------------

def grabar_video(cap, gesto, numero):
    nombre = f"{RECORDER_NAME}_{gesto.lower()}_{numero:03d}.mp4"
    ruta   = os.path.join(DATASET_DIR, gesto, nombre)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        ruta, fourcc, VIDEO_FPS,
        (RECORD_WIDTH, RECORD_HEIGHT)
    )

    frames_grabados = 0

    while frames_grabados < FRAME_TOTAL:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        recorte = recortar_roi(frame)
        writer.write(recorte)

        display  = frame.copy()
        progreso = int((frames_grabados / FRAME_TOTAL) * (ROI_X2 - ROI_X1))

        dibujar_roi(display, grabando=True)

        cv2.rectangle(display,
                      (ROI_X1, ROI_Y2 + 5),
                      (ROI_X2, ROI_Y2 + 15),
                      COLOR_BLACK, -1)
        cv2.rectangle(display,
                      (ROI_X1, ROI_Y2 + 5),
                      (ROI_X1 + progreso, ROI_Y2 + 15),
                      RECORDING_COLOR, -1)

        draw_text(display, "GRABANDO",
                  (20, 30), RECORDING_COLOR, scale=0.7)
        draw_text(display, f"{gesto}",
                  (20, 60), COLOR_WHITE, scale=0.6)

        cv2.imshow("gestflow — grabacion", display)
        cv2.waitKey(1)

        frames_grabados += 1

    writer.release()
    return ruta


# -------------------------------------------------------------
#  Pantalla de confirmacion post grabacion
# -------------------------------------------------------------

def pantalla_confirmacion(cap, ruta_video):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        dibujar_roi(frame)

        panel = frame.copy()
        cv2.rectangle(panel, (0, h - 100), (w, h), COLOR_BLACK, -1)
        cv2.addWeighted(panel, 0.7, frame, 0.3, 0, frame)

        draw_text(frame, "Video guardado correctamente",
                  (20, h - 76), READY_COLOR, scale=0.7)
        draw_text(frame,
                  "ESPACIO continuar  |  R repetir  |  ESC salir",
                  (20, h - 36), COLOR_YELLOW, scale=0.6)

        cv2.imshow("gestflow — grabacion", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            return "salir"
        if key == ord("r"):
            if os.path.exists(ruta_video):
                os.remove(ruta_video)
            return "repetir"
        if key == 32:
            return "continuar"


# -------------------------------------------------------------
#  Resumen final
# -------------------------------------------------------------

def resumen_final():
    print("\n  Resumen de grabacion:")
    print(f"  {'Gesto':<22} {'Grabados':>10} {'Faltantes':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10}")

    total_grabados  = 0
    total_faltantes = 0

    for gesto in GESTURES:
        grabados        = contar_videos(gesto)
        faltantes       = max(0, VIDEOS_PER_GESTURE - grabados)
        total_grabados  += grabados
        total_faltantes += faltantes
        print(f"  {gesto:<22} {grabados:>10} {faltantes:>10}")

    print(f"  {'-'*22} {'-'*10} {'-'*10}")
    print(f"  {'TOTAL':<22} {total_grabados:>10} {total_faltantes:>10}")

    if total_faltantes == 0:
        print("\n  Dataset completo. Puedes ejecutar 02_train.py")
    else:
        print(f"\n  Faltan {total_faltantes} videos para completar el dataset.")


# -------------------------------------------------------------
#  Main
# -------------------------------------------------------------

def main():
    print("\n  gestflow — Script de grabacion")
    print(f"  Usuario      : {RECORDER_NAME}")
    print(f"  Zona de mano : ({ROI_X1},{ROI_Y1}) → ({ROI_X2},{ROI_Y2})")
    print("  Inicializando camara...")

    crear_carpetas()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RECORD_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RECORD_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          VIDEO_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print(f"\n  Error: No se pudo abrir la camara (indice {CAMERA_INDEX})")
        sys.exit(1)

    ret, _ = cap.read()
    if not ret:
        print("\n  Error: La camara se abrio pero no devuelve frames.")
        cap.release()
        sys.exit(1)

    print("  Camara lista.")

    continuar = pantalla_bienvenida(cap)
    if not continuar:
        cap.release()
        cv2.destroyAllWindows()
        return

    for gesto in GESTURES:
        videos_existentes = contar_videos(gesto)

        if videos_existentes >= VIDEOS_PER_GESTURE:
            print(f"\n  {gesto} ya tiene {videos_existentes} videos. Saltando.")
            continue

        videos_restantes = VIDEOS_PER_GESTURE - videos_existentes
        print(f"\n  Grabando {gesto} ({videos_restantes} videos restantes)")

        saltar_gesto = False

        while contar_videos(gesto) < VIDEOS_PER_GESTURE:
            numero = siguiente_numero(gesto)
            accion = pantalla_gesto(cap, gesto, numero, VIDEOS_PER_GESTURE)

            if accion == "salir":
                cap.release()
                cv2.destroyAllWindows()
                resumen_final()
                return

            if accion == "saltar":
                saltar_gesto = True
                break

            cuenta_regresiva(cap, gesto)
            ruta = grabar_video(cap, gesto, numero)
            confirmacion = pantalla_confirmacion(cap, ruta)

            if confirmacion == "salir":
                cap.release()
                cv2.destroyAllWindows()
                resumen_final()
                return

            if confirmacion == "repetir":
                continue

        if saltar_gesto:
            print(f"  {gesto} saltado.")
            continue

        print(f"  {gesto} completado.")

    cap.release()
    cv2.destroyAllWindows()
    resumen_final()


if __name__ == "__main__":
    main()