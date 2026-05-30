# =============================================================
#  gestflow — training/01_record.py
#  Script de grabacion del dataset
#  Guia al usuario gesto por gesto con cuenta regresiva
#  Guarda los videos en dataset/raw_videos/[GESTO]/
#
#  Uso:
#      python training/01_record.py
# =============================================================

import os
import sys
import cv2
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATASET_DIR,
    GESTURES,
    CAMERA_INDEX,
    VIDEO_FPS,
    VIDEO_DURATION,
    VIDEOS_PER_GESTURE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    RECORD_WIDTH,
    RECORD_HEIGHT,
    COUNTDOWN_SECONDS,
    RECORDING_COLOR,
    READY_COLOR,
    FONT_SCALE,
    FONT_THICKNESS,
)


# -------------------------------------------------------------
#  Constantes visuales internas
# -------------------------------------------------------------

FONT          = cv2.FONT_HERSHEY_SIMPLEX
COLOR_WHITE   = (255, 255, 255)
COLOR_BLACK   = (0,   0,   0  )
COLOR_YELLOW  = (0,   255, 255)
FRAME_TOTAL   = int(VIDEO_FPS * VIDEO_DURATION)


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
    path   = os.path.join(DATASET_DIR, gesto)
    videos = [
        f for f in os.listdir(path)
        if f.lower().endswith(".mp4")
    ]
    return len(videos)


# -------------------------------------------------------------
#  Siguiente numero de video disponible
# -------------------------------------------------------------

def siguiente_numero(gesto):
    return contar_videos(gesto) + 1


# -------------------------------------------------------------
#  Dibuja texto con fondo negro para legibilidad
# -------------------------------------------------------------

def draw_text(frame, texto, pos, color=COLOR_WHITE,
              scale=None, thickness=None):
    scale     = scale     or FONT_SCALE
    thickness = thickness or FONT_THICKNESS

    (tw, th), _ = cv2.getTextSize(texto, FONT, scale, thickness)
    x, y        = pos

    cv2.rectangle(
        frame,
        (x - 4, y - th - 4),
        (x + tw + 4, y + 4),
        COLOR_BLACK,
        -1,
    )
    cv2.putText(
        frame, texto, (x, y),
        FONT, scale, color, thickness,
        cv2.LINE_AA,
    )


# -------------------------------------------------------------
#  Pantalla de bienvenida
#  Muestra estado de cada gesto antes de empezar
# -------------------------------------------------------------

def pantalla_bienvenida(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        overlay        = frame.copy()
        overlay_height = min(60 + len(GESTURES) * 36 + 60, h)
        cv2.rectangle(overlay, (0, 0), (w, overlay_height), COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        draw_text(frame, "gestflow — Grabacion de dataset",
                  (20, 36), COLOR_YELLOW, scale=0.8)

        for i, gesto in enumerate(GESTURES):
            count  = contar_videos(gesto)
            estado = f"{count:3d} / {VIDEOS_PER_GESTURE}"
            color  = READY_COLOR if count >= VIDEOS_PER_GESTURE else COLOR_WHITE
            draw_text(
                frame,
                f"  {gesto:<20} {estado}",
                (20, 72 + i * 36),
                color,
                scale=0.65,
            )

        draw_text(
            frame,
            "Presiona ESPACIO para empezar  |  ESC para salir",
            (20, overlay_height - 16),
            COLOR_YELLOW,
            scale=0.6,
        )

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

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 180), COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        draw_text(frame, f"Gesto : {gesto}",
                  (20, 40), COLOR_YELLOW, scale=0.9)
        draw_text(frame, f"Video : {numero} de {total}",
                  (20, 80), COLOR_WHITE, scale=0.7)
        draw_text(frame,
                  "Preparate y mantente fuera del frame",
                  (20, 116), COLOR_WHITE, scale=0.65)
        draw_text(frame,
                  "ESPACIO grabar  |  S saltar gesto  |  ESC salir",
                  (20, 152), COLOR_YELLOW, scale=0.55)

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

            draw_text(frame, f"Gesto : {gesto}",
                      (20, 40), COLOR_YELLOW, scale=0.9)
            draw_text(frame, f"Grabando en : {i}",
                      (w // 2 - 100, h // 2),
                      COLOR_YELLOW, scale=1.2, thickness=3)
            draw_text(frame,
                      "Entra al frame YA con el gesto formado",
                      (20, h - 30), COLOR_WHITE, scale=0.6)

            cv2.imshow("gestflow — grabacion", frame)
            cv2.waitKey(1)


# -------------------------------------------------------------
#  Grabacion de un video
#  Captura FRAME_TOTAL frames y los guarda como MP4
#  Graba en RECORD_WIDTH x RECORD_HEIGHT
#  El modelo los reducira a FRAME_WIDTH x FRAME_HEIGHT
# -------------------------------------------------------------

def grabar_video(cap, gesto, numero):
    nombre = f"{gesto.lower()}_{numero:03d}.mp4"
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
        resized = cv2.resize(frame, (RECORD_WIDTH, RECORD_HEIGHT))
        writer.write(resized)

        progreso = int((frames_grabados / FRAME_TOTAL) * (RECORD_WIDTH - 40))

        display = resized.copy()
        cv2.rectangle(display, (20, RECORD_HEIGHT - 30),
                      (RECORD_WIDTH - 20, RECORD_HEIGHT - 14),
                      COLOR_BLACK, -1)
        cv2.rectangle(display, (20, RECORD_HEIGHT - 30),
                      (20 + progreso, RECORD_HEIGHT - 14),
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

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 100), (w, h), COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        draw_text(frame, "Video guardado correctamente",
                  (20, h - 72), READY_COLOR, scale=0.7)
        draw_text(frame,
                  "ESPACIO continuar  |  R repetir  |  ESC salir",
                  (20, h - 36), COLOR_YELLOW, scale=0.6)

        cv2.imshow("gestflow — grabacion", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            return "salir"
        if key == ord("r"):
            os.remove(ruta_video)
            return "repetir"
        if key == 32:
            return "continuar"


# -------------------------------------------------------------
#  Resumen final al terminar la sesion
# -------------------------------------------------------------

def resumen_final():
    print("\n  Resumen de grabacion:")
    print(f"  {'Gesto':<22} {'Grabados':>10} {'Faltantes':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10}")

    total_grabados  = 0
    total_faltantes = 0

    for gesto in GESTURES:
        grabados   = contar_videos(gesto)
        faltantes  = max(0, VIDEOS_PER_GESTURE - grabados)
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
        print(f"  Verifica CAMERA_INDEX en config.py")
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
            total  = VIDEOS_PER_GESTURE

            accion = pantalla_gesto(cap, gesto, numero, total)

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