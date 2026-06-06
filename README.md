# gestflow

Sistema de control del escritorio mediante gestos de mano en tiempo real.
Utiliza una arquitectura CNN + LSTM con MobileNetV2 preentrenada para clasificar
9 gestos directamente desde los pixeles del video, sin extraccion manual de caracteristicas.

---

## Requisitos

- Python 3.10.11
- Windows 10 / 11
- Camara web

---

## Instalacion

1. Clona o descarga el repositorio

2. Crea el entorno virtual con Python 3.10 especificamente

        py -3.10 -m venv venv

3. Activa el entorno virtual

        venv\Scripts\activate

4. Actualiza pip

        python -m pip install --upgrade pip setuptools wheel

5. Instala las dependencias

        pip install -r requirements.txt

---

## Estructura del proyecto

    gestflow/
    ├── dataset/
    │   └── raw_videos/
    │       ├── MOVE_CURSOR/
    │       ├── LEFT_CLICK/
    │       ├── RIGHT_CLICK/
    │       ├── SCROLL/
    │       ├── DRAG/
    │       ├── ALT_TAB/
    │       ├── PAUSE/
    │       ├── SAFE_MODE/
    │       └── OPEN_APP/
    ├── training/
    │   ├── 01_record.py
    │   ├── 02_train.py
    │   ├── augmentation.py
    │   ├── callbacks.py
    │   ├── model.py
    │   ├── dataloader.py
    │   ├── evaluate.py
    │   └── model/
    │       └── gesture_model.keras
    ├── app/
    │   ├── main.py
    │   ├── gesture_recognizer.py
    │   ├── mouse_controller.py
    │   └── overlay.py
    ├── config.py
    ├── requirements.txt
    └── README.md

---

## Uso

El proyecto se usa en tres pasos en orden.

### Paso 1 — Grabar el dataset

    python training/01_record.py

El script guia al usuario gesto por gesto con cuenta regresiva.
Graba 60 videos por gesto en dataset/raw_videos/.

Controles durante la grabacion:

    ESPACIO   Iniciar grabacion
    R         Repetir el video actual
    S         Saltar al siguiente gesto
    ESC       Salir y guardar progreso

### Paso 2 — Entrenar el modelo

    python training/02_train.py

Carga el dataset, aplica data augmentation, entrena el modelo CNN + LSTM
con validacion cruzada 5-fold y guarda el modelo en training/model/gesture_model.keras.

### Paso 3 — Usar el sistema

    python app/main.py

Abre la ventana flotante con el feed de la camara, el gesto detectado
y la precision en tiempo real. El sistema controla el escritorio automaticamente.

---

## Gestos disponibles

    MOVE_CURSOR   Indice apuntando al techo, palma al frente         Mueve el cursor
    LEFT_CLICK    Indice y medio juntos, palma al frente             Clic izquierdo
    RIGHT_CLICK   Indice y menique arriba, palma al frente           Clic derecho
    SCROLL        Indice y medio en V moviendose, palma al frente    Scroll hacia arriba
    DRAG          Pellizco cerrado moviendose, palma al frente       Arrastrar elemento
    ALT_TAB       Indice, medio y anular arriba, palma al frente     Cambiar ventana
    PAUSE         Palma completamente abierta al frente              Pausar el sistema
    SAFE_MODE     Puno cerrado al frente                             Activar modo seguro
    OPEN_APP      V abierta estatica, palma al frente                Abrir menu inicio

Orientacion para todos los gestos: palma mirando directamente hacia la camara.

---

## Arquitectura del modelo

    Input (15, 112, 112, 3)
        -> TimeDistributed(MobileNetV2 + GlobalAveragePooling2D)
        -> Masking
        -> LSTM(128, return_sequences=True)
        -> Dropout(0.4)
        -> BatchNormalization
        -> LSTM(64, return_sequences=False)
        -> Dropout(0.4)
        -> BatchNormalization
        -> Dense(64, relu)
        -> Dropout(0.4)
        -> Dense(9, softmax)

El entrenamiento se realiza en dos fases:

    Fase 1   MobileNetV2 completamente congelada, solo se entrenan las capas LSTM y Dense
    Fase 2   Fine-tuning de las ultimas 30 capas de MobileNetV2 con learning rate reducido

---

## Data Augmentation

Se aplican 6 tecnicas sobre los frames crudos del video.
Cada video del dataset original genera 6 muestras de entrenamiento.

    Original            Sin modificacion
    Gaussian Jitter     Ruido gaussiano sobre los pixeles
    Spatial Scaling     Zoom in/out desde el centro del frame
    Horizontal Mirror   Espejo horizontal del video completo
    Time Warping        Remuestreo temporal de los frames
    Brightness Shift    Variacion del brillo global del video

Factor de augmentation: 6x
Con 60 videos por gesto se obtienen 360 muestras de entrenamiento por clase.

---

## Configuracion

Todos los parametros del proyecto estan centralizados en config.py.
Ningun otro archivo tiene valores hardcodeados.

    VIDEOS_PER_GESTURE     60          Videos a grabar por gesto
    FRAME_COUNT            15          Frames extraidos por video
    FRAME_WIDTH            112         Ancho de entrada al modelo en px
    FRAME_HEIGHT           112         Alto de entrada al modelo en px
    RECORD_WIDTH           640         Ancho de grabacion en px
    RECORD_HEIGHT          480         Alto de grabacion en px
    KFOLD_SPLITS           5           Folds de validacion cruzada
    EPOCHS                 100         Epocas maximas de entrenamiento
    BATCH_SIZE             8           Tamano del batch
    LEARNING_RATE          0.001       Learning rate inicial
    AUGMENTATION_FACTOR    6           Multiplicador del dataset
    CONFIDENCE_THRESHOLD   0.75        Precision minima para ejecutar accion
    PREDICTION_MARGIN      0.20        Margen minimo entre top-1 y top-2 (rechazo)
    ACTION_COOLDOWN        0.5         Segundos entre acciones consecutivas
    SCROLL_SPEED           3           Unidades de scroll por deteccion
    CURSOR_SMOOTHING       7           Factor de suavizado del cursor
    DRAG_THRESHOLD         0.80        Precision minima especifica para DRAG
    PYAUTOGUI_FAILSAFE     True        Esquina de pantalla aborta toda accion
    PANIC_HOTKEY           ctrl+alt+p  Pausa global + suelta el raton
    KILL_HOTKEY            ctrl+alt+q  Apagado de emergencia global

---

## Seguridad

El sistema controla el raton y el teclado de forma automatica, por lo que
incorpora varias salvaguardas:

    Rechazo por margen    Si el modelo duda entre dos gestos (top-1 y top-2
                          muy cercanos) la prediccion se descarta. Evita
                          acciones espurias cuando no hay un gesto claro,
                          ya que el modelo no tiene una clase NEUTRAL.

    Failsafe pyautogui    Llevar el cursor a una esquina de la pantalla
                          aborta cualquier accion en curso.

    Hotkey de panico      ctrl+alt+p pausa el sistema y suelta el raton al
                          instante, aunque la ventana no tenga el foco.

    Hotkey de apagado     ctrl+alt+q cierra el sistema por completo.

    Gestos PAUSE/SAFE     Pausan o restringen el sistema desde la propia mano.

---

## Resultados

    Accuracy validacion cruzada   96.67%
    Accuracy test set             88.89%
    Dataset de prueba             10 videos por gesto
    Tiempo de entrenamiento       1h 08m 38s

Los resultados finales se actualizaran cuando se entrene
con el dataset completo de 60 videos por gesto.

---

## Tecnologias utilizadas

    TensorFlow     2.15.0   Modelo, capas, callbacks, fine-tuning
    Keras          2.15.0   API de alto nivel
    OpenCV         4.9.0    Captura de camara, procesamiento de video, overlay
    NumPy          1.26.4   Operaciones matriciales, data augmentation
    SciPy          1.11.4   Interpolacion para time warping
    scikit-learn   1.3.2    Validacion cruzada, metricas, confusion matrix
    PyAutoGUI      0.9.54   Control del cursor y teclado
    pygetwindow    0.0.9    Gestion de ventanas
    pywin32        306      Integracion con Windows
    pycaw          20230407 Control de audio
    Matplotlib     3.8.4    Curvas de entrenamiento por fold
    Seaborn        0.13.2   Visualizacion de matriz de confusion
    tqdm           4.66.2   Barra de progreso durante la carga del dataset
    pandas         2.1.4    Analisis y resumen del dataset
    keyboard       0.13.5   Deteccion de tecla ESC para pausar el sistema

---

## Autores

Proyecto desarrollado para el curso de Deep Learning.