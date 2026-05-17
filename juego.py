"""
juego.py
========

Juego de Piedra-Papel-Tijera contra un bot adaptativo, usando la
webcam como interfaz.

Componentes
-----------
* **Detección de la mano**: MediaPipe Hands extrae los 21 landmarks.
* **Clasificación del gesto**: el modelo SVM-RBF entrenado en
  `entrenar.py` (`modelo_gestos.joblib`) decide qué jugada has hecho a
  partir de los landmarks normalizados.
* **Bot adaptativo**: `bot_rps.HedgeBot`, ensemble de 24 predictores
  con pesos exponenciales que aprende tus patrones a medida que juegas.

Flujo de una ronda
------------------
1. Pulsas SPACE para empezar.
2. El bot elige su jugada **antes** de la cuenta atrás (sin ver la
   tuya: no hace trampas).
3. Aparece la cuenta atrás 3 → 2 → 1 → ¡YA!
4. En "¡YA!" se captura el frame, se clasifica tu mano y se muestra
   el resultado durante un par de segundos.
5. Si no se ve mano en ese frame, la ronda se anula.

Teclas
------
    SPACE : jugar una ronda
    r     : reiniciar marcador y memoria del bot
    q     : salir
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from bot_rps import MOVES, HedgeBot, result
from entrenar import normalizar_landmarks

MODELO_PATH = Path("modelo_gestos.joblib")
FOTOS_DIR = Path("fotos")
FOTO_FILES = {0: "Piedras.png", 1: "Papel.png", 2: "Tijeras.png"}
ICON_SIZE = 220                  # alto/ancho al que se reescalan los iconos
NUM_PUNTOS = 21

# Mapa entre las etiquetas de texto del clasificador y los enteros del bot.
LABEL_TO_INT = {"piedra": 0, "papel": 1, "tijera": 2}

# Duraciones (segundos) de cada fase de la ronda.
T_COUNTDOWN_STEP = 0.8
T_SHOW_RESULT = 2.5


def cargar_modelo(path: Path):
    if not path.exists():
        sys.exit(f"No se encontró {path}. Ejecuta primero entrenar.py.")
    datos = joblib.load(path)
    return datos["modelo"]


def landmarks_a_vector(hand_landmarks) -> np.ndarray:
    fila = np.empty(NUM_PUNTOS * 2, dtype=np.float32)
    for i, lm in enumerate(hand_landmarks.landmark):
        fila[2 * i]     = lm.x
        fila[2 * i + 1] = lm.y
    return normalizar_landmarks(fila.reshape(1, -1))


def dibujar_texto(frame, texto, pos, escala=1.0, color=(255, 255, 255),
                  grosor=2, centrado=False):
    """`cv2.putText` con sombra y opción de centrar horizontalmente."""
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(texto, fuente, escala, grosor)
    x, y = pos
    if centrado:
        x = int(x - w / 2)
    cv2.putText(frame, texto, (x + 2, y + 2), fuente, escala, (0, 0, 0),
                grosor + 2, cv2.LINE_AA)
    cv2.putText(frame, texto, (x, y), fuente, escala, color,
                grosor, cv2.LINE_AA)


def cargar_iconos() -> dict[int, np.ndarray]:
    """Carga los PNG de piedra/papel/tijera y los reescala a `ICON_SIZE`.

    Devuelve un dict {0,1,2 -> imagen BGRA cuadrada}. Si una imagen no
    trae canal alfa, le añadimos uno opaco para que el overlay sea
    uniforme."""
    iconos: dict[int, np.ndarray] = {}
    for k, nombre in FOTO_FILES.items():
        path = FOTOS_DIR / nombre
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            sys.exit(f"No se pudo leer {path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img = cv2.resize(img, (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA)
        iconos[k] = img
    return iconos


def overlay_bgra(frame: np.ndarray, icon: np.ndarray, x: int, y: int) -> None:
    """Pega `icon` (BGRA) sobre `frame` (BGR) en la posición (x, y),
    recortando si se sale del borde y respetando la transparencia."""
    h_i, w_i = icon.shape[:2]
    h_f, w_f = frame.shape[:2]

    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w_i, w_f), min(y + h_i, h_f)
    if x0 >= x1 or y0 >= y1:
        return

    icon_crop = icon[y0 - y: y1 - y, x0 - x: x1 - x]
    alpha = icon_crop[..., 3:4].astype(np.float32) / 255.0
    region = frame[y0:y1, x0:x1].astype(np.float32)
    mezcla = region * (1 - alpha) + icon_crop[..., :3].astype(np.float32) * alpha
    frame[y0:y1, x0:x1] = mezcla.astype(np.uint8)


def dibujar_panel_resultado(frame, icono_humano, icono_bot,
                            titulo: str, color_titulo) -> None:
    """Pinta el panel grande con GANAS/PIERDES/EMPATE, los dos iconos y
    un 'VS' en el centro, sobre una banda semitransparente."""
    h, w = frame.shape[:2]

    # Banda oscura semitransparente para que el texto se lea bien.
    panel_h = ICON_SIZE + 140
    panel_y0 = (h - panel_h) // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y0), (w, panel_y0 + panel_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Título (GANAS / PIERDES / EMPATE).
    dibujar_texto(frame, titulo, (w // 2, panel_y0 + 55),
                  escala=1.6, color=color_titulo, grosor=4, centrado=True)

    # Iconos: humano a la izquierda, bot a la derecha.
    icon_y = panel_y0 + 80
    gap = 90
    x_humano = w // 2 - ICON_SIZE - gap // 2
    x_bot    = w // 2 + gap // 2
    overlay_bgra(frame, icono_humano, x_humano, icon_y)
    overlay_bgra(frame, icono_bot, x_bot, icon_y)

    # Etiquetas debajo de cada icono.
    dibujar_texto(frame, "TU", (x_humano + ICON_SIZE // 2, icon_y + ICON_SIZE + 35),
                  escala=0.8, color=(255, 255, 255), grosor=2, centrado=True)
    dibujar_texto(frame, "BOT", (x_bot + ICON_SIZE // 2, icon_y + ICON_SIZE + 35),
                  escala=0.8, color=(255, 255, 255), grosor=2, centrado=True)

    # "VS" en el centro.
    dibujar_texto(frame, "VS", (w // 2, icon_y + ICON_SIZE // 2 + 15),
                  escala=1.4, color=(0, 200, 255), grosor=4, centrado=True)


def main(usuario: str | None = None) -> None:
    modelo = cargar_modelo(MODELO_PATH)
    iconos = cargar_iconos()
    bot = HedgeBot()
    saludo = f"Hola, {usuario}!" if usuario else "Juego cargado"
    print(f"{saludo}  SPACE = jugar, r = reiniciar, q = salir")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # La cámara puede tardar 1-2 s en quedar liberada después del módulo
    # de login facial. Reintentamos un par de veces antes de rendirnos.
    cap = None
    for intento in range(8):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            # Validamos también que entreguen frames reales.
            ok, _ = cap.read()
            if ok:
                print(f"Cámara lista (intento {intento + 1}).")
                break
        cap.release()
        print(f"Cámara aún no disponible, reintentando... ({intento + 1}/8)")
        time.sleep(0.6)
    else:
        sys.exit("No se pudo abrir la webcam tras varios intentos.")

    # Aseguramos que la ventana principal exista antes de pintar nada.
    cv2.namedWindow("Piedra / Papel / Tijera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Piedra / Papel / Tijera", 960, 720)
    print("Ventana del juego creada. Esperando frames...")

    # Splash de bienvenida si entramos con usuario identificado.
    if usuario:
        t_splash_fin = time.time() + 1.8
    else:
        t_splash_fin = 0.0

    # Estado del juego ---------------------------------------------------
    wins = losses = ties = 0       # marcador (perspectiva del humano)
    estado = "IDLE"                # IDLE | COUNTDOWN | RESULT
    t_fase = 0.0                   # instante en que empezó la fase actual
    bot_move: int | None = None    # jugada que el bot ya ha decidido
    bot_preds = None               # predicciones para alimentar el update
    titulo_resultado = ""          # GANAS / PIERDES / EMPATE
    color_resultado = (255, 255, 255)
    human_move_ult: int | None = None    # para el panel de la fase RESULT
    mensaje_simple = ""            # mensaje sin iconos (rondas anuladas)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = hands.process(rgb)

            hand_landmarks = None
            if resultados.multi_hand_landmarks:
                hand_landmarks = resultados.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

            ahora = time.time()

            # ---- Splash de bienvenida ------------------------------------
            if usuario and ahora < t_splash_fin:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h // 2 - 90), (w, h // 2 + 90),
                              (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                dibujar_texto(frame, f"Bienvenido, {usuario}!",
                              (w // 2, h // 2 - 10), escala=1.4,
                              color=(0, 255, 0), grosor=4, centrado=True)
                dibujar_texto(frame, "Pulsa SPACE para empezar",
                              (w // 2, h // 2 + 55), escala=0.7,
                              color=(220, 220, 220), grosor=2, centrado=True)
                cv2.imshow("Piedra / Papel / Tijera", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            # ---- Lógica de estados ---------------------------------------
            if estado == "COUNTDOWN":
                transcurrido = ahora - t_fase
                paso = int(transcurrido // T_COUNTDOWN_STEP)  # 0,1,2 -> 3,2,1 ; 3 -> ¡YA!
                if paso < 3:
                    dibujar_texto(frame, str(3 - paso), (w // 2, h // 2 + 30),
                                  escala=6.0, color=(0, 255, 255), grosor=10,
                                  centrado=True)
                else:
                    # Momento de capturar: clasificamos la mano y resolvemos.
                    if hand_landmarks is None:
                        titulo_resultado = ""
                        mensaje_simple = "No se detecto mano - ronda anulada"
                        color_resultado = (0, 200, 255)
                        human_move_ult = None
                    else:
                        X = landmarks_a_vector(hand_landmarks)
                        etiqueta = modelo.predict(X)[0]
                        human_move = LABEL_TO_INT[etiqueta]
                        human_move_ult = human_move
                        mensaje_simple = ""

                        res = result(human_move, bot_move)
                        bot.update(bot_move, human_move, bot_preds)

                        if res == 0:
                            titulo_resultado = "EMPATE"
                            color_resultado = (200, 200, 200)
                            ties += 1
                        elif res == -1:
                            titulo_resultado = "GANAS"
                            color_resultado = (0, 255, 0)
                            wins += 1
                        else:
                            titulo_resultado = "PIERDES"
                            color_resultado = (0, 0, 255)
                            losses += 1

                    estado = "RESULT"
                    t_fase = ahora

            elif estado == "RESULT":
                if human_move_ult is not None:
                    dibujar_panel_resultado(
                        frame,
                        iconos[human_move_ult],
                        iconos[bot_move],
                        titulo_resultado,
                        color_resultado,
                    )
                else:
                    dibujar_texto(frame, mensaje_simple, (w // 2, h // 2),
                                  escala=1.0, color=color_resultado, grosor=3,
                                  centrado=True)
                if ahora - t_fase >= T_SHOW_RESULT:
                    estado = "IDLE"

            # ---- HUD (siempre visible) -----------------------------------
            dibujar_texto(frame, f"Tu {wins}   Empates {ties}   Bot {losses}",
                          (10, 30), escala=0.7, color=(255, 255, 255))
            if usuario:
                dibujar_texto(frame, f"Jugador: {usuario}",
                              (w - 10 - 8 * len(usuario) - 110, 30),
                              escala=0.6, color=(0, 220, 255))
            if estado == "IDLE":
                dibujar_texto(frame, "SPACE = jugar    r = reiniciar    q = salir",
                              (10, h - 15), escala=0.55, color=(200, 200, 200))

            cv2.imshow("Piedra / Papel / Tijera", frame)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q"):
                break
            if tecla == ord("r"):
                wins = losses = ties = 0
                bot.reset()
                estado = "IDLE"
                print("Marcador y bot reiniciados.")
            if tecla == ord(" ") and estado == "IDLE":
                # El bot decide AHORA, antes de ver tu mano: nada de trampas.
                bot_move, bot_preds = bot.choose()
                estado = "COUNTDOWN"
                t_fase = ahora

    cap.release()
    cv2.destroyAllWindows()

    total = wins + losses + ties
    if total:
        print(f"\nResultado final tras {total} rondas:")
        print(f"  Tú:      {wins}  ({wins / total:.1%})")
        print(f"  Empates: {ties}  ({ties / total:.1%})")
        print(f"  Bot:     {losses}  ({losses / total:.1%})")


if __name__ == "__main__":
    import sys as _sys
    main(_sys.argv[1] if len(_sys.argv) > 1 else None)
