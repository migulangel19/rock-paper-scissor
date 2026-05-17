"""
Reconocimiento facial con YuNet (detección) + SFace (embeddings).
Los modelos se descargan automáticamente la primera vez (~35 MB en total).
"""
import cv2
import pickle
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import numpy as np

# Compatibilidad con pickles guardados con numpy 2.x (renombró `numpy.core`
# como `numpy._core`). En este venv usamos numpy 1.26 (lo exige mediapipe
# 0.10.21), así que registramos alias para que `pickle.load` los encuentre.
class _CompatUnpickler(pickle.Unpickler):
    """Carga pickles guardados con numpy 2.x en un entorno con numpy 1.26.

    numpy 2 renombró `numpy.core` -> `numpy._core`; aquí remapeamos al
    vuelo para que los embeddings antiguos sigan siendo legibles."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


DB_PATH = "usuarios.pkl"   # {username: [emb1, emb2, ...]}
_TEMP   = tempfile.gettempdir()

YUNET_PATH = os.path.join(_TEMP, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(_TEMP, "face_recognition_sface_2021dec.onnx")
YUNET_URL  = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL  = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

COS_THRESHOLD = 0.363   # umbral oficial SFace (cosine)
N_CAPTURAS    = 50


# ── Descarga de modelos ───────────────────────────────────────────────────────

def _descargar(ruta, url, nombre):
    if os.path.exists(ruta) and os.path.getsize(ruta) > 1000:
        return True
    print(f"Descargando modelo {nombre}...", flush=True)
    tmp = ruta + ".tmp"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, ruta)
        print(f"✓ {nombre} listo", flush=True)
        return True
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"No se pudo descargar {nombre}: {e}")
        return False

def modelos_disponibles():
    return os.path.exists(YUNET_PATH) and os.path.getsize(YUNET_PATH) > 1000 \
       and os.path.exists(SFACE_PATH) and os.path.getsize(SFACE_PATH) > 1000

def descargar_modelos():
    """Descarga YuNet y SFace si no están en %TEMP%. Devuelve True si ok."""
    ok = _descargar(YUNET_PATH, YUNET_URL, "YuNet detector (~200 KB)")
    ok = _descargar(SFACE_PATH, SFACE_URL, "SFace reconocedor (~35 MB)") and ok
    return ok

def _crear_detector():
    return cv2.FaceDetectorYN.create(
        YUNET_PATH, "", (320, 320),
        score_threshold=0.6, nms_threshold=0.3
    )

def _crear_recognizer():
    return cv2.FaceRecognizerSF.create(SFACE_PATH, "")


# ── DB ────────────────────────────────────────────────────────────────────────

def cargar_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return _CompatUnpickler(f).load()
    return {}

def guardar_db(db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)


# ── Detección y embedding ─────────────────────────────────────────────────────

def _detectar(detector, frame):
    """Detecta caras con YuNet. Devuelve lista de filas (15 valores cada una)."""
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    return [faces[i] for i in range(faces.shape[0])]

def _embedding(recognizer, frame, face):
    """Extrae embedding SFace (128-d). Devuelve vector normalizado."""
    face_row = np.array(face, dtype=np.float32).reshape(1, -1)
    aligned  = recognizer.alignCrop(frame, face_row)
    feat     = recognizer.feature(aligned).flatten()
    return feat / (np.linalg.norm(feat) + 1e-9)

def _bbox(face):
    return int(face[0]), int(face[1]), int(face[2]), int(face[3])

def _mejor_coseno(emb, candidatos):
    """Cosine similarity máxima entre emb y la lista de embeddings almacenados."""
    scores = []
    for c in candidatos:
        if isinstance(c, np.ndarray) and c.ndim == 1:
            scores.append(float(np.dot(emb, c)))
    return max(scores) if scores else -1.0

def _pct(score):
    """Mapea score coseno → porcentaje: umbral=50%, 0.85=100%."""
    pct = (score - COS_THRESHOLD) / (0.85 - COS_THRESHOLD) * 50 + 50
    return max(0.0, min(100.0, round(pct, 1)))


# ── Utilidades visuales ───────────────────────────────────────────────────────

def _barra(frame, actual, total):
    h, w   = frame.shape[:2]
    BAR_W  = 320
    x0, y0 = (w - BAR_W) // 2, h - 45
    cv2.rectangle(frame, (x0, y0), (x0 + BAR_W, y0 + 18), (40, 40, 40), -1)
    cv2.rectangle(frame, (x0, y0),
                  (x0 + int(BAR_W * min(actual, total) / total), y0 + 18),
                  (0, 220, 80), -1)
    cv2.putText(frame, f"{actual}/{total}", (x0 + BAR_W + 8, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def _esquinas(img, x, y, w, h, color, grosor=3, largo=22):
    for v, p1, p2 in [
        ((x,   y),   (x+largo,   y),     (x,   y+largo)),
        ((x+w, y),   (x+w-largo, y),     (x+w, y+largo)),
        ((x,   y+h), (x+largo,   y+h),   (x,   y+h-largo)),
        ((x+w, y+h), (x+w-largo, y+h),   (x+w, y+h-largo)),
    ]:
        cv2.line(img, v, p1, color, grosor)
        cv2.line(img, v, p2, color, grosor)


# ── Registro ──────────────────────────────────────────────────────────────────

def registrar(nombre):
    if not modelos_disponibles():
        if not descargar_modelos():
            raise RuntimeError("No se pudieron descargar los modelos de reconocimiento facial")

    detector  = _crear_detector()
    recognizer = _crear_recognizer()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    win = f"Registro - {nombre}"

    # ── Esperar ESPACIO con cara visible ──────────────────────────────────────
    listo = False
    while not listo:
        ret, frame = cap.read()
        if not ret:
            break
        caras = _detectar(detector, frame)
        display = frame.copy()
        for face in caras:
            x, y, w, h = _bbox(face)
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(display, "ESPACIO: iniciar | ESC: cancelar",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            if caras:
                listo = True
            else:
                cv2.putText(display, "No se detecta cara, acercate",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow(win, display)
                cv2.waitKey(800)
        elif key == 27:
            cap.release(); cv2.destroyAllWindows(); return False

    if not listo:
        cap.release(); cv2.destroyAllWindows(); return False

    # ── Capturar embeddings libremente ────────────────────────────────────────
    embeddings = []
    while len(embeddings) < N_CAPTURAS:
        ret, frame = cap.read()
        if not ret:
            break
        caras = _detectar(detector, frame)
        display = frame.copy()
        if caras:
            face    = caras[0]
            x, y, w, h = _bbox(face)
            emb = _embedding(recognizer, frame, face)
            embeddings.append(emb)
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 220, 80), 2)
        cv2.putText(display, "Muevete con naturalidad",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 80), 2)
        _barra(display, len(embeddings), N_CAPTURAS)
        cv2.imshow(win, display)
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release(); cv2.destroyAllWindows(); return False

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) < N_CAPTURAS // 2:
        raise RuntimeError("Pocas capturas, inténtalo de nuevo")

    db = cargar_db()
    db[nombre] = embeddings
    guardar_db(db)
    print(f"✓ '{nombre}' registrado con {len(embeddings)} embeddings SFace")
    return True


# ── Login ─────────────────────────────────────────────────────────────────────

def login(streak_secs: float = 5.0, welcome_secs: float = 1.5):
    """Bucle de login.

    Cuando detecta al mismo usuario con ≥ 80% de confianza durante
    `streak_secs` segundos seguidos, lo da por identificado, muestra
    "Bienvenido, ..." durante `welcome_secs` y devuelve el nombre.

    Pensado para una demo: con `streak_secs=5` el público ve durante
    unos segundos cómo la caja verde sigue la cara con el nombre y la
    confianza antes de pasar al juego.
    """
    db = cargar_db()
    if not db:
        raise RuntimeError("No hay usuarios registrados")

    if not modelos_disponibles():
        if not descargar_modelos():
            raise RuntimeError("No se pudieron descargar los modelos")

    detector   = _crear_detector()
    recognizer = _crear_recognizer()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    cache = {"nombre": None, "pct": 0.0, "color": (0, 0, 255)}
    streak_nombre: str | None = None      # nombre que se está repitiendo
    streak_inicio: float | None = None    # instante en que empezó el streak

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        caras = _detectar(detector, frame)
        frame_nombre_ok: str | None = None
        for face in caras:
            emb = _embedding(recognizer, frame, face)
            mejor_nombre, mejor_score = None, -1.0
            for nombre, candidatos in db.items():
                if not isinstance(candidatos, list) or not candidatos \
                        or not isinstance(candidatos[0], np.ndarray):
                    continue
                score = _mejor_coseno(emb, candidatos)
                if score > mejor_score:
                    mejor_score, mejor_nombre = score, nombre

            x, y, w, h = _bbox(face)
            pct = _pct(mejor_score)
            if pct >= 80.0:
                color    = (0, 255, 0)
                etiqueta = f"{mejor_nombre}  {pct:.1f}%"
                cache    = {"nombre": mejor_nombre, "pct": pct, "color": color}
                frame_nombre_ok = mejor_nombre
            else:
                color    = (0, 0, 255)
                etiqueta = f"Desconocido  {pct:.1f}%"
                cache    = {"nombre": None, "pct": pct, "color": color}

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)
            _esquinas(frame, x, y, w, h, color)
            cv2.putText(frame, etiqueta, (x, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # ---- Auto-cierre por streak de reconocimientos ----------------
        if streak_secs > 0:
            ahora = time.time()
            if frame_nombre_ok and frame_nombre_ok == streak_nombre:
                pass                                  # se mantiene el streak
            elif frame_nombre_ok:
                streak_nombre = frame_nombre_ok
                streak_inicio = ahora                 # nuevo streak
            else:
                streak_nombre, streak_inicio = None, None

            # Barra de progreso del streak en la parte inferior central.
            if streak_inicio is not None:
                progreso = min(1.0, (ahora - streak_inicio) / streak_secs)
                fh, fw = frame.shape[:2]
                bar_w, bar_h = 360, 14
                x0 = (fw - bar_w) // 2
                y0 = fh - 50
                cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h),
                              (40, 40, 40), -1)
                cv2.rectangle(frame, (x0, y0),
                              (x0 + int(bar_w * progreso), y0 + bar_h),
                              (0, 220, 80), -1)
                restante = max(0.0, streak_secs - (ahora - streak_inicio))
                cv2.putText(frame, f"Identificando: {streak_nombre}  ({restante:.1f}s)",
                            (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 220, 80), 2)

            if streak_inicio is not None and (ahora - streak_inicio) >= streak_secs:
                fh, fw = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, fh // 2 - 90), (fw, fh // 2 + 90),
                              (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
                texto = f"Bienvenido, {streak_nombre}!"
                (tw, _), _ = cv2.getTextSize(
                    texto, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 4)
                cv2.putText(frame, texto, ((fw - tw) // 2, fh // 2 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 4,
                            cv2.LINE_AA)
                sub = "Lanzando el juego..."
                (sw, _), _ = cv2.getTextSize(
                    sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.putText(frame, sub, ((fw - sw) // 2, fh // 2 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2,
                            cv2.LINE_AA)
                t_fin = time.time() + welcome_secs
                while time.time() < t_fin:
                    cv2.imshow("Login facial", frame)
                    cv2.waitKey(50)
                cache["nombre"] = streak_nombre
                break

        cv2.putText(frame, "ESC: salir",
                    (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 180, 180), 1)
        cv2.imshow("Login facial", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    # Damos tiempo a Qt a procesar la destrucción y al driver de la cámara
    # a liberar el dispositivo antes de que `juego.py` lo reabra.
    for _ in range(10):
        cv2.waitKey(50)
    time.sleep(0.4)
    return cache["nombre"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("══════════════════════════════════════════")
    print("     LOGIN POR RECONOCIMIENTO FACIAL      ")
    print("══════════════════════════════════════════")
    db = cargar_db()
    print(f"Usuarios: {', '.join(db.keys()) if db else 'ninguno'}")
    print("\n[1] Iniciar sesión  [2] Registrar usuario")
    opcion = input("Opción: ").strip()
    if opcion == "2":
        nombre = input("Nombre: ").strip()
        if not nombre or not registrar(nombre):
            return
    identificado = login()
    print(f"\n{'✓ Bienvenido, ' + identificado + '!' if identificado else '✗ No reconocido.'}")

if __name__ == "__main__":
    main()
