# Trabajo Final — Aprendizaje Automático

Juego de **Piedra · Papel · Tijera** controlado por gestos de la mano,
con **acceso por reconocimiento facial** y un **bot adaptativo** que
aprende los patrones del jugador en tiempo real.

El proyecto integra tres bloques de aprendizaje automático muy
distintos en una única aplicación:

| Bloque                 | Técnica                                                        | Quién la implementó |
| ---------------------- | --------------------------------------------------------------- | --------------------- |
| Login biométrico      | YuNet + SFace (deep learning preentrenado)                      | Compañero            |
| Bot RPS adaptativo     | Ensemble Hedge sobre 24 predictores + Random Forest supervisado | Compañero            |
| Clasificador de gestos | MediaPipe Hands (21 landmarks) + SVM-RBF                        | Nosotros              |

---

## 1. Visión general del sistema

```
                     ┌────────────────────┐
   Usuario abre app  │      app.py        │  (Tkinter)
   ───────────────► │                     │
                     │  Login facial      │
                     │  (YuNet + SFace)   │ ─► reconocido en 5 s
                     └────────────────────┘
                              │
                              ▼  subprocess (juego.py "<nombre>")
                     ┌────────────────────┐
                     │      juego.py      │  (OpenCV)
                     │                    │
                     │  Webcam            │
                     │   ├─ MediaPipe     │ ─► 21 landmarks
                     │   ├─ Normalización │
                     │   └─ SVM-RBF       │ ─► piedra / papel / tijera
                     │                    │
                     │  HedgeBot          │ ─► jugada del bot
                     │                    │
                     │  Marcador + iconos │
                     └────────────────────┘
```

El flujo completo de una partida:

1. **`app.py`** abre un GUI con dos opciones: *Iniciar sesión* y
   *Registrar usuario*. Internamente usa `login_facial.py`.
2. La cámara se enciende, detecta tu cara con **YuNet**, extrae un
   embedding de 128 dimensiones con **SFace** y compara con la base
   de datos `usuarios.pkl` por similitud coseno.
3. Cuando reconoce al mismo usuario durante **5 s seguidos** con
   ≥ 80 % de confianza, muestra **"Bienvenido, ..."** y cierra el
   login.
4. `app.py` lanza el juego como **subproceso aparte**
   (`python juego.py <nombre>`).
5. `juego.py` abre una nueva ventana de cámara. Pulsas `SPACE`, el
   bot decide su jugada, aparece la cuenta atrás **3 → 2 → 1 → ¡YA!**,
   se clasifica tu mano y se muestra **GANAS / PIERDES / EMPATE** con
   los iconos de la jugada de cada uno.
6. El bot actualiza sus pesos según lo que jugaste para ir
   anticipándote en futuras rondas.

---

## 2. Estructura del repositorio

```
.
├── venv/                      # Entorno virtual de Python
├── app.py                     # Punto de entrada (GUI Tk + lanzador del juego)
├── login_facial.py            # YuNet + SFace + DB de usuarios
├── usuarios.pkl               # DB de embeddings (Pablo Barranco, Miguel Merino)
│
├── capturar_gestos.py         # Captura del dataset de gestos
├── datos_gestos.csv           # 1498 muestras de gestos (≈500 por clase)
├── entrenar.py                # Pipeline ML del clasificador de gestos
├── modelo_gestos.joblib       # SVM-RBF entrenado + lista de etiquetas
├── jugar.py                   # Inferencia simple (test del modelo, sin juego)
│
├── bot_rps.py                 # Bot adaptativo Hedge (RPS)
├── juego.py                   # Juego completo con webcam + bot + iconos
│
├── fotos/                     # Iconos del panel de resultado
│   ├── Piedras.png
│   ├── Papel.png
│   └── Tijeras.png
│
├── Proyecto Aprendizaje Automatico/   # Carpeta original del compañero (referencia)
├── proyecto_rps_ml.ipynb              # Notebook original del bot (referencia)
│
└── README.md
```

---

## 3. Entorno

Todas las dependencias viven en `venv/`. Versiones fijadas:

| Librería                 | Versión  | Por qué esa                                      |
| ------------------------- | --------- | ------------------------------------------------- |
| `opencv-python`         | 4.11.0.86 | Compatible con numpy < 2                          |
| `opencv-contrib-python` | 4.11.0.86 | Trae `FaceDetectorYN` y `FaceRecognizerSF`    |
| `mediapipe`             | 0.10.21   | La 0.10.35 eliminó `mp.solutions` (API legacy) |
| `numpy`                 | 1.26.4    | Lo exige mediapipe 0.10.21                        |
| `scikit-learn`          | 1.8.0     | KNN, SVM, RandomForest, Pipeline                  |
| `pandas`                | 3.0.3     | Carga del CSV                                     |
| `joblib`                | 1.5.3     | Serialización del modelo                         |

Y dependencias del sistema (no se instalan con `pip`):

- **python3-tk** — el GUI de login usa Tkinter.
  `sudo apt-get install -y python3-tk` en Debian/Ubuntu.

### Reproducir el entorno

```bash
python3 -m venv venv
source venv/bin/activate
pip install "mediapipe==0.10.21" "opencv-python<4.12" \
            opencv-contrib-python==4.11.0.86 \
            scikit-learn pandas joblib
sudo apt-get install -y python3-tk
```

### Notas técnicas que nos dieron guerra

- **Pickle de numpy 2 → numpy 1**: el `usuarios.pkl` se generó en un
  PC con numpy 2.x, donde el módulo interno se renombró
  `numpy.core` → `numpy._core`. En nuestro venv (numpy 1.26)
  `pickle.load` fallaba con `ModuleNotFoundError: numpy._core.numeric`.
  Lo resolvemos con un `_CompatUnpickler` en
  [login_facial.py](login_facial.py) que sobreescribe `find_class`
  para remapear el módulo al vuelo.
- **Tkinter + OpenCV en el mismo proceso**: al destruir la ventana
  Tk y llamar inmediatamente a `cv2.imshow` la ventana de OpenCV se
  quedaba muda (Qt arrastraba estado de la sesión anterior). Por eso
  `app.py` arranca `juego.py` como **subproceso aparte** con
  `subprocess.run`. Tiene además la ventaja de aislar fallos: si el
  juego peta, la app de login no se cae.

---

## 4. Bloque 1 — Reconocimiento facial (login)

### Modelos

| Tarea              | Modelo                                                    | Origen     | Tamaño |
| ------------------ | --------------------------------------------------------- | ---------- | ------- |
| Detección de cara | **YuNet** (`face_detection_yunet_2023mar.onnx`)   | OpenCV Zoo | ~200 KB |
| Embedding facial   | **SFace** (`face_recognition_sface_2021dec.onnx`) | OpenCV Zoo | ~35 MB  |

Ambos modelos están preentrenados; **no entrenamos nada nuevo aquí**.
La primera vez que se ejecuta el login, [login_facial.py](login_facial.py)
los descarga automáticamente a `tempfile.gettempdir()` desde
[opencv/opencv_zoo](https://github.com/opencv/opencv_zoo).

### Pipeline

1. **YuNet** localiza una o varias caras en el frame y devuelve, para
   cada una, bounding box + 5 puntos faciales (ojos, nariz, esquinas de
   la boca). Es una red ligerísima diseñada para tiempo real.
2. **SFace** recibe la cara recortada y alineada (`alignCrop` usa los
   5 puntos de YuNet) y devuelve un **vector de 128 floats** que
   representa la identidad de esa persona.
3. **Comparación**: para cada usuario en `usuarios.pkl`, guardamos
   ~50 embeddings tomados con cabeza en distintas posiciones. Al
   identificar, calculamos la **similitud coseno** del embedding actual
   con cada uno de los almacenados y nos quedamos con el mejor por
   usuario.
4. **Umbral**: 0.363 es el valor oficial de SFace para decidir que dos
   embeddings son de la misma persona. Lo convertimos a un % "amistoso"
   (umbral → 50 %, coseno 0.85 → 100 %) y exigimos ≥ 80 % para dar el
   login por bueno.
5. **Streak temporal**: para que el reconocimiento se note durante la
   demo, exigimos que la misma persona se identifique **5 segundos
   seguidos** antes de cerrar el login. Una barra verde abajo muestra
   el progreso. Luego aparece "Bienvenido, ..." durante 1.5 s.

### Registro de un usuario nuevo

- Espera a `SPACE` con tu cara visible.
- Captura **50 embeddings** moviéndote con naturalidad (gira ligeramente
  la cabeza, cambia la expresión). Esos 50 embeddings se guardan en
  `usuarios.pkl` y son el "perfil" del usuario.
- Por qué 50 y no 1: la similitud coseno se compara contra el **mejor**
  de la lista, lo que hace el matching mucho más robusto a variaciones
  de pose, iluminación, gafas, etc.

---

## 5. Bloque 2 — Clasificador de gestos (piedra/papel/tijera)

Esta parte la hicimos nosotros desde cero. Es el equivalente a un
*tutorial completo de ML clásico* aplicado a vídeo en tiempo real.

### 5.1 Por qué landmarks y no píxeles

En lugar de entrenar una CNN sobre imágenes de manos (mucho data
necesario, GPU, etc.), delegamos la extracción de *features* a
**MediaPipe Hands**, que para cada mano detectada devuelve las
coordenadas (x, y, z) de **21 puntos clave** (muñeca, nudillos, falanges
y puntas de los cinco dedos). Esto nos da:

- 42 *features* (usamos solo x e y, ignoramos z).
- Independencia del fondo y de la iluminación.
- Resolución del problema con modelos clásicos en milisegundos por
  predicción.

### 5.2 Captura del dataset

[capturar_gestos.py](capturar_gestos.py) abre la webcam y guarda una
fila en `datos_gestos.csv` cada vez que pulsas:

| Tecla | Etiqueta |
| ----- | -------- |
| `0` | piedra   |
| `1` | papel    |
| `2` | tijera   |
| `q` | salir    |

Formato del CSV: 42 columnas (`x0, y0, …, x20, y20`) + columna
`etiqueta`. El archivo final tiene **1498 muestras** (~500 por clase)
capturadas a distintas distancias, ángulos y rotaciones de muñeca.

### 5.3 Preprocesado: normalización de landmarks

Es la decisión más importante del modelo. Sin esto, el clasificador
aprendería a clasificar por **dónde** está tu mano en el encuadre o por
**lo grande** que aparece, no por el gesto. La normalización tiene dos
pasos (función `normalizar_landmarks` en [entrenar.py](entrenar.py)):

1. **Traslación**: a todos los puntos se les resta la muñeca (punto 0),
   de forma que la muñeca pase a ser el origen `(0, 0)`.
2. **Escala**: dividimos por la distancia muñeca → nudillo del dedo
   medio (punto 9). Esa distancia es una medida estable del "tamaño"
   de la mano en la imagen y casi independiente del gesto.

Tras este preprocesado, dos manos haciendo el mismo gesto a distancias
distintas producen vectores de 42 floats muy parecidos.

### 5.4 Entrenamiento y elección de modelo

[entrenar.py](entrenar.py) hace un split estratificado 80/20 y entrena
tres pipelines (cada uno con `StandardScaler`):

| Modelo            | Hiperparámetros                                  | Accuracy en test  |
| ----------------- | ------------------------------------------------- | ----------------- |
| KNN               | `n_neighbors=5`                                 | 98.33 %           |
| **SVM-RBF** | `C=10`, `gamma="scale"`, `probability=True` | **99.33 %** |
| RandomForest      | `n_estimators=300`                              | 99.33 %           |

Elegimos **SVM-RBF** (empate técnico con RandomForest, pero algo más
ligero en inferencia y devuelve probabilidades calibradas). El pipeline
completo (scaler + clasificador) se serializa en `modelo_gestos.joblib`,
junto con la lista de etiquetas.

**Reporte de clasificación:**

```
              precision    recall  f1-score   support
       papel     0.9900    0.9900    0.9900       100
      piedra     0.9901    1.0000    0.9950       100
      tijera     1.0000    0.9900    0.9950       100
    accuracy                         0.9933       300
```

**Matriz de confusión** (filas = real, columnas = predicho):

```
            papel  piedra  tijera
papel         99     1       0
piedra         0   100       0
tijera         1     0      99
```

Sólo 2 errores en 300 muestras, ambos confundiendo *papel* y *tijera*
(visualmente son los gestos con más dedos extendidos y más
configuraciones intermedias).

### 5.5 Inferencia

Importante: **la inferencia reusa exactamente la misma función de
normalización que el entrenamiento**, importándola con
`from entrenar import normalizar_landmarks`. Así evitamos el clásico
*training/serving skew*.

---

## 6. Bloque 3 — Bot adaptativo de RPS

Implementación basada en el notebook
[proyecto_rps_ml.ipynb](proyecto_rps_ml.ipynb) del compañero. La lógica
algorítmica se extrajo y se modulariza en [bot_rps.py](bot_rps.py),
quitando matplotlib y todo lo de exploración.

### Idea

Desde la teoría de juegos, jugar uniforme aleatorio garantiza un 33 %
de victorias (equilibrio de Nash). Pero los humanos **no** somos
aleatorios: sesgo a piedra en la primera tirada, win-stay/lose-shift,
ciclos R→P→T, anti-repetición... El bot intenta detectar y explotar
estos patrones en tiempo real.

### Componentes

- **24 predictores** especializados (8 base × 3 meta-niveles):

  - `UniformPredictor` (baseline)
  - `FrequencyPredictor` (frecuencias acumuladas con Laplace)
  - `RecentFrequencyPredictor(window=10)` y `(window=30)`
  - `MarkovN(1)`, `MarkovN(2)`, `MarkovN(3)`
  - `WSLSPredictor` (Win-Stay / Lose-Shift)
  - **Meta-niveles tipo *Iocaine Powder*** (`MetaShift`): cada
    predicción se rota 0, 1 o 2 posiciones para protegerse de un
    humano que intente contra-anticiparse.
- **Meta-selector Hedge** (Freund & Schapire, 1997): mantiene un peso
  exponencial por predictor. En cada ronda:

  1. Mezcla las predicciones ponderadas.
  2. Elige la jugada que maximiza la utilidad esperada
     (`P(gana) − P(pierde)`).
  3. Tras ver la jugada real, premia a los predictores que le
     asignaron alta probabilidad: `w_i *= exp(η · (p_i[real] − 1/3))`.
  4. Aplica un decay `γ = 0.99` para olvidar el pasado y adaptarse a
     cambios de estrategia.

### Resultados (del notebook, contra arquetipos sintéticos)

| Bot                       | Edge promedio   |
| ------------------------- | --------------- |
| Aleatorio                 | 0.00            |
| Frecuencia                | −0.06          |
| Markov-2                  | +0.36           |
| Random Forest supervisado | +0.27           |
| **Ensemble Hedge**  | **+0.37** |

El Hedge gana o iguala al mejor predictor único contra cualquier
arquetipo y nunca tiene puntos débiles (edge ≥ +0.08 en el peor caso).

### Integración con el juego

En cada ronda, [juego.py](juego.py):

1. Llama a `bot.choose()` **antes** de la cuenta atrás, así el bot
   decide su jugada sin ver tu mano (no hace trampas).
2. Tras clasificar tu gesto, llama a
   `bot.update(bot_move, human_move, preds)` para que el ensemble
   actualice sus pesos.

---

## 7. Cómo se usa

```bash
source venv/bin/activate
python app.py
```

1. **Registrar usuario** (solo la primera vez): escribe un nombre,
   pulsa el botón, mira a cámara, pulsa `SPACE` y muévete suavemente
   mientras la barra inferior se completa (50 capturas).
2. **Iniciar sesión y jugar**: pulsa el botón principal. La cámara se
   abre, te identifica durante 5 s con una barra verde, te saluda y
   lanza el juego.
3. **En el juego**:
   - `SPACE` → empieza la cuenta atrás 3-2-1-¡YA!
   - Saca tu jugada justo cuando aparezca el "¡YA!".
   - El bot mostrará su jugada y quién ha ganado, con iconos.
   - `r` → reinicia marcador y bot.
   - `q` → salir.

Si no hay base de usuarios y solo quieres probar el juego sin login:

```bash
python juego.py             # sin nombre, sin saludo
python juego.py "Tu Nombre" # con saludo personalizado
```

---

## 8. Decisiones y aprendizajes

- **Landmarks en vez de píxeles** → modelo pequeño, sin GPU, sin
  necesidad de dataset enorme. Cambia el problema "clasificar una
  imagen" por "clasificar un vector de 42 floats".
- **Normalización por muñeca + escala** → el cambio que más
  contribuye a la generalización. Sin ella el modelo aprende dónde
  está la mano, no qué gesto hace.
- **Reutilizar el preprocesado entre training y serving** → importar
  la función desde el módulo de entrenamiento elimina una clase entera
  de bugs.
- **Pipeline de sklearn + `StandardScaler`** → aunque los datos ya
  están normalizados geométricamente, estandarizar ayuda especialmente
  a SVM y KNN.
- **Split estratificado** → buena práctica aunque las clases estén
  balanceadas; deja el flujo preparado para datasets desbalanceados.
- **Subproceso para el juego** → evita que el estado Qt de Tkinter
  contamine OpenCV, y aísla fallos.
- **Compatibilidad numpy 2 ↔ 1** → un `Unpickler` con `find_class`
  custom es la forma menos invasiva de leer pickles "del futuro".
- **Streak temporal en el login** → priorizar la experiencia visible
  del público (5 s con barra) sobre la pura conveniencia.

---

## 9. Posibles mejoras

- **Random Forest del notebook integrado**: actualmente el bot del
  juego usa solo Hedge; sería trivial añadir el RF supervisado del
  notebook como predictor extra del ensemble.
- **Landmark 3D**: usar también la coordenada `z` de MediaPipe para
  distinguir mejor gestos con dedos solapados.
- **Más gestos**: añadir *lagarto* y *Spock* (piedra-papel-tijera-
  lagarto-Spock) — basta con capturar más muestras y reentrenar.
- **Validación cruzada + GridSearchCV** sobre `C` y `gamma` del SVM.
- **Dashboard del bot**: mostrar en pantalla los pesos del ensemble en
  tiempo real para que el público vea qué patrón está explotando.
- **Recolectar partidas humanas reales** para entrenar/validar el bot
  fuera de los arquetipos sintéticos del notebook.

---

## 10. Referencias

- **Wang, Xu, Zhou (2014).** *Social cycling and conditional responses
  in the Rock-Paper-Scissors game.* Nature Scientific Reports.
- **Egnor (2000).** *Iocaine Powder.* International RoShamBo
  Programming Competition.
- **Freund & Schapire (1997).** *A decision-theoretic generalization of
  on-line learning.* JCSS 55(1).
- **OpenCV Zoo** — YuNet y SFace:
  https://github.com/opencv/opencv_zoo
- **MediaPipe Hands** — Google: https://developers.google.com/mediapipe
