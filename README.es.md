# Rock · Paper · Scissors — ML edition

[English](README.md) · **Español**

Juego de piedra-papel-tijera por webcam que mete **tres paradigmas de machine
learning muy distintos** en una sola aplicación: entras con tu cara, sacas tu
jugada con la mano y un bot adaptativo aprende tus patrones y te contraataca.

| Bloque | Paradigma | Técnica |
| --- | --- | --- |
| **Login facial** | Aprendizaje profundo preentrenado | YuNet (detección) + SFace (embeddings de 128-D) + similitud coseno |
| **Clasificador de gestos** | Aprendizaje supervisado clásico | MediaPipe Hands (21 landmarks) + SVM-RBF |
| **Bot adaptativo** | Aprendizaje online con expertos | Ensemble Hedge sobre 24 predictores (Markov, frecuencia, WSLS) |

Todo funciona en tiempo real sobre CPU. No hace falta GPU.

---

## Cómo funciona

```
app.py ──► login facial (YuNet + SFace) ──► juego.py
                                              ├─ MediaPipe → 21 landmarks
                                              ├─ SVM-RBF   → piedra/papel/tijera
                                              └─ HedgeBot  → jugada del bot
```

1. `app.py` abre un GUI de Tkinter con dos opciones: registrar usuario o
   iniciar sesión.
2. La webcam detecta tu cara con **YuNet**, extrae un embedding de 128
   dimensiones con **SFace** y lo compara por similitud coseno contra la base
   local de usuarios.
3. Una vez te reconoce, lanza `juego.py` en un proceso aparte. Pulsas `SPACE`,
   corre la cuenta atrás 3-2-1, **MediaPipe** extrae los 21 landmarks de tu
   mano, el **SVM** clasifica el gesto y el **bot** revela su jugada.
4. El bot actualiza los pesos de sus expertos con tu jugada para anticiparte en
   rondas posteriores.

---

## Puesta en marcha

Requiere Python 3.10 o superior (el código usa la sintaxis de tipos `str | None`).

```bash
git clone https://github.com/migulangel19/rock-paper-scissor.git
cd rock-paper-scissor

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Tkinter no se instala con pip:
sudo apt-get install -y python3-tk     # Debian / Ubuntu
```

Los modelos faciales (YuNet ~200 KB, SFace ~35 MB) se **descargan solos** desde
[OpenCV Zoo](https://github.com/opencv/opencv_zoo) la primera vez que ejecutas
el login, al directorio temporal del sistema.

---

## Uso

```bash
source venv/bin/activate
python app.py
```

**La primera vez — registrarse.** Escribe un nombre, pulsa el botón de
registro, mira a cámara, pulsa `SPACE` y mueve la cabeza con suavidad mientras
se completa la barra inferior. Captura 50 embeddings para que el matching sea
robusto a pose, iluminación y expresión.

**Después — iniciar sesión y jugar.** Pulsa el botón principal. La cámara te
identifica, mantiene un streak de confianza de 1,5 s con una barra verde, te
saluda y lanza el juego.

**En el juego:**

| Tecla | Acción |
| --- | --- |
| `SPACE` | Arranca la cuenta atrás 3-2-1 — saca tu jugada en el "¡YA!" |
| `r` | Reinicia el marcador y los pesos aprendidos del bot |
| `q` | Salir |

Para probar el juego sin pasar por el login:

```bash
python juego.py                # sin saludo
python juego.py "Tu Nombre"    # con saludo personalizado
```

---

## Los tres bloques

### 1. Login facial — `login_facial.py`

Los dos modelos están **preentrenados**; aquí no se entrena nada. YuNet
devuelve la bounding box más 5 puntos faciales, que `alignCrop` usa para pasarle
a SFace un recorte bien alineado. SFace devuelve un vector de identidad de 128
floats.

Cada usuario guarda ~50 embeddings. Al identificar se compara el embedding
actual contra todos ellos y gana la mejor similitud coseno por usuario. El
umbral oficial de SFace para "misma persona" es **0,363**, que mapeamos a un
porcentaje más amistoso (umbral → 50 %, coseno 0,85 → 100 %) y exigimos ≥ 80 %.
Se requiere un **streak de 1,5 s** de reconocimiento sostenido antes de cerrar
el login, lo que descarta falsos positivos puntuales y hace el proceso visible
durante una demo.

### 2. Clasificador de gestos — `entrenar.py`, `capturar_gestos.py`

En vez de entrenar una CNN sobre píxeles, la extracción de características se
delega en **MediaPipe Hands**, que convierte una imagen ruidosa en 21 landmarks
(x, y). El problema deja de ser visión por computador y pasa a ser una
clasificación en un espacio de 42 dimensiones bien estructurado.

La decisión que más impacta es la **normalización**: se trasladan los landmarks
para que la muñeca sea el origen y se escalan por la distancia muñeca → nudillo
medio. Sin ella, esos mismos 42 floats codifican *dónde* está la mano y *cómo
de grande* se ve, en lugar de *qué gesto* hace, y el clasificador aprende a
clasificar por distancia a la cámara.

Dataset: **1.498 muestras** recogidas por los autores (500 papel, 499 piedra,
499 tijera), split estratificado 80/20 y `StandardScaler` dentro de un
`Pipeline` de sklearn.

### 3. Bot adaptativo — `bot_rps.py`

24 predictores simples (uniforme, frecuencia, frecuencia reciente, Markov de
órdenes 1-3, win-stay/lose-shift), cada uno envuelto en una capa meta-shift que
juega también las anti-estrategias, combinados por el algoritmo **Hedge**. En
cada ronda los pesos se mueven exponencialmente hacia el experto que mejor te
esté prediciendo.

Por eso gana a un Random Forest supervisado entrenado offline: el bosque aplica
una política fija aprendida sobre un oponente *promedio*, mientras que Hedge
repondera contra el oponente *concreto* que tiene delante. En entornos no
estacionarios, el aprendizaje online con expertos domina.

---

## Resultados

**Clasificador de gestos** — test estratificado del 20 %, 300 muestras:

| Modelo | Accuracy |
| --- | --- |
| KNN (k=5) | 98,33 % |
| **SVM-RBF** (elegido) | **99,33 %** |
| Random Forest (300 árboles) | 99,33 % |

La accuracy es alta porque MediaPipe ya hace el trabajo duro. Eso es en sí la
lección: encuadrar bien el problema y elegir buenas características suele
importar más que elegir el clasificador.

**Bot** — 1.000 rondas × 3 ensayos contra arquetipos humanos sintéticos.
*Edge = P(gana el bot) − P(pierde el bot)*, más alto es mejor:

| Estrategia | Edge medio | Edge en el peor caso |
| --- | --- | --- |
| Aleatorio | 0,00 | 0,00 |
| Frecuencia | −0,06 | −0,30 |
| Markov-1 | +0,33 | −0,01 |
| Markov-2 | +0,36 | +0,08 |
| WSLS solo | +0,13 | −0,20 |
| Random Forest supervisado | +0,27 | +0,03 |
| **Ensemble Hedge** | **+0,37** | **+0,08** |

La latencia queda por debajo de 60 ms por frame en un portátil sin GPU.

---

## Estructura del repositorio

```
app.py                  # Punto de entrada: GUI Tk (login + registro) y lanzador del juego
login_facial.py         # YuNet + SFace, base de usuarios, bucles de registro y login
juego.py                # El juego: webcam, inferencia de gestos, bot, marcador
bot_rps.py              # Ensemble Hedge sobre 24 predictores
capturar_gestos.py      # Herramienta de captura del dataset
entrenar.py             # Pipeline de entrenamiento y comparativa de modelos
datos_gestos.csv        # 1.498 muestras de gestos
modelo_gestos.joblib    # Pipeline SVM-RBF entrenado + lista de etiquetas
proyecto_rps_ml.ipynb   # Notebook: diseño del bot, arquetipos y evaluación
MEMORIA.md              # Memoria académica completa
fotos/                  # Iconos de las jugadas para el panel de resultado
```

## Reentrenar el modelo de gestos

```bash
python capturar_gestos.py   # 0 = piedra, 1 = papel, 2 = tijera; añade al CSV
python entrenar.py          # compara KNN / SVM / RF y guarda el mejor en .joblib
```

`juego.py` importa `normalizar_landmarks` directamente de `entrenar.py`, así que
entrenamiento e inferencia no pueden desincronizarse.

---

## Privacidad

Los embeddings faciales son datos biométricos. La base de usuarios
(`usuarios.pkl`) está **en el .gitignore y no se publica nunca**: solo existe en
la máquina donde se registraron los usuarios. Si clonas el repo empiezas con la
base vacía; te registras con `app.py` y los datos se quedan en local.

## Limitaciones

- El bot **nunca se ha evaluado contra humanos reales**, solo contra arquetipos
  sintéticos. Eso da evidencia indirecta de que supera a las baselines y de que
  se adapta, pero no garantiza el mismo edge contra jugadores de carne y hueso.
- El clasificador de gestos está entrenado con las manos de los autores y con
  iluminación razonable. Los gestos a mitad de camino entre papel y tijera son
  donde la matriz de confusión enseña sus únicos errores.
- La base de caras es pequeña (tres usuarios registrados durante el
  desarrollo), así que la tasa de falsos positivos no está medida de forma
  significativa.

## Créditos

Proyecto académico de la asignatura de Aprendizaje Automático, por **Miguel
Merino**, **Pablo Barranco** y **Pablo Rodríguez**.

El login facial y el notebook del bot vienen del trabajo de un compañero; el
clasificador de gestos y la integración de los tres bloques se construyeron
para este proyecto.

Referencias principales: Wang, Xu y Zhou (2014), *Social cycling and
conditional responses in the Rock-Paper-Scissors game*; Freund y Schapire
(1997), *A decision-theoretic generalization of on-line learning*; Egnor
(2000), *Iocaine Powder*; y [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
para YuNet y SFace.

## Licencia

[MIT](LICENSE).
