# Memoria — Trabajo Final de Aprendizaje Automático

> **Sistema integrado de visión por computador con login biométrico,
> reconocimiento de gestos y bot adaptativo: juego de
> Piedra · Papel · Tijera por webcam.**

Este documento sigue los apartados mínimos pedidos en el enunciado
del trabajo (Introducción, Definición del problema, Marco teórico,
Metodología, Modelos/algoritmos, Datos, Resultados, Discusión) y se
ha estructurado dando algo más de extensión a los apartados que
mayor peso tienen en la rúbrica:

| Criterio de la rúbrica | Peso | Apartado(s) de esta memoria |
| --- | --- | --- |
| Marco teórico | 20 % | §3 |
| Implementación / Análisis | 20 % | §5, §9, apéndices |
| Definición del problema | 15 % | §2 |
| Metodología | 15 % | §4 |
| Resultados y discusión | 10 % | §7, §8 |

El [README.md](README.md) cubre la parte de instalación y uso. Aquí
nos centramos en el contenido académico.

---

## 1. Introducción y motivación

El proyecto integra **tres bloques de aprendizaje automático muy
distintos** en una sola aplicación, lo que nos ha permitido cubrir
buena parte del temario de la asignatura en un único entregable:

| Bloque | Paradigma | Técnica |
| --- | --- | --- |
| **Login biométrico** | Aprendizaje profundo (modelos preentrenados) | YuNet (detección facial) + SFace (embeddings) + similitud coseno |
| **Clasificador de gestos** | Aprendizaje supervisado clásico | MediaPipe Hands (21 landmarks) + SVM-RBF / KNN / RandomForest |
| **Bot RPS adaptativo** | Aprendizaje *online* con expertos | Ensemble Hedge sobre 24 predictores (Markov, frecuencia, WSLS) |

La motivación es triple:

- **Pedagógica**: probar en un mismo proyecto modelos *preentrenados*
  (SFace), *modelos clásicos entrenados con nuestros datos* (SVM) y
  *aprendizaje online* (Hedge), comparando las virtudes de cada
  paradigma. Nos obliga a entender cuándo conviene cada uno y cómo
  encajar pipelines de naturaleza distinta.
- **Práctica**: construir algo *demostrable en vivo* el día de la
  presentación, con la cámara del portátil y sin requerir GPU.
- **Lúdica**: el juego permite que el público interactúe con el
  sistema y compruebe por sí mismo cómo el bot aprende sus patrones.

---

## 2. Definición del problema

El sistema completo encadena **tres problemas de ML** bien
diferenciados:

### 2.1 Login biométrico (verificación facial 1:N)

Dado un flujo de vídeo y una base de datos de usuarios registrados,
decidir si la cara que aparece corresponde a alguno de ellos (y a
cuál). Es un problema de **verificación 1:N** con clase abierta:
puede aparecer una cara *desconocida* que el sistema debe rechazar.
Restricciones:

- Tiempo real (≥ 10 fps) sobre webcam estándar.
- Pocas muestras por usuario (~50 embeddings).
- Funcionar sobre CPU.

### 2.2 Clasificación de gestos

Dada una imagen de una mano capturada por la webcam, clasificarla en
una de tres clases: `piedra`, `papel`, `tijera`. Restricciones:

- Robustez a la **posición, distancia y rotación** de la mano en el
  encuadre.
- Latencia < 50 ms por predicción.
- Dataset propio recogido por los autores.

### 2.3 Bot adaptativo de RPS

Dado el historial de jugadas humanas en una partida, elegir la
**mejor jugada del bot para la siguiente ronda**. Es un problema
de **aprendizaje secuencial contra un adversario no estacionario**
(el humano puede cambiar de estrategia o intentar engañar al bot).
La métrica objetivo es el *edge*:

$$\text{edge} = P(\text{bot gana}) - P(\text{bot pierde}) \in [-1, +1]$$

con baseline 0 (jugar uniforme aleatorio garantiza edge = 0 por
equilibrio de Nash en RPS).

---

## 3. Marco teórico

### 3.1 Detección y reconocimiento facial

- **Detección — YuNet** (Wu et al., 2023). Red convolucional ligera
  (≈ 200 KB) diseñada para tiempo real sobre CPU. Por cada cara
  devuelve una *bounding box* y cinco puntos faciales (ojos, nariz,
  comisuras), entrenada por OpenCV Zoo sobre WIDER FACE.
- **Reconocimiento — SFace** (Zhong et al., 2021). Red de embedding
  facial. Toma la cara alineada (~ 112×112 px) y la proyecta en un
  vector de **128 dimensiones** sobre la esfera unidad. Su función de
  coste es una *softmax con margen adaptativo* diseñada para que
  vectores de la misma persona estén cerca por *similitud coseno* y
  los de personas distintas, lejos. La verificación se reduce a un
  producto escalar entre vectores normalizados.
- **Métrica**: similitud coseno
  $\cos(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v}$, con
  ambos vectores L2-normalizados. Umbral oficial publicado: 0.363.

### 3.2 Extracción de features de la mano

**MediaPipe Hands** (Google, 2020) es un *pipeline* de dos etapas.
Primero un detector de palmas (BlazePalm) localiza la mano; después
un regresor predice **21 *landmarks*** (x, y, z) por mano. Ambos
modelos están entrenados con datasets masivos sintéticos y reales,
y optimizados para móvil. La salida es un vector estructurado y
casi libre de ruido que codifica la *pose* de la mano: muñeca,
nudillos, falanges y puntas de los cinco dedos.

### 3.3 Aprendizaje supervisado clásico

Modelos comparados (todos en `scikit-learn`):

- **KNN** (`k=5`): clasifica por mayoría entre los 5 vecinos más
  cercanos en el espacio de features estandarizadas. Sin entrenamiento
  real (lazy learning).
- **SVM con kernel RBF**: encuentra el hiperplano de margen máximo
  en un espacio implícito de dimensión infinita gracias al *kernel
  trick*:
  $$K(\mathbf{x}, \mathbf{x}') = \exp\bigl(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\bigr)$$
  Parámetros: `C = 10`, `gamma = "scale"`. Tiende a funcionar muy
  bien cuando el número de features (42 en nuestro caso) es del
  mismo orden o mayor que el ruido del problema.
- **Random Forest** (300 árboles): conjunto de árboles de decisión
  entrenados con *bagging* + aleatoriedad en las features.
  Robusto frente al sobreajuste, da importancia de features
  *de regalo*.

Cada modelo va dentro de un `Pipeline` que comienza con
`StandardScaler`, para que SVM y KNN no se vean perjudicados por
escalas distintas en los componentes del vector.

### 3.4 Aprendizaje online con expertos: el algoritmo Hedge

Para el bot usamos **Hedge** (Freund & Schapire, 1997), un algoritmo
clásico del *online learning*. Mantenemos $N$ expertos (aquí 24
predictores), cada uno con un peso $w_i$. En cada ronda:

1. Combinamos sus predicciones de la jugada del humano:
   $\hat{p} = \sum_i \frac{w_i}{\sum_j w_j} \cdot p_i$.
2. El bot elige la acción que maximiza la utilidad esperada
   (P(gana) − P(pierde)) contra $\hat{p}$.
3. Tras observar la jugada real $a$, actualizamos los pesos
   premiando a los expertos que le asignaron alta probabilidad:
   $$w_i \leftarrow w_i \cdot \exp\bigl(\eta \cdot (p_i[a] - 1/3)\bigr)$$
4. Aplicamos un *decay* $\gamma = 0.99$ para "olvidar"
   rendimientos antiguos y poder adaptarse si el humano cambia de
   estrategia.

**Garantía teórica de Hedge**: su *regret* frente al mejor experto
en hindsight es sublineal —concretamente $O(\sqrt{T \log N})$— lo
que en la práctica significa que **converge al mejor predictor sin
necesidad de saber a priori cuál es**.

### 3.5 Meta-niveles tipo *Iocaine Powder*

Idea de Egnor (2000): si el bot predice $X$ y juega $\text{beats}(X)$,
un humano avispado puede contraatacar jugando
$\text{beats}(\text{beats}(X))$. Para protegernos, cada predictor se
duplica en **tres meta-niveles**:

- Nivel 0: predicción directa.
- Nivel 1: rotamos la predicción una posición en el ciclo
  R → P → T.
- Nivel 2: rotamos dos posiciones.

Así pasamos de 8 predictores base a un ensemble de 24, y dejamos
que Hedge descubra qué nivel funciona mejor contra el oponente
actual.

### 3.6 ¿Por qué clasificar landmarks y no píxeles?

Una CNN entrenada sobre imágenes de manos resolvería el problema
pero exigiría mucha más data, GPU para entrenar, y aprendería
correlaciones espurias con el fondo y la iluminación. Al delegar
la extracción de features a MediaPipe, convertimos el problema en
**clasificar un vector de 42 floats**, que cualquier modelo clásico
resuelve trivialmente. Es un ejemplo de libro de **transfer
learning implícito** + **feature engineering por descomposición del
problema**.

---

## 4. Metodología

### 4.1 Reconocimiento facial

1. **Registro de usuarios**: cada usuario captura ~50 embeddings con
   la cabeza en distintas poses. Almacenamos los 50 vectores
   íntegros (no un *centroide*) para que la verificación se haga
   contra el *mejor* candidato — esto da mucha más robustez a
   variaciones de pose/luz.
2. **Login**: por cada frame, YuNet → SFace → coseno contra el
   conjunto de embeddings de cada usuario. Solo etiquetamos al
   *mejor* usuario si el coseno supera 0.363; mapeamos a un % para
   feedback visual y exigimos ≥ 80 %.
3. **Streak temporal**: para evitar falsos positivos transitorios y
   dar tiempo a la demostración, exigimos **5 segundos consecutivos**
   de reconocimiento del mismo usuario antes de cerrar el login.

### 4.2 Clasificación de gestos

1. **Recolección del dataset** con `capturar_gestos.py`: la mano se
   ve por webcam, MediaPipe extrae los 21 landmarks y se graban en
   CSV cuando pulsamos `0`/`1`/`2`. Recogimos **~500 muestras por
   clase**, variando distancia, ángulo, rotación de muñeca,
   condiciones de luz y alternando manos. Tras la captura inicial,
   limpiamos manualmente filas mal etiquetadas hasta dejar el dataset
   balanceado en 1498 muestras.
2. **Preprocesado**: normalización geométrica (ver §5.2). Es la
   decisión que más impacta al rendimiento.
3. **Entrenamiento**: split estratificado 80/20, `StandardScaler` +
   modelo, comparación por accuracy en test. El ganador se guarda en
   `modelo_gestos.joblib`.
4. **Inferencia**: misma función de normalización que en training,
   importada directamente del módulo de entrenamiento para evitar
   *training/serving skew*.

### 4.3 Bot RPS

Implementado en [bot_rps.py](bot_rps.py), extraído del notebook
[proyecto_rps_ml.ipynb](proyecto_rps_ml.ipynb). La metodología:

1. **Generación de arquetipos humanos sintéticos** (sesgo a piedra,
   WSLS, ciclos, anti-repetición, mezcla realista) con sesgos
   calibrados a partir de la literatura (Wang et al., 2014; Sato et
   al., 2022).
2. **Definición de 8 predictores base** + 3 meta-niveles ⇒ 24.
3. **Hedge** como meta-selector.
4. **Evaluación**: 1000 rondas × 3 ensayos por (bot × arquetipo).
5. **Comparación contra una baseline supervisada**: Random Forest
   entrenado offline con features *hand-engineered* del contexto
   reciente.

En el juego, el bot:

1. Decide su jugada **antes** de la cuenta atrás → no ve la mano
   humana (no hace trampas).
2. Tras clasificar el gesto, recibe el `update` con la jugada real.

---

## 5. Modelos / algoritmos utilizados

### 5.1 Login facial

| Modelo | Tipo | Tamaño | Origen | Uso |
| --- | --- | --- | --- | --- |
| **YuNet** | CNN ligera de detección | ~200 KB (ONNX) | OpenCV Zoo | Localizar la cara y sus 5 puntos clave |
| **SFace** | CNN de embedding 128-d | ~35 MB (ONNX) | OpenCV Zoo | Vectorizar la identidad para comparar por coseno |

Ambos son **preentrenados, no los entrenamos nosotros**. Se
descargan automáticamente la primera vez desde
`https://github.com/opencv/opencv_zoo`.

### 5.2 Clasificador de gestos

- **MediaPipe Hands** como extractor de 21 landmarks
  (preentrenado).
- **Normalización geométrica** (`normalizar_landmarks` en
  [entrenar.py](entrenar.py)):
  1. Trasladar todos los puntos para que la muñeca (punto 0) sea el
     origen `(0, 0)`.
  2. Escalar dividiendo por la distancia muñeca → nudillo del dedo
     medio (punto 9), una medida estable del "tamaño" de la mano
     en la imagen.
- Tres candidatos en `Pipeline(StandardScaler + clasificador)`:
  - `KNeighborsClassifier(n_neighbors=5)`
  - `SVC(kernel='rbf', C=10, gamma='scale', probability=True)` *(ganador)*
  - `RandomForestClassifier(n_estimators=300, random_state=42)`

### 5.3 Bot RPS — ensemble Hedge

| Predictor | Idea |
| --- | --- |
| `UniformPredictor` | Baseline aleatorio |
| `FrequencyPredictor` | Frecuencias acumuladas con suavizado de Laplace |
| `RecentFrequencyPredictor(10)` y `(30)` | Frecuencias en ventana móvil |
| `MarkovN(1)`, `(2)`, `(3)` | Cadenas de Markov de órdenes 1–3 |
| `WSLSPredictor` | Detector explícito de Win-Stay / Lose-Shift |
| `MetaShift(base, 0/1/2)` | Rota la predicción (protección anti-contra-anticipación) |

Total: **8 base × 3 meta-niveles = 24 predictores**.
Meta-selector: **HedgeBot** con `η = 0.3` y `decay = 0.99`.

---

## 6. Datos empleados

### 6.1 Dataset de gestos (propio)

- **Fichero**: [datos_gestos.csv](datos_gestos.csv)
- **Tamaño**: 1498 filas (≈ 500 por clase).
- **Features**: 42 floats por fila (`x0, y0, …, x20, y20`),
  valores en [0, 1] (coordenadas normalizadas por MediaPipe respecto
  al frame).
- **Etiqueta**: texto `piedra` / `papel` / `tijera`.
- **Recolección**: hecha por los autores con la webcam, variando
  distancia, ángulo y posición, en distintas condiciones de luz, y
  alternando entre ambas manos. Se descartaron manualmente capturas
  mal etiquetadas para llegar al balance final.

### 6.2 Base de embeddings faciales

- **Fichero**: `usuarios.pkl` (no versionado)
- **Formato**: `dict[username -> list[np.ndarray(128,)]]`
- **Contenido**: 50 embeddings por usuario registrado. Durante el
  desarrollo se registraron 3 usuarios.
- **Privacidad**: al tratarse de datos biométricos, el fichero está en
  el `.gitignore` y no se publica. Se genera en local con
  `app.py -> Registrar usuario`.

### 6.3 Datos del bot RPS (sintéticos)

- 5 arquetipos humanos sintéticos calibrados con la literatura.
- 1000–2000 rondas por arquetipo, 3 ensayos por configuración.
- **No hay dataset humano real**: la evaluación es íntegramente
  *in silico*. Es una limitación intencionada y reconocida (§8).

---

## 7. Resultados

### 7.1 Clasificador de gestos (test estratificado 20 %, 300 muestras)

| Modelo | Accuracy |
| --- | --- |
| KNN (k=5) | 98.33 % |
| **SVM-RBF** | **99.33 %** |
| RandomForest (300 árboles) | 99.33 % |

Reporte de clasificación del modelo elegido:

```
              precision    recall  f1-score   support
       papel     0.9900    0.9900    0.9900       100
      piedra     0.9901    1.0000    0.9950       100
      tijera     1.0000    0.9900    0.9950       100
    accuracy                         0.9933       300
```

Matriz de confusión (filas = real, columnas = predicho):

```
            papel  piedra  tijera
papel         99     1       0
piedra         0   100       0
tijera         1     0      99
```

### 7.2 Bot RPS (1000 rondas × 3 ensayos por par)

*Edge = P(bot gana) − P(bot pierde)*, mayor es mejor.

| Bot | Edge promedio | Edge peor caso |
| --- | --- | --- |
| Aleatorio | 0.00 | 0.00 |
| Frecuencia | −0.06 | −0.30 (vs WSLS) |
| Markov-1 | +0.33 | −0.01 |
| Markov-2 | +0.36 | +0.08 |
| WSLS único | +0.13 | −0.20 (vs Cycler) |
| Random Forest supervisado | +0.27 | +0.03 |
| **Ensemble Hedge** | **+0.37** | **+0.08** |

**Curva de aprendizaje**: contra **Cycler** y **WSLS** el ensemble
alcanza ~80 % de edge en menos de 50 rondas; contra arquetipos más
ruidosos (`RockBiased`, `AntiRepeater`, `MixedHuman`) el edge se
estabiliza en 10–20 % tras unas pocas decenas de rondas.

### 7.3 Sistema integrado (cualitativo)

- **Login**: con 50 embeddings por usuario, el sistema identifica
  correctamente en ≈ 1 segundo (mostramos 5 s por requisito de UX).
  No hemos observado falsos positivos entre los dos usuarios
  registrados.
- **Juego**: en condiciones normales (mano frontal, luz uniforme),
  el clasificador acierta el gesto en todas las rondas que jugamos
  durante el desarrollo. Las dudas aparecen cuando se hace el gesto
  *a mitad de camino* entre papel y tijera, congruente con la matriz
  de confusión.
- **Latencia**: < 60 ms por frame en un portátil sin GPU, holgado
  para 25–30 fps reales.

---

## 8. Discusión de resultados

### Por qué el clasificador funciona tan bien (¿demasiado bien?)

99.33 % de accuracy con solo 1500 muestras y modelos clásicos puede
parecer sospechosamente alto. La explicación es que **el trabajo
duro ya lo hace MediaPipe**: su CNN preentrenada convierte una
imagen ruidosa en 42 floats casi libres de ruido que codifican
exactamente la información relevante (la pose de la mano). El
problema deja de ser visión por computador y pasa a ser una
clasificación trivial en un espacio bien estructurado.

Esto ilustra una idea importante de la asignatura: **encuadrar bien
el problema y elegir buenas features es a menudo más valioso que
elegir el clasificador**.

### Por qué la normalización es la decisión que más impacta

Sin la traslación y el escalado por la longitud muñeca → nudillo
medio, los mismos 42 floats codifican **dónde** está la mano en el
encuadre y **lo grande** que se ve, no qué gesto hace. Una mano de
papel a 30 cm y otra a 80 cm producen vectores muy distintos, y el
clasificador aprende a clasificar por distancia. Tras normalizar,
ambas producen vectores casi idénticos. Es un caso de libro de
*feature engineering*.

### Por qué el Random Forest del notebook pierde frente a Hedge

Cuando intentamos *aprender el oponente* con un modelo supervisado
clásico, lo entrenamos con un mix de arquetipos *offline*. En
*test time*, el modelo no sabe **a qué oponente concreto** se
enfrenta y aplica una política fija aprendida en promedio.

Hedge, en cambio, **adapta sus pesos cada ronda**: si el oponente
parece un Cycler, todo el peso migra a `Markov-1`; si parece WSLS,
migra al `WSLSPredictor`. Esto refleja una diferencia conceptual
profunda entre **aprendizaje supervisado fuera de línea** y
**aprendizaje online con expertos**: para entornos no estacionarios,
el segundo paradigma domina.

### Por qué confiamos en el bot sin haberlo probado contra humanos

No lo hemos hecho. Es la principal limitación del trabajo y la
recogemos como tal. La evaluación contra arquetipos sintéticos da
**evidencia indirecta** de que el bot supera a las baselines y se
adapta, pero no garantiza el mismo rendimiento contra humanos
reales, que pueden:

- Adaptarse al bot (carrera adaptativa que los meta-niveles ayudan
  pero no resuelven del todo).
- Tener sesgos no contemplados en nuestros arquetipos.

Una extensión natural sería una demo web que recogiera partidas
reales y se reentrenase con ellas.

### Sobre la integración de los tres bloques

La aportación más interesante a nivel de ingeniería es probablemente
**enchufar los tres bloques**. Cada uno toma decisiones a un ritmo
distinto (login: ~1 s para acumular confianza; gestos: una
predicción cada ~30 ms; bot: una predicción por ronda) y tiene un
ciclo de vida distinto (login una sola vez, el resto durante toda
la sesión).

Decisiones técnicas no triviales:

- **Subproceso para el juego**: Tkinter y OpenCV/Qt no se llevan
  bien en el mismo proceso. Tras destruir la ventana de Tk, la
  siguiente `cv2.imshow` se quedaba congelada. Arrancar el juego
  con `subprocess.run([sys.executable, "juego.py", nombre])` da un
  intérprete limpio y aísla fallos.
- **Reutilización de la normalización entre training y serving**:
  `jugar.py` y `juego.py` importan `normalizar_landmarks` directamente
  de `entrenar.py`, eliminando una clase entera de bugs.
- **Compatibilidad numpy 2 → numpy 1**: el `usuarios.pkl` se
  serializó en un PC con numpy 2.x; aquí usamos numpy 1.26 (lo
  exige mediapipe 0.10.21). Un `_CompatUnpickler` con `find_class`
  remapea `numpy._core.*` → `numpy.core.*` al vuelo.

### Análisis crítico de las elecciones

| Decisión | Alternativa | Por qué la nuestra |
| --- | --- | --- |
| Landmarks + SVM | CNN end-to-end | Mucho menos data, sin GPU, fácilmente interpretable |
| 50 embeddings/usuario | Solo el centroide | Robustez a pose/luz; cuesta poca memoria |
| Streak 5 s en login | Cierre instantáneo | UX y demo visibles |
| Subproceso para el juego | Mismo proceso | Estabilidad: Qt + Tk en el mismo proceso es frágil |
| Hedge | Único predictor "el bueno" | Adaptación a oponentes no estacionarios |

### Limitaciones honestas

- Dataset pequeño y homogéneo (2 personas, mismo portátil): el
  modelo podría no generalizar a manos muy distintas. Mitigación:
  añadir más muestras en la captura.
- Solo 2 usuarios faciales registrados; el sistema no está
  estresado para 1:N grandes.
- Bot evaluado solo *in silico*.

---

## 9. Conclusiones

El proyecto cumple los tres objetivos planteados, alcanza una
*accuracy* del **99.3 %** en el clasificador de gestos y un **edge
de +0.37 promedio** en el bot, y combina los tres bloques en una
aplicación demostrable en vivo. Lo más valioso desde el punto de
vista didáctico ha sido contrastar **tres paradigmas distintos** de
ML en un mismo entregable: modelos preentrenados (SFace),
aprendizaje supervisado clásico (SVM) y aprendizaje online con
expertos (Hedge), entendiendo cuándo y por qué conviene cada uno.

---

## 10. Referencias

- **Wang, Z., Xu, B., Zhou, H.-J. (2014).** *Social cycling and
  conditional responses in the Rock-Paper-Scissors game.* Nature
  Scientific Reports 4:5830.
- **Sato et al. (2022).** *Human Randomness in the Rock-Paper-Scissors
  Game.* MDPI Applied Sciences 12(23):12192.
- **Egnor, D. (2000).** *Iocaine Powder.* International RoShamBo
  Programming Competition.
- **Freund, Y., Schapire, R. (1997).** *A decision-theoretic
  generalization of on-line learning and an application to boosting.*
  Journal of Computer and System Sciences 55(1).
- **Zhong, Y. et al. (2021).** *SFace: Sigmoid-Constrained
  Hypersphere Loss for Robust Face Recognition.* IEEE TIP.
- **Wu, W. et al. (2023).** *YuNet: A Tiny Millisecond-level Face
  Detector.* Machine Intelligence Research.
- **MediaPipe Hands** — Google Research, 2020.
- **OpenCV Zoo** — repositorio oficial de modelos preentrenados
  para OpenCV: https://github.com/opencv/opencv_zoo

---

## Apéndice A — Posibles mejoras / trabajo futuro

- **Integrar el Random Forest del notebook como predictor extra del
  Hedge**, en vez de tenerlos como modelos paralelos.
- **Landmark 3D**: usar también la `z` de MediaPipe para distinguir
  gestos con solapamiento (papel vs tijera vistas oblicuamente).
- **Más gestos**: lagarto, Spock, etc. (basta con capturar más datos
  y reentrenar).
- **GridSearchCV** sobre `C` y `gamma` del SVM y `n_estimators` /
  `max_depth` del Random Forest.
- **Dashboard del bot**: visualizar en pantalla la evolución de los
  pesos del ensemble en tiempo real, para mostrar al público qué
  patrón está explotando.
- **Recolectar partidas reales** mediante una demo web para evaluar
  el bot fuera de los arquetipos sintéticos.

## Apéndice B — Preguntas frecuentes (preparación para la defensa)

- **¿Por qué SVM y no una red neuronal?**
  Con 1500 muestras y 42 features ya normalizadas geométricamente,
  una NN no aporta valor; el SVM converge en milisegundos y es
  determinista. Si fuésemos a clasificar píxeles directamente, sí
  habría que usar una CNN.

- **¿Por qué SFace y no, p. ej., FaceNet o ArcFace?**
  SFace tiene un wrapper directo en `cv2.FaceRecognizerSF`,
  funciona en CPU y los embeddings son comparables vía coseno con
  un umbral oficial. Las alternativas darían un rendimiento
  parecido pero requerirían más cableado.

- **¿Y si el bot se enfrenta a un humano que juega aleatorio?**
  El edge esperado tiende a 0 (no se puede ganar al azar puro). Los
  pesos del Hedge convergerían al `UniformPredictor`. Esto es
  consistente con la teoría: contra un oponente sin sesgo, ninguna
  estrategia bate al equilibrio de Nash.

- **¿Cómo se evita el sobreajuste en el SVM con `C=10`?**
  El split estratificado deja un 20 % fuera (300 muestras), y el
  accuracy en test (99.3 %) es coherente con el del train, sin
  brecha apreciable. La normalización geométrica reduce además la
  varianza de los inputs.

- **¿Qué pasa si MediaPipe no detecta la mano cuando llega "¡YA!"?**
  La ronda se anula explícitamente (mensaje "no se detectó mano,
  ronda anulada") y no penaliza ni a humano ni a bot.
