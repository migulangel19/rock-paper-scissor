# Rock · Paper · Scissors — ML edition

**English** · [Español](README.es.md)

A webcam game of rock-paper-scissors that puts **three very different machine
learning paradigms** into a single application: you log in with your face, you
throw your move with your hand, and an adaptive bot learns your patterns and
counters them.

| Block | Paradigm | Technique |
| --- | --- | --- |
| **Face login** | Pretrained deep learning | YuNet (detection) + SFace (128-D embeddings) + cosine similarity |
| **Gesture classifier** | Classical supervised learning | MediaPipe Hands (21 landmarks) + SVM-RBF |
| **Adaptive bot** | Online learning with experts | Hedge ensemble over 24 predictors (Markov, frequency, WSLS) |

Everything runs in real time on CPU. No GPU required.

---

## How it works

```
app.py ──► face login (YuNet + SFace) ──► juego.py
                                            ├─ MediaPipe → 21 hand landmarks
                                            ├─ SVM-RBF   → rock / paper / scissors
                                            └─ HedgeBot  → the bot's move
```

1. `app.py` opens a Tkinter GUI with two options: register a user or log in.
2. The webcam detects your face with **YuNet**, extracts a 128-dimensional
   embedding with **SFace**, and compares it by cosine similarity against the
   local user database.
3. Once you are recognised, it launches `juego.py` in a separate process. You
   press `SPACE`, a 3-2-1 countdown runs, **MediaPipe** extracts the 21
   landmarks of your hand, the **SVM** classifies the gesture, and the **bot**
   reveals its move.
4. The bot updates its expert weights with your move so it can anticipate you
   in later rounds.

---

## Quickstart

Requires Python 3.10 or newer (the code uses `str | None` type syntax).

```bash
git clone https://github.com/migulangel19/rock-paper-scissor.git
cd rock-paper-scissor

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Tkinter is not a pip package:
sudo apt-get install -y python3-tk     # Debian / Ubuntu
```

The face models (YuNet ~200 KB, SFace ~35 MB) are **downloaded automatically**
from [OpenCV Zoo](https://github.com/opencv/opencv_zoo) the first time you run
the login, into your system temp directory.

---

## Usage

```bash
source venv/bin/activate
python app.py
```

### 1. Register — first time only

Type a name, press **Registrar usuario**, look at the camera and press `SPACE`.
Move your head gently while the bottom bar fills up: it captures 50 embeddings,
which is what makes the match robust to pose, lighting and expression. Press
`ESC` to cancel.

### 2. Log in

Press **Iniciar sesión y jugar**. The camera identifies you and a green bar
tracks a 1.5 s streak of sustained recognition; hold still until it completes.
It greets you and launches the game in its own window.

### 3. Play a round

1. Press `SPACE` to start the round.
2. The bot **locks in its move before the countdown**, without looking at your
   hand. It predicts you from your past moves, never by peeking at the current
   frame.
3. A `3 → 2 → 1 → ¡YA!` countdown runs, 0.8 s per step.
4. On **"¡YA!"** a single frame is captured: MediaPipe extracts your 21 hand
   landmarks and the SVM classifies the gesture.
5. The result panel shows both moves as icons plus **GANAS / PIERDES / EMPATE**
   for 2.5 s, then the game returns to idle, ready for the next round.

Hold your hand **flat and facing the camera**. Distance does not matter — the
landmarks are scale-normalised — but a hand seen edge-on or half out of frame
will not be detected. If no hand is visible at the exact "¡YA!" instant the
round is voided and nothing is scored; just press `SPACE` again.

A running scoreboard (`Tu / Empates / Bot`) stays on screen the whole time.

| Key | Action |
| --- | --- |
| `SPACE` | Play a round |
| `r` | Reset the scoreboard **and** the bot's learned weights |
| `q` | Quit and print the final tally to the terminal |

The bot needs data before it can exploit you. Over five rounds you are close to
coin-flipping; the weights only start concentrating on a pattern after a few
dozen rounds, so play a long session if you want to watch it adapt. Pressing
`r` wipes that memory and starts it from scratch.

To try the game without the login step:

```bash
python juego.py                # no greeting
python juego.py "Your Name"    # personalised greeting
```

---

## The three blocks

### 1. Face login — `login_facial.py`

Both models are **pretrained**; nothing is trained here. YuNet returns a
bounding box plus 5 facial points, which `alignCrop` uses to feed SFace a
properly aligned crop. SFace returns a 128-float identity vector.

Each user stores ~50 embeddings. At login time the current embedding is
compared against all of them and the best cosine similarity per user wins.
SFace's official same-person threshold is **0.363**, which we map to a
friendlier percentage (threshold → 50 %, cosine 0.85 → 100 %) and require
≥ 80 %. A **1.5 s streak** of sustained recognition is required before the
login closes, which rejects one-off false positives and makes the process
visible during a demo.

### 2. Gesture classifier — `entrenar.py`, `capturar_gestos.py`

Instead of training a CNN on raw pixels, feature extraction is delegated to
**MediaPipe Hands**, which turns a noisy image into 21 (x, y) landmarks. The
problem stops being computer vision and becomes a classification task in a
well-structured 42-dimensional space.

The decision that matters most is **normalisation**: landmarks are translated
so the wrist is the origin, then scaled by the wrist → middle-knuckle distance.
Without it the same 42 floats encode *where* the hand is and *how big* it looks
rather than *what gesture* it makes, and the classifier learns to classify by
distance to the camera.

Dataset: **1,498 samples** collected by the authors (500 paper, 499 rock,
499 scissors), stratified 80/20 split, `StandardScaler` inside an sklearn
`Pipeline`.

### 3. Adaptive bot — `bot_rps.py`

24 simple predictors (uniform, frequency, recent frequency, Markov of orders
1-3, win-stay/lose-shift), each wrapped in a meta-shift layer that also plays
the anti-strategies, combined by the **Hedge** algorithm. Every round, weights
move exponentially towards whichever expert has been predicting you well.

This is why it beats a supervised Random Forest trained offline: the forest
applies a fixed policy learned on an *average* opponent, while Hedge re-weights
against the *specific* opponent in front of it. For non-stationary
environments, online learning with experts dominates.

---

## Results

**Gesture classifier** — stratified 20 % test set, 300 samples:

| Model | Accuracy |
| --- | --- |
| KNN (k=5) | 98.33 % |
| **SVM-RBF** (chosen) | **99.33 %** |
| Random Forest (300 trees) | 99.33 % |

The accuracy is high because MediaPipe already does the hard part. That is
itself the lesson: framing the problem and choosing good features often
matters more than choosing the classifier.

**Bot** — 1,000 rounds × 3 trials against synthetic human archetypes.
*Edge = P(bot wins) − P(bot loses)*, higher is better:

| Strategy | Mean edge | Worst-case edge |
| --- | --- | --- |
| Random | 0.00 | 0.00 |
| Frequency | −0.06 | −0.30 |
| Markov-1 | +0.33 | −0.01 |
| Markov-2 | +0.36 | +0.08 |
| WSLS alone | +0.13 | −0.20 |
| Supervised Random Forest | +0.27 | +0.03 |
| **Hedge ensemble** | **+0.37** | **+0.08** |

Latency is under 60 ms per frame on a laptop with no GPU.

---

## Repository layout

```
app.py                  # Entry point: Tk GUI (login + register) and game launcher
login_facial.py         # YuNet + SFace, user database, register and login loops
juego.py                # The game: webcam, gesture inference, bot, scoreboard
bot_rps.py              # Hedge ensemble over 24 predictors
capturar_gestos.py      # Dataset capture tool
entrenar.py             # Training pipeline and model comparison
datos_gestos.csv        # 1,498 gesture samples
modelo_gestos.joblib    # Trained SVM-RBF pipeline + label list
proyecto_rps_ml.ipynb   # Notebook: bot design, archetypes and evaluation
MEMORIA.md              # Full academic write-up (Spanish)
fotos/                  # Move icons for the result panel
```

## Retraining the gesture model

```bash
python capturar_gestos.py   # 0 = rock, 1 = paper, 2 = scissors; appends to the CSV
python entrenar.py          # compares KNN / SVM / RF, saves the best to .joblib
```

`juego.py` imports `normalizar_landmarks` directly from `entrenar.py`, so
training and serving can never drift apart.

---

## Privacy

Face embeddings are biometric data. The user database (`usuarios.pkl`) is
**git-ignored and never published** — it only ever exists on the machine where
the users registered. Clone the repo and you start with an empty database;
register yourself with `app.py` and the data stays local.

## Limitations

- The bot has **never been evaluated against real humans**, only against
  synthetic archetypes. That gives indirect evidence it beats the baselines and
  adapts, but no guarantee of the same edge against real players.
- The gesture classifier is trained on the authors' hands under reasonable
  lighting. Gestures made halfway between paper and scissors are where the
  confusion matrix shows its only errors.
- The face database is small (three registered users during development), so
  the false-positive rate is not meaningfully measured.

## Credits

Academic project for the Machine Learning course, by **Miguel Merino**,
**Pablo Barranco** and **Pablo Rodríguez**.

The face login and the bot notebook come from a teammate's work; the gesture
classifier and the integration of the three blocks were built for this project.

Key references: Wang, Xu & Zhou (2014), *Social cycling and conditional
responses in the Rock-Paper-Scissors game*; Freund & Schapire (1997),
*A decision-theoretic generalization of on-line learning*; Egnor (2000),
*Iocaine Powder*; and [OpenCV Zoo](https://github.com/opencv/opencv_zoo) for
YuNet and SFace.

## License

[MIT](LICENSE).
