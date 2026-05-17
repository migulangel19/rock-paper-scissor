"""
bot_rps.py
==========

Bot adaptativo de Piedra-Papel-Tijera basado en un *ensemble Hedge* de
predictores (Markov de varios órdenes, frecuencia, win-stay/lose-shift,
etc.) con tres meta-niveles tipo *Iocaine Powder*.

Es una versión modularizada del bot que mi compañero desarrolló en el
notebook `proyecto_rps_ml.ipynb`. Aquí solo extraemos lo necesario para
poder usar el bot desde el juego en tiempo real, sin Jupyter ni
gráficas. La lógica algorítmica y los hiperparámetros son los mismos.

Codificación
------------
    0 = Piedra,  1 = Papel,  2 = Tijera

Resultado desde la perspectiva del bot:
    +1  bot gana
     0  empate
    -1  humano gana
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

ROCK, PAPER, SCISSORS = 0, 1, 2
MOVES = ["Piedra", "Papel", "Tijera"]


def result(human_move: int, bot_move: int) -> int:
    """Resultado de una ronda desde la perspectiva del bot."""
    if human_move == bot_move:
        return 0
    return 1 if (bot_move - human_move) % 3 == 1 else -1


def beats(move: int) -> int:
    """Jugada que vence a `move`."""
    return (move + 1) % 3


# ---------------------------------------------------------------------------
# Predictores: cada uno estima P(próxima jugada del humano).
# ---------------------------------------------------------------------------

class Predictor:
    def predict(self, hh, bh, rs):
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__


class UniformPredictor(Predictor):
    def predict(self, hh, bh, rs):
        return np.array([1 / 3, 1 / 3, 1 / 3])


class FrequencyPredictor(Predictor):
    """Frecuencia acumulada con suavizado de Laplace."""

    def predict(self, hh, bh, rs):
        if not hh:
            return np.array([1 / 3, 1 / 3, 1 / 3])
        c = np.array([hh.count(i) for i in range(3)], dtype=float) + 1
        return c / c.sum()


class RecentFrequencyPredictor(Predictor):
    """Frecuencia en ventana móvil (capta cambios de estrategia)."""

    def __init__(self, window: int = 20):
        self.window = window

    @property
    def name(self) -> str:
        return f"RecentFreq-{self.window}"

    def predict(self, hh, bh, rs):
        if not hh:
            return np.array([1 / 3, 1 / 3, 1 / 3])
        recent = hh[-self.window:]
        c = np.array([recent.count(i) for i in range(3)], dtype=float) + 1
        return c / c.sum()


class MarkovN(Predictor):
    """Cadena de Markov de orden N sobre la historia del humano."""

    def __init__(self, n: int = 1):
        self.n = n
        self._trans = defaultdict(lambda: np.ones(3))
        self._last_seen = 0

    @property
    def name(self) -> str:
        return f"Markov-{self.n}"

    def predict(self, hh, bh, rs):
        for i in range(self._last_seen, len(hh) - self.n):
            key = tuple(hh[i:i + self.n])
            self._trans[key][hh[i + self.n]] += 1
        self._last_seen = max(0, len(hh) - self.n)
        if len(hh) < self.n + 1:
            return np.array([1 / 3, 1 / 3, 1 / 3])
        c = self._trans[tuple(hh[-self.n:])]
        return c / c.sum()


class WSLSPredictor(Predictor):
    """Detector explícito de Win-Stay/Lose-Shift."""

    @property
    def name(self) -> str:
        return "WinStayLoseShift"

    def predict(self, hh, bh, rs):
        if not rs:
            return np.array([1 / 3, 1 / 3, 1 / 3])
        last = rs[-1]
        if last == -1:                       # humano ganó la ronda anterior
            p = np.full(3, 0.10)
            p[hh[-1]] = 0.80                 # tenderá a repetir
        elif last == 1:                      # humano perdió
            p = np.full(3, 0.15)
            p[beats(bh[-1])] = 0.70          # tenderá a jugar la que ganaba al bot
        else:
            p = np.array([1 / 3, 1 / 3, 1 / 3])
        return p / p.sum()


class MetaShift(Predictor):
    """Envuelve un predictor y rota su predicción `shift` posiciones
    (idea del *Iocaine Powder*: contraataca al oponente que se anticipa)."""

    def __init__(self, base: Predictor, shift: int = 0):
        self.base = base
        self.shift = shift

    @property
    def name(self) -> str:
        return f"{self.base.name}_L{self.shift}"

    def predict(self, hh, bh, rs):
        return np.roll(self.base.predict(hh, bh, rs), self.shift)


def build_ensemble() -> list[Predictor]:
    """Construye los 24 predictores del ensemble (8 base × 3 meta-niveles)."""
    base = [
        UniformPredictor(),
        FrequencyPredictor(),
        RecentFrequencyPredictor(window=10),
        RecentFrequencyPredictor(window=30),
        MarkovN(1), MarkovN(2), MarkovN(3),
        WSLSPredictor(),
    ]
    return [MetaShift(p, s) for p in base for s in (0, 1, 2)]


# ---------------------------------------------------------------------------
# Meta-selector: algoritmo Hedge (Freund & Schapire, 1997).
# ---------------------------------------------------------------------------

class HedgeBot:
    """Combina los predictores con pesos exponenciales adaptativos.

    En cada ronda:
      1. Mezcla las predicciones de todos los predictores ponderadas por
         su peso actual.
      2. Elige la jugada que maximiza la utilidad esperada (P(gana) − P(pierde)).
      3. Tras ver la jugada real del humano, premia a los predictores que
         le asignaron mayor probabilidad y aplica un decay para olvidar
         rendimientos antiguos.
    """

    def __init__(self, predictors=None, eta: float = 0.3, decay: float = 0.99):
        self.predictors = predictors or build_ensemble()
        self.eta = eta
        self.decay = decay
        self.reset()

    def reset(self) -> None:
        self.weights = np.ones(len(self.predictors))
        self.hh: list[int] = []   # historial del humano
        self.bh: list[int] = []   # historial del bot
        self.rs: list[int] = []   # historial de resultados

    def choose(self):
        """Decide la jugada del bot a partir del histórico.

        Devuelve `(bot_move, preds)`. `preds` se necesita después en
        :meth:`update` para no recomputar las predicciones.
        """
        if not self.hh:
            # Sin información todavía: jugada aleatoria.
            bm = int(np.random.choice([0, 1, 2]))
            return bm, None

        preds = np.array([p.predict(self.hh, self.bh, self.rs)
                          for p in self.predictors])
        w = self.weights / self.weights.sum()
        combined = (w[:, None] * preds).sum(axis=0)

        # Utilidad esperada de cada jugada del bot:
        # bot m vence al humano (m-1) % 3 y pierde con (m+1) % 3.
        utils = np.array([combined[(m - 1) % 3] - combined[(m + 1) % 3]
                          for m in range(3)])
        return int(np.argmax(utils)), preds

    def update(self, bot_move: int, human_move: int, preds) -> None:
        """Registra la ronda y actualiza los pesos del ensemble."""
        res = result(human_move, bot_move)
        self.hh.append(human_move)
        self.bh.append(bot_move)
        self.rs.append(res)

        if preds is None:
            return  # primera ronda: nada que actualizar
        for i, p in enumerate(preds):
            self.weights[i] *= np.exp(self.eta * (p[human_move] - 1 / 3))
        self.weights *= self.decay
        self.weights = np.maximum(self.weights, 1e-4)
        self.weights /= self.weights.max()
