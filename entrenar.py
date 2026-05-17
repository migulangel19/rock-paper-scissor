"""
entrenar.py
===========

Entrena un clasificador de gestos (piedra / papel / tijera) a partir del
dataset capturado con `capturar_gestos.py`.

Pipeline
--------
1. Carga `datos_gestos.csv` (21 landmarks (x, y) + columna `etiqueta`).
2. Normaliza los landmarks para que el modelo no dependa de la posición
   ni del tamaño de la mano en la imagen:
       - Se traslada el origen al punto 0 (la muñeca).
       - Se escala dividiendo por la distancia muñeca -> nudillo del dedo
         medio (punto 9). Esta distancia es una medida estable del
         "tamaño" de la mano en la imagen.
3. Divide en train/test estratificado (80/20).
4. Entrena tres modelos clásicos (KNN, SVM-RBF, RandomForest) dentro de
   un `Pipeline` con `StandardScaler` y compara su accuracy en test.
5. Guarda el mejor modelo en `modelo_gestos.joblib` (el pipeline completo,
   listo para usarse en inferencia con los vectores ya normalizados).

Uso
---
    python entrenar.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CSV_PATH = Path("datos_gestos.csv")
MODELO_PATH = Path("modelo_gestos.joblib")
NUM_PUNTOS = 21
SEED = 42


def normalizar_landmarks(coords: np.ndarray) -> np.ndarray:
    """Normaliza un lote de landmarks de mano.

    Parameters
    ----------
    coords : np.ndarray
        Array de forma ``(n_muestras, 42)`` con el formato
        ``[x0, y0, x1, y1, ..., x20, y20]``.

    Returns
    -------
    np.ndarray
        Array de la misma forma, con cada mano centrada en la muñeca
        (punto 0) y escalada por la distancia muñeca -> nudillo medio
        (punto 9). De esta forma, dos manos que hagan el mismo gesto
        a distinta distancia o en distinta posición del encuadre
        producen vectores muy parecidos.
    """
    pts = coords.reshape(-1, NUM_PUNTOS, 2).astype(np.float32)

    # 1) Trasladar: la muñeca pasa a ser (0, 0).
    munieca = pts[:, 0:1, :]
    pts = pts - munieca

    # 2) Escalar: dividir por la distancia muñeca -> nudillo del dedo medio.
    #    Es una referencia estable del tamaño de la mano en la imagen.
    escala = np.linalg.norm(pts[:, 9, :], axis=1, keepdims=True)
    escala = np.where(escala == 0, 1.0, escala)  # evita divisiones por cero
    pts = pts / escala[:, :, None]

    return pts.reshape(-1, NUM_PUNTOS * 2)


def cargar_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Carga el CSV y devuelve `(X_normalizado, y)`."""
    if not path.exists():
        sys.exit(f"No se encontró {path}. Ejecuta primero capturar_gestos.py.")

    df = pd.read_csv(path)
    if "etiqueta" not in df.columns:
        sys.exit("El CSV no contiene la columna 'etiqueta'.")

    y = df["etiqueta"].to_numpy()
    X = df.drop(columns=["etiqueta"]).to_numpy(dtype=np.float32)
    X = normalizar_landmarks(X)

    print(f"Dataset cargado: {len(X)} muestras")
    print("Distribución de clases:")
    for clase, n in df["etiqueta"].value_counts().items():
        print(f"  {clase:8s} {n}")

    return X, y


def construir_modelos() -> dict[str, Pipeline]:
    """Devuelve los candidatos a entrenar (todos como `Pipeline`)."""
    return {
        "KNN":          Pipeline([("scaler", StandardScaler()),
                                  ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "SVM-RBF":      Pipeline([("scaler", StandardScaler()),
                                  ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                                              probability=True))]),
        "RandomForest": Pipeline([("scaler", StandardScaler()),
                                  ("clf", RandomForestClassifier(
                                      n_estimators=300, random_state=SEED))]),
    }


def main() -> None:
    X, y = cargar_dataset(CSV_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"\nTrain: {len(X_train)}   Test: {len(X_test)}\n")

    resultados: list[tuple[str, float, Pipeline]] = []
    for nombre, modelo in construir_modelos().items():
        modelo.fit(X_train, y_train)
        acc = modelo.score(X_test, y_test)
        resultados.append((nombre, acc, modelo))
        print(f"{nombre:13s} accuracy = {acc:.4f}")

    # Selecciona el mejor por accuracy en test.
    nombre, acc, mejor = max(resultados, key=lambda r: r[1])
    print(f"\nMejor modelo: {nombre} (accuracy={acc:.4f})\n")

    y_pred = mejor.predict(X_test)
    etiquetas = sorted(np.unique(y))
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Matriz de confusión (filas = real, columnas = predicho):")
    print(f"Etiquetas: {etiquetas}")
    print(confusion_matrix(y_test, y_pred, labels=etiquetas))

    joblib.dump({"modelo": mejor, "etiquetas": etiquetas}, MODELO_PATH)
    print(f"\nModelo guardado en {MODELO_PATH}")


if __name__ == "__main__":
    main()
