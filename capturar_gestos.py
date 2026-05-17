import csv
import os

import cv2
import mediapipe as mp

CSV_PATH = "datos_gestos.csv"
ETIQUETAS = {ord("0"): "piedra", ord("1"): "papel", ord("2"): "tijera"}
NUM_PUNTOS = 21


def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    nuevo_archivo = not os.path.exists(CSV_PATH)
    csv_file = open(CSV_PATH, "a", newline="")
    writer = csv.writer(csv_file)
    if nuevo_archivo:
        encabezados = []
        for i in range(NUM_PUNTOS):
            encabezados.extend([f"x{i}", f"y{i}"])
        encabezados.append("etiqueta")
        writer.writerow(encabezados)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la webcam.")
        csv_file.close()
        return

    conteo = {v: 0 for v in ETIQUETAS.values()}

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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = hands.process(rgb)

            landmarks_actuales = None
            if resultados.multi_hand_landmarks:
                hand_landmarks = resultados.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                landmarks_actuales = hand_landmarks.landmark

            y = 25
            cv2.putText(
                frame,
                "0=piedra  1=papel  2=tijera  q=salir",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            for etiqueta, n in conteo.items():
                y += 25
                cv2.putText(
                    frame,
                    f"{etiqueta}: {n}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Captura de gestos", frame)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q"):
                break

            if tecla in ETIQUETAS and landmarks_actuales is not None:
                etiqueta = ETIQUETAS[tecla]
                fila = []
                for lm in landmarks_actuales:
                    fila.extend([lm.x, lm.y])
                fila.append(etiqueta)
                writer.writerow(fila)
                csv_file.flush()
                conteo[etiqueta] += 1
                print(f"Guardado: {etiqueta} (total {conteo[etiqueta]})")

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()


if __name__ == "__main__":
    main()
