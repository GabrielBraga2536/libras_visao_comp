import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import os

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

hands = mp.solutions.hands.Hands(max_num_hands=2)
model_path = 'keras_model.h5'
model = load_model(model_path)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Threshold de confiança (padrão 0.6). Pode ser sobrescrito via variável de ambiente CONF_THRESHOLD
CONF_THRESHOLD = float(os.getenv('CONF_THRESHOLD', '0.6'))
SHOW_LOWCONF = os.getenv('SHOW_LOWCONF', '1')  # '1' exibe rótulo mesmo abaixo do threshold (em cor diferente)

# Inferir número de saídas do modelo (classes)
try:
    num_outputs = model.output_shape[-1] if isinstance(model.output_shape, tuple) else None
except Exception:
    num_outputs = None


def load_labels(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        labels = [ln.strip() for ln in f.readlines()]
    labels = [l for l in labels if l]
    return labels if labels else None


# Carregar classes de labels.txt se existir; caso contrário, usar padrão
labels_path = 'labels.txt'
classes = load_labels(labels_path)
if classes is None:
    classes = ['A', 'B', 'C', 'D']

if num_outputs is not None and len(classes) != num_outputs:
    print(f"Aviso: nº de rótulos ({len(classes)}) difere do nº de saídas do modelo ({num_outputs}).")
    print("Revise seu labels.txt ou ajuste o modelo para refletir o nº de classes.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) or 30

output_video_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    handsPoints = results.multi_hand_landmarks

    if handsPoints is not None:
        for hand in handsPoints:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in hand.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_max = max(x_max, x)
                x_min = min(x_min, x)
                y_max = max(y_max, y)
                y_min = min(y_min, y)

            # corrige bordas
            x1 = max(x_min - 50, 0)
            y1 = max(y_min - 50, 0)
            x2 = min(x_max + 50, w)
            y2 = min(y_max + 50, h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            try:
                if x2 > x1 and y2 > y1:
                    # recorta da imagem original (BGR) e converte para RGB, pois o modelo
                    # usa MobileNetV2 preprocess_input internamente (espera RGB 0..255)
                    imgCrop = frame[y1:y2, x1:x2]
                    imgCrop = cv2.resize(imgCrop, (224, 224))
                    imgCropRGB = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB)

                    # NÃO normalizar aqui; o modelo já aplica preprocess_input
                    data[0] = imgCropRGB.astype(np.float32)

                    prediction = model.predict(data, verbose=0)
                    if isinstance(prediction, list):
                        prediction = prediction[0]

                    indexVal = int(np.argmax(prediction))
                    conf = float(np.max(prediction))

                    if indexVal < len(classes):
                        label = classes[indexVal]
                    else:
                        label = str(indexVal)

                    # Mostra label: acima do threshold (vermelho), abaixo (opcional em laranja)
                    text = f"{label} ({conf:.2f})"
                    if conf >= CONF_THRESHOLD:
                        color = (0, 0, 255)  # vermelho
                        cv2.putText(frame, text, (x1, max(y1 - 10, 20)),
                                    cv2.FONT_HERSHEY_COMPLEX, 1.2, color, 3)
                    else:
                        if SHOW_LOWCONF == '1':
                            color = (0, 165, 255)  # laranja
                            cv2.putText(frame, text, (x1, max(y1 - 10, 20)),
                                        cv2.FONT_HERSHEY_COMPLEX, 1.2, color, 2)
            except Exception:
                pass

    # mostra o frame processado
    cv2.imshow("Frame", frame)
    out.write(frame)

    # pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
