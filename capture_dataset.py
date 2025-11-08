#!/usr/bin/env python3
"""
Captura imagens da webcam para construir um dataset no formato aceito por train.py
(pastas por classe). Por padrão, salva 50 fotos automaticamente com um intervalo
configurável e mostra uma prévia com contadores na tela.

Uso básico:
  python3 capture_dataset.py --label A --count 50 --out-dir dataset

Opções úteis:
  --interval 0.5     # segundos entre capturas
  --start-delay 3    # segundos de contagem regressiva antes de começar
  --hand-crop        # tenta recortar a mão usando MediaPipe (se instalado)
  --require-crop     # só salva se houver recorte (mão via MediaPipe ou ROI manual)
  --margin 40        # margem em pixels ao redor do bbox da mão
  --square           # força o recorte a ser quadrado (útil p/ modelos tipo 224x224)
  --roi 100,100,224,224  # recorte manual (x,y,w,h) aplicado antes de salvar
  --mirror           # espelha a imagem horizontalmente (selfie)

Teclas durante a execução:
  q       Sai imediatamente
  space  Pausa/retoma a captura automática
  s      Salva uma foto manualmente (além do modo automático)

Requisitos:
  - OpenCV (cv2)
  - MediaPipe (opcional, apenas se usar --hand-crop)
"""

import argparse
import os
import time
from datetime import datetime
from typing import Optional, Tuple

import cv2

try:
  import mediapipe as mp  # type: ignore
  _HAS_MEDIAPIPE = True
except Exception:
  _HAS_MEDIAPIPE = False


def ensure_dir(path: str) -> None:
  os.makedirs(path, exist_ok=True)


def parse_roi(roi_str: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
  if not roi_str:
    return None
  try:
    x, y, w, h = map(int, roi_str.split(','))
    return max(x, 0), max(y, 0), max(w, 1), max(h, 1)
  except Exception:
    raise argparse.ArgumentTypeError("ROI deve estar no formato x,y,w,h (inteiros)")


def hand_crop(frame, hands, margin: int = 40) -> Optional[Tuple[int, int, int, int]]:
  """Retorna (x1, y1, x2, y2) do bbox da mão com uma margem, se detectada."""
  h, w, _ = frame.shape
  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  res = hands.process(rgb)
  if not res.multi_hand_landmarks:
    return None
  x_min, y_min = w, h
  x_max, y_max = 0, 0
  for hand in res.multi_hand_landmarks:
    for lm in hand.landmark:
      x, y = int(lm.x * w), int(lm.y * h)
      x_min = min(x_min, x)
      y_min = min(y_min, y)
      x_max = max(x_max, x)
      y_max = max(y_max, y)
  x1 = max(x_min - margin, 0)
  y1 = max(y_min - margin, 0)
  x2 = min(x_max + margin, w)
  y2 = min(y_max + margin, h)
  if x2 <= x1 or y2 <= y1:
    return None
  return (x1, y1, x2, y2)


def make_square(x1: int, y1: int, x2: int, y2: int, frame_shape) -> Tuple[int, int, int, int]:
  """Ajusta um bbox para quadrado mantendo centro e dentro dos limites."""
  h, w = frame_shape[:2]
  cw = x2 - x1
  ch = y2 - y1
  side = max(cw, ch)
  cx = (x1 + x2) // 2
  cy = (y1 + y2) // 2
  nx1 = max(cx - side // 2, 0)
  ny1 = max(cy - side // 2, 0)
  nx2 = nx1 + side
  ny2 = ny1 + side
  # Ajusta se estourar borda
  if nx2 > w:
    shift = nx2 - w
    nx1 = max(nx1 - shift, 0)
    nx2 = w
  if ny2 > h:
    shift = ny2 - h
    ny1 = max(ny1 - shift, 0)
    ny2 = h
  return int(nx1), int(ny1), int(nx2), int(ny2)


def save_image(img, out_dir: str, label: str, idx: int) -> str:
  ensure_dir(os.path.join(out_dir, label))
  ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
  filename = f"{label}_{ts}_{idx:03d}.jpg"
  out_path = os.path.join(out_dir, label, filename)
  cv2.imwrite(out_path, img)
  return out_path


def main():
  ap = argparse.ArgumentParser(description="Capturar imagens da webcam para dataset")
  ap.add_argument('--label', required=True, help='Nome da classe (subpasta em out-dir)')
  ap.add_argument('--out-dir', default='dataset', help='Diretório base do dataset')
  ap.add_argument('--count', type=int, default=50, help='Número de fotos automáticas')
  ap.add_argument('--interval', type=float, default=0.5, help='Intervalo entre fotos (segundos)')
  ap.add_argument('--start-delay', type=float, default=3.0, help='Contagem regressiva antes de começar (segundos)')
  ap.add_argument('--camera', type=int, default=0, help='Índice da câmera')
  ap.add_argument('--width', type=int, default=640)
  ap.add_argument('--height', type=int, default=480)
  ap.add_argument('--mirror', action='store_true', help='Espelha horizontalmente (selfie)')
  ap.add_argument('--roi', type=str, default=None, help='Recorte manual x,y,w,h')
  ap.add_argument('--hand-crop', action='store_true', help='Recorta automaticamente a mão (requer MediaPipe)')
  ap.add_argument('--resize', type=int, default=None, help='Redimensiona para NxN antes de salvar (ex.: 224)')
  ap.add_argument('--require-crop', action='store_true', help='Só salva se encontrar mão (ou ROI manual)')
  ap.add_argument('--margin', type=int, default=40, help='Margem em pixels para o bbox da mão')
  ap.add_argument('--square', action='store_true', help='Força recorte quadrado (centralizado)')
  args = ap.parse_args()

  if args.hand_crop and not _HAS_MEDIAPIPE:
    print("Aviso: MediaPipe não encontrado. Desativando --hand-crop.")
    args.hand_crop = False

  cap = cv2.VideoCapture(args.camera)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

  if not cap.isOpened():
    print("Erro: não foi possível acessar a câmera.")
    return 1

  hands = mp.solutions.hands.Hands(max_num_hands=2) if args.hand_crop else None

  # Contagem regressiva antes de iniciar
  start_time = time.time()
  paused = False
  taken = 0
  last_shot = 0.0

  roi_rect = parse_roi(args.roi)

  print("Controles: 'q' = sair, 'space' = pausar/retomar, 's' = salvar manualmente")

  while True:
    ok, frame = cap.read()
    if not ok:
      print("Falha ao ler frame da câmera.")
      break

    if args.mirror:
      frame = cv2.flip(frame, 1)

    overlay = frame.copy()

    # ROI manual (desenha)
    if roi_rect is not None:
      x, y, w, h = roi_rect
      cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Hand-crop bbox (desenha)
    bbox = None
    if args.hand_crop and hands is not None:
      bbox = hand_crop(frame, hands, margin=args.margin)
      if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 0), 2)

    # Infos na tela
    elapsed = time.time() - start_time
    if elapsed < args.start_delay:
      msg = f"Começando em {int(args.start_delay - elapsed) + 1}s"
    else:
      msg = ("PAUSADO" if paused else "CAPTURANDO") + f"  {taken}/{args.count}"
    cv2.putText(overlay, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 220, 50), 2)
    cv2.putText(overlay, f"Label: {args.label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

    alpha = 0.85
    frame_disp = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.imshow('Capture Dataset', frame_disp)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
      break
    elif key == ord(' '):  # space
      paused = not paused
    elif key == ord('s'):
      # salva manual
      crop_img = None
      if bbox:
        x1, y1, x2, y2 = bbox
        if args.square:
          x1, y1, x2, y2 = make_square(x1, y1, x2, y2, frame.shape)
        crop_img = frame[y1:y2, x1:x2]
      elif roi_rect is not None:
        x, y, w, h = roi_rect
        x1, y1, x2, y2 = x, y, x + w, y + h
        if args.square:
          x1, y1, x2, y2 = make_square(x1, y1, x2, y2, frame.shape)
        crop_img = frame[y1:y2, x1:x2]
      elif not args.require_crop:
        crop_img = frame
      else:
        print("Sem recorte (mão/ROI) - não salvando (require-crop ativo)")

      if crop_img is not None:
        if args.resize:
          crop_img = cv2.resize(crop_img, (args.resize, args.resize))
        out_path = save_image(crop_img, args.out_dir, args.label, taken + 1)
        taken += 1
        print(f"Foto salva manualmente: {out_path}")

    # Captura automática
    now = time.time()
    if not paused and elapsed >= args.start_delay and taken < args.count:
      if now - last_shot >= args.interval:
        crop_img = None
        if bbox:
          x1, y1, x2, y2 = bbox
          if args.square:
            x1, y1, x2, y2 = make_square(x1, y1, x2, y2, frame.shape)
          crop_img = frame[y1:y2, x1:x2]
        elif roi_rect is not None:
          x, y, w, h = roi_rect
          x1, y1, x2, y2 = x, y, x + w, y + h
          if args.square:
            x1, y1, x2, y2 = make_square(x1, y1, x2, y2, frame.shape)
          crop_img = frame[y1:y2, x1:x2]
        elif not args.require_crop:
          crop_img = frame

        if crop_img is not None:
          if args.resize:
            crop_img = cv2.resize(crop_img, (args.resize, args.resize))
          out_path = save_image(crop_img, args.out_dir, args.label, taken + 1)
          print(f"Foto salva: {out_path}")
          taken += 1
          last_shot = now
        else:
          print("Sem recorte (mão/ROI) - pulando frame (require-crop ativo)")

    if taken >= args.count:
      cv2.putText(frame_disp, "Concluído! Pressione 'q' para sair.", (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 220, 220), 2)
      cv2.imshow('Capture Dataset', frame_disp)
      # não dá break automaticamente para permitir revisão na tela

  if hands is not None:
    hands.close()
  cap.release()
  cv2.destroyAllWindows()
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
