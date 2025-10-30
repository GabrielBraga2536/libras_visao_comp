"""
Ferramentas de linha de comando para inspecionar e manipular modelos Keras (.h5).

Funcionalidades principais:
- summary: imprime (ou salva) o summary do modelo
- info: imprime shapes de entrada/saída, dtypes e contagem de parâmetros
- layers: lista camadas com índice, nome, tipo e shapes
- io: checa rapidamente se o modelo aceita uma entrada 224x224x3 e quantas classes retorna
- to-savedmodel: converte .h5 para TensorFlow SavedModel
- to-tflite: converte .h5 para TensorFlow Lite (.tflite), com opção de float16
- replace-last: substitui a última camada por uma nova Dense com N classes (exige re-treino)

Uso:
  python model_utils.py summary --model keras_model.h5
  python model_utils.py to-tflite --model keras_model.h5 --out-file model.tflite --float16

Observação: "replace-last" cria um novo arquivo .h5 com a última camada trocada, mas
sem re-treinar: os pesos dessa nova camada são aleatórios. Treino adicional é necessário.
"""

import argparse
import io
import os
from typing import List, Tuple

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception as e:
    raise RuntimeError("TensorFlow/Keras não disponíveis no ambiente: " + str(e))


def load_keras_model(path: str) -> keras.Model:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    model = keras.models.load_model(path)
    return model


def print_summary(model: keras.Model, save_path: str | None = None) -> None:
    buf = io.StringIO()
    model.summary(print_fn=lambda x: buf.write(x + "\n"))
    text = buf.getvalue()
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Summary salvo em: {save_path}")
    else:
        print(text)


def human_params(n: int) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n}{unit}"
        n //= 1000
    return str(n)


def info(model: keras.Model) -> None:
    inputs = model.inputs
    outputs = model.outputs
    print("Entradas:")
    for t in inputs:
        print(f" - name={t.name}, shape={t.shape}, dtype={t.dtype}")
    print("Saídas:")
    for t in outputs:
        print(f" - name={t.name}, shape={t.shape}, dtype={t.dtype}")
    print(f"Total de parâmetros: {model.count_params()} ({human_params(model.count_params())})")


def list_layers(model: keras.Model) -> None:
    for i, layer in enumerate(model.layers):
        out_shape = getattr(layer, 'output_shape', None)
        print(f"[{i:03d}] name={layer.name:30s} type={layer.__class__.__name__:20s} out_shape={out_shape}")


def quick_io_check(model: keras.Model, h: int = 224, w: int = 224) -> None:
    # Tenta inferir a forma de entrada: usa uma imagem dummy 224x224x3
    x = np.zeros((1, h, w, 3), dtype=np.float32)
    try:
        y = model.predict(x, verbose=0)
        if isinstance(y, list):
            shapes = [arr.shape for arr in y]
            print("Predição OK. Shapes de saída:", shapes)
        else:
            print("Predição OK. Shape de saída:", y.shape)
    except Exception as e:
        print("Falha na predição com tensor 224x224x3:", e)


def to_savedmodel(model: keras.Model, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    tf.saved_model.save(model, out_dir)
    print(f"SavedModel salvo em: {out_dir}")


def to_tflite(model: keras.Model, out_file: str, float16: bool = False) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if float16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(out_file, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite salvo em: {out_file} (float16={float16})")


def replace_last_layer(model: keras.Model, num_classes: int, activation: str = "softmax", name: str = "predictions") -> keras.Model:
    if len(model.layers) < 2:
        raise ValueError("Modelo muito pequeno para substituir a última camada.")
    # Usa a penúltima camada como base. Isso supõe um grafo sequencial no final.
    base_output = model.layers[-2].output
    new_output = keras.layers.Dense(num_classes, activation=activation, name=name)(base_output)
    new_model = keras.Model(inputs=model.input, outputs=new_output)
    return new_model


def main():
    parser = argparse.ArgumentParser(description="Utilitários para modelos Keras (.h5)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sum = sub.add_parser("summary", help="Imprime ou salva o summary do modelo")
    p_sum.add_argument("--model", required=True, help="Caminho do .h5")
    p_sum.add_argument("--save", help="Caminho para salvar o summary em texto")

    p_info = sub.add_parser("info", help="Mostra shapes e parâmetros")
    p_info.add_argument("--model", required=True)

    p_layers = sub.add_parser("layers", help="Lista camadas")
    p_layers.add_argument("--model", required=True)

    p_io = sub.add_parser("io", help="Checagem rápida de I/O com tensor 224x224x3")
    p_io.add_argument("--model", required=True)
    p_io.add_argument("--h", type=int, default=224)
    p_io.add_argument("--w", type=int, default=224)

    p_sm = sub.add_parser("to-savedmodel", help="Converte .h5 para SavedModel")
    p_sm.add_argument("--model", required=True)
    p_sm.add_argument("--out-dir", required=True)

    p_tfl = sub.add_parser("to-tflite", help="Converte .h5 para .tflite")
    p_tfl.add_argument("--model", required=True)
    p_tfl.add_argument("--out-file", required=True)
    p_tfl.add_argument("--float16", action="store_true", help="Ativa quantização float16")

    p_rep = sub.add_parser("replace-last", help="Substitui a última camada por Dense(N)")
    p_rep.add_argument("--model", required=True)
    p_rep.add_argument("--num-classes", type=int, required=True)
    p_rep.add_argument("--activation", default="softmax")
    p_rep.add_argument("--out", required=True, help="Caminho do novo .h5")

    args = parser.parse_args()

    model = load_keras_model(args.model)

    if args.cmd == "summary":
        print_summary(model, args.save)
    elif args.cmd == "info":
        info(model)
    elif args.cmd == "layers":
        list_layers(model)
    elif args.cmd == "io":
        quick_io_check(model, h=getattr(args, 'h', 224), w=getattr(args, 'w', 224))
    elif args.cmd == "to-savedmodel":
        to_savedmodel(model, args.out_dir)
    elif args.cmd == "to-tflite":
        to_tflite(model, args.out_file, float16=args.float16)
    elif args.cmd == "replace-last":
        new_model = replace_last_layer(model, args.num_classes, activation=args.activation)
        new_model.save(args.out)
        print(f"Novo modelo salvo em: {args.out}")
        print("Atenção: treine novamente ao menos a última camada para obter boas predições.")
    else:
        parser.error("Comando não reconhecido")


if __name__ == "__main__":
    main()
