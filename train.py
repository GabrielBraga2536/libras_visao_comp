"""
Treino de modelo Keras para reconhecimento de gestos/letras a partir de pastas de imagens.

Estrutura de dados esperada (um diretório com subpastas por classe):

dataset/
  A/
    img001.jpg
    ...
  B/
  C/
  ...

O script divide automaticamente em treino/validação via validation_split.

Saídas:
- Modelo salvo (padrão: keras_model.h5)
- Arquivo de rótulos (padrão: labels.txt), na ordem das classes usadas no treino

Exemplos:
    python3 train.py --data-dir dataset --epochs 15 --batch-size 32
    python3 train.py --data-dir dataset --epochs 5 --ft-epochs 10 --fine-tune 50

Notas:
- A normalização usada é compatível com o libras.py ((x/127.5)-1.0 via preprocess_input do MobileNetV2).
- "fine-tune" permite destravar as últimas N camadas do backbone para refinar o modelo após treinar a cabeça.
"""

import argparse
import os
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_datasets(data_dir: str, img_size: int, batch_size: int, val_split: float, seed: int = 42):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset='training',
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='categorical',
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset='validation',
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='categorical',
    )

    # `image_dataset_from_directory` attaches `class_names` to the returned dataset
    # object. We need to capture it before transforming the dataset (e.g. prefetch),
    # because operations like `.prefetch()` return a new Dataset wrapper that does
    # not carry the attribute.
    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def build_model(num_classes: int, img_size: int, dropout: float = 0.2) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
    base.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    # Normalização compatível com libras.py
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    if dropout and dropout > 0:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    model = keras.Model(inputs, outputs)
    return model


def unfreeze_last_n_layers(model: tf.keras.Model, n_layers: int, base_name: str = 'mobilenetv2_1.00_224'):
    # Tenta localizar o backbone pelo nome padrão do MobileNetV2
    base = None
    for layer in model.layers:
        if base_name in layer.name:
            base = layer
            break
    if base is None:
        # fallback: procura a camada com mais pesos (provável backbone)
        base = max(model.layers, key=lambda l: len(l.weights))

    # Destrava as últimas N camadas do backbone
    total = len(base.layers) if hasattr(base, 'layers') else 0
    if total == 0:
        # Se não conseguir acessar subcamadas, destrava o modelo todo
        model.trainable = True
        return

    for i, l in enumerate(base.layers):
        # Congele por padrão
        l.trainable = False
        # Descongele últimas N
        if i >= total - n_layers:
            l.trainable = True


def train(args):
    train_ds, val_ds, class_names = build_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
    )

    # `class_names` was captured from the original dataset before `prefetch()`
    num_classes = len(class_names)
    print("Classes:", class_names)

    model = build_model(num_classes=num_classes, img_size=args.img_size, dropout=args.dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=args.output_model,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=max(3, args.epochs // 3),
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Fine-tuning opcional (destrava últimas N camadas)
    if args.fine_tune and args.fine_tune > 0 and args.ft_epochs > 0:
        print(f"Iniciando fine-tuning das últimas {args.fine_tune} camadas do backbone...")
        unfreeze_last_n_layers(model, n_layers=args.fine_tune)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=max(args.lr * 0.1, 1e-6)),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.ft_epochs,
            callbacks=callbacks,
            verbose=1,
        )

    # Garante que a melhor versão está salva (ModelCheckpoint já salvou a melhor)
    # Mas também podemos salvar o estado final atual se desejado:
    if args.save_final:
        final_path = os.path.splitext(args.output_model)[0] + "_final.h5"
        model.save(final_path)
        print("Modelo final salvo em:", final_path)

    # Salva labels.txt na ordem usada
    with open(args.output_labels, 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(name + "\n")
    print("Rótulos salvos em:", args.output_labels)

    print("Treino concluído. Modelo (melhor val_accuracy):", args.output_model)


def parse_args():
    p = argparse.ArgumentParser(description="Treinar modelo Keras para reconhecimento de gestos/letras")
    p.add_argument('--data-dir', required=True, help='Diretório com subpastas por classe')
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--val-split', type=float, default=0.2)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--ft-epochs', type=int, default=0, help='Épocas adicionais de fine-tuning')
    p.add_argument('--fine-tune', type=int, default=0, help='Qtde de camadas finais do backbone a destravar')
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-model', default='keras_model.h5')
    p.add_argument('--output-labels', default='labels.txt')
    p.add_argument('--save-final', action='store_true', help='Também salva o estado final do modelo em *_final.h5')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
