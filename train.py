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

import numpy as np

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


def build_model(num_classes: int, img_size: int, dropout: float = 0.2,
                use_augmentation: bool = True, flip: bool = False) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
    base.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))

    # Data augmentation (aplicado apenas em treino). Evitamos flip horizontal por padrão,
    # pois pode inverter lateralidade de sinais; pode ser ligado com --flip.
    if use_augmentation:
        aug_layers = [
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.1),
        ]
        if flip:
            aug_layers.insert(0, layers.RandomFlip("horizontal"))
        x = keras.Sequential(aug_layers, name="augment")(inputs)
    else:
        x = inputs

    # Normalização compatível com libras.py
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
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


def count_files_per_class(data_dir: str, class_names: list[str]) -> dict:
    counts = {}
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        n = 0
        if os.path.isdir(cls_dir):
            for root, _, files in os.walk(cls_dir):
                for fn in files:
                    # conta qualquer arquivo; assumimos que diretórios tem apenas imagens válidas
                    if not fn.startswith('.'):
                        n += 1
        counts[cls] = n
    return counts


def make_class_weight(class_counts: dict, class_names: list[str]) -> dict:
    # Pesos inversamente proporcionais à frequência
    counts = np.array([class_counts.get(c, 0) for c in class_names], dtype=np.float32)
    counts = np.maximum(counts, 1.0)
    total = float(np.sum(counts))
    num_classes = len(class_names)
    weights = total / (counts * num_classes)
    return {i: float(w) for i, w in enumerate(weights)}


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

    # Class weights para lidar com desbalanceamento
    class_counts = count_files_per_class(args.data_dir, class_names)
    class_weight = make_class_weight(class_counts, class_names)
    print("Amostras por classe:", class_counts)
    print("Pesos por classe:", class_weight)

    model = build_model(
        num_classes=num_classes,
        img_size=args.img_size,
        dropout=args.dropout,
        use_augmentation=not args.no_augment,
        flip=args.flip,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
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
        class_weight=class_weight,
        verbose=1,
    )

    # Fine-tuning opcional (destrava últimas N camadas)
    if args.fine_tune and args.fine_tune > 0 and args.ft_epochs > 0:
        print(f"Iniciando fine-tuning das últimas {args.fine_tune} camadas do backbone...")
        unfreeze_last_n_layers(model, n_layers=args.fine_tune)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=max(args.lr * 0.1, 1e-6)),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
            metrics=['accuracy'],
        )
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.ft_epochs,
            callbacks=callbacks,
            class_weight=class_weight,
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

    # Avaliação e relatório (matriz de confusão e métricas por classe)
    try:
        evaluate_and_report(model, val_ds, class_names, prefix=args.report_prefix)
    except Exception as e:
        print("Falha ao gerar relatório de avaliação:", e)


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
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-model', default='keras_model.h5')
    p.add_argument('--output-labels', default='labels.txt')
    p.add_argument('--save-final', action='store_true', help='Também salva o estado final do modelo em *_final.h5')
    p.add_argument('--no-augment', action='store_true', help='Desativa data augmentation')
    p.add_argument('--flip', action='store_true', help='Ativa RandomFlip horizontal na augmentation (cuidado com lateralidade)')
    p.add_argument('--report-prefix', default='eval', help='Prefixo para arquivos de relatório (ex.: eval_confusion_matrix.png)')
    return p.parse_args()


def evaluate_and_report(model: tf.keras.Model, val_ds: tf.data.Dataset, class_names: list[str], prefix: str = 'eval'):
    """Gera matriz de confusão e métricas por classe usando o conjunto de validação."""
    import matplotlib.pyplot as plt

    # Coleta rótulos verdadeiros e predições
    y_true = []
    for _, y in val_ds:
        y_true.append(y.numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_true_lbl = np.argmax(y_true, axis=1)

    y_pred = model.predict(val_ds, verbose=0)
    if isinstance(y_pred, list):
        y_pred = y_pred[0]
    y_pred_lbl = np.argmax(y_pred, axis=1)

    num_classes = len(class_names)
    cm = tf.math.confusion_matrix(y_true_lbl, y_pred_lbl, num_classes=num_classes).numpy()

    # Métricas por classe (precisão, revocação, F1)
    tp = np.diag(cm).astype(np.float32)
    support_true = cm.sum(axis=1).astype(np.float32)  # por classe (linhas)
    support_pred = cm.sum(axis=0).astype(np.float32)  # por classe (colunas)
    precision = np.divide(tp, np.maximum(support_pred, 1), where=support_pred>0)
    recall = np.divide(tp, np.maximum(support_true, 1), where=support_true>0)
    f1 = np.divide(2*precision*recall, np.maximum(precision+recall, 1e-8))

    # Salva matriz de confusão como imagem
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title('Matriz de Confusão (validação)')
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_png = f"{prefix}_confusion_matrix.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("Matriz de confusão salva em:", out_png)

    # Salva métricas por classe em CSV
    out_csv = f"{prefix}_metrics.csv"
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write('class,precision,recall,f1,support_true,support_pred\n')
        for i, name in enumerate(class_names):
            f.write(f"{name},{precision[i]:.4f},{recall[i]:.4f},{f1[i]:.4f},{int(support_true[i])},{int(support_pred[i])}\n")
    print("Métricas por classe salvas em:", out_csv)

    # Principais confusões (pares fora da diagonal)
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    flat_indices = np.dstack(np.unravel_index(np.argsort(cm_copy.ravel())[::-1], cm_copy.shape))[0]
    print("Top confusões (verdadeiro -> predito : contagem):")
    shown = 0
    for i, j in flat_indices:
        if cm[i, j] <= 0:
            break
        print(f" - {class_names[i]} -> {class_names[j]} : {cm[i, j]}")
        shown += 1
        if shown >= 10:
            break


if __name__ == '__main__':
    args = parse_args()
    train(args)
