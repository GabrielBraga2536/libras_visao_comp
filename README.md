# Reconhecimento de Gestos de Mão em Vídeo com MediaPipe e Keras

Este código demonstra como realizar o reconhecimento de gestos de mão em um vídeo utilizando a biblioteca MediaPipe e um modelo de aprendizado profundo carregado com o Keras. O objetivo é identificar gestos de mão específicos em cada quadro do vídeo e exibir os resultados visualmente em um novo vídeo de saída.

## Pré-requisitos

Antes de executar o código, você precisará ter os seguintes pré-requisitos instalados:

- Python (versão utilizada: 3.7+)
- Bibliotecas Python: cv2, mediapipe, keras, numpy
- TensorFlow (versão utilizada: 2.9.1)

## Como usar

1. **Código Fonte**: Certifique-se de que você tenha o código fonte em mãos.

2. **Vídeo de Entrada**: Você precisa fornecer um vídeo como entrada para o código. No exemplo fornecido, o caminho para o vídeo é definido como 'V.mp4'. Certifique-se de que seu vídeo esteja no mesmo diretório ou atualize o caminho do arquivo de vídeo conforme necessário.

3. **Modelo Keras**: Você também precisa ter treinado um modelo de aprendizado profundo para o reconhecimento de gestos de mão usando o Keras. Certifique-se de carregar o modelo corretamente no código (por exemplo, 'keras_model.h5').

4. **Execução**: Execute o código em um ambiente Python compatível (como o Jupyter Notebook). Ele processará cada quadro do vídeo de entrada, identificará gestos de mão e exibirá os resultados visualmente em um novo vídeo de saída.

5. **Resultados**: O vídeo de saída conterá os quadros do vídeo de entrada com os rótulos indicando o gesto de mão identificado.

## Melhorando o Modelo

Para melhorar o modelo de reconhecimento de gestos de mão, considere as seguintes estratégias:

- Aumentar o tamanho do conjunto de dados de treinamento.
- Aumentar o número de classes de gestos de mão.
- Ajustar hiperparâmetros do modelo.
- Experimentar diferentes arquiteturas de rede.
- Aumentar a quantidade de dados de treinamento sintético.
- Ajustar os critérios de pós-processamento dos resultados.

## Manipular o modelo (.h5)

O arquivo `keras_model.h5` pode ser inspecionado e convertido com o utilitário `model_utils.py`.

Exemplos:

- Ver summary do modelo:
  
	```bash
	python model_utils.py summary --model keras_model.h5
	```

- Checar shapes de entrada/saída e parâmetros:
  
	```bash
	python model_utils.py info --model keras_model.h5
	```

- Listar camadas:
  
	```bash
	python model_utils.py layers --model keras_model.h5
	```

- Testar predição com um tensor dummy 224x224x3:
  
	```bash
	python model_utils.py io --model keras_model.h5
	```

- Converter para SavedModel (TensorFlow):
  
	```bash
	python model_utils.py to-savedmodel --model keras_model.h5 --out-dir saved_model/
	```

- Converter para TFLite (opcionalmente com float16):
  
	```bash
	python model_utils.py to-tflite --model keras_model.h5 --out-file model.tflite --float16
	```

- Substituir a última camada por uma nova Dense com N classes (requer re-treino posterior):
  
	```bash
	python model_utils.py replace-last --model keras_model.h5 --num-classes 4 --activation softmax --out novo_modelo.h5
	```

Observações:

- A operação `replace-last` cria um novo modelo com a última camada trocada, mas os pesos desta camada
	são iniciais/aleatórios; será necessário re-treinar ao menos essa camada para obter boas predições.
- O arquivo `.h5` geralmente não contém o mapeamento de rótulos (nomes das classes). Mantenha os rótulos
	em um arquivo separado (por exemplo, `labels.txt`) e garanta que a ordem corresponda às saídas do modelo.

## Treinar um novo modelo

Estrutura de dados esperada (um diretório com subpastas por classe):

```
dataset/
	A/
		img001.jpg
		...
	B/
	C/
	...
```

O script `train.py` divide automaticamente em treino/validação via `validation_split`.

Exemplos de uso:

```bash
# Treino básico
python3 train.py --data-dir dataset --epochs 15 --batch-size 32

# Treino + fine-tuning (destrava últimas 50 camadas do backbone por 10 épocas adicionais)
python3 train.py --data-dir dataset --epochs 5 --ft-epochs 10 --fine-tune 50
```

Saídas geradas:
- `keras_model.h5` — modelo treinado (melhor val_accuracy)
- `labels.txt` — rótulos na ordem usada pelo modelo

Observações:
- A normalização do `train.py` é compatível com o `libras.py` (escala para [-1, 1]).
- Para melhor desempenho, garanta balanceamento entre classes e considere aumentar dados (data augmentation).

## Usando um dataset do Kaggle

Este repositório inclui um utilitário para baixar e preparar datasets do Kaggle no formato aceito pelo `train.py` (pastas por classe): `kaggle_download_and_prepare.py`.

Pré-requisitos:
- Ter a CLI do Kaggle instalada e configurada.
	- Instalação: `pip install kaggle`
	- Credenciais: baixe `kaggle.json` em https://www.kaggle.com/ (Account -> Create API Token) e salve em `~/.kaggle/kaggle.json` (permissão 600) ou exporte `KAGGLE_USERNAME` e `KAGGLE_KEY`.

Casos suportados:
- Dataset já organizado em pastas por classe (ex.: `A/`, `B/`, ...): o script detecta e copia para `dataset/`.
- Dataset com CSV (arquivo -> rótulo) e imagens soltas: o script reconstrói a estrutura de pastas.

Exemplos de uso:

```bash
# 1) Baixar e detectar pastas por classe automaticamente
python kaggle_download_and_prepare.py \
	--dataset owner/dataset-slug \
	--target-dir .kaggle_raw \
	--unzip-dir .kaggle_unzip \
	--output-dir dataset

# 2) Baixar e reconstruir a partir de CSV
python kaggle_download_and_prepare.py \
	--dataset owner/dataset-slug \
	--target-dir .kaggle_raw \
	--unzip-dir .kaggle_unzip \
	--output-dir dataset \
	--csv-path labels.csv \
	--images-dir images \
	--filename-col filename \
	--label-col label

# Depois de preparado, treine normalmente
python3 train.py --data-dir dataset --epochs 15 --batch-size 32
```

Notas:
- Use `--symlink` para criar links simbólicos em vez de copiar arquivos (mais rápido e economiza espaço).
- Se o dataset vier em outro layout, especifique `--csv-path/--images-dir` e os nomes das colunas no CSV.

## Capturar seu próprio dataset com a webcam

Você pode usar a webcam para capturar rapidamente imagens e criar um dataset no formato esperado (`dataset/<classe>/imagem.jpg`) com o utilitário `capture_dataset.py`.

Exemplos:

```bash
# Captura 50 imagens da classe "A" para dataset/A/
python3 capture_dataset.py --label A --count 50 --out-dir dataset --resize 224

# Começa com 3s de contagem regressiva, tira 100 fotos, intervalo de 0.4s
python3 capture_dataset.py --label B --count 100 --interval 0.4 --start-delay 3 --resize 224

# Tenta recortar automaticamente a mão usando MediaPipe (se instalado)
python3 capture_dataset.py --label C --hand-crop --resize 224

# Define um ROI manual (x,y,w,h) e espelha imagem (selfie)
python3 capture_dataset.py --label D --roi 100,120,300,300 --mirror --resize 224
```

Atalhos de teclado durante a captura:
- `q`: sair
- `space`: pausar/retomar a captura automática
- `s`: salvar uma foto manualmente (útil para ajustes finos)

Observações:
- O `--resize 224` garante imagens quadradas prontas para o treino com MobileNetV2, mas é opcional
	(o `train.py` também redimensiona ao carregar). Manter uma resolução consistente ajuda.
- Para melhores resultados, mantenha o enquadramento similar entre classes (distância, iluminação, fundo).
- Capture variações (ângulos, pequenas rotações, posições) para melhorar a robustez do modelo.
