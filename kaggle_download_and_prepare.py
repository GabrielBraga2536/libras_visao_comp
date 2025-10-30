#!/usr/bin/env python3
"""
Baixar dataset do Kaggle e prepará-lo no formato esperado por train.py:

Estrutura de saída esperada (um diretório com subpastas por classe):

output_dir/
  A/
    img001.jpg
  B/
  ...

Modos de preparo suportados:
1) Dataset já vem com pastas por classe -> detecta e copia/move para output_dir
2) Dataset vem com CSV (arquivo -> rótulo) e imagens soltas -> reconstrói pastas por classe

Requisitos (uma das opções):
- Ter a CLI do Kaggle instalada (pip install kaggle) e credenciais configuradas (~/.kaggle/kaggle.json)
- OU ter as variáveis de ambiente KAGGLE_USERNAME e KAGGLE_KEY definidas

Exemplos de uso:

# Baixar e tentar detectar pastas por classe automaticamente
python kaggle_download_and_prepare.py \
  --dataset owner/dataset-slug \
  --target-dir .kaggle_raw \
  --unzip-dir .kaggle_unzip \
  --output-dir dataset

# Baixar e reconstruir a partir de CSV
python kaggle_download_and_prepare.py \
  --dataset owner/dataset-slug \
  --target-dir .kaggle_raw \
  --unzip-dir .kaggle_unzip \
  --output-dir dataset \
  --csv-path labels.csv \
  --images-dir images \
  --filename-col filename \
  --label-col label

"""
from __future__ import annotations
import argparse
import csv
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}


def run(cmd: List[str], env: Optional[dict] = None) -> None:
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        raise SystemExit(f"Comando falhou com código {res.returncode}: {' '.join(cmd)}")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def has_kaggle_cli() -> bool:
    from shutil import which
    return which('kaggle') is not None


def get_local_kaggle_config_dir() -> Optional[Path]:
    """Retorna diretório .kaggle local do projeto se existir (./.kaggle/kaggle.json)."""
    here = Path.cwd() / '.kaggle'
    if (here / 'kaggle.json').exists():
        return here
    return None


def check_kaggle_credentials() -> None:
    # CLI do Kaggle requer ~/.kaggle/kaggle.json com permissão 600
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    local_cfg = get_local_kaggle_config_dir()
    if local_cfg:
        try:
            os.chmod(local_cfg / 'kaggle.json', 0o600)
        except Exception:
            pass
        # Define variável de ambiente para a CLI usar o config local
        os.environ['KAGGLE_CONFIG_DIR'] = str(local_cfg)
        return
    if kaggle_json.exists():
        try:
            os.chmod(kaggle_json, 0o600)
        except Exception:
            pass
        return
    # ou variáveis de ambiente
    if os.getenv('KAGGLE_USERNAME') and os.getenv('KAGGLE_KEY'):
        return
    print("[AVISO] Credenciais do Kaggle não encontradas.")
    print("- Baixe seu kaggle.json em https://www.kaggle.com/ -> Account -> Create API Token")
    print(f"- Salve em {kaggle_json} (permissão 600) OU exporte KAGGLE_USERNAME e KAGGLE_KEY")
    # não aborta; a CLI pode pedir login dependendo do ambiente


def download_kaggle_dataset(dataset: str, target_dir: Path) -> List[Path]:
    """Baixa todos os arquivos do dataset Kaggle para target_dir.
    Retorna a lista de zips baixados (se houver)."""
    ensure_dir(target_dir)
    check_kaggle_credentials()
    if not has_kaggle_cli():
        raise SystemExit("A CLI do Kaggle não está instalada. Instale com: pip install kaggle")

    # Baixa todos os arquivos; -o sobrescreve se já existir
    # Propaga KAGGLE_CONFIG_DIR se estiver setado para suportar config local ao projeto
    env = os.environ.copy()
    run(['kaggle', 'datasets', 'download', '-d', dataset, '-p', str(target_dir), '-o'], env=env)

    zips = list(target_dir.glob('*.zip'))
    if not zips:
        print("Nenhum .zip encontrado; o dataset pode ter sido baixado como arquivo único.")
    else:
        print(f"Zips baixados: {[z.name for z in zips]}")
    return zips


def unzip_all(zips: List[Path], unzip_dir: Path) -> List[Path]:
    ensure_dir(unzip_dir)
    extracted_roots: List[Path] = []
    for z in zips:
        print(f"Extraindo {z} -> {unzip_dir}")
        with zipfile.ZipFile(z, 'r') as zf:
            zf.extractall(unzip_dir)
        extracted_roots.append(unzip_dir)
    return extracted_roots


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def detect_class_folders(root: Path) -> Optional[Path]:
    """Tenta detectar uma pasta que contenha subpastas por classe com imagens."""
    # caso 1: o próprio root tem subpastas por classe
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if subdirs and all(any(is_image_file(f) for f in d.rglob('*')) for d in subdirs):
        return root
    # caso 2: há uma pasta interna que contém as classes
    for d in root.rglob('*'):
        if d.is_dir():
            subdirs = [s for s in d.iterdir() if s.is_dir()]
            if subdirs and all(any(is_image_file(f) for f in s.rglob('*')) for s in subdirs):
                return d
    return None


def copytree_class_folders(src_root: Path, dst_root: Path) -> None:
    ensure_dir(dst_root)
    for cls_dir in sorted([d for d in src_root.iterdir() if d.is_dir()]):
        dst_cls = dst_root / cls_dir.name
        ensure_dir(dst_cls)
        count = 0
        for f in cls_dir.rglob('*'):
            if f.is_file() and is_image_file(f):
                shutil.copy2(f, dst_cls / f.name)
                count += 1
        print(f"Classe {cls_dir.name}: {count} imagens")


def prepare_from_csv(csv_path: Path, images_dir: Path, output_dir: Path,
                      filename_col: str, label_col: str, use_symlink: bool=False) -> None:
    ensure_dir(output_dir)
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    by_label: Dict[str, List[Path]] = {}
    for r in rows:
        fn = r[filename_col]
        lab = r[label_col]
        img = images_dir / fn
        if not img.exists():
            print(f"[AVISO] Arquivo não encontrado: {img}")
            continue
        by_label.setdefault(lab, []).append(img)
    for lab, files in by_label.items():
        dst = output_dir / lab
        ensure_dir(dst)
        for img in files:
            dst_file = dst / img.name
            if use_symlink:
                try:
                    if dst_file.exists():
                        dst_file.unlink()
                    os.symlink(os.path.abspath(img), dst_file)
                except Exception:
                    shutil.copy2(img, dst_file)
            else:
                shutil.copy2(img, dst_file)
        print(f"Classe {lab}: {len(files)} imagens")


def count_images_per_class(output_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for d in sorted([p for p in output_dir.iterdir() if p.is_dir()]):
        n = sum(1 for f in d.rglob('*') if f.is_file() and is_image_file(f))
        counts[d.name] = n
    return counts


def main():
    ap = argparse.ArgumentParser(description="Baixar dataset do Kaggle e preparar em pastas por classe")
    ap.add_argument('--dataset', required=True, help='Slug do Kaggle: owner/dataset-slug')
    ap.add_argument('--target-dir', default='.kaggle_raw', help='Diretório para baixar os .zip')
    ap.add_argument('--unzip-dir', default='.kaggle_unzip', help='Diretório para descompactar arquivos')
    ap.add_argument('--output-dir', default='dataset', help='Diretório de saída no formato classes/')
    ap.add_argument('--csv-path', help='Se fornecido, CSV com mapeamento filename->label')
    ap.add_argument('--images-dir', help='Diretório onde estão as imagens referenciadas no CSV')
    ap.add_argument('--filename-col', default='filename')
    ap.add_argument('--label-col', default='label')
    ap.add_argument('--symlink', action='store_true', help='Criar symlinks ao invés de copiar arquivos')

    args = ap.parse_args()

    target_dir = Path(args.target_dir)
    unzip_dir = Path(args.unzip_dir)
    output_dir = Path(args.output_dir)

    zips = download_kaggle_dataset(args.dataset, target_dir)
    if zips:
        unzip_all(zips, unzip_dir)
    else:
        # pode ser que o Kaggle tenha baixado um único arquivo grande não-zipado
        # neste caso, apenas tente usar unzip_dir = target_dir
        unzip_dir = target_dir

    if args.csv_path:
        if not args.images_dir:
            raise SystemExit('--images-dir é obrigatório quando --csv-path é usado')
        csv_path = unzip_dir / args.csv_path if not os.path.isabs(args.csv_path) else Path(args.csv_path)
        images_dir = unzip_dir / args.images_dir if not os.path.isabs(args.images_dir) else Path(args.images_dir)
        if not csv_path.exists():
            raise SystemExit(f'CSV não encontrado: {csv_path}')
        if not images_dir.exists():
            raise SystemExit(f'Diretório de imagens não encontrado: {images_dir}')
        prepare_from_csv(csv_path, images_dir, output_dir, args.filename_col, args.label_col, use_symlink=args.symlink)
    else:
        detected = detect_class_folders(unzip_dir)
        if detected is None:
            print('[ERRO] Não foi possível detectar estrutura em pastas por classe. Informe --csv-path e colunas.')
            sys.exit(1)
        print(f'Estrutura de classes detectada em: {detected}')
        copytree_class_folders(detected, output_dir)

    counts = count_images_per_class(output_dir)
    total = sum(counts.values())
    print('Resumo por classe:')
    for k, v in counts.items():
        print(f'  {k}: {v}')
    print('Total de imagens:', total)
    print('Pronto! Agora rode: python3 train.py --data-dir', str(output_dir))


if __name__ == '__main__':
    main()
