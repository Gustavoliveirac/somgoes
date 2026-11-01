#!/usr/bin/env bash
# Fase 1: SOM com probabilidade suave e vizinhos no espaço latente, dois estágios sem vazamento
# e variáveis novas do GOES (temperatura de brilho, iniciação, deslocamento das nuvens) e dos
# pluviômetros. Coloque fase1_som.py e este arquivo na pasta do somgoes, junto com
# modelos_tese.py, analise_tese.py, goes_geometria.npz e goes_planck.json.
#   conda activate <ambiente do somgoes>
#   bash roda_fase1.sh            (cerca de 1 h; não interrompa)
#   SEM_CATBOOST=1 bash roda_fase1.sh   (mais rápido, sem a referência do CatBoost)
# Ao final, envie só o arquivo log_fase1.txt.
set -u
cd "$(dirname "$0")"
{
  echo "=== fase1_som.py  $(date)"
  python fase1_som.py || echo "fase1_som.py terminou com erro"
  echo "=== figura 1 com o recorte do GOES  $(date)"
  python figuras_tese.py fig01
  echo "=== fim  $(date)"
} 2>&1 | tee log_fase1.txt
echo "pronto. envie log_fase1.txt"
