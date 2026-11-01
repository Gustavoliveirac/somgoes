#!/usr/bin/env bash
# Fase 1: SOM com probabilidade suave e vizinhos no espaço latente, dois estágios sem vazamento
# e variáveis novas do GOES (temperatura de brilho, iniciação, deslocamento das nuvens) e dos
# pluviômetros. Precisa de goes/goes_geometria.npz e goes/goes_planck.json.
#   conda activate <ambiente do somgoes>
#   bash scripts/roda_fase1.sh            (cerca de 1 h; não interrompa)
#   SEM_CATBOOST=1 bash scripts/roda_fase1.sh   (mais rápido, sem a referência do CatBoost)
# Ao final, envie só o arquivo logs/log_fase1.txt.
set -u
cd "$(dirname "$0")/.."      # raiz do projeto (este script fica em scripts/)
mkdir -p logs
{
  echo "=== fase1_som.py  $(date)"
  python -m tese.fase1_som || echo "fase1_som.py terminou com erro"
  echo "=== figura 1 com o recorte do GOES  $(date)"
  python -m tese.figuras_tese fig01
  echo "=== fim  $(date)"
} 2>&1 | tee logs/log_fase1.txt
echo "pronto. envie logs/log_fase1.txt"
