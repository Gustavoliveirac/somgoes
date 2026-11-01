#!/usr/bin/env bash
# Roda a comparação de classificadores (modelos_tese.py) e refaz as figuras novas.
# Coloque modelos_tese.py, analise_tese.py, figuras_tese.py e este arquivo na pasta do somgoes.
#   conda activate <ambiente do somgoes>
#   pip install catboost          (uma vez; sem ele o CatBoost é pulado)
#   bash roda_modelos.sh          (1 a 2 h; não interrompa)
# Ao final, envie só o arquivo log_modelos.txt.
set -u
cd "$(dirname "$0")"
{
  echo "=== modelos_tese.py  $(date)"
  python modelos_tese.py || echo "modelos_tese.py terminou com erro; seguindo para as figuras"
  echo
  echo "=== figuras_tese.py  $(date)"
  python figuras_tese.py fig01 fig14 fig15
  echo "=== fim  $(date)"
} 2>&1 | tee log_modelos.txt
echo "pronto. envie log_modelos.txt"
