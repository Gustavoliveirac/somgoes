#!/usr/bin/env bash
# Roda a comparação de classificadores (modelos_tese.py) e refaz as figuras novas.
#   conda activate <ambiente do somgoes>
#   pip install catboost          (uma vez; sem ele o CatBoost é pulado)
#   bash scripts/roda_modelos.sh          (1 a 2 h; não interrompa)
# Ao final, envie só o arquivo logs/log_modelos.txt.
set -u
cd "$(dirname "$0")/.."      # raiz do projeto (este script fica em scripts/)
mkdir -p logs
{
  echo "=== modelos_tese.py  $(date)"
  python -m tese.modelos_tese || echo "modelos_tese.py terminou com erro; seguindo para as figuras"
  echo
  echo "=== figuras_tese.py  $(date)"
  python -m tese.figuras_tese fig01 fig14 fig15
  echo "=== fim  $(date)"
} 2>&1 | tee logs/log_modelos.txt
echo "pronto. envie logs/log_modelos.txt"
