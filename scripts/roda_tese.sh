#!/usr/bin/env bash
# Roda as análises e as figuras do capítulo. Da raiz do somgoes:
#   bash scripts/roda_tese.sh            rodada completa (30 a 60 min; não interrompa)
#   SEM_MODELOS=1 bash scripts/roda_tese.sh   rodada rápida, sem logística e gradient boosting
# Ao final, envie só o arquivo logs/log_tese.txt.
set -u
cd "$(dirname "$0")/.."      # raiz do projeto (este script fica em scripts/)
mkdir -p logs
{
  echo "=== figuras que não dependem da análise  $(date)"
  python -m tese.figuras_tese fig01 fig02 fig04 fig05 fig06 fig07 fig13
  echo
  echo "=== analise_tese.py  $(date)"
  python -m tese.analise_tese || echo "analise_tese.py terminou com erro; seguindo para as figuras"
  echo
  echo "=== figuras_tese.py (todas)  $(date)"
  python -m tese.figuras_tese
  echo "=== fim  $(date)"
} 2>&1 | tee logs/log_tese.txt
echo "pronto. envie logs/log_tese.txt (figuras em figuras_tese/pt e figuras_tese/en)"
