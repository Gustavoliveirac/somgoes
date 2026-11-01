#!/usr/bin/env bash
# Roda as análises e as figuras do capítulo. Coloque este arquivo, analise_tese.py e
# figuras_tese.py na pasta do somgoes (a mesma de config_somgoes.py) e rode:
#   bash roda_tese.sh            rodada completa (30 a 60 min; não interrompa)
#   SEM_MODELOS=1 bash roda_tese.sh   rodada rápida, sem logística e gradient boosting
# Ao final, envie só o arquivo log_tese.txt.
set -u
cd "$(dirname "$0")"
{
  echo "=== figuras que não dependem da análise  $(date)"
  python figuras_tese.py fig01 fig02 fig04 fig05 fig06 fig07 fig13
  echo
  echo "=== analise_tese.py  $(date)"
  python analise_tese.py || echo "analise_tese.py terminou com erro; seguindo para as figuras"
  echo
  echo "=== figuras_tese.py (todas)  $(date)"
  python figuras_tese.py
  echo "=== fim  $(date)"
} 2>&1 | tee log_tese.txt
echo "pronto. envie log_tese.txt (figuras em figuras_tese/pt e figuras_tese/en)"
