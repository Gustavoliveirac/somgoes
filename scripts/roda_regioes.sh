#!/usr/bin/env bash
# Transições entre as 8 regiões do SOM e POD e FAR por limiar (sem treino, cerca de 15 min).
# Da raiz do somgoes:  bash scripts/roda_regioes.sh
# Ao final, o log é acrescentado ao logs/versionar.log, que é o único arquivo a enviar.
set -u
cd "$(dirname "$0")/.."      # raiz do projeto (este script fica em scripts/)
mkdir -p logs resultados
[ -f "$HOME/venvs/somgoes/bin/activate" ] && source "$HOME/venvs/somgoes/bin/activate"
LOG=logs/regioes.log
{
  echo "=== regiões e POD/FAR  $(date)"
  python -m tese.regioes_transicoes || echo "regioes_transicoes terminou com erro"
  echo "=== fim  $(date)"
} 2>&1 | tee "$LOG"
{ echo; echo "[$(date '+%F %T')] ================ REGIÕES (scripts/roda_regioes.sh) ================"; cat "$LOG"; } >> logs/versionar.log
[ -f etapa_a/regioes/regioes.log ] && cp etapa_a/regioes/regioes.log "resultados/regioes_$(date +%Y%m%d).log"
echo "pronto. envie logs/versionar.log"
