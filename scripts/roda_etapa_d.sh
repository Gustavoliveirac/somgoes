#!/usr/bin/env bash
# Etapa D: SOM com as mesmas entradas do gradient boosting e SOM como correção do
# gradient boosting (seis temporadas). Da raiz do somgoes:
#   nohup bash scripts/roda_etapa_d.sh > /dev/null 2>&1 &     (cerca de 10 a 12 h; tail -f logs/etapa_d.log)
# Primeiro roda um teste curto; se falhar, a rodada completa não começa. PULAR_TESTE=1 pula o teste.
# Ao final, o log é acrescentado ao logs/versionar.log, que é o único arquivo a enviar.
set -u
cd "$(dirname "$0")/.."      # raiz do projeto (este script fica em scripts/)
mkdir -p logs resultados
[ -f "$HOME/venvs/somgoes/bin/activate" ] && source "$HOME/venvs/somgoes/bin/activate"
LOG=logs/etapa_d.log
{
  echo "=== etapa D  $(date)  python $(command -v python)"
  ok=1
  if [ "${PULAR_TESTE:-0}" != "1" ]; then
    echo "=== teste curto  $(date)"
    ETAPA_D_TESTE=1 python -m tese.etapa_d || { ok=0; echo "TESTE FALHOU; a rodada completa não foi iniciada"; }
  fi
  if [ "$ok" = "1" ]; then
    echo "=== rodada completa  $(date)"
    python -m tese.etapa_d || echo "rodada completa terminou com erro"
  fi
  echo "=== fim  $(date)"
} 2>&1 | tee "$LOG"
{ echo; echo "[$(date '+%F %T')] ================ ETAPA D (scripts/roda_etapa_d.sh) ================"; cat "$LOG"; } >> logs/versionar.log
[ -f etapa_d/etapa_d.log ] && cp etapa_d/etapa_d.log "resultados/etapa_d_$(date +%Y%m%d).log"
echo "pronto. envie logs/versionar.log"
