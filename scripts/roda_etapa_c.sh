#!/usr/bin/env bash
# Etapa C: SOM treinado do zero com cabeça de previsão (triagem de variantes numa
# temporada e rodada final nas seis). Da raiz do somgoes:
#   nohup bash scripts/roda_etapa_c.sh > /dev/null 2>&1 &     (cerca de 8 h; tail -f logs/etapa_c.log)
# Primeiro roda um teste curto; se falhar, a rodada completa não começa. PULAR_TESTE=1 pula o teste.
# Ao final, o log é acrescentado ao logs/versionar.log, que é o único arquivo a enviar.
set -u
cd "$(dirname "$0")/.."      # raiz do projeto (este script fica em scripts/)
mkdir -p logs resultados
[ -f "$HOME/venvs/somgoes/bin/activate" ] && source "$HOME/venvs/somgoes/bin/activate"
LOG=logs/etapa_c.log
{
  echo "=== etapa C  $(date)  python $(command -v python)"
  ok=1
  if [ "${PULAR_TESTE:-0}" != "1" ]; then
    echo "=== teste curto  $(date)"
    ETAPA_C_TESTE=1 python -m tese.etapa_c || { ok=0; echo "TESTE FALHOU; a rodada completa não foi iniciada"; }
  fi
  if [ "$ok" = "1" ]; then
    echo "=== rodada completa  $(date)"
    python -m tese.etapa_c || echo "rodada completa terminou com erro"
  fi
  echo "=== fim  $(date)"
} 2>&1 | tee "$LOG"
{ echo; echo "[$(date '+%F %T')] ================ ETAPA C (scripts/roda_etapa_c.sh) ================"; cat "$LOG"; } >> logs/versionar.log
[ -f etapa_c/etapa_c.log ] && cp etapa_c/etapa_c.log "resultados/etapa_c_$(date +%Y%m%d).log"
echo "pronto. envie logs/versionar.log"
