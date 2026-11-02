#!/usr/bin/env bash
# Etapa B: gradient boosting com transições no SOM (seis temporadas) e SOM com cabeça treinados
# juntos (uma temporada, FOLD=2 por padrão). Da raiz do somgoes:
#   nohup bash scripts/roda_etapa_b.sh > /dev/null 2>&1 &     (algumas horas; tail -f logs/etapa_b.log)
# Primeiro roda um teste curto; se falhar, a rodada completa não começa. PULAR_TESTE=1 pula o teste.
# Ao final, o log é acrescentado ao logs/versionar.log, que é o único arquivo a enviar.
set -u
cd "$(dirname "$0")/.."      # raiz do projeto (este script fica em scripts/)
mkdir -p logs resultados
[ -f "$HOME/venvs/somgoes/bin/activate" ] && source "$HOME/venvs/somgoes/bin/activate"
LOG=logs/etapa_b.log
{
  echo "=== etapa B  $(date)  python $(command -v python)"
  ok=1
  if [ "${PULAR_TESTE:-0}" != "1" ]; then
    echo "=== teste curto  $(date)"
    ETAPA_B_TESTE=1 python -m tese.etapa_b || { ok=0; echo "TESTE FALHOU; a rodada completa não foi iniciada"; }
  fi
  if [ "$ok" = "1" ]; then
    echo "=== rodada completa  $(date)"
    python -m tese.etapa_b || echo "rodada completa terminou com erro"
  fi
  echo "=== fim  $(date)"
} 2>&1 | tee "$LOG"
{ echo; echo "[$(date '+%F %T')] ================ ETAPA B (scripts/roda_etapa_b.sh) ================"; cat "$LOG"; } >> logs/versionar.log
[ -f etapa_b/etapa_b.log ] && cp etapa_b/etapa_b.log "resultados/etapa_b_$(date +%Y%m%d).log"
echo "pronto. envie logs/versionar.log"
