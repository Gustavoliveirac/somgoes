#!/usr/bin/env bash
# Etapa A do capítulo (sem retreinar o SOM). Da raiz do somgoes:
#   nohup bash scripts/roda_etapa_a.sh > /dev/null 2>&1 &      (algumas horas; acompanhar com tail -f logs/etapa_a.log)
# Primeiro roda um teste curto (h30, anéis e área central, 2 temporadas). Se o teste falhar,
# a rodada completa não começa. PULAR_TESTE=1 pula o teste; SEM_TABPFN=1 e SEM_SEQ=1 desligam partes.
# Usa o ambiente ~/venvs/somgoes se ele existir. Ao final, o log é acrescentado ao
# logs/versionar.log, que é o único arquivo a enviar.
set -u
cd "$(dirname "$0")/.."      # raiz do projeto (este script fica em scripts/)
mkdir -p logs resultados
[ -f "$HOME/venvs/somgoes/bin/activate" ] && source "$HOME/venvs/somgoes/bin/activate"
LOG=logs/etapa_a.log
{
  echo "=== etapa A  $(date)  python $(command -v python)"
  python -c "import torch, sklearn; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| sklearn', sklearn.__version__)"
  python -c "import tabpfn; print('tabpfn', getattr(tabpfn, '__version__', '?'))" 2>/dev/null || echo "tabpfn não instalado"
  ok=1
  if [ "${PULAR_TESTE:-0}" != "1" ]; then
    echo "=== teste curto  $(date)"
    ETAPA_A_TESTE=1 python -m tese.etapa_a || { ok=0; echo "TESTE FALHOU; a rodada completa não foi iniciada"; }
  fi
  if [ "$ok" = "1" ]; then
    echo "=== rodada completa  $(date)"
    python -m tese.etapa_a || echo "rodada completa terminou com erro"
  fi
  echo "=== fim  $(date)"
} 2>&1 | tee "$LOG"
{ echo; echo "[$(date '+%F %T')] ================ ETAPA A (scripts/roda_etapa_a.sh) ================"; cat "$LOG"; } >> logs/versionar.log
[ -f etapa_a/etapa_a.log ] && cp etapa_a/etapa_a.log "resultados/etapa_a_$(date +%Y%m%d).log"
echo "pronto. envie logs/versionar.log"
