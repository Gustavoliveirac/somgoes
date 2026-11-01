#!/usr/bin/env bash
# roda_tudo.sh — prepara o cache e treina cenários e grades, na ordem.
#   ./roda_tudo.sh          rodada completa (retomável)
#   ./roda_tudo.sh rapido   conferência rápida
# Requer o disco externo montado. Ao final envie de volta: pacote_somgoes.zip
set -e
cd "$(dirname "$0")"
export GFUSE_ROOT="${GFUSE_ROOT:-$HOME/Documentos/gfuse}"
if [ ! -f cache_somgoes/amostras.parquet ]; then
    echo "[1/2] preparando cache completo..."; python prep_somgoes.py
elif ! grep -q entrada_colar cache_somgoes/meta_cache.json; then
    echo "[1/2] cache antigo: refazendo chuva e alvos, reaproveitando o GOES..."; python prep_somgoes.py --so-chuva
else
    echo "[1/2] cache atualizado (apague cache_somgoes/ para refazer)"
fi
echo "[2/2] treinando..."
if [ "$1" = "rapido" ]; then python train_somgoes.py --rapido; else python train_somgoes.py; fi
echo "Pronto. Envie de volta: pacote_somgoes.zip"
