#!/usr/bin/env bash
# roda_analise.sh — estados, validação, síntese, motor de alerta e figuras.
# Requer o cache e os resultados da rodada completa. Ao final envie: pacote_analise.zip
set -e
cd "$(dirname "$0")/.."      # raiz do projeto (este script fica em scripts/)
export GFUSE_ROOT="${GFUSE_ROOT:-$HOME/Documentos/gfuse}"
echo "[1/6] estados (modelo final; reaproveita se já treinado)..."; python -m analise.estados_somgoes
echo "[2/6] estados, níveis e similaridade fora da temporada..."; python -m analise.validacao_estados
echo "[3/6] síntese (valor das fontes, antecedência, rotas)..."; python -m analise.sintese_somgoes
echo "[4/6] exportando o motor de alerta..."; python -m alerta.motor_alerta exportar
echo "[5/6] replay dos 8 dias de maior chuva..."; python -m alerta.motor_alerta eventos 8
echo "[6/6] figuras e pacote..."; python -m analise.analise_somgoes
echo "Para o artigo (cerca de 23 h na GPU, uma vez): python -m core.rodada_publicacao e depois rode este script de novo"
echo "Pronto. Figuras em analise_somgoes/figuras e motor/. Envie de volta: pacote_analise.zip"
