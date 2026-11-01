#!/usr/bin/env bash
# =====================================================================
# versionar.sh - controle de versao do somgoes com git (espelho no GitHub)
#
# Tudo acontece dentro de <raiz do somgoes>/logs. As entregas das conversas
# (entrega_*.zip) sao colocadas em logs/, o log unico e logs/versionar.log
# e a pasta para o Project sai em logs/contexto/.
#
# Uso, sempre de dentro da pasta do somgoes (a de config_somgoes.py)
#   ./versionar.sh iniciar               uma vez so, cria o repositorio e o primeiro commit
#   ./versionar.sh aplicar [zip]         aplica uma entrega (sem argumento, a entrega_*.zip
#                                        mais recente em logs/)
#   ./versionar.sh fechar ["nota"]       fim da conversa, registra o que estiver pendente,
#                                        anota no historico e monta logs/contexto
#   ./versionar.sh desfazer              reverte o ultimo commit (cria um commit novo, nada se perde)
#   ./versionar.sh marcar v1.1 "texto"   cria uma tag de versao estavel
#   ./versionar.sh status                mostra o que mudou desde o ultimo commit
#   ./versionar.sh contexto              so monta logs/contexto, sem commit (para conferir o tamanho)
#   ./versionar.sh remoto <url>          liga o repositorio ao GitHub e envia tudo (uma vez so)
#
# Mensagens de commit em ingles. Se houver remoto 'origin', cada aplicar, fechar,
# marcar e desfazer termina com git push (commits e tags).
# Esta pasta e o repositorio publicado no GitHub. As subpastas orientacoes/ (modo de
# trabalho, HISTORICO.md e contexto_fora.txt), escrita/ (guia e transicao), referencias/
# e resultados/ ficam fora do git pelo .gitignore e sobem so para o Project.
# O contexto do Project junta tudo numa pasta unica, sem subpastas, com prefixos.
#
# O historico de tudo (inclusive erros) fica em orientacoes/HISTORICO.md, fora do git.
# =====================================================================
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
RAIZ="$(pwd -P)"
mkdir -p logs
LOG="$RAIZ/logs/versionar.log"
REG="$RAIZ/logs/entregas.txt"
PAI="$RAIZ"   # orientacoes/, escrita/, referencias/ e resultados/ ficam dentro da pasta, ignoradas pelo git
HIST="$PAI/orientacoes/HISTORICO.md"
CTX="$RAIZ/logs/contexto"
TMP=""
trap '[ -n "$TMP" ] && rm -rf "$TMP"' EXIT

log()   { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
falha() { log "ERRO: $*"; log "Me envie: $LOG"; exit 1; }
cab()   { echo >> "$LOG"; log "================ $* ================"; }
trim()  { sed -e 's/\r$//' -e 's/#.*//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }
versao(){ git describe --tags --always --dirty 2>/dev/null; }
precisa_repo() { [ -d .git ] || falha "repositorio nao existe, rode antes ./versionar.sh iniciar"; }
juntar() { local IFS=","; echo "$*" | sed 's/,/, /g'; }

hist() {   # hist <titulo> ; o corpo vem pela entrada padrao
    mkdir -p "$(dirname "$HIST")"
    [ -f "$HIST" ] || printf '# Historico do somgoes\n\nRegistro automatico do versionar.sh, do mais antigo para o mais recente, incluindo entregas recusadas e erros.\n' > "$HIST"
    { echo; echo "## $(date '+%F %H:%M') $1"; echo; cat; } >> "$HIST"
}
commit_hist() { :; }   # o historico fica fora do git (orientacoes/), nao ha o que commitar

[ -f config_somgoes.py ] || [ -f src/config_somgoes.py ] || falha "rode de dentro da pasta do somgoes (a que tem config_somgoes.py ou src/config_somgoes.py)"
command -v git >/dev/null || falha "git nao instalado (sudo apt install git)"

enviar() {   # envia commits e tags ao GitHub, se houver remoto origin
    git remote get-url origin >/dev/null 2>&1 || return 0
    if git push -q origin HEAD --follow-tags >> "$LOG" 2>&1; then log "enviado ao GitHub ($(git remote get-url origin))"
    else log "AVISO: push ao GitHub falhou (detalhe acima); os commits continuam locais, rode git push depois"; fi
}

cmd_remoto() {
    cab "REMOTO"
    precisa_repo
    local U="${1:-}"; [ -n "$U" ] || falha "informe a url, ex.: ./versionar.sh remoto git@github.com:USUARIO/somgoes.git"
    if git remote get-url origin >/dev/null 2>&1; then git remote set-url origin "$U"; else git remote add origin "$U"; fi
    git push -q -u origin HEAD --tags >> "$LOG" 2>&1 || falha "push falhou; confira a chave SSH ou o token do GitHub"
    hist "remoto" <<< "Repositorio ligado a $U."
    commit_hist "history: remote set" && git push -q origin HEAD >> "$LOG" 2>&1
    log "repositorio ligado e enviado: $U"
}

# ------------------------------------------------------------------ iniciar
cmd_iniciar() {
    cab "INICIAR"
    if [ -d .git ]; then
        log "repositorio ja existe, nada a fazer (ultimo commit $(git log -1 --format='%h %s'))"
        return 0
    fi
    [ -f .gitignore ] || falha ".gitignore nao esta nesta pasta"
    git init -q -b main 2>/dev/null || git init -q || falha "git init falhou"
    git config user.name  >/dev/null || git config user.name "Gustavo Correa"
    git config user.email >/dev/null || git config user.email "somgoes@$(hostname)"

    local D="_arquivo/$(date +%Y%m%d_%H%M%S)_versionamento" movidos=()
    for f in atualizar_pacote.sh limpar_projeto.sh preparar_contexto.sh; do
        if [ -f "$f" ]; then
            mkdir -p "$D" && mv "$f" "$D/" && movidos+=("$f") && log "movido para $D: $f (substituido pelo versionar.sh)"
        fi
    done
    if [ -f README.md ] && grep -q 'atualizar_pacote.sh' README.md; then
        awk '/^\| `atualizar_pacote.sh`/ {print "| `versionar.sh` | controle de versao. `aplicar` aplica uma entrega de logs/, `fechar` encerra a conversa e monta logs/contexto (ver docs/VERSIONAMENTO.md) |"; next}
             /^\| `limpar_projeto.sh`/ {next}
             {gsub(/atualizar_pacote.sh/, "versionar.sh"); print}' README.md > README.md.novo && mv README.md.novo README.md
        log "README.md atualizado (tabela de scripts)"
    fi

    hist "iniciar" <<EOT
Repositorio git criado em $RAIZ com tag v1.0.
Scripts movidos para $D por terem sido substituidos pelo versionar.sh: $(juntar "${movidos[@]:-nenhum}").
EOT
    git add -A
    while IFS= read -r g; do
        [ -f "$g" ] || continue
        if [ "$(du -k "$g" | cut -f1)" -gt 2048 ]; then
            git rm -q --cached -- "$g"
            log "AVISO: $g tem mais de 2 MB e ficou fora do git (se for dado, inclua no .gitignore)"
            echo "Arquivo com mais de 2 MB deixado fora do git: $g" >> "$HIST"
        fi
    done < <(git diff --cached --name-only)

    git commit -q -m "Initial commit of somgoes under version control" || falha "commit inicial falhou"
    git tag -a v1.0 -m "State on $(date +%F) when version control started"
    log "commit inicial $(git rev-parse --short HEAD), tag v1.0, $(git ls-files | wc -l) arquivos versionados"
    git ls-files | sed 's/^/    /' >> "$LOG"
    montar_contexto
}

# ------------------------------------------------------------------ aplicar
cmd_aplicar() {
    cab "APLICAR ENTREGA"
    precisa_repo
    local Z="${1:-}"
    [ -n "$Z" ] || Z=$(ls -t logs/entrega_*.zip 2>/dev/null | head -1)
    [ -n "$Z" ] && [ -f "$Z" ] || falha "nenhuma entrega_*.zip em logs/"
    Z=$(readlink -f "$Z")
    local ID; ID=$(basename "$Z" .zip)
    log "entrega $ID"
    if [ -f "$REG" ] && grep -q "^$ID .*aplicada" "$REG"; then
        falha "a entrega $ID ja foi aplicada ($(grep "^$ID " "$REG" | tail -1)); para reaplicar, renomeie o zip"
    fi

    TMP=$(mktemp -d)
    unzip -q -o "$Z" -d "$TMP" || falha "nao consegui descompactar $Z"
    local B="$TMP"
    [ -f "$B/ENTREGA.txt" ] || B=$(dirname "$(find "$TMP" -maxdepth 2 -name ENTREGA.txt | head -1)" 2>/dev/null)
    [ -f "$B/ENTREGA.txt" ] || falha "zip sem ENTREGA.txt, nao esta no formato de entrega"
    local MSG DESC
    MSG=$(head -1 "$B/ENTREGA.txt" | tr -d '\r')
    DESC=$(tail -n +2 "$B/ENTREGA.txt" | tr -d '\r')
    log "descricao: $MSG"

    local LISTA=()
    if [ -d "$B/arquivos" ]; then
        while IFS= read -r f; do
            case "$f" in /*|*..*) falha "caminho invalido na entrega: $f";; esac
            LISTA+=("$f")
        done < <(cd "$B/arquivos" && find . -type f -not -path '*/__pycache__/*' | sed 's|^\./||' | sort)
    fi

    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        local mud; mud=$(git status --porcelain --untracked-files=no | awk '{print $2}' | tr '\n' ' ')
        hist "alteracoes manuais no servidor" <<< "Arquivos alterados fora de uma entrega, registrados antes de $ID: $mud"
        git add -u && git commit -q -m "Manual changes on the machine before $ID"
        log "havia alteracoes manuais nao registradas, guardadas no commit $(git rev-parse --short HEAD)"
    fi
    local ANTES; ANTES=$(git rev-parse HEAD)

    local TOCADOS=() NOVOS=() ALTERADOS=() REMOVIDOS=()
    for f in "${LISTA[@]}"; do
        if [ -f "$f" ] && cmp -s "$B/arquivos/$f" "$f"; then log "igual       $f"; continue; fi
        if [ -e "$f" ]; then ALTERADOS+=("$f"); log "alterado    $f"; else NOVOS+=("$f"); log "novo        $f"; fi
        mkdir -p "$(dirname "$f")"
        cp "$B/arquivos/$f" "$f" || falha "copia falhou em $f"
        case "$f" in *.sh) chmod +x "$f";; esac
        TOCADOS+=("$f")
    done
    if [ -f "$B/REMOVER.txt" ]; then
        while IFS= read -r f; do
            f=$(echo "$f" | trim); [ -z "$f" ] && continue
            case "$f" in /*|*..*) log "AVISO: ignorado caminho invalido no REMOVER.txt: $f"; continue;; esac
            if [ -e "$f" ]; then
                local dst="_arquivo/removidos/$ID/$f"
                mkdir -p "$(dirname "$dst")" && mv "$f" "$dst"
                log "removido    $f (copia em _arquivo/removidos/$ID)"
                TOCADOS+=("$f"); REMOVIDOS+=("$f")
            else
                log "ja ausente  $f"
            fi
        done < "$B/REMOVER.txt"
    fi

    local ERRTXT="" YML=0 saida
    for f in "${TOCADOS[@]}"; do
        [ -f "$f" ] || continue
        case "$f" in
            *.py) saida=$(python3 -m py_compile "$f" 2>&1) || ERRTXT+="$f: $(echo "$saida" | tail -4 | tr '\n' ' ')"$'\n' ;;
            *.sh) saida=$(bash -n "$f" 2>&1)            || ERRTXT+="$f: $(echo "$saida" | head -2 | tr '\n' ' ')"$'\n' ;;
            docker-compose*.yml) YML=1 ;;
        esac
    done
    if [ "$YML" -eq 1 ] && command -v docker >/dev/null; then
        saida=$(docker compose config -q 2>&1) || ERRTXT+="docker compose config: $(echo "$saida" | head -2 | tr '\n' ' ')"$'\n'
    fi
    if [ -n "$ERRTXT" ]; then
        echo "$ERRTXT" >> "$LOG"
        git reset -q --hard "$ANTES"
        for f in "${NOVOS[@]}"; do rm -f "$f"; done
        for f in "${REMOVIDOS[@]}"; do [ -e "_arquivo/removidos/$ID/$f" ] && mv "_arquivo/removidos/$ID/$f" "$f"; done
        hist "$ID RECUSADA" <<EOT
$MSG

Erro de sintaxe, nada foi aplicado e os arquivos voltaram ao estado anterior.
\`\`\`
$ERRTXT\`\`\`
EOT
        commit_hist "history: $ID rejected"
        echo "$ID $(date '+%F %T') recusada $MSG" >> "$REG"
        falha "a entrega tem erro de sintaxe; tudo voltou ao estado anterior (detalhe acima e no historico)"
    fi

    local POS="nao havia"
    if [ -f "$B/pos_aplicar.sh" ]; then
        log "rodando pos_aplicar.sh"
        local ini; ini=$(wc -l < "$LOG")
        if (cd "$RAIZ" && bash "$B/pos_aplicar.sh") >> "$LOG" 2>&1; then
            POS="concluido"; log "pos_aplicar.sh concluido"
        else
            POS="FALHOU, ultimas linhas abaixo"
            log "AVISO: pos_aplicar.sh terminou com erro (o codigo fica registrado mesmo assim)"
            POS+=$'\n```\n'"$(tail -n +"$((ini+1))" "$LOG" | grep -v 'AVISO: pos_aplicar' | tail -8)"$'\n```'
        fi
    fi

    hist "$ID aplicada" <<EOT
$MSG
${DESC:+
$DESC
}
Novos: $(juntar "${NOVOS[@]:-nenhum}").
Alterados: $(juntar "${ALTERADOS[@]:-nenhum}").
Removidos: $(juntar "${REMOVIDOS[@]:-nenhum}").
pos_aplicar.sh: $POS
EOT
    local VERS=()
    for f in "${TOCADOS[@]}"; do git check-ignore -q -- "$f" || VERS+=("$f"); done
    [ "${#VERS[@]}" -gt 0 ] && git add -A -- "${VERS[@]}"
    git add -u
    { echo "$MSG [$ID]"; if [ -n "$DESC" ]; then echo; echo "$DESC"; fi; } | git commit -q --allow-empty -F - || falha "commit falhou"
    local C; C=$(git rev-parse --short HEAD)
    log "commit $C, versao $(versao)"
    git show --stat --format= HEAD | sed 's/^/    /' >> "$LOG"
    echo "$ID $(date '+%F %T') aplicada commit=$C $MSG" >> "$REG"
    enviar
    log "entrega registrada; no fim da conversa rode ./versionar.sh fechar"
}

# ------------------------------------------------------------------ fechar
cmd_fechar() {
    cab "FECHAR CONVERSA"
    precisa_repo
    local NOTA="${*:-}"
    local mud; mud=$(git status --porcelain --untracked-files=no | awk '{print $2}' | tr '\n' ' ')
    local nsoltos soltos; nsoltos=$(git ls-files --others --exclude-standard | wc -l)
    soltos=$(git ls-files --others --exclude-standard | head -40 | tr '\n' ' ')
    local ents=""; [ -f "$REG" ] && ents=$(awk -v d="$(git log -1 --format=%cd --date=format:'%F %T' --grep='^Session close' 2>/dev/null)" '($2" "$3) > d' "$REG")
    hist "fechamento da conversa" <<EOT
${NOTA:+$NOTA

}Entregas desde o ultimo fechamento:
\`\`\`
${ents:-nenhuma}
\`\`\`
Alteracoes manuais registradas agora: ${mud:-nenhuma}.
Arquivos soltos fora do git ($nsoltos, ate 40 listados): ${soltos:-nenhum}.
EOT
    git add -u
    git commit -q --allow-empty -m "Session close${NOTA:+: $NOTA}" || falha "commit de fechamento falhou"
    log "commit $(git rev-parse --short HEAD), versao $(versao)"
    [ -n "$mud" ] && log "alteracoes manuais registradas: $mud"
    [ -n "$soltos" ] && log "AVISO: arquivos soltos fora do git (decidir se sao essenciais ou teste): $soltos"
    enviar
    montar_contexto
}

CTX_FORA="$PAI/orientacoes/contexto_fora.txt"
CTX_HIST_KB=150

fora_do_contexto() {   # 0 se o arquivo casa com um padrao de docs/contexto_fora.txt
    [ -f "$CTX_FORA" ] || return 1
    local p
    while IFS= read -r p; do
        p=$(echo "$p" | trim); [ -z "$p" ] && continue
        case "$1" in $p) return 0;; esac
    done < "$CTX_FORA"
    return 1
}

copiar_ctx() {   # copiar_ctx <origem> <destino>; escreve na saida a observacao para o indice
    local f="$1" d="$2" tam; tam=$(stat -c %s "$f")
    if [ "$f" = "$HIST" ]; then
        local t="$d.tmp" tam2 lim ini
        LC_ALL=C awk '{ if (length($0) > 2000) printf "[linha de %d bytes omitida na copia para o Project]\n", length($0); else print }' "$f" > "$t"
        tam2=$(stat -c %s "$t")
        if [ "$tam2" -gt $((CTX_HIST_KB*1024)) ]; then
            lim=$((tam2 - CTX_HIST_KB*1024))
            ini=$(LC_ALL=C awk -v lim="$lim" '{b+=length($0)+1} /^## 20[0-9][0-9]-/ && b>lim {print NR; exit}' "$t")
            { echo "# Historico do somgoes (trecho final)"; echo
              echo "Copia para o Project so com as ultimas entradas (ate ${CTX_HIST_KB} kB). O arquivo completo fica em $HIST no servidor."
              echo; tail -n +"${ini:-1}" "$t"; } > "$d"
            rm -f "$t"; echo "trecho final, completo no servidor"
        else
            mv "$t" "$d"; echo "linhas longas omitidas, completo no servidor"
        fi
        return
    fi
    if fora_do_contexto "$f"; then
        { head -c 4000 "$f"; printf '\n\n[... truncado para o Project, arquivo completo (%s bytes) em %s no servidor]\n' "$tam" "$f"; } > "$d"
        echo "truncado, completo no servidor"; return
    fi
    cp "$f" "$d"
}

montar_contexto() {
    local NOVA="$CTX.novo"; rm -rf "$NOVA"; mkdir -p "$NOVA"
    local E="$NOVA/00_ESTADO.md" V; V=$(versao)
    {
        echo "# Estado do somgoes"
        echo
        echo "Gerado em $(date '+%F %T') na maquina $(hostname), pasta $RAIZ, pelo versionar.sh."
        echo "Versao $V. Ultimo commit $(git log -1 --format='%h de %ci, %s')."
        echo "Ponto de partida de qualquer conversa nova. Ler tambem ori__VERSIONAMENTO.md (como trabalhamos),"
        echo "ori__HISTORICO.md (o que foi feito, inclusive erros) e esc__transicao_somgoes_capitulo.md."
        echo "Prefixos: sem prefixo = codigo versionado no git (subpastas viram __); ori__ = orientacoes do"
        echo "trabalho; esc__ = orientacoes de escrita do artigo; ref__ = referencias; res__ = resultados."
        echo "ori, esc, ref e res ficam fora do git (subpastas ignoradas pelo .gitignore)."
        echo
        echo "## Ultimos 20 commits"
        echo '```'; git log -20 --format='%h %cd %s' --date=format:'%Y-%m-%d %H:%M'; echo '```'
        echo
        echo "## Tags"
        echo '```'; git tag -n1 --sort=-creatordate | head -10; echo '```'
        echo
        echo "## Arquivos soltos (nem versionados nem ignorados)"
        local ns; ns=$(git ls-files --others --exclude-standard | wc -l)
        echo "$ns arquivos, ate 40 listados"
        echo '```'; git ls-files --others --exclude-standard | head -40; echo '```'
        echo
        echo "## Remoto"
        echo '```'; git remote -v 2>/dev/null | head -2; echo '```'
        echo
        echo "## Pasta teste (fora do git)"
        echo '```'; [ -d teste ] && find teste -maxdepth 2 -mindepth 1 | sort | head -40; echo '```'
        echo
        echo "## Arquivos versionados"
        echo
        echo "| Arquivo | Tamanho | sha256 |"
        echo "|---|---|---|"
        git ls-files | while IFS= read -r f; do
            [ -f "$f" ] && printf '| %s | %s | %s |\n' "$f" "$(du -h "$f" | cut -f1)" "$(sha256sum "$f" | cut -c1-10)"
        done
    } > "$E"
    local n=0 pulados=0 nome obs
    { echo "| Arquivo | Origem |"; echo "|---|---|"; } > "$NOVA/.indice"
    while IFS= read -r f; do
        [ -f "$f" ] || continue
        if [ ! -s "$f" ] || [ "$(stat -c %s "$f")" -gt 1500000 ] || ! grep -Iq . "$f"; then pulados=$((pulados+1)); continue; fi
        nome=$(echo "$f" | sed 's#/#__#g'); case "$nome" in .*) nome="_${nome#.}";; esac
        obs=$(copiar_ctx "$f" "$NOVA/$nome") && echo "| $nome | $f${obs:+ ($obs)} |" >> "$NOVA/.indice"; n=$((n+1))
    done < <(git ls-files)
    local nfora=0
    for par in "orientacoes:ori" "escrita:esc" "referencias:ref" "resultados:res"; do
        local pasta="$PAI/${par%%:*}" pre="${par##*:}"
        [ -d "$pasta" ] || continue
        while IFS= read -r f; do
            if [ ! -s "$f" ] || [ "$(stat -c %s "$f")" -gt 1500000 ] || ! grep -Iq . "$f"; then pulados=$((pulados+1)); continue; fi
            nome="${pre}__$(echo "${f#$pasta/}" | sed 's#/#__#g')"
            obs=$(copiar_ctx "$f" "$NOVA/$nome") && echo "| $nome | ${par%%:*}/${f#$pasta/} (fora do git)${obs:+ ($obs)} |" >> "$NOVA/.indice"
            nfora=$((nfora+1))
        done < <(find "$pasta" -type f -not -name '.*' | sort)
    done
    if [ -f .env ]; then
        grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env | sed -E 's/=.*/=/' > "$NOVA/env_chaves.txt"
        echo "| env_chaves.txt | .env (so os nomes das variaveis) |" >> "$NOVA/.indice"
    fi
    { echo "# Indice dos arquivos para o Project"; echo
      echo "Gerado em $(date '+%F %T'), versao $V. Comece por 00_ESTADO.md."; echo
      cat "$NOVA/.indice"; } > "$NOVA/00_indice_arquivos.md"
    rm -f "$NOVA/.indice"
    rm -rf "$CTX"; mv "$NOVA" "$CTX"
    log "contexto pronto em $CTX ($n arquivos do git, $nfora de orientacoes, escrita, referencias e resultados, $pulados binarios ou grandes deixados de fora)"
    log "apague os arquivos do Project e suba todo o conteudo de $CTX"
}

# ------------------------------------------------------------------ outros
cmd_marcar() {
    cab "MARCAR"
    precisa_repo
    local T="${1:-}"; [ -n "$T" ] || falha "informe a versao, ex.: ./versionar.sh marcar v1.1 \"coletor geral estavel\""
    shift
    git tag -a "$T" -m "${*:-$T}" || falha "nao consegui criar a tag $T"
    hist "tag $T" <<< "Versao estavel marcada em $(git rev-parse --short HEAD). ${*:-}"
    commit_hist "history: tag $T"
    log "tag $T"
    enviar
}

cmd_desfazer() {
    cab "DESFAZER"
    precisa_repo
    [ -z "$(git status --porcelain --untracked-files=no)" ] || falha "ha alteracoes sem commit; veja ./versionar.sh status"
    local alvo; alvo=$(git log -1 --format='%h %s' -E --invert-grep --grep='^(history:|Session close|Revert)')
    local h; h=$(echo "$alvo" | cut -d' ' -f1)
    [ -n "$h" ] || falha "nao achei commit para desfazer"
    if ! git revert --no-commit "$h" >> "$LOG" 2>&1; then
        if git diff --name-only --diff-filter=U | grep -q .; then
            git revert --abort 2>/dev/null; git reset -q --hard HEAD
            falha "o commit $h nao pode ser revertido automaticamente (conflito com mudancas posteriores)"
        fi
    fi
    hist "desfeito" <<< "Revertido o commit $alvo."
    git add -u
    git commit -q -m "Revert $alvo" || falha "commit da reversao falhou"
    log "revertido $alvo, versao $(versao)"
    enviar
}

cmd_status() {
    precisa_repo
    echo "versao $(versao)"
    git status --short
    git diff --stat
}

case "${1:-}" in
    iniciar)  cmd_iniciar ;;
    aplicar)  shift; cmd_aplicar "$@" ;;
    fechar|estado) shift; cmd_fechar "$@" ;;
    marcar)   shift; cmd_marcar "$@" ;;
    contexto) precisa_repo; cab "CONTEXTO"; montar_contexto ;;
    remoto)   shift; cmd_remoto "$@" ;;
    desfazer) cmd_desfazer ;;
    status)   cmd_status ;;
    *) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
