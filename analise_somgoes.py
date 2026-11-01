#!/usr/bin/env python
"""
analise_somgoes.py — figuras e tabelas de todo o estudo, cada figura com o seu CSV.

Entradas: resultados_somgoes/metricas_consolidadas.csv (validação cruzada por
temporada) e resultados_somgoes/estados/<cenário>/ (estados_somgoes.py).
Saídas:   analise_somgoes/figuras/Fxx_*.png
          analise_somgoes/dados_figuras/Fxx_*.csv   (os dados exatos de cada figura)
          analise_somgoes/tabela_resumo.csv
          pacote_analise.zip  (todos os CSVs e logs, sem as figuras)
Uso:      python analise_somgoes.py
"""
from __future__ import annotations

import json
import logging
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config_somgoes as cfg

OUT = cfg.ROOT / "analise_somgoes"
FIG, DAD = OUT / "figuras", OUT / "dados_figuras"
EST = cfg.RESULTS_DIR / "estados"
log = logging.getLogger("analise")
CH_NOME = {"nenhuma": "sem chuva", "colar": "chuva no colar", "colar_nucleo": "colar e núcleo"}
PE_NOME = {"mi": "informação mútua", "duas_etapas": "duas etapas", "duas_etapas_markov2": "duas etapas + transição"}
COND = {"cond_seco_chuva": "seco -> vai chover", "cond_seco_forte": "seco -> chuva forte",
        "cond_fraca_forte": "fraca -> intensifica", "cond_fraca_seco": "fraca -> para",
        "cond_forte_persiste": "forte -> persiste"}
COR = {"nenhuma": "C0", "colar": "C1", "colar_nucleo": "C3"}
TRANS_NOME = {"SS": "seco > seco", "inicio": "seco > fraca/moderada", "inicio_forte": "seco > forte",
              "intensifica": "fraca/moderada > forte", "CC": "fraca/moderada > fraca/moderada",
              "termina": "chuva > seco", "forte_persiste": "forte > forte", "enfraquece": "forte > fraca/moderada"}
_EM = {"temperature": "temperatura média das EMs (°C)", "umidity": "umidade relativa média das EMs (%)",
       "pression": "pressão média das EMs (hPa)", "wind_speed": "vento médio das EMs (m/s)",
       "radiation": "radiação solar média das EMs (W/m²)", "vento_sin": "direção do vento, componente leste-oeste",
       "vento_cos": "direção do vento, componente norte-sul"}
_Q = {"tot": "todo o anel", "N": "quadrante norte", "L": "quadrante leste", "S": "quadrante sul", "O": "quadrante oeste"}
_IN = {"max60": "chuva máxima em 60 min (mm)", "frac60": "fração das estações com chuva em 60 min",
       "max10": "chuva máxima nos últimos 10 min (mm)", "max30": "chuva máxima em 30 min (mm)",
       "media60": "chuva média em 60 min (mm)", "fracforte60": "fração das estações com 10 mm ou mais em 60 min"}


def nome_var(v):
    """nome legível de cada variável usada nas tabelas e figuras"""
    if v.startswith("em_"):
        return _EM.get(v[3:], v)
    if v.startswith("g") and "_" in v and v[1].isdigit():
        c, est = int(v[1]), v.split("_", 1)[1]
        canal = cfg.GOES_NOMES[c]
        un = "mm/h" if c == 7 else "radiância"
        return {"media": f"{canal} média no campo ({un})", "nucleo_min": f"{canal} mínimo sobre o núcleo ({un})",
                "nucleo_max": f"{canal} máximo sobre o núcleo ({un})", "p10": f"{canal} percentil 10 no campo ({un})",
                "desvio": f"{canal} desvio no campo", "min": f"{canal} mínimo no campo", "max": f"{canal} máximo no campo"}.get(est, v)
    if v.startswith("in_nuc_"):
        return "núcleo, " + _IN.get(v[7:], v)
    if v.startswith(("in_c1_", "in_c2_")):
        anel = "colar 13-60 km" if v[3:5] == "c1" else "colar 60-100 km"
        q, est = v[6:].split("_", 1)
        return f"{anel}, {_Q.get(q, q)}, {_IN.get(est, est)}"
    if v.startswith("colar"):
        return ("colar 13-60 km" if v.startswith("colar1") else "colar 60-100 km") + ", chuva máxima em 60 min (mm)"
    return {"hora_sin": "hora do dia (seno)", "hora_cos": "hora do dia (cosseno)", "agora": "chuva máx. 60 min no núcleo agora",
            "prox": "chuva máx. 60 min no núcleo depois"}.get(v, v)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def salvar(nome, fig, df):
    fig.tight_layout()
    fig.savefig(FIG / f"{nome}.png", bbox_inches="tight"); plt.close(fig)
    df.to_csv(DAD / f"{nome}.csv", sep=";", index=False)
    log.info(f"  {nome}")


def K_de(grade):
    a, b = str(grade).split("x"); return int(a) * int(b)


def carregar():
    M = pd.read_csv(cfg.RESULTS_DIR / "metricas_consolidadas.csv", sep=";", low_memory=False)
    M = M[M.fold >= 0].copy()
    M["chuva"] = M.get("chuva", pd.Series("nenhuma", index=M.index)).fillna("nenhuma")
    M["horizonte"] = M.get("horizonte", pd.Series(60, index=M.index)).fillna(60).astype(int)
    nf = M.groupby("exp_id").fold.transform("nunique")
    M = M[nf == M.fold.nunique()].copy()          # só modelos avaliados em todas as temporadas
    M["K"] = M.grade.map(K_de)
    M["modelo"] = np.where(M.familia.str.startswith("kmeans"), "k-means", "SOM " + M.perda.map(PE_NOME).fillna(M.perda))
    return M


def media_ep(d, col):
    g = d.groupby(["horizonte", "chuva", "modelo", "K"])[col]
    return pd.DataFrame({"media": g.mean(), "ep": g.std() / np.sqrt(g.size()), "n_temporadas": g.size()}).reset_index()


# ═══════════════════════════════ desempenho (validação cruzada) ══════════════
def f01_cenarios(M):
    d = M[(M.familia == "cenarios") & (M.K == 100)]
    T = media_ep(d, "forte_prox.ap_skill")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for a, h in zip(ax, (60, 30)):
        t = T[T.horizonte == h]
        mods = sorted(t.modelo.unique())
        x = np.arange(len(cfg.CHUVA_ENTRADA)); w = 0.8 / max(1, len(mods))
        for i, m in enumerate(mods):
            tt = t[t.modelo == m].set_index("chuva").reindex(cfg.CHUVA_ENTRADA)
            a.bar(x + i * w, tt.media, w, yerr=tt.ep, label=m, capsize=2)
        a.set_xticks(x + w * (len(mods) - 1) / 2, [CH_NOME[c] for c in cfg.CHUVA_ENTRADA])
        a.set_title(f"chuva forte nos próximos {h} min"); a.set_ylabel("skill (0 = acaso, 1 = perfeito)")
    ax[0].legend(fontsize=7)
    salvar("F01_skill_por_cenario_e_perda", fig, T)


def f02_lift_recall(M):
    sel = M[((M.familia == "cenarios") & (M.perda == cfg.ESTADOS_PERDA) & (M.K == 100)) |
            ((M.familia == "kmeans_cenarios") & (M.K == 100))]
    L = []
    for (h, ch, mod), d in sel.groupby(["horizonte", "chuva", "modelo"]):
        pts = [(0.0, 0.0)]
        for f in cfg.FRACS_TEMPO:
            pts.append((f, d[f"forte_prox.cob_{int(f*100)}pct"].mean()))
        for r in (50, 80, 90):
            pts.append((d[f"forte_prox.tempo_para_{r}pct"].mean(), r / 100))
        pts.append((1.0, 1.0))
        for f, rc in sorted(pts):
            L.append(dict(horizonte=h, chuva=ch, modelo=mod, frac_tempo=round(f, 4), recall=round(rc, 4),
                          lift=round(rc / f, 3) if f > 0 else np.nan))
    D = pd.DataFrame(L)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for j, h in enumerate((60, 30)):
        for ch in cfg.CHUVA_ENTRADA:
            for mod, ls in (("k-means", "--"), (f"SOM {PE_NOME[cfg.ESTADOS_PERDA]}", "-")):
                t = D[(D.horizonte == h) & (D.chuva == ch) & (D.modelo == mod)]
                if t.empty:
                    continue
                lab = f"{CH_NOME[ch]} | {mod}"
                t = t[t.frac_tempo > 0]
                ax[0, j].plot(t.frac_tempo, t.recall, ls, color=COR[ch], marker="o", ms=3, label=lab)
                ax[1, j].plot(t.frac_tempo, t.lift, ls, color=COR[ch], marker="o", ms=3, label=lab)
        ax[0, j].plot([0, 1], [0, 1], "k:", lw=0.8)
        ax[0, j].set(title=f"recall de chuva forte ({h} min)", xlabel="fração do tempo sob monitoramento",
                     ylabel="fração dos eventos cobertos", xscale="log", xlim=(0.009, 1))
        ax[1, j].set(title=f"lift ({h} min)", xlabel="fração do tempo sob monitoramento",
                     ylabel="lift (vezes a taxa base)", xscale="log", xlim=(0.009, 1))
    from matplotlib.lines import Line2D
    hand = [Line2D([], [], color=COR[c], lw=2, label=CH_NOME[c]) for c in cfg.CHUVA_ENTRADA] + \
           [Line2D([], [], color="k", ls="-", label="SOM (linha cheia)"), Line2D([], [], color="k", ls="--", label="k-means (tracejado)")]
    ax[0, 0].legend(handles=hand, fontsize=7, loc="lower right")
    salvar("F02_curvas_recall_e_lift", fig, D)


def f03_grade(M):
    d = M[(M.familia.isin(["grades", "kmeans_cenarios"]))]
    T = media_ep(d, "forte_prox.ap_skill")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for a, h in zip(ax, (60, 30)):
        for ch in cfg.CHUVA_ENTRADA:
            for mod, ls in (("k-means", "--"),):
                t = T[(T.horizonte == h) & (T.chuva == ch) & (T.modelo == mod)].sort_values("K")
                a.errorbar(t.K, t.media, t.ep, fmt=ls, color=COR[ch], marker="s", ms=3, label=f"{CH_NOME[ch]} | k-means")
            t = T[(T.horizonte == h) & (T.chuva == ch) & (T.modelo != "k-means")].sort_values("K")
            a.errorbar(t.K, t.media, t.ep, fmt="-", color=COR[ch], marker="o", ms=3, label=f"{CH_NOME[ch]} | SOM")
        a.set(xscale="log", xlabel="número de células (2x2 a 25x25)", title=f"chuva forte em {h} min",
              ylabel="skill")
    ax[0].legend(fontsize=6)
    salvar("F03_skill_por_tamanho_de_grade", fig, T)


def f04_nos_fortes(M):
    d = M[M.familia.isin(["grades", "kmeans_cenarios"])]
    T = d.groupby(["horizonte", "chuva", "modelo", "K"])[["no_forte_top3.recall", "no_forte_top3.precisao"]]\
        .mean().round(4).reset_index()
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for j, h in enumerate((60, 30)):
        for ch in cfg.CHUVA_ENTRADA:
            for mod, ls in (("k-means", "--"),):
                pass
            for kind, ls in (("SOM", "-"), ("k-means", "--")):
                t = T[(T.horizonte == h) & (T.chuva == ch) & ((T.modelo == "k-means") == (kind == "k-means"))].sort_values("K")
                ax[0, j].plot(t.K, t["no_forte_top3.recall"], ls, color=COR[ch], marker="o", ms=3, label=f"{CH_NOME[ch]} | {kind}")
                ax[1, j].plot(t.K, t["no_forte_top3.precisao"], ls, color=COR[ch], marker="o", ms=3)
        ax[0, j].set(xscale="log", title=f"as 3 células de maior taxa de chuva forte, {h} min",
                     ylabel="fração dos eventos que caem nelas")
        ax[1, j].set(xscale="log", xlabel="tamanho do mapa (nº total de células, de 2x2 a 25x25)",
                     ylabel="fração dos horários nelas com chuva forte")
    ax[0, 0].legend(fontsize=6)
    salvar("F04_nos_concentradores_por_grade", fig, T)


def f05_condicionais(M):
    sel = M[((M.familia == "cenarios") & (M.K == 100)) | ((M.familia == "kmeans_cenarios") & (M.K == 100))]
    cols = [f"{c}.ap_skill" for c in COND]
    T = sel.groupby(["horizonte", "chuva", "modelo"])[cols].mean().round(3).reset_index()
    T.columns = [COND.get(c.split(".")[0], c) for c in T.columns]
    fig, ax = plt.subplots(2, 1, figsize=(9, 9))
    for a, h in zip(ax, (60, 30)):
        t = T[T.horizonte == h].copy()
        t.index = [f"{CH_NOME[c]} | {m}" for c, m in zip(t.chuva, t.modelo)]
        v = t[list(COND.values())]
        im = a.imshow(v.to_numpy(float), cmap="viridis", vmin=0, vmax=0.7, aspect="auto")
        a.set_xticks(range(v.shape[1]), v.columns, rotation=35, ha="right"); a.set_yticks(range(len(v)), v.index)
        for i in range(v.shape[0]):
            for k in range(v.shape[1]):
                a.text(k, i, f"{v.iat[i, k]:.2f}", ha="center", va="center", color="w", fontsize=7)
        a.set_title(f"skill condicional ao estado atual, {h} min"); a.grid(False)
        a.tick_params(axis="y", labelsize=7)
    fig.colorbar(im, ax=ax, shrink=0.5, location="right", pad=0.02)
    salvar("F05_skill_condicional", fig, T)


def f06_temporadas(M):
    sel = M[((M.familia == "cenarios") & (M.perda == cfg.ESTADOS_PERDA) & (M.K == 100)) |
            ((M.familia == "kmeans_cenarios") & (M.K == 100))]
    temps = json.load(open(cfg.CACHE_DIR / "meta_cache.json"))["temporadas"]
    T = sel[["horizonte", "chuva", "modelo", "fold", "forte_prox.ap_skill", "forte_prox.base"]].copy()
    T["temporada"] = T.fold.map(lambda f: temps[int(f)] if int(f) < len(temps) else f"fold{int(f)}")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for a, h in zip(ax, (60, 30)):
        for (ch, mod), t in T[T.horizonte == h].groupby(["chuva", "modelo"]):
            t = t.sort_values("temporada")
            a.plot(t.temporada, t["forte_prox.ap_skill"], "--" if mod == "k-means" else "-", color=COR[ch], marker="o", ms=3,
                   label=f"{CH_NOME[ch]} | {mod}")
        a.set(title=f"skill por temporada, {h} min", ylabel="skill"); a.tick_params(axis="x", rotation=30)
    ax[0].legend(fontsize=6)
    salvar("F06_skill_por_temporada", fig, T.sort_values(["horizonte", "chuva", "modelo", "temporada"]))


def tabela_resumo(M):
    sel = M[((M.familia == "cenarios") & (M.K == 100)) | ((M.familia == "kmeans_cenarios") & (M.K == 100))]
    cols = {"forte_prox.base": "taxa_base_forte", "forte_prox.ap_skill": "skill_forte",
            "forte_prox.cob_5pct": "recall_5pct_tempo", "forte_prox.lift_5pct": "lift_5pct_tempo",
            "forte_prox.cob_10pct": "recall_10pct_tempo", "forte_prox.lift_10pct": "lift_10pct_tempo",
            "forte_prox.tempo_para_80pct": "tempo_para_recall_80pct",
            "no_forte_top3.recall": "recall_3_celulas", "no_forte_top3.precisao": "precisao_3_celulas",
            **{f"{c}.ap_skill": f"skill_{n.replace(' -> ', '_').replace(' ', '_')}" for c, n in COND.items()},
            "celulas_efetivas": "celulas_efetivas"}
    T = sel.groupby(["horizonte", "chuva", "modelo"])[list(cols)].mean().round(4).rename(columns=cols).reset_index()
    T.to_csv(OUT / "tabela_resumo.csv", sep=";", index=False)
    log.info("tabela_resumo.csv")


# ═══════════════════════════════ estados ═════════════════════════════════════
def borda_regioes(a, reg, H, W):
    for i in range(H):
        for j in range(W):
            if j + 1 < W and reg[i, j] != reg[i, j + 1]:
                a.plot([j + .5, j + .5], [i - .5, i + .5], "k", lw=1.5)
            if i + 1 < H and reg[i, j] != reg[i + 1, j]:
                a.plot([j - .5, j + .5], [i + .5, i + .5], "k", lw=1.5)


def f_estados(cen):
    d = EST / cen
    if not (d / "celulas.csv").exists():
        return
    C = pd.read_csv(d / "celulas.csv", sep=";")
    R = pd.read_csv(d / "regioes.csv", sep=";")
    H, W = cfg.ESTADOS_GRADE
    reg = np.full((H, W), -1); tx = np.full((H, W), np.nan); dom = np.full((H, W), -1)
    for r in C.itertuples():
        reg[r.linha, r.coluna], tx[r.linha, r.coluna] = r.regiao, r.taxa_forte_prox
        dom[r.linha, r.coluna] = cfg.TRANSICOES.index(r.transicao_dominante)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    im = ax[0].imshow(tx, cmap="magma_r", vmin=0); fig.colorbar(im, ax=ax[0], shrink=0.8, label="taxa de chuva forte depois")
    im2 = ax[1].imshow(np.ma.masked_less(dom, 0), cmap="tab10", vmin=0, vmax=9)
    for a in ax:
        borda_regioes(a, reg, H, W); a.grid(False)
        for r in np.unique(reg[reg > 0]):
            ii, jj = np.where(reg == r)
            a.text(jj.mean(), ii.mean(), f"R{r}", ha="center", va="center", fontsize=9, weight="bold",
                   color="w", bbox=dict(boxstyle="round", fc="k", alpha=0.5))
    ax[0].set_title("taxa de chuva forte na janela seguinte por célula")
    hand = [plt.Rectangle((0, 0), 1, 1, color=plt.cm.tab10(k / 9)) for k in range(len(cfg.TRANSICOES))]
    ax[1].legend(hand, [TRANS_NOME[t] for t in cfg.TRANSICOES], fontsize=6, loc="upper left", bbox_to_anchor=(1, 1),
                 title="agora > depois", title_fontsize=6)
    ax[1].set_title("transição dominante por célula")
    salvar(f"F07_mapa_grade_{cen}", fig, C[["celula", "linha", "coluna", "regiao", "n", "taxa_forte_prox",
                                            "taxa_chuva_prox", "transicao_dominante"]])

    for nome in ("horizonte", "10min"):
        T = pd.read_csv(d / f"transicao_{nome}.csv", sep=";", index_col=0)
        fig, a = plt.subplots(figsize=(5.5, 4.6))
        im = a.imshow(T.to_numpy(float), cmap="Blues", vmin=0, vmax=1)
        a.set_xticks(range(T.shape[1]), [f"R{c}" for c in T.columns]); a.set_yticks(range(T.shape[0]), [f"R{c}" for c in T.index])
        for i in range(T.shape[0]):
            for k in range(T.shape[1]):
                v = T.iat[i, k]
                if v >= 0.02:
                    a.text(k, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="w" if v > 0.5 else "k")
        a.set(xlabel="região depois", ylabel="região agora", title=f"probabilidade de transição ({nome})"); a.grid(False)
        fig.colorbar(im, ax=a, shrink=0.8)
        salvar(f"F08_transicao_{nome}_{cen}", fig, T.reset_index())

    P = pd.read_csv(d / "perfil_regioes.csv", sep=";", index_col=0)
    sel = [c for c in P.columns if not c.endswith("_media") or c.startswith("em_") or c.startswith("in_")]
    sel = [c for c in sel if not (c.startswith(("in_c1_", "in_c2_")) and not c.split("_")[2] in ("tot", "S", "O"))]
    Pt = P[sel].T
    fig, a = plt.subplots(figsize=(7, 0.22 * len(Pt) + 1.5))
    im = a.imshow(Pt.to_numpy(float), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    a.set_xticks(range(Pt.shape[1]), [f"R{r}" for r in Pt.columns]); a.set_yticks(range(len(Pt)), [nome_var(v) for v in Pt.index], fontsize=6)
    a.set_title("perfil dos estados: quanto cada variável fica acima (vermelho) ou abaixo (azul)\nda média geral, em desvios padrão", fontsize=8)
    a.grid(False); fig.colorbar(im, ax=a, shrink=0.6)
    salvar(f"F09_perfil_regioes_{cen}", fig, Pt.reset_index().rename(columns={"index": "variavel"})
           .assign(nome=lambda x: x.variavel.map(nome_var)))
    # descrição de cada estado em texto
    lin = []
    for r in R.itertuples():
        z = P.loc[r.regiao].drop(labels=[c for c in P.columns if c.startswith("hora")], errors="ignore")
        top = z.reindex(z.abs().sort_values(ascending=False).index).head(5)
        lin.append(dict(estado=f"R{int(r.regiao)}", nome=r.nome, frac_tempo=r.frac_tempo, taxa_forte_depois=r.taxa_forte_prox,
                        taxa_chuva_depois=r.taxa_chuva_prox, permanencia_media_min=r.permanencia_media_min,
                        hora_media=r.hora_media, transicao_mais_comum=TRANS_NOME.get(r.transicao_dominante),
                        agora_seco=r.frac_agora_seco, agora_fraca_moderada=r.frac_agora_chuva_fraca,
                        agora_forte=r.frac_agora_chuva_forte,
                        caracteristicas="; ".join(f"{nome_var(k)} {'acima' if v > 0 else 'abaixo'} ({v:+.1f} dp)"
                                                  for k, v in top.items())))
    pd.DataFrame(lin).to_csv(DAD / f"T_descricao_estados_{cen}.csv", sep=";", index=False)

    PM = pd.read_csv(d / "perfil_mudanca.csv", sep=";")
    if len(PM):
        alvo = int(R.loc[R.taxa_forte_prox.idxmax(), "regiao"])
        sub = PM[PM.destino == alvo]
        origens = sub.groupby("regiao").n_mudam.first().sort_values(ascending=False).index[:3]
        fig, axs = plt.subplots(1, max(1, len(origens)), figsize=(4.2 * max(1, len(origens)), 4), squeeze=False)
        rows = []
        for a, o in zip(axs[0], origens):
            t = sub[sub.regiao == o].reindex(sub[sub.regiao == o].dif_padronizada.abs().sort_values(ascending=False).index).head(12)
            a.barh([nome_var(v) for v in t.variavel[::-1]], t.dif_padronizada[::-1],
                   color=np.where(t.dif_padronizada[::-1] > 0, "C3", "C0"))
            a.set_title(f"R{o} -> R{alvo} (n={t.n_mudam.iloc[0]}) contra permanecer", fontsize=8)
            a.set_xlabel("quem muda menos quem permanece (desvios padrão)"); a.tick_params(axis="y", labelsize=6)
            rows.append(t.assign(nome=t.variavel.map(nome_var)))
        salvar(f"F10_o_que_leva_a_regiao_forte_{cen}", fig, pd.concat(rows) if rows else sub.head(0))

    G = pd.read_csv(d / "goes_composto.csv", sep=";")
    canais = [c for c in (7, 0, 6) if c in G.canal.unique()]
    regs = sorted(G.regiao.unique())
    fig, axs = plt.subplots(len(canais), len(regs), figsize=(1.6 * len(regs), 1.7 * len(canais)), squeeze=False)
    for i, c in enumerate(canais):
        gc = G[G.canal == c]; vmin, vmax = np.percentile(gc.valor, [2, 98])
        for j, r in enumerate(regs):
            g = gc[gc.regiao == r]
            img = np.full((g.i.max() + 1, g.j.max() + 1), np.nan); img[g.i, g.j] = g.valor
            axs[i, j].imshow(img, cmap="viridis" if c == 7 else "gray_r", vmin=vmin, vmax=vmax)
            axs[i, j].set_xticks([]); axs[i, j].set_yticks([]); axs[i, j].grid(False)
            n = img.shape[0]
            axs[i, j].add_patch(plt.Circle((n / 2 - .5, n / 2 - .5), 13 / (cfg.KM_PER_PX * cfg.IMG_SIZE / cfg.GOES_RES),
                                           fill=False, color="r", lw=0.8))
            if i == 0:
                axs[i, j].set_title(f"R{r}", fontsize=8)
        axs[i, 0].set_ylabel(cfg.GOES_NOMES[c] + ("" if c == 7 else " (radiância)"), fontsize=8)
    fig.suptitle("campo GOES médio por região (círculo = núcleo)", fontsize=9)
    salvar(f"F11_goes_medio_por_regiao_{cen}", fig, G[G.canal.isin(canais)])

    CD = pd.read_csv(d / "ciclo_diurno.csv", sep=";", index_col=0)
    fig, a = plt.subplots(figsize=(8, 3))
    im = a.imshow(CD.to_numpy(float), cmap="YlOrRd", aspect="auto")
    a.set_yticks(range(len(CD)), [f"R{r}" for r in CD.index]); a.set_xticks(range(CD.shape[1]), CD.columns)
    a.set(xlabel="hora (BRT)", title="frequência de cada região por hora do dia"); a.grid(False)
    fig.colorbar(im, ax=a, shrink=0.8)
    salvar(f"F12_ciclo_diurno_{cen}", fig, CD.reset_index())


def f18_maquina_estados(cen):
    d = EST / cen
    if not (d / "regioes.csv").exists():
        return
    R = pd.read_csv(d / "regioes.csv", sep=";").set_index("regiao")
    C = pd.read_csv(d / "celulas.csv", sep=";")
    T = pd.read_csv(d / "transicao_horizonte.csv", sep=";", index_col=0); T.columns = T.columns.astype(int)
    PM = pd.read_csv(d / "perfil_mudanca.csv", sep=";") if (d / "perfil_mudanca.csv").exists() else pd.DataFrame()
    # posição com significado: x = fração do tempo em que já chove no estado, y = risco depois
    pos = pd.DataFrame({"x": R.frac_agora_chuva_fraca + R.frac_agora_chuva_forte, "y": R.taxa_forte_prox})
    pos = pos.sort_values("y")
    for k in range(1, len(pos)):                       # afasta estados sobrepostos
        i, j = pos.index[k - 1], pos.index[k]
        if abs(pos.at[j, "y"] - pos.at[i, "y"]) < 0.06 and abs(pos.at[j, "x"] - pos.at[i, "x"]) < 0.15:
            pos.at[j, "x"] += 0.16
    h = int(cen.split("_h")[-1])
    fig, a = plt.subplots(figsize=(11, 8.5))
    arestas = []
    for de in T.index:
        saidas = T.loc[de].drop(de, errors="ignore").sort_values(ascending=False).head(2)
        for para in T.columns:
            p = T.at[de, para]
            if de == para or p < 0.08 or para not in saidas.index or de not in pos.index or para not in pos.index:
                continue
            fat = ""
            if len(PM) and R.at[para, "taxa_forte_prox"] > R.at[de, "taxa_forte_prox"]:
                t = PM[(PM.regiao == de) & (PM.destino == para)]
                t = t[~t.variavel.str.startswith("hora")]
                t = t.reindex(t.dif_padronizada.abs().sort_values(ascending=False).index).head(2)
                fat = "\n".join(f"{'+' if x > 0 else '-'} {nome_var(v)[:38]}" for v, x in zip(t.variavel, t.dif_padronizada))
            arestas.append(dict(de=f"R{de}", para=f"R{para}", prob=round(p, 3), fatores=fat.replace("\n", " | ")))
            x0, y0 = pos.loc[de, ["x", "y"]]; x1, y1 = pos.loc[para, ["x", "y"]]
            a.annotate("", xy=(x1, y1), xytext=(x0, y0),
                       arrowprops=dict(arrowstyle="-|>", lw=0.8 + 6 * p, color="C3" if R.at[para, "taxa_forte_prox"] > 0.2 else "C7",
                                       alpha=0.7, shrinkA=28, shrinkB=28, connectionstyle="arc3,rad=0.18"))
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            a.text(xm, ym, f"{p:.0%}" + (f"\n{fat}" if fat else ""), fontsize=5.5, ha="center", va="center",
                   bbox=dict(boxstyle="round", fc="w", ec="0.8", alpha=0.85))
    for r, (x, y) in pos[["x", "y"]].iterrows():
        tx = R.at[r, "taxa_forte_prox"]
        a.scatter(x, y, s=900 + 3000 * R.at[r, "frac_tempo"] ** 0.5, c=[plt.cm.magma_r(min(1, tx / 0.8))],
                  edgecolors="k", zorder=3)
        fica = T.at[r, r] if r in T.index and r in T.columns else np.nan
        a.text(x, y, f"R{r}\n{100*tx:.0f}% forte\nfica {100*fica:.0f}%", ha="center", va="center", fontsize=7,
               zorder=4, color="w" if tx > 0.3 else "k", weight="bold")
    a.set(xlim=(-0.15, 1.35), ylim=(-0.08, 1.05), xlabel="fração do tempo em que já chove no núcleo nesse estado",
          ylabel=f"probabilidade de chuva forte nos próximos {h} min")
    a.set_title(f"máquina de estados ({cen.replace('_h', ', ')} min): setas = probabilidade de estar no outro estado "
                f"{h} min depois (as 2 saídas mais prováveis de cada estado),\nnas setas para risco maior, as 2 variáveis que mais distinguem quem faz a transição "
                f"(+ acima, - abaixo de quem permanece); cor = taxa de chuva forte depois", fontsize=8)
    salvar(f"F18_maquina_de_estados_{cen}", fig, pd.DataFrame(arestas))


def f19_pareto(M):
    f = 0.05
    sel = M.copy()
    T = sel.groupby(["familia", "exp_id", "modelo", "chuva", "horizonte", "grade"])[
        [f"forte_prox.cob_{int(f*100)}pct"]].mean().reset_index().rename(columns={f"forte_prox.cob_{int(f*100)}pct": "recall"})
    T["lift"] = T.recall / f
    for h in (60, 30):
        t = T[T.horizonte == h].sort_values("recall", ascending=False).head(30).reset_index(drop=True)
        fig, a = plt.subplots(figsize=(11, 4.5))
        a.bar(range(len(t)), t.recall, color=[COR.get(c, "C7") for c in t.chuva],
              edgecolor=["k" if m == "k-means" else "none" for m in t.modelo], hatch=None)
        a.set_xticks(range(len(t)), [f"{m[:22]} {g}" for m, g in zip(t.modelo, t.grade)], rotation=75, fontsize=6)
        a.set(ylabel=f"recall com {int(f*100)}% do tempo monitorado",
              title=f"os 30 modelos de maior recall a {int(f*100)}% do tempo ({h} min); cor = entrada de chuva")
        b = a.twinx(); b.set_ylim(np.array(a.get_ylim()) / f); b.set_ylabel("lift (= recall / fração do tempo)"); b.grid(False)
        salvar(f"F19_pareto_recall_lift_h{h}", fig, t)


def f13_mapa_estacoes():
    if not cfg.META_FILE.exists():
        return
    m = pd.read_csv(cfg.META_FILE, sep=";")
    m["Codigo"] = m.Codigo.astype(str)
    excl = cfg.CACHE_DIR / "estacoes_excluidas.csv"
    ex = set(pd.read_csv(excl, sep=";").query("excluida").codigo.astype(str)) if excl.exists() else set()
    R0 = json.load(open(cfg.CACHE_DIR / "meta_cache.json"))["raio_nucleo_km"]
    kmlat, kmlon = 111.0, 111.0 * np.cos(np.radians(cfg.BH_LAT))
    m["x_km"] = (m.Longitude - cfg.BH_LON) * kmlon; m["y_km"] = (m.Latitude - cfg.BH_LAT) * kmlat
    m["dist_km"] = np.hypot(m.x_km, m.y_km)
    m["papel"] = np.select([m.Codigo.isin(ex), m.Tipo == "meteorologica", m.dist_km <= R0,
                            m.dist_km <= 60, m.dist_km <= 100],
                           ["excluída", "EM (entrada)", "núcleo (alvo)", "colar 1 (13-60 km)", "colar 2 (60-100 km)"],
                           "fora")
    fig, a = plt.subplots(figsize=(6.5, 6.5))
    for p, mk in (("colar 2 (60-100 km)", "."), ("colar 1 (13-60 km)", "."), ("núcleo (alvo)", "o"),
                  ("EM (entrada)", "^"), ("excluída", "x"), ("fora", ".")):
        t = m[m.papel == p]
        a.scatter(t.x_km, t.y_km, s=30 if mk in "o^" else 10, marker=mk, label=f"{p} ({len(t)})")
    for r in (R0, 60, 100):
        a.add_patch(plt.Circle((0, 0), r, fill=False, ls="--", color="k", lw=0.7))
    half = cfg.IMG_SIZE * cfg.KM_PER_PX / 2
    a.add_patch(plt.Rectangle((-half, -half), 2 * half, 2 * half, fill=False, color="C3", lw=1, label="campo GOES"))
    a.set(xlim=(-130, 130), ylim=(-130, 130), aspect="equal", xlabel="km (leste)", ylabel="km (norte)",
          title="estações e regiões do estudo (centro de BH na origem)")
    a.legend(fontsize=7, loc="lower left")
    salvar("F13_mapa_estacoes", fig, m[["Codigo", "Nome", "Fonte", "Tipo", "Frequencia_norm", "Latitude",
                                        "Longitude", "x_km", "y_km", "dist_km", "papel"]].round(3))


def main():
    for p in (FIG, DAD):
        p.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    r = logging.getLogger(); r.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "analise.log", mode="w")):
        h.setFormatter(fmt); r.addHandler(h)
    r.setLevel(logging.INFO)
    M = carregar()
    log.info(f"métricas: {len(M)} linhas, famílias {sorted(M.familia.unique())}")
    pd.DataFrame({"variavel": sorted(set(M.columns) | set())}).head(0)
    for f in (f01_cenarios, f02_lift_recall, f03_grade, f04_nos_fortes, f05_condicionais, f06_temporadas, tabela_resumo):
        try:
            f(M)
        except Exception as e:
            log.info(f"  {f.__name__} falhou: {e}")
    try:
        f19_pareto(M)
    except Exception as e:
        log.info(f"  pareto falhou: {e}")
    for ch, h in cfg.ESTADOS_CENARIOS:
        try:
            f18_maquina_estados(f"{ch}_h{h}")
        except Exception as e:
            log.info(f"  máquina de estados {ch}_h{h} falhou: {e}")
        try:
            f_estados(f"{ch}_h{h}")
        except Exception as e:
            log.info(f"  estados {ch}_h{h} falhou: {e}")
    try:
        f13_mapa_estacoes()
    except Exception as e:
        log.info(f"  mapa de estações falhou: {e}")

    z = cfg.ROOT / "pacote_analise.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in list(OUT.rglob("*.csv")) + list(OUT.rglob("*.log")):
            zf.write(f, f.relative_to(cfg.ROOT))
        for f in EST.rglob("*"):
            if f.suffix in (".csv", ".log") and "modelo" not in f.parts:
                zf.write(f, f.relative_to(cfg.ROOT))
        for f in list(OUT.glob("*.csv.gz")) + list((cfg.ROOT / "motor").glob("*.csv")) + \
                list((cfg.ROOT / "motor").glob("*.json")):
            zf.write(f, f.relative_to(cfg.ROOT))
        for n in ("metricas_consolidadas.csv", "ranking.csv", "escolha_grade.csv"):
            if (cfg.RESULTS_DIR / n).exists():
                zf.write(cfg.RESULTS_DIR / n, (cfg.RESULTS_DIR / n).relative_to(cfg.ROOT))
    log.info(f"pacote: {z} ({z.stat().st_size/1e6:.1f} MB) — figuras ficam em {FIG}")


if __name__ == "__main__":
    main()
