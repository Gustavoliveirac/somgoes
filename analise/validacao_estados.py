#!/usr/bin/env python
"""
validacao_estados.py — estados e níveis de alerta avaliados FORA da temporada de treino.

Usa os modelos já treinados na validação cruzada (cenarios/.../fold*/melhor.pt). Em cada
fold: (1) agrupa as células em regiões só com o mapa daquele fold, (2) numera as regiões
pela taxa de chuva forte nas temporadas de TREINO (R1 = menor risco, RN = maior), o que
torna as regiões comparáveis entre folds, (3) mede tudo na temporada deixada de fora.
Níveis de alerta: cada célula recebe um nível pela sua taxa de chuva forte no TREINO
(cfg.NIVEIS_ALERTA); o desempenho de cada nível é medido na temporada deixada de fora.

Similaridade entre os modelos dos folds (viabilidade operacional): os 6 modelos, treinados
com conjuntos diferentes de temporadas, são aplicados aos MESMOS horários e comparados
em células e regiões (índice de Rand ajustado), nível de alerta (concordância e kappa de
Cohen) e escore de risco (correlação de postos de Spearman). O consenso conta quantos
modelos colocam cada horário em alerta ou acima, com a taxa real de chuva forte.

Saídas em analise_somgoes/validacao/ (CSVs) e analise_somgoes/figuras/F14 a F17:
  regioes_por_fold.csv      taxa no treino e fora, fração do tempo, fração dos eventos
  regioes_resumo.csv        média e desvio entre temporadas
  niveis_por_fold.csv       o mesmo por nível de alerta
  niveis_resumo.csv         média e desvio entre temporadas
  similaridade_pares.csv    concordância entre cada par de modelos
  similaridade_resumo.csv   média, mínimo e máximo por cenário
  consenso.csv              nº de modelos em alerta ou acima x taxa de chuva forte
Uso: python -m analise.validacao_estados
"""
from __future__ import annotations

import logging
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from itertools import combinations

from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score

from core import config_somgoes as cfg
from core import somgoes_nucleo as nu
from analise.estados_somgoes import conectividade

OUT = cfg.ROOT / "analise_somgoes" / "validacao"
FIG = cfg.ROOT / "analise_somgoes" / "figuras"
DAD = cfg.ROOT / "analise_somgoes" / "dados_figuras"
log = logging.getLogger("validacao")


def avaliar_grupos(grupo_tr, grupo_va, y_tr, y_va, rotulos):
    """estatísticas por grupo (região ou nível), treino e fora da temporada"""
    f_tr, f_va = y_tr["prox"] >= 3, y_va["prox"] >= 3
    t_va = y_va["transicao"]
    linhas = []
    for g in rotulos:
        mt, mv = grupo_tr == g, grupo_va == g
        linhas.append(dict(
            grupo=g,
            taxa_forte_treino=f_tr[mt].mean() if mt.any() else np.nan,
            taxa_forte_fora=f_va[mv].mean() if mv.any() else np.nan,
            frac_tempo_fora=mv.mean(),
            frac_eventos_forte_fora=f_va[mv].sum() / max(1, f_va.sum()),
            frac_inicio_forte_fora=((t_va == 2) & mv).sum() / max(1, (t_va == 2).sum()),
            frac_intensifica_fora=((t_va == 3) & mv).sum() / max(1, (t_va == 3).sum()),
            taxa_chuva_fora=(y_va["prox"][mv] >= 1).mean() if mv.any() else np.nan,
            n_fora=int(mv.sum())))
    return pd.DataFrame(linhas)


def rodar(cache, chuva, h):
    base = cfg.RESULTS_DIR / "cenarios" / f"cenarios_{chuva}_h{h}_{cfg.ESTADOS_PERDA}_" \
                                          f"{cfg.ESTADOS_GRADE[0]}x{cfg.ESTADOS_GRADE[1]}"
    folds = sorted(int(p.name[4:]) for p in base.glob("fold*") if (p / "melhor.pt").exists())
    if not folds:
        log.info(f"  sem modelos salvos em {base}"); return None, None
    g = cfg.ESTADOS_GRADE
    conn = conectividade(*g)
    N = cfg.N_REGIOES_PADRAO
    R, L, todos = [], [], {}
    comum = np.where(cache._val(h))[0]
    if folds and folds[0] < 0:
        comum = comum[:cfg.RAPIDO["max_amostras"]]
    for fold in folds:
        tr, va = cache.split(fold, None if fold >= 0 else cfg.RAPIDO["max_amostras"], h)
        norm = cache.normalizador(tr)
        model = nu.SOMGoes(cache, cfg.GOES_CENARIOS, "ae", g, chuva=chuva)
        st = torch.load(base / f"fold{fold}" / "melhor.pt", map_location=nu.DEV, weights_only=False)
        model.load_state_dict(st["model"]); model.to(nu.DEV)
        a_tr, a_va = nu.bmus(model, cache, tr, norm), nu.bmus(model, cache, va, norm)
        y_tr, y_va = cache.rotulos(tr, h), cache.rotulos(va, h)
        # regiões definidas só com o mapa deste fold, numeradas pelo risco no treino
        lab = AgglomerativeClustering(n_clusters=N, linkage="ward", connectivity=conn)\
            .fit(model.prot.detach().cpu().numpy()).labels_
        f_tr = y_tr["prox"] >= 3
        risco = pd.Series(f_tr).groupby(lab[a_tr]).mean().reindex(range(N)).fillna(0)
        ren = {r: i + 1 for i, r in enumerate(risco.sort_values().index)}
        reg = np.array([ren[x] for x in lab])
        d = avaliar_grupos(reg[a_tr], reg[a_va], y_tr, y_va, range(1, N + 1))
        d.insert(0, "fold", fold); R.append(d)
        # níveis de alerta por célula, pela taxa de chuva forte no treino
        K = g[0] * g[1]
        n_cel = np.bincount(a_tr, minlength=K)
        tx = np.bincount(a_tr[f_tr], minlength=K) / np.maximum(n_cel, 1)
        limites = [x for x, _ in cfg.NIVEIS_ALERTA]
        niv_cel = np.digitize(tx, limites[:-1])
        nomes = [n for _, n in cfg.NIVEIS_ALERTA]
        d = avaliar_grupos(niv_cel[a_tr], niv_cel[a_va], y_tr, y_va, range(len(nomes)))
        d["grupo"] = d.grupo.map(dict(enumerate(nomes)))
        d.insert(0, "fold", fold); L.append(d)
        # mesmos horários para todos os modelos (similaridade entre folds)
        a_c = nu.bmus(model, cache, comum, norm)
        todos[fold] = dict(celula=a_c, regiao=reg[a_c], nivel=niv_cel[a_c], escore=tx[a_c])
    S, C = similaridade(todos, cache.rotulos(comum, h)["prox"] >= 3, chuva, h)
    R, L = pd.concat(R), pd.concat(L)
    for D in (R, L):
        D.insert(0, "horizonte", h); D.insert(0, "chuva", chuva)
    return R, L, S, C


def similaridade(todos, forte, chuva, h):
    linhas = []
    for fa, fb in combinations(sorted(todos), 2):
        A, B = todos[fa], todos[fb]
        alto_a, alto_b = A["nivel"] >= 2, B["nivel"] >= 2
        linhas.append(dict(chuva=chuva, horizonte=h, fold_a=fa, fold_b=fb,
                           rand_ajustado_celulas=adjusted_rand_score(A["celula"], B["celula"]),
                           rand_ajustado_regioes=adjusted_rand_score(A["regiao"], B["regiao"]),
                           concordancia_nivel=float((A["nivel"] == B["nivel"]).mean()),
                           kappa_nivel=cohen_kappa_score(A["nivel"], B["nivel"]),
                           concordancia_alerta_ou_acima=float((alto_a == alto_b).mean()),
                           kappa_alerta_ou_acima=cohen_kappa_score(alto_a, alto_b),
                           spearman_escore=spearmanr(A["escore"], B["escore"]).correlation))
    S = pd.DataFrame(linhas).round(4)
    n_alto = np.sum([v["nivel"] >= 2 for v in todos.values()], 0)
    C = pd.DataFrame({"n_modelos_alerta_ou_acima": n_alto, "forte": forte}).groupby("n_modelos_alerta_ou_acima")\
        .agg(n_horarios=("forte", "size"), taxa_forte=("forte", "mean"), eventos_forte=("forte", "sum")).reset_index()
    C["frac_tempo"] = (C.n_horarios / C.n_horarios.sum()).round(5)
    C["frac_eventos_forte"] = (C.eventos_forte / max(1, C.eventos_forte.sum())).round(4)
    C.insert(0, "n_modelos", len(todos)); C.insert(0, "horizonte", h); C.insert(0, "chuva", chuva)
    return S, C.round(4)


def resumo(D):
    num = [c for c in D.columns if c not in ("chuva", "horizonte", "fold", "grupo")]
    g = D.groupby(["chuva", "horizonte", "grupo"], sort=False)[num]
    S = g.mean().round(4).join(g.std().round(4), rsuffix="_dp").reset_index()
    S["n_temporadas"] = D.groupby(["chuva", "horizonte", "grupo"], sort=False).fold.nunique().values
    return S


def figuras(RS, LS):
    for (ch, h), d in RS.groupby(["chuva", "horizonte"]):
        fig, a = plt.subplots(figsize=(6.5, 3.6))
        x = np.arange(len(d))
        a.bar(x - 0.2, d.taxa_forte_treino, 0.4, label="temporadas de treino")
        a.bar(x + 0.2, d.taxa_forte_fora, 0.4, yerr=d.taxa_forte_fora_dp, capsize=2, label="temporada fora")
        a.set_xticks(x, [f"R{int(r)}" for r in d.grupo])
        a.set(ylabel=f"taxa de chuva forte em {h} min", title=f"regiões, treino x fora da temporada ({ch}, {h} min)")
        a.legend(fontsize=7); fig.tight_layout()
        nome = f"F14_regioes_fora_da_temporada_{ch}_h{h}"
        fig.savefig(FIG / f"{nome}.png", bbox_inches="tight"); plt.close(fig)
        d.to_csv(DAD / f"{nome}.csv", sep=";", index=False)
    for (ch, h), d in LS.groupby(["chuva", "horizonte"]):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        x = np.arange(len(d))
        ax[0].bar(x, d.frac_tempo_fora, yerr=d.frac_tempo_fora_dp, capsize=2, color="C7", label="fração do tempo")
        ax[0].bar(x, d.frac_eventos_forte_fora, 0.45, yerr=d.frac_eventos_forte_fora_dp, capsize=2, color="C3",
                  label="fração dos eventos de chuva forte")
        ax[1].bar(x, d.taxa_forte_fora, yerr=d.taxa_forte_fora_dp, capsize=2, color="C3")
        for a in ax:
            a.set_xticks(x, d.grupo, rotation=15)
        ax[0].legend(fontsize=7); ax[0].set_title(f"níveis de alerta fora da temporada ({ch}, {h} min)")
        ax[1].set(title="taxa de chuva forte em cada nível", ylabel="precisão")
        fig.tight_layout()
        nome = f"F15_niveis_de_alerta_{ch}_h{h}"
        fig.savefig(FIG / f"{nome}.png", bbox_inches="tight"); plt.close(fig)
        d.to_csv(DAD / f"{nome}.csv", sep=";", index=False)


def main():
    for p in (OUT, FIG, DAD):
        p.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    r = logging.getLogger(); r.handlers.clear()
    for hd in (logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "validacao.log", mode="w")):
        hd.setFormatter(fmt); r.addHandler(hd)
    r.setLevel(logging.INFO)
    cache = nu.Cache()
    RR, LL, SS, CC = [], [], [], []
    for chuva, h in cfg.ESTADOS_CENARIOS:
        log.info(f"cenário {chuva} h{h}")
        res = rodar(cache, chuva, h)
        if res[0] is not None:
            RR.append(res[0]); LL.append(res[1]); SS.append(res[2]); CC.append(res[3])
    if not RR:
        log.info("nenhum modelo encontrado"); return
    R, L = pd.concat(RR), pd.concat(LL)
    R.round(5).to_csv(OUT / "regioes_por_fold.csv", sep=";", index=False)
    L.round(5).to_csv(OUT / "niveis_por_fold.csv", sep=";", index=False)
    RS, LS = resumo(R), resumo(L)
    RS.to_csv(OUT / "regioes_resumo.csv", sep=";", index=False)
    LS.to_csv(OUT / "niveis_resumo.csv", sep=";", index=False)
    log.info("NÍVEIS DE ALERTA (fora da temporada, média entre temporadas)\n" +
             LS[["chuva", "horizonte", "grupo", "frac_tempo_fora", "taxa_forte_fora",
                 "frac_eventos_forte_fora", "frac_inicio_forte_fora", "n_temporadas"]].to_string(index=False))
    figuras(RS, LS)
    S, C = pd.concat(SS), pd.concat(CC)
    S.to_csv(OUT / "similaridade_pares.csv", sep=";", index=False)
    C.to_csv(OUT / "consenso.csv", sep=";", index=False)
    met = [c for c in S.columns if c not in ("chuva", "horizonte", "fold_a", "fold_b")]
    g = S.groupby(["chuva", "horizonte"])[met]
    SR = pd.concat({"media": g.mean(), "minimo": g.min(), "maximo": g.max()}, axis=1).round(4)
    SR.columns = [f"{b}_{a}" for a, b in SR.columns]
    SR = SR.reset_index(); SR.to_csv(OUT / "similaridade_resumo.csv", sep=";", index=False)
    log.info("SIMILARIDADE ENTRE MODELOS (média dos pares)\n" +
             SR[["chuva", "horizonte"] + [f"{m}_media" for m in met]].to_string(index=False))
    # o consenso dos 6 modelos sobre os mesmos horários inclui modelos que viram aqueles dados;
    # ele NÃO é usado como medida de desempenho (ver S12, comitê sem vazamento, na síntese)
    figuras_similaridade(S, C.iloc[0:0])


def figuras_similaridade(S, C):
    for (ch, h), d in S.groupby(["chuva", "horizonte"]):
        folds = sorted(set(d.fold_a) | set(d.fold_b))
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
        for a, col, tit in zip(ax, ("rand_ajustado_regioes", "kappa_nivel", "spearman_escore"),
                               ("regiões (Rand ajustado)", "nível de alerta (kappa)", "escore de risco (Spearman)")):
            M = np.eye(len(folds))
            for r in d.itertuples():
                i, j = folds.index(r.fold_a), folds.index(r.fold_b)
                M[i, j] = M[j, i] = getattr(r, col)
            im = a.imshow(M, cmap="viridis", vmin=0, vmax=1)
            a.set_xticks(range(len(folds)), [f"f{f}" for f in folds]); a.set_yticks(range(len(folds)), [f"f{f}" for f in folds])
            for i in range(len(folds)):
                for j in range(len(folds)):
                    a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if M[i, j] < 0.6 else "k")
            a.set_title(tit, fontsize=9); a.grid(False)
        fig.suptitle(f"concordância entre os modelos de cada fold ({ch}, {h} min)", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8); 
        nome = f"F16_similaridade_entre_folds_{ch}_h{h}"
        fig.savefig(FIG / f"{nome}.png", bbox_inches="tight"); plt.close(fig)
        d.to_csv(DAD / f"{nome}.csv", sep=";", index=False)
    for (ch, h), d in C.groupby(["chuva", "horizonte"]):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        ax[0].bar(d.n_modelos_alerta_ou_acima, d.frac_tempo, color="C7", label="fração do tempo")
        ax[0].bar(d.n_modelos_alerta_ou_acima, d.frac_eventos_forte, 0.45, color="C3", label="fração dos eventos fortes")
        ax[0].set(xlabel="nº de modelos em alerta ou acima", yscale="log", title=f"consenso ({ch}, {h} min)")
        ax[0].legend(fontsize=7)
        ax[1].plot(d.n_modelos_alerta_ou_acima, d.taxa_forte, "o-", color="C3")
        ax[1].set(xlabel="nº de modelos em alerta ou acima", ylabel="taxa de chuva forte", title="precisão por consenso")
        fig.tight_layout()
        nome = f"F17_consenso_entre_modelos_{ch}_h{h}"
        fig.savefig(FIG / f"{nome}.png", bbox_inches="tight"); plt.close(fig)
        d.to_csv(DAD / f"{nome}.csv", sep=";", index=False)


if __name__ == "__main__":
    main()
