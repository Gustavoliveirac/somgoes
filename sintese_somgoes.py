#!/usr/bin/env python
"""
sintese_somgoes.py — síntese de todo o estudo para o artigo e para a operação.

Tudo o que é desempenho usa previsões FORA da temporada de treino (modelos da validação
cruzada). Saídas em analise_somgoes/figuras/Sxx_*.png e dados_figuras/Sxx_*.csv:

  S01 todas as estratégias no plano recall x lift (nós concentradores e 5% do tempo)
  S02 catálogo das estratégias, o que cada uma faz e por que funcionou ou não (CSV)
  S03 ganho do GOES e dos quadros anteriores (nenhum, t0, 6 quadros, convlstm)
  S04 30 x 60 minutos, por entrada de chuva
  S05 valor de cada fonte: remoção de EMs, GOES, colar 1, colar 2 e núcleo no modelo treinado
  S06 confiabilidade (probabilidade prevista x frequência observada)
  S07 valor econômico relativo do alerta (modelo custo/perda)
  S08 antecedência dos alertas antes do início da chuva forte e alarmes falsos por temporada
  S09 probabilidade de chuva forte em até 10..120 min a partir de cada estado
  S10 rotas de estados mais frequentes antes do início da chuva forte
Uso: python sintese_somgoes.py
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
from sklearn.metrics import average_precision_score, roc_curve

import config_somgoes as cfg
import somgoes_nucleo as nu

OUT = cfg.ROOT / "analise_somgoes"
FIG, DAD = OUT / "figuras", OUT / "dados_figuras"
log = logging.getLogger("sintese")
CH = {"nenhuma": "sem chuva", "colar": "colar", "colar_nucleo": "colar e núcleo"}
PERDA_OP = cfg.ESTADOS_PERDA
LIM_ALERTA, LIM_PROVAVEL = cfg.NIVEIS_ALERTA[1][0], cfg.NIVEIS_ALERTA[2][0]


def salvar(nome, fig, df):
    if fig is not None:
        fig.tight_layout(); fig.savefig(FIG / f"{nome}.png", bbox_inches="tight"); plt.close(fig)
    df.to_csv(DAD / f"{nome}.csv", sep=";", index=False)
    log.info(f"  {nome}")


def skill(y, s):
    b = y.mean()
    return (average_precision_score(y, s) - b) / (1 - b) if 0 < b < 1 else np.nan


# ═══════════════════════════════ previsões fora da temporada ═════════════════
GRUPOS = {"EMs": "em", "GOES": "goes", "colar 1 (13-60 km)": "in_c1", "colar 2 (60-100 km)": "in_c2",
          "núcleo": "in_nuc"}


def bmus_mod(model, cache, idx, norm, grupo=None, lote=1024):
    cc = cache.cols_chuva[model.chuva]
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(idx), lote):
            b = cache.lote(idx[s:s + lote], norm, model.goes_modo, model.chuva)
            if grupo == "em":                       # fonte indisponível = máscara zerada
                for k in ("x10", "m10", "x1h", "m1h"):
                    b[k] = torch.zeros_like(b[k])
            elif grupo == "goes" and "goes" in b:
                b["goes"] = torch.zeros_like(b["goes"])
            elif grupo and grupo.startswith("in_") and "chuva" in b:
                j = [i for i, c in enumerate(cc) if c.startswith(grupo)]
                b["chuva"][:, j] = 0; b["mchuva"][:, j] = 0
            b = {k: v.to(nu.DEV) for k, v in b.items()}
            out.append(model.dist(model.codificar(b)[0]).argmin(1).cpu().numpy())
    return np.concatenate(out)


def oof(cache, chuva, h, ablacao=False):
    base = cfg.RESULTS_DIR / "cenarios" / f"cenarios_{chuva}_h{h}_{PERDA_OP}_10x10"
    folds = sorted(int(p.name[4:]) for p in base.glob("fold*") if (p / "melhor.pt").exists())
    linhas, abl = [], []
    for fold in folds:
        tr, va = cache.split(fold, None if fold >= 0 else cfg.RAPIDO["max_amostras"], h)
        norm = cache.normalizador(tr)
        m = nu.SOMGoes(cache, cfg.GOES_CENARIOS, "ae", (10, 10), chuva=chuva)
        m.load_state_dict(torch.load(base / f"fold{fold}" / "melhor.pt", map_location=nu.DEV,
                                     weights_only=False)["model"]); m.to(nu.DEV)
        ytr, yva = cache.rotulos(tr, h), cache.rotulos(va, h)
        a_tr = bmus_mod(m, cache, tr, norm)
        f_tr = ytr["prox"] >= 3
        n = np.bincount(a_tr, minlength=100)
        tx = (np.bincount(a_tr[f_tr], minlength=100) + 5 * f_tr.mean()) / (n + 5)
        a_va = bmus_mod(m, cache, va, norm)
        linhas.append(pd.DataFrame({"idx": va, "fold": fold, "escore": tx[a_va], "forte": yva["prox"] >= 3,
                                    "agora": yva["agora"], "transicao": yva["transicao"]}))
        if ablacao:
            fva = yva["prox"] >= 3
            ref = skill(fva, tx[a_va]); cob_ref = cob5(fva, tx[a_va])
            for nome, g in GRUPOS.items():
                s = tx[bmus_mod(m, cache, va, norm, g)]
                abl.append(dict(chuva=chuva, horizonte=h, fold=fold, fonte_removida=nome, skill_completo=ref,
                                skill_sem_fonte=skill(fva, s), recall5_completo=cob_ref, recall5_sem_fonte=cob5(fva, s)))
    D = pd.concat(linhas) if linhas else pd.DataFrame()
    if len(D):
        D["ts"] = cache.A.ts.to_numpy()[D.idx.to_numpy()]
        D["chuva"], D["horizonte"] = chuva, h
    return D, pd.DataFrame(abl)


def cob5(y, s, f=0.05):
    o = np.argsort(-s, kind="stable")
    k = max(1, int(round(f * len(s))))
    return y[o][:k].sum() / max(1, y.sum())


# ═══════════════════════════════ figuras de comparação (métricas) ════════════
def s01_s02_s03_s04(M):
    M = M[M.fold >= 0].copy()
    nf = M.groupby("exp_id").fold.transform("nunique")
    M["completo"] = nf == 6
    M["chuva"] = M.get("chuva").fillna("nenhuma"); M["horizonte"] = M.get("horizonte").fillna(60).astype(int)
    M["estrategia"] = np.where(M.familia.str.startswith("kmeans"), "k-means",
                       np.where(M.familia == "som_classico", "SOM clássico " + M.perda.astype(str),
                                M.goes.astype(str) + " | " + M.perda.astype(str)))
    g = M.groupby(["familia", "exp_id", "estrategia", "chuva", "horizonte", "grade"])
    P = g.agg(n_temporadas=("fold", "nunique"), skill=("forte_prox.ap_skill", "mean"),
              recall_5pct=("forte_prox.cob_5pct", "mean"), lift_5pct=("forte_prox.lift_5pct", "mean"),
              recall_nos=("no_forte_top3.recall", "mean"), precisao_nos=("no_forte_top3.precisao", "mean"),
              base=("forte_prox.base", "mean")).reset_index()
    P["lift_nos"] = P.precisao_nos / P.base
    P = P.round(4)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for fam, mk in (("kmeans", "s"), ("kmeans_cenarios", "s"), ("som_classico", "D"), ("etapa1", "o"),
                    ("etapa2", "^"), ("cenarios", "*"), ("grades", "P")):
        t = P[(P.familia == fam) & (P.n_temporadas >= 1)]
        cor = t.chuva.map({"nenhuma": "C0", "colar": "C1", "colar_nucleo": "C3"})
        tam = np.where(t.n_temporadas == 6, 40, 14)
        ax[0].scatter(t.recall_nos, t.lift_nos, c=cor, marker=mk, s=tam, alpha=0.75, label=fam)
        ax[1].scatter(t.recall_5pct, t.skill, c=cor, marker=mk, s=tam, alpha=0.75, label=fam)
    for a in ax:
        a.legend(fontsize=7, title="família (cor = entrada de chuva)", title_fontsize=7)
    ax[0].set(xlabel="recall nas 3 células mais fortes", ylabel="lift nas 3 células mais fortes",
              title="nós concentradores (cada ponto é um experimento)")
    ax[1].set(xlabel="recall a 5% do tempo monitorado", ylabel="skill",
              title="ordenação geral (pontos grandes = 6 temporadas)")
    salvar("S01_estrategias_recall_lift", fig, P)

    def med(fam=None, perda=None, goes=None, chuva="nenhuma", h=60, grade="10x10", seis=True):
        t = M[(M.chuva == chuva) & (M.horizonte == h) & (M.grade == grade)]
        if fam: t = t[t.familia == fam]
        if perda: t = t[t.perda == perda]
        if goes: t = t[t.goes == goes]
        if fam == "etapa1": t = t[t.ae == "ae"]
        if seis: t = t[t.completo]
        return round(t["forte_prox.ap_skill"].mean(), 3) if len(t) else np.nan, t.fold.nunique()

    CAT = [
        ("k-means", "kmeans_cenarios", None, None, "agrupa os horários por distância nas entradas, sem saber da chuva",
         "referência; não tem como priorizar o que importa para a chuva forte, então os grupos seguem a variabilidade dominante (ciclo diurno, temperatura)"),
        ("SOM clássico", "som_classico", "baseline", None, "SOM online nas features planas, sem chuva na perda",
         "organiza bem o espaço, mas as features planas e a ausência da chuva na perda limitam a separação"),
        ("SOM clássico supervisionado", "som_classico", "supervisionado", None, "SOM com a transição acrescentada ao vetor só no treino",
         "a chuva orienta a organização, mas o BMU final usa só a entrada, e parte do ganho se perde"),
        ("reconstrução", "etapa1", "recon", "t0", "SOM profundo que só aprende a reconstruir a entrada",
         "aprende o que é mais variável na entrada, não o que antecede a chuva; serve para medir o ganho de pôr a chuva na perda"),
        ("informação mútua (mi)", "etapa1", "mi", "t0", "maximiza quanto a célula informa sobre a transição",
         "é o lift médio em forma logarítmica; recompensa exatamente concentrar cada transição em células próprias"),
        ("mi com peso nas fortes / peso dobrado", "etapa1", "mi_forte", "t0", "a mesma mi com mais peso nas transições para forte ou no termo todo",
         "equivalente à mi; a mi já dá destaque às classes raras porque o lote é balanceado"),
        ("mi + âncora dos cantos", "etapa1", "mi_canto", "t0", "puxa os 4 estados básicos para os cantos da grade",
         "skill um pouco menor, nós mais concentradores; impõe uma geometria que nem sempre coincide com a dos dados"),
        ("mi + lift relativo / separação", "etapa1", "mi_lift", "t0", "termos extras de concentração relativa ou de sobreposição",
         "redundantes com a mi e competem com ela, sem ganho"),
        ("concentração por entropia", "etapa1", "mi_conc", "t0", "cada transição em poucas células",
         "pior resultado: premia jogar os eventos numa célula grande junto com muitos horários sem chuva"),
        ("VAE", "etapa1", "mi", "t0", "versão variacional (latente probabilístico)",
         "um pouco abaixo do AE; o ruído do latente borra a fronteira entre células vizinhas"),
        ("condicional (cmi)", "cenarios", "cmi", None, "informação sobre a próxima janela dentro de cada estado atual",
         "sem organizar antes pelo estado atual, o mapa fica sem estrutura de base"),
        ("duas etapas", "cenarios", "duas_etapas", None, "organiza pelo estado atual e depois subdivide pelo futuro",
         "escolhida; iguala a mi no geral e é melhor em antecipar a chuva que ainda não começou"),
        ("duas etapas + transição (original)", "cenarios", "duas_etapas_markov", None, "modelo de Markov entre células com suavidade",
         "colapsou: a suavidade premiava ficar na mesma célula"),
        ("duas etapas + transição (corrigida)", "cenarios", "duas_etapas_markov2", None, "Markov com peso menor e uso do mapa",
         "não colapsa, mas não supera a de duas etapas; a dinâmica é lida depois, pelas transições do mapa treinado"),
    ]
    linhas = []
    for nome, fam, perda, goes, oque, porque in CAT:
        s, nf_ = med(fam, perda, goes, chuva="nenhuma", h=60, grade="10x10", seis=False)
        if nome == "VAE":
            t = M[(M.familia == "etapa1") & (M.ae == "vae") & (M.goes == "t0") & (M.perda == "mi")]
            s, nf_ = round(t["forte_prox.ap_skill"].mean(), 3), t.fold.nunique()
        linhas.append(dict(estrategia=nome, familia=fam, o_que_faz=oque, skill_sem_chuva_60min_10x10=s,
                           temporadas=nf_, por_que=porque))
    salvar("S02_catalogo_estrategias", None, pd.DataFrame(linhas))

    G = []
    for goes in ("nenhum", "t0", "seq_cnn", "convlstm"):
        for perda in ("recon", "mi"):
            t = M[(M.familia == "etapa1") & (M.goes == goes) & (M.perda == perda) & (M.ae == "ae")]
            t3 = t[t.fold == 3]
            G.append(dict(goes=goes, perda=perda, skill_fold3=t3["forte_prox.ap_skill"].mean(),
                          skill_6temporadas=t["forte_prox.ap_skill"].mean() if t.fold.nunique() == 6 else np.nan))
    t = M[(M.familia == "etapa2") & (M.goes == "seq_cnn") & (M.grade == "10x10")]
    G.append(dict(goes="seq_cnn", perda="mi", skill_fold3=np.nan,
                  skill_6temporadas=t["forte_prox.ap_skill"].mean() if t.fold.nunique() == 6 else np.nan))
    G = pd.DataFrame(G).groupby(["goes", "perda"]).max().reset_index().round(3)
    fig, a = plt.subplots(figsize=(6.5, 3.6))
    nomes = {"nenhum": "sem GOES", "t0": "quadro atual", "seq_cnn": "6 quadros empilhados", "convlstm": "6 quadros convlstm"}
    for k, (perda, d) in enumerate(G.groupby("perda")):
        d = d.set_index("goes").reindex(list(nomes))
        x = np.arange(len(nomes)) + 0.2 * (k - 0.5)
        a.bar(x, d.skill_fold3, 0.38, label=f"{perda} (temporada 2022/23)")
        a.scatter(x, d.skill_6temporadas, color="k", zorder=3, s=15)
    a.set_xticks(range(len(nomes)), list(nomes.values()), rotation=10)
    a.set(ylabel="skill chuva forte 60 min", title="ganho do GOES e dos quadros anteriores (ponto = média de 6 temporadas)")
    a.legend(fontsize=7)
    salvar("S03_goes_e_quadros", fig, G)

    C = M[(M.familia == "cenarios") & M.completo & (M.perda == PERDA_OP)]
    cols = {"forte_prox.ap_skill": "skill", "forte_prox.cob_5pct": "recall_5pct", "forte_prox.base": "taxa_base",
            "cond_seco_chuva.ap_skill": "seco_vai_chover", "cond_fraca_forte.ap_skill": "fraca_intensifica",
            "cond_fraca_seco.ap_skill": "fraca_para", "cond_forte_persiste.ap_skill": "forte_persiste"}
    H = C.groupby(["chuva", "horizonte"])[list(cols)].mean().rename(columns=cols).round(3).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, col in zip(ax, ("skill", "recall_5pct")):
        for k, h in enumerate((60, 30)):
            d = H[H.horizonte == h].set_index("chuva").reindex(cfg.CHUVA_ENTRADA)
            a.bar(np.arange(3) + 0.38 * k, d[col], 0.36, label=f"{h} min")
        a.set_xticks(np.arange(3) + 0.19, [CH[c] for c in cfg.CHUVA_ENTRADA]); a.set_title(col); a.legend(fontsize=7)
    salvar("S04_30_vs_60_min", fig, H)


# ═══════════════════════════════ figuras operacionais (fora da temporada) ═════
def s05(A):
    if A.empty:
        return
    A = A.assign(perda_skill=A.skill_completo - A.skill_sem_fonte, perda_recall5=A.recall5_completo - A.recall5_sem_fonte)
    R = A.groupby(["horizonte", "fonte_removida"])[["skill_completo", "skill_sem_fonte", "perda_skill",
                                                    "recall5_completo", "recall5_sem_fonte", "perda_recall5"]].mean().round(4).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, h in zip(ax, (60, 30)):
        d = R[R.horizonte == h].sort_values("perda_skill")
        a.barh(d.fonte_removida, d.perda_skill, color="C3")
        a.set(title=f"quanto a skill cai sem cada fonte ({h} min)", xlabel="queda de skill")
    salvar("S05_valor_de_cada_fonte", fig, R)


def s06_s07_s08(D, cache):
    linhas_c, linhas_v, linhas_l, linhas_f = [], [], [], []
    for (ch, h), d in D.groupby(["chuva", "horizonte"]):
        q = pd.qcut(d.escore.rank(method="first"), 12, labels=False)
        c = d.groupby(q).agg(prob_prevista=("escore", "mean"), freq_observada=("forte", "mean"), n=("forte", "size"))
        c.insert(0, "horizonte", h); c.insert(0, "chuva", ch); linhas_c.append(c.reset_index(drop=True))
        y, s = d.forte.to_numpy(), d.escore.to_numpy()
        F, Hh, th = roc_curve(y, s)
        b = y.mean()
        for alpha in np.r_[np.linspace(0.01, 0.1, 10), np.linspace(0.15, 0.95, 17)]:
            num = np.minimum(alpha, b) - F * alpha * (1 - b) + Hh * b * (1 - alpha) - b
            V = num / (np.minimum(alpha, b) - b * alpha)
            k = int(np.nanargmax(V))
            linhas_v.append(dict(chuva=ch, horizonte=h, custo_perda=round(alpha, 3), valor_relativo=round(V[k], 4),
                                 limiar_escore=round(min(th[k], 1.0), 4)))
        # antecedência e alarmes falsos, por temporada
        sf = "" if h == 60 else f"_{h}"
        A = cache.A
        est = A[f"estado_agora{sf}"].fillna(-1).to_numpy()
        intens = A[f"agora{sf}"].fillna(0).to_numpy()            # mm/h no núcleo
        colar1 = A["colar1_max"].fillna(0).to_numpy() if "colar1_max" in A else np.zeros(len(A))
        ts = A.ts.to_numpy()
        for fold, dd in d.groupby("fold"):
            dd = dd.sort_values("idx")
            sc = pd.Series(dd.escore.to_numpy(), index=dd.idx.to_numpy())
            idx_f = dd.idx.to_numpy()
            ini = [i for i in idx_f if est[i] >= 3 and i - 1 >= 0 and est[i - 1] < 3 and
                   (ts[i] - ts[i - 1]) == np.timedelta64(10, "m")]
            for nivel, lim in (("alerta", LIM_ALERTA), ("chuva forte provável", LIM_PROVAVEL)):
                leads = []
                for i in ini:
                    ld = 0
                    for k in range(1, 13):
                        j = i - k
                        if j < 0 or (ts[i] - ts[j]) != np.timedelta64(10 * k, "m") or j not in sc.index:
                            break
                        if sc[j] >= lim:
                            ld = 10 * k
                    leads.append(ld)
                leads = np.array(leads)
                ativo = (sc >= lim).to_numpy()
                ixs = sc.index.to_numpy()
                # regra de manutenção: interrupções de até 30 min não encerram o episódio
                ativo_m = ativo.copy()
                ult = -99
                for k in range(len(ativo)):
                    if ativo[k]:
                        if 0 < k - ult <= 3:
                            ativo_m[ult:k] = True
                        ult = k
                ep_ini_m = np.where(ativo_m & ~np.r_[False, ativo_m[:-1]])[0]
                ep_fim_m = np.where(ativo_m & ~np.r_[ativo_m[1:], False])[0]
                set_ini0 = set(ini)
                falsos_m, tipos = 0, {"forte no colar 13-60 km": 0, "quase forte no núcleo (5 a 10 mm/h)": 0,
                                       "chuva fraca ou moderada no núcleo": 0, "seco no núcleo e no colar": 0}
                for a0, a1 in zip(ep_ini_m, ep_fim_m):
                    jan = np.arange(ixs[a0], min(len(est), ixs[a1] + h // 10 + 1))
                    if any(j in set_ini0 or est[j] >= 3 for j in jan):
                        continue
                    falsos_m += 1
                    if colar1[jan].max() >= cfg.LIMITES[2]:
                        tipos["forte no colar 13-60 km"] += 1
                    elif intens[jan].max() >= cfg.LIMITES[2] / 2:
                        tipos["quase forte no núcleo (5 a 10 mm/h)"] += 1
                    elif intens[jan].max() >= cfg.BASCULA:
                        tipos["chuva fraca ou moderada no núcleo"] += 1
                    else:
                        tipos["seco no núcleo e no colar"] += 1
                ep_ini = np.where(ativo & ~np.r_[False, ativo[:-1]])[0]
                ep_fim = np.where(ativo & ~np.r_[ativo[1:], False])[0]
                falsos = 0
                set_ini = set(ini)
                for a0, a1 in zip(ep_ini, ep_fim):
                    jan = range(ixs[a0], ixs[a1] + h // 10 + 1)
                    if not any(j in set_ini or (est[j] >= 3 if j < len(est) else False) for j in jan):
                        falsos += 1
                linhas_l.append(dict(chuva=ch, horizonte=h, fold=fold, nivel=nivel, inicios_forte=len(ini),
                                     detectados_antes=int((leads > 0).sum()),
                                     frac_detectados_antes=round(float((leads > 0).mean()), 4) if len(ini) else np.nan,
                                     frac_10min_ou_mais=round(float((leads >= 10).mean()), 4) if len(ini) else np.nan,
                                     frac_30min_ou_mais=round(float((leads >= 30).mean()), 4) if len(ini) else np.nan,
                                     antecedencia_mediana_min=float(np.median(leads[leads > 0])) if (leads > 0).any() else np.nan,
                                     episodios_alerta=len(ep_ini), episodios_falsos=falsos,
                                     frac_episodios_falsos=round(falsos / max(1, len(ep_ini)), 4),
                                     episodios_com_manutencao=len(ep_ini_m), falsos_com_manutencao=falsos_m,
                                     frac_falsos_com_manutencao=round(falsos_m / max(1, len(ep_ini_m)), 4),
                                     **{f"falsos: {k}": v for k, v in tipos.items()}))
    C = pd.concat(linhas_c).round(4); V = pd.DataFrame(linhas_v); L = pd.DataFrame(linhas_l)
    fig, a = plt.subplots(figsize=(5, 4.5))
    for (ch, h), c in C.groupby(["chuva", "horizonte"]):
        a.plot(c.prob_prevista, c.freq_observada, "o-", ms=3, label=f"{CH[ch]}, {h} min")
    a.plot([0, 1], [0, 1], "k:"); a.set(xlabel="probabilidade prevista (taxa da célula no treino)",
                                         ylabel="frequência observada fora da temporada", title="confiabilidade")
    a.legend(fontsize=7)
    salvar("S06_confiabilidade", fig, C)
    fig, a = plt.subplots(figsize=(6, 4))
    for (ch, h), v in V.groupby(["chuva", "horizonte"]):
        a.plot(v.custo_perda, v.valor_relativo.clip(lower=0), "-", label=f"{CH[ch]}, {h} min")
    a.set(xscale="log", xlabel="razão custo da ação / perda evitada", ylabel="valor econômico relativo (0 a 1)",
          title="valor do alerta para quem decide", ylim=(0, 1)); a.legend(fontsize=7)
    salvar("S07_valor_economico", fig, V)
    LR = L.groupby(["chuva", "horizonte", "nivel"]).agg(
        inicios_forte=("inicios_forte", "sum"), frac_detectados_antes=("frac_detectados_antes", "mean"),
        frac_10min_ou_mais=("frac_10min_ou_mais", "mean"), frac_30min_ou_mais=("frac_30min_ou_mais", "mean"),
        antecedencia_mediana_min=("antecedencia_mediana_min", "mean"),
        episodios_alerta_por_temporada=("episodios_alerta", "mean"),
        episodios_falsos_por_temporada=("episodios_falsos", "mean"),
        frac_episodios_falsos=("frac_episodios_falsos", "mean"),
        episodios_com_manutencao_por_temporada=("episodios_com_manutencao", "mean"),
        falsos_com_manutencao_por_temporada=("falsos_com_manutencao", "mean"),
        frac_falsos_com_manutencao=("frac_falsos_com_manutencao", "mean"),
        **{f"falsos_{k}": (f"falsos: {k}", "mean") for k in ("forte no colar 13-60 km",
            "quase forte no núcleo (5 a 10 mm/h)", "chuva fraca ou moderada no núcleo", "seco no núcleo e no colar")}
    ).round(3).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for (ch, h), t in LR[LR.nivel == "alerta"].groupby(["chuva", "horizonte"]):
        ax[0].bar(f"{CH[ch]}\n{h} min", t.frac_detectados_antes, color="C3")
        ax[1].bar(f"{CH[ch]}\n{h} min", t.frac_episodios_falsos, color="C7")
    ax[0].set(title="inícios de chuva forte com alerta antes", ylabel="fração")
    ax[1].set(title="episódios de alerta sem chuva forte", ylabel="fração")
    salvar("S08_antecedencia_e_falsos", fig, pd.concat([LR.assign(tipo="resumo"), L.assign(tipo="por_temporada")]))


def s09_s10(cache):
    for ch, h in cfg.ESTADOS_CENARIOS:
        d = cfg.RESULTS_DIR / "estados" / f"{ch}_h{h}"
        if not (d / "modelo" / "melhor.pt").exists() or not (d / "celulas.csv").exists():
            continue
        cel = pd.read_csv(d / "celulas.csv", sep=";").set_index("celula").regiao
        val = np.where(cache._val(h))[0]
        rng = np.random.default_rng(cfg.SEED); para = rng.random(len(val)) < 0.10
        norm = cache.normalizador(val[~para])
        m = nu.SOMGoes(cache, cfg.GOES_CENARIOS, "ae", cfg.ESTADOS_GRADE, chuva=ch)
        m.load_state_dict(torch.load(d / "modelo" / "melhor.pt", map_location=nu.DEV, weights_only=False)["model"])
        m.to(nu.DEV)
        reg = pd.Series(cel.reindex(bmus_mod(m, cache, val, norm)).to_numpy(), index=val)
        sf = "" if h == 60 else f"_{h}"
        est = cache.A[f"estado_agora{sf}"].fillna(-1).to_numpy()
        prox = cache.prox_idx
        # probabilidade de chuva forte em até k minutos a partir de cada região (descritivo)
        linhas = []
        fut = np.arange(len(est))
        atingiu = np.zeros((len(val), 12), bool)
        cur = val.copy()
        acum = np.zeros(len(val), bool)
        for k in range(12):
            cur = np.where(cur >= 0, prox[np.clip(cur, 0, None)], -1)
            acum |= (cur >= 0) & (est[np.clip(cur, 0, None)] >= 3)
            atingiu[:, k] = acum
        T = pd.DataFrame(atingiu, columns=[f"{10*(k+1)}" for k in range(12)])
        T["regiao"] = reg.to_numpy()
        S = T.groupby("regiao").mean().round(4).reset_index()
        S.insert(0, "horizonte", h); S.insert(0, "chuva", ch)
        fig, a = plt.subplots(figsize=(6, 4))
        for _, r in S.iterrows():
            a.plot(range(10, 130, 10), [r[str(m)] for m in range(10, 130, 10)], "o-", ms=3, label=f"R{int(r.regiao)}")
        a.set(xlabel="minutos à frente", ylabel="prob. de ocorrer chuva forte em até k minutos",
              title=f"tempo até a chuva forte por estado ({CH[ch]}, {h} min)")
        a.legend(fontsize=7, ncol=2)
        salvar(f"S09_tempo_ate_forte_{ch}_h{h}", fig, S)
        # rotas antes do início da chuva forte (regiões 60, 30 e 10 min antes)
        ini = [i for i in val if est[i] >= 3 and est[i - 1] < 3]
        rr = []
        for i in ini:
            seq = []
            for k in (6, 3, 1):
                j = i - k
                seq.append(f"R{int(reg[j])}" if j in reg.index else "?")
            rr.append(" > ".join(seq))
        Rt = pd.Series(rr).value_counts().rename_axis("rota_60_30_10min_antes").reset_index(name="inicios")
        Rt["frac"] = (Rt.inicios / max(1, len(ini))).round(4)
        Rt.insert(0, "horizonte", h); Rt.insert(0, "chuva", ch)
        fig, a = plt.subplots(figsize=(6, 4))
        t = Rt.head(12)
        a.barh(t.rota_60_30_10min_antes[::-1], t.frac[::-1], color="C3")
        a.set(xlabel="fração dos inícios de chuva forte", title=f"rotas mais frequentes ({CH[ch]}, {h} min)")
        salvar(f"S10_rotas_ate_forte_{ch}_h{h}", fig, Rt)


def s12_comite_aninhado(cache):
    """Comitê avaliado sem vazamento: para cada temporada s, os 5 modelos do comitê foram
    treinados SEM s (cada um também sem uma segunda temporada t, usada para parar o treino).
    Compara com o modelo único da validação cruzada na mesma temporada."""
    base = cfg.RESULTS_DIR / "comite_aninhado"
    linhas, cons = [], []
    for ch, h in cfg.COMITE_CENARIOS:
        d = base / f"{ch}_h{h}"
        if not d.exists():
            continue
        for s_ in sorted({int(p.name.split("_")[0][4:]) for p in d.glob("fora*_sem*") if (p / "melhor.pt").exists()}):
            mods = sorted(p for p in d.glob(f"fora{s_}_sem*") if (p / "melhor.pt").exists())
            if len(mods) < 2:
                continue
            f = cache.A.fold.to_numpy(); val = cache._val(h)
            va = np.where(val & (f == s_))[0]
            yva = cache.rotulos(va, h)["prox"] >= 3
            P = []
            for p in mods:
                t_ = int(p.name.split("_sem")[1])
                tr = np.where(val & (f != s_) & (f != t_))[0]
                norm = cache.normalizador(tr)
                m = nu.SOMGoes(cache, cfg.GOES_CENARIOS, "ae", (10, 10), chuva=ch)
                m.load_state_dict(torch.load(p / "melhor.pt", map_location=nu.DEV, weights_only=False)["model"]); m.to(nu.DEV)
                a_tr = bmus_mod(m, cache, tr, norm); fo = cache.rotulos(tr, h)["prox"] >= 3
                n = np.bincount(a_tr, minlength=100)
                tx = (np.bincount(a_tr[fo], minlength=100) + 5 * fo.mean()) / (n + 5)
                P.append(tx[bmus_mod(m, cache, va, norm)])
            P = np.array(P)
            prob = P.mean(0); votos = (P >= LIM_ALERTA).sum(0)
            linhas.append(dict(chuva=ch, horizonte=h, temporada_fora=s_, n_modelos=len(P),
                               skill_comite=skill(yva, prob), recall5_comite=cob5(yva, prob),
                               skill_cada_modelo_media=np.mean([skill(yva, x) for x in P]),
                               recall5_cada_modelo_media=np.mean([cob5(yva, x) for x in P])))
            c = pd.DataFrame({"votos": votos, "forte": yva}).groupby("votos").agg(
                n=("forte", "size"), taxa_forte=("forte", "mean"), eventos=("forte", "sum")).reset_index()
            c.insert(0, "temporada_fora", s_); c.insert(0, "horizonte", h); c.insert(0, "chuva", ch); cons.append(c)
    if not linhas:
        log.info("  comitê aninhado ainda não treinado (rodada_publicacao.py)"); return
    L = pd.DataFrame(linhas).round(4)
    C = pd.concat(cons)
    CR = C.groupby(["chuva", "horizonte", "votos"]).agg(n=("n", "sum"), eventos=("eventos", "sum")).reset_index()
    CR["taxa_forte"] = (CR.eventos / CR.n).round(4)
    CR["frac_tempo"] = CR.groupby(["chuva", "horizonte"]).n.transform(lambda x: x / x.sum()).round(4)
    CR["frac_eventos"] = CR.groupby(["chuva", "horizonte"]).eventos.transform(lambda x: x / max(1, x.sum())).round(4)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for (ch, h), t in CR.groupby(["chuva", "horizonte"]):
        ax[0].plot(t.votos, t.taxa_forte, "o-", label=f"{CH[ch]}, {h} min")
    ax[0].set(xlabel="nº de modelos em alerta (nenhum viu a temporada)", ylabel="taxa de chuva forte", title="consenso sem vazamento")
    ax[0].legend(fontsize=7)
    R = L.groupby(["chuva", "horizonte"])[["skill_comite", "skill_cada_modelo_media"]].mean()
    R.plot.bar(ax=ax[1], rot=0); ax[1].set(title="comitê x modelo isolado (média das temporadas)", ylabel="skill")
    salvar("S12_comite_sem_vazamento", fig, pd.concat([L.assign(tabela="por_temporada"), CR.assign(tabela="consenso")]))


def main():
    for p in (FIG, DAD):
        p.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    r = logging.getLogger(); r.handlers.clear()
    for hd in (logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "sintese.log", mode="w")):
        hd.setFormatter(fmt); r.addHandler(hd)
    r.setLevel(logging.INFO)
    M = pd.read_csv(cfg.RESULTS_DIR / "metricas_consolidadas.csv", sep=";", low_memory=False)
    for f in (lambda: s01_s02_s03_s04(M),):
        try:
            f()
        except Exception as e:
            log.info(f"  comparações falharam: {e}")
    cache = nu.Cache()
    DD, AA = [], []
    for ch in cfg.CHUVA_ENTRADA:
        for h in cfg.HORIZONTES:
            log.info(f"previsões fora da temporada: {ch} h{h}")
            D, A = oof(cache, ch, h, ablacao=(ch == "colar_nucleo"))
            if len(D):
                DD.append(D); AA.append(A)
    if DD:
        D = pd.concat(DD)
        D.drop(columns=["idx"]).to_csv(OUT / "previsoes_fora_da_temporada.csv.gz", sep=";", index=False)
        for f in (lambda: s05(pd.concat(AA)), lambda: s06_s07_s08(D, cache)):
            try:
                f()
            except Exception as e:
                log.info(f"  figura operacional falhou: {e}")
    try:
        s12_comite_aninhado(cache)
    except Exception as e:
        log.info(f"  comitê aninhado falhou: {e}")
    try:
        s09_s10(cache)
    except Exception as e:
        log.info(f"  rotas falharam: {e}")


if __name__ == "__main__":
    main()
