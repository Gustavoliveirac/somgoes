#!/usr/bin/env python
"""
regioes_transicoes.py — transições entre as 8 regiões do SOM e POD e FAR por limiar.

Não treina nada. Usa os latentes da etapa A (etapa_a/cache) e os escores fora da temporada
já salvos (etapa_a/escores_h*_*.csv.gz).

  R1  regiões do mapa de cada temporada (agrupamento de Ward dos 100 protótipos em 8), numeradas
      de 1 a 8 pela frequência de chuva forte nos horários de treino, de modo que a região 8 é
      a de maior frequência em todas as temporadas e as tabelas podem ser somadas
  R2  probabilidade de chuva forte estimada no treino por
        região agora
        região 30 min antes e região agora (64 pares)
        nó agora e região 30 min antes (800 pares, com o nó agora como valor de partida)
  R3  tabela de transições (região 30 min antes para região agora) nos horários fora do treino,
      com o número de horários e a fração seguida de chuva forte
  R4  POD, FAR e CSI de todos os modelos para limiares fixos de probabilidade (0,05 a 0,50) e
      para alertas em 2, 5 e 10% do tempo, por horário de 10 min

Saídas em etapa_a/regioes/ (CSVs) e log mestre etapa_a/regioes/regioes.log.
Uso (da raiz):  python -m tese.regioes_transicoes
"""
from __future__ import annotations

import logging
import sys
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import average_precision_score

from core import config_somgoes as cfg
from tese import etapa_a as ea

at = ea.at
mt = ea.mt
OUT = cfg.ROOT / "etapa_a" / "regioes"
N_REG = ea.N_REG
LIMIARES = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
FRACOES = (0.02, 0.05, 0.10)
ALFAS = ea.ALFAS
log = logging.getLogger("regioes")


def configurar_log():
    OUT.mkdir(parents=True, exist_ok=True)
    at.OUT = OUT
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    r = logging.getLogger(); r.handlers.clear()
    for hd in (logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "regioes.log", mode="w")):
        hd.setFormatter(fmt); r.addHandler(hd)
    r.setLevel(logging.INFO)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 500)


def tabela(chave, base, y, ajuste, pred, alfa, ncat):
    """Frequência de chuva forte por categoria `chave` nos horários `ajuste`, com `base` (uma
    probabilidade por horário) como valor de partida de peso alfa; chave < 0 usa só a base."""
    v = ajuste[chave[ajuste] >= 0]
    n = np.bincount(chave[v], minlength=ncat); f = np.bincount(chave[v][y[v]], minlength=ncat)
    c = chave[pred]; b = base[pred]
    return np.where(c >= 0, (f[np.clip(c, 0, None)] + alfa * b) / (n[np.clip(c, 0, None)] + alfa), b)


def escolher_alfa(chave, base_fn, y, itr, s, ncat):
    """Peso alfa por validação interna: uma temporada de treino fora por vez."""
    notas = {a: [] for a in ALFAS}
    for si_ in np.unique(s[itr]):
        a = itr[s[itr] != si_]; b = itr[s[itr] == si_]
        if y[b].sum() == 0:
            continue
        base = base_fn(a)
        for al in ALFAS:
            notas[al].append(average_precision_score(y[b], tabela(chave, base, y, a, b, al, ncat)))
    return max(ALFAS, key=lambda a: np.mean(notas[a]) if notas[a] else -1)


def dobra(R, yfull, temp, jan, ncache):
    idx, ntr, d, prot = R["idx"], R["ntr"], R["d"], R["prot"]
    n = len(idx); itr = np.arange(ntr); ite = np.arange(ntr, n)
    bmu = d.argmin(1); y = yfull[idx]; s = temp[idx]
    posmap = np.full(ncache, -1, np.int64); posmap[idx] = np.arange(n)
    jp = jan[idx, -4]
    prev = np.where(jp >= 0, posmap[np.clip(jp, 0, None)], -1)
    bprev = np.where(prev >= 0, bmu[np.clip(prev, 0, None)], -1)
    reg_no = fcluster(linkage(prot.astype(np.float64), "ward"), N_REG, "maxclust") - 1
    # regiões numeradas de 1 a 8 pela frequência de chuva forte no treino
    r0 = reg_no[bmu]
    taxa = np.array([y[itr][r0[itr] == r].mean() if (r0[itr] == r).any() else 0 for r in range(N_REG)])
    ordem = np.empty(N_REG, int); ordem[np.argsort(taxa)] = np.arange(1, N_REG + 1)
    rg = ordem[r0]
    rgp = np.where(bprev >= 0, ordem[reg_no[np.clip(bprev, 0, None)]], 0)
    media = lambda a: np.full(n, (y[a].sum() + 1) / (len(a) + 2))
    # região agora
    p_reg = tabela(rg, media(itr), y, itr, np.arange(n), 5, N_REG + 1)
    def base_reg(a):
        return tabela(rg, media(a), y, a, np.arange(n), 5, N_REG + 1)
    # par de regiões (30 min antes, agora)
    ch_par = np.where(rgp > 0, rgp * (N_REG + 1) + rg, -1)
    al_par = escolher_alfa(ch_par, base_reg, y, itr, s, (N_REG + 1) ** 2)
    p_par = tabela(ch_par, p_reg, y, itr, ite, al_par, (N_REG + 1) ** 2)
    # nó agora e região 30 min antes, com o nó como valor de partida
    def base_no(a):
        return ea.f1.prob_nos(bmu[a], y[a])[bmu]
    ch_nr = np.where(rgp > 0, bmu * (N_REG + 1) + rgp, -1)
    al_nr = escolher_alfa(ch_nr, base_no, y, itr, s, 100 * (N_REG + 1))
    p_nr = tabela(ch_nr, base_no(itr), y, itr, ite, al_nr, 100 * (N_REG + 1))
    return dict(posmap=posmap, ntr=ntr, ite=ite, y=y, rg=rg, rgp=rgp, p_reg=p_reg[ite], p_par=p_par,
                p_nr=p_nr, al_par=al_par, al_nr=al_nr, taxa_regiao=np.sort(taxa))


def pod_far(y, s, nome, h, conj):
    lin = []
    ok = np.isfinite(s); y, s = y[ok], s[ok]
    if not len(s):
        return lin
    def cat(a, ponto):
        hit = int((a & y).sum()); fa = int((a & ~y).sum()); miss = int((~a & y).sum())
        lin.append(dict(horizonte=h, entradas=conj, modelo=nome, ponto=ponto, frac_tempo_com_alerta=round(a.mean(), 4),
                        POD=round(hit / max(1, hit + miss), 3), FAR=round(fa / max(1, hit + fa), 3),
                        CSI=round(hit / max(1, hit + fa + miss), 3)))
    for f in FRACOES:
        thr = np.quantile(s, 1 - f)
        cat(s > thr if (s > thr).mean() > 0 else s >= thr, f"alerta em {int(f * 100)}% do tempo")
    if not nome.startswith("persist"):
        for lim in LIMIARES:
            cat(s >= lim, f"probabilidade >= {lim:.2f}")
    return lin


def main():
    configurar_log()
    t00 = time.time()
    cache = mt.cache_som()
    A = at.carregar()
    PF, TR, PAR = [], [], []
    for h in (30, 60):
        E = at.conjunto(A, h)
        yfull = cache.A[f"estado_prox{at.sufixo(h)}"].to_numpy() >= 3
        temp = cache.A.fold.to_numpy()
        for conj, chuva in (("sem chuva", "nenhuma"), ("anéis e área central", "colar_nucleo")):
            arq = cfg.ROOT / "etapa_a" / f"escores_h{h}_{chuva}.csv.gz"
            if not arq.exists():
                log.info(f"sem {arq.name}; pulado"); continue
            Sc = pd.read_csv(arq, sep=";", parse_dates=["ts"])
            Sc = E[["ts", "pos"]].merge(Sc, on="ts", how="left")
            n = len(E); y = E.forte.to_numpy(); fE = E.fold.to_numpy(); pE = E.pos.to_numpy()
            novos = {k: np.full(n, np.nan) for k in ("SOM, região agora", "SOM, região 30 min antes e região agora",
                                                     "SOM, nó agora e região 30 min antes")}
            L = ea.latentes(chuva, h, cache)
            for k, R in L.items():
                D = dobra(R, yfull, temp, cache.jan, len(cache.A))
                j = D["posmap"][pE]; te = (fE == k) & (j >= D["ntr"])
                jt = j[te] - D["ntr"]
                novos["SOM, região agora"][te] = D["p_reg"][jt]
                novos["SOM, região 30 min antes e região agora"][te] = D["p_par"][jt]
                novos["SOM, nó agora e região 30 min antes"][te] = D["p_nr"][jt]
                yi, ri, rp = D["y"][D["ite"]][jt], D["rg"][D["ite"]][jt], D["rgp"][D["ite"]][jt]
                TR.append(pd.DataFrame(dict(horizonte=h, entradas=conj, temporada=k, regiao_30min_antes=rp,
                                            regiao_agora=ri, forte=yi)))
                PAR.append(dict(horizonte=h, entradas=conj, temporada=k, alfa_par_regioes=D["al_par"],
                                alfa_no_e_regiao=D["al_nr"],
                                taxa_forte_regioes_1_a_8=" ".join(f"{t:.3f}" for t in D["taxa_regiao"])))
            log.info(f"h{h} {conj}: regiões prontas")
            usadas = np.isfinite(Sc["gradient boosting, variáveis novas (referência)"].to_numpy(float))
            modelos = [c for c in Sc.columns if c not in ("ts", "pos", "fold", "forte")]
            for nome in modelos:
                PF += pod_far(y[usadas], Sc[nome].to_numpy(float)[usadas], nome, h, conj)
            for nome, s in novos.items():
                PF += pod_far(y[usadas], s[usadas], nome, h, conj)
    at.salvar("R_parametros_por_temporada", pd.DataFrame(PAR))
    T = pd.concat(TR)
    T = T[T.regiao_30min_antes > 0]
    tab = T.groupby(["horizonte", "entradas", "regiao_30min_antes", "regiao_agora"]).forte \
        .agg(horarios="size", seguidos_de_chuva_forte="sum").reset_index()
    tab["frac_chuva_forte"] = (tab.seguidos_de_chuva_forte / tab.horarios).round(4)
    at.salvar("R_transicoes_entre_regioes", tab, max_linhas=300)
    PF = pd.DataFrame(PF)
    at.salvar("R_pod_far_por_limiar", PF, mostrar=False)
    principais = ["SOM, probabilidade do nó", "SOM, probabilidade suave", "SOM, região agora",
                  "SOM, região 30 min antes e região agora", "SOM, nó agora e região 30 min antes",
                  "SOM, trajetória (nó agora e 30 min antes)", "mistura de especialistas guiada pelo SOM",
                  "gradient boosting, variáveis novas (referência)", "persistência"]
    for (h, conj), g in PF[PF.modelo.isin(principais)].groupby(["horizonte", "entradas"]):
        w = g.pivot_table(index="modelo", columns="ponto", values=["POD", "FAR"]).round(2)
        w = w.reindex([m for m in principais if m in w.index])
        log.info(f"\n### POD e FAR, h{h}, {conj}\n{w.to_string()}\n")
    log.info(f"fim em {(time.time() - t00) / 60:.1f} min; saídas em {OUT}")


if __name__ == "__main__":
    main()
