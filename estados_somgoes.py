#!/usr/bin/env python
"""
estados_somgoes.py — estados atmosféricos e transições a partir do modelo escolhido.

Para cada cenário (entrada de chuva x horizonte) definido em cfg.ESTADOS_CENARIOS:
  1. treina o modelo final (perda e grade de cfg.ESTADOS_PERDA / cfg.ESTADOS_GRADE)
     em todas as temporadas; 10% dos horários ficam só para o critério de parada.
     O desempenho preditivo honesto é o da validação cruzada (train_somgoes.py);
     este modelo serve para DESCREVER os estados.
  2. agrupa as células em regiões contíguas (agrupamento hierárquico de Ward dos
     protótipos, com vizinhança da grade), para cada número em cfg.N_REGIOES
  3. grava, para o número padrão de regiões (cfg.N_REGIOES_PADRAO):
       celulas.csv              célula, linha, coluna, região, contagens e taxas por
                                transição, médias das variáveis de entrada
       regioes.csv              o mesmo por região, com nome descritivo, permanência
                                média e probabilidade de chuva forte na próxima janela
       transicao_10min.csv      probabilidade região -> região em 10 min
       transicao_horizonte.csv  probabilidade região -> região em h minutos
       perfil_regioes.csv       médias padronizadas das variáveis por região
       perfil_mudanca.csv       para cada região, o que distingue os horários que vão
                                para cada destino dos que permanecem (diferença padronizada)
       temporadas.csv           taxa de chuva forte por região em cada temporada
       ciclo_diurno.csv         ocorrência de cada região por hora do dia
       goes_composto.csv        campo GOES médio por região (8 canais, 32x32)
       regioes_N.csv            célula -> região para cada número de regiões testado

Uso:  python estados_somgoes.py            (todos os cenários)
      python estados_somgoes.py --rapido   (poucas épocas, só para conferir)
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd
import torch
from scipy.sparse import lil_matrix
from sklearn.cluster import AgglomerativeClustering

import config_somgoes as cfg
import somgoes_nucleo as nu

log = logging.getLogger("estados")
NOMES_AGORA = {0: "seco", 1: "chuva fraca/moderada", 2: "chuva forte"}


def setup_log():
    (cfg.RESULTS_DIR / "estados").mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(name)s %(message)s", "%Y-%m-%d %H:%M:%S")
    r = logging.getLogger(); r.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(cfg.RESULTS_DIR / "estados" / "estados.log", mode="a")):
        h.setFormatter(fmt); r.addHandler(h)
    r.setLevel(logging.INFO)


def conectividade(H, W):
    C = lil_matrix((H * W, H * W))
    for i in range(H):
        for j in range(W):
            k = i * W + j
            for di, dj in ((1, 0), (0, 1)):
                ii, jj = i + di, j + dj
                if ii < H and jj < W:
                    C[k, ii * W + jj] = C[ii * W + jj, k] = 1
    return C.tocsr()


def variaveis(cache):
    em = [c for c in cache.A.columns if c.startswith("em_")]
    goes = [c for c in cache.meta["goes_stat_cols"] if c.endswith(("nucleo_min", "nucleo_max", "media"))]
    ch = cache.cols_chuva["colar_nucleo"]
    return em + goes + ch + ["hora_sin", "hora_cos"]


def rodar(cache, chuva, h, prm):
    out = cfg.RESULTS_DIR / "estados" / f"{chuva}_h{h}"
    out.mkdir(parents=True, exist_ok=True)
    log.info(f"=== cenário chuva={chuva} horizonte={h} min ===")
    val = np.where(cache._val(h))[0]
    if prm["max"]:
        val = val[:prm["max"]]
    rng = np.random.default_rng(cfg.SEED)
    para = rng.random(len(val)) < 0.10
    tr, va = val[~para], val[para]
    norm = cache.normalizador(tr)
    g = cfg.ESTADOS_GRADE
    torch.manual_seed(cfg.SEED)
    model = nu.SOMGoes(cache, cfg.GOES_CENARIOS, "ae", g, chuva=chuva)
    nu.treinar(model, cache, tr, va, norm, cfg.ESTADOS_PERDA, prm["epocas"], prm["lotes"], out / "modelo", h)
    a = nu.bmus(model, cache, val, norm)
    y = cache.rotulos(val, h)
    K = g[0] * g[1]

    # ── regiões ─────────────────────────────────────────────────────────────
    P = model.prot.detach().cpu().numpy()
    conn = conectividade(*g)
    regs = {}
    for n in cfg.N_REGIOES:
        regs[n] = AgglomerativeClustering(n_clusters=n, linkage="ward", connectivity=conn).fit(P).labels_
        pd.DataFrame({"celula": np.arange(K), "linha": np.arange(K) // g[1], "coluna": np.arange(K) % g[1],
                      "regiao": regs[n]}).to_csv(out / f"regioes_{n}.csv", sep=";", index=False)
    reg_cel = regs[cfg.N_REGIOES_PADRAO]

    A = cache.A.iloc[val].reset_index(drop=True)
    A["celula"] = a
    A["regiao_bruta"] = reg_cel[a]
    A["agora3"] = nu._g3(y["agora"]); A["prox3"] = nu._g3(y["prox"])
    A["transicao_h"] = y["transicao"]
    A["forte_prox"] = (y["prox"] >= 3).astype(int)
    A["hora"] = A.ts.dt.hour
    # renumera regiões pela taxa de chuva forte na próxima janela (1 = menor)
    ordem = A.groupby("regiao_bruta").forte_prox.mean().sort_values().index
    ren = {r: i + 1 for i, r in enumerate(ordem)}
    A["regiao"] = A.regiao_bruta.map(ren)
    reg_cel_ren = np.array([ren.get(r, -1) for r in reg_cel])

    VARS = variaveis(cache)
    trans = [f"n_{t}" for t in cfg.TRANSICOES]
    for i, t in enumerate(cfg.TRANSICOES):
        A[f"n_{t}"] = (A.transicao_h == i).astype(int)

    def tabela(chave):
        gg = A.groupby(chave)
        T = gg[trans + ["forte_prox"]].sum()
        T.insert(0, "n", gg.size())
        T["frac_tempo"] = (T.n / len(A)).round(5)
        T["taxa_forte_prox"] = (T.forte_prox / T.n).round(4)
        T["taxa_chuva_prox"] = gg.prox3.apply(lambda s: (s >= 1).mean()).round(4)
        for k, nm in NOMES_AGORA.items():
            T[f"frac_agora_{nm.split('/')[0].replace(' ', '_')}"] = gg.agora3.apply(lambda s: (s == k).mean()).round(4)
        T["transicao_dominante"] = T[trans].idxmax(1).str[2:]
        T["hora_media"] = gg.apply(lambda d: (np.degrees(np.arctan2(d.hora_sin.mean(), d.hora_cos.mean())) / 15) % 24).round(1)
        T = T.join(gg[VARS].mean().round(4))
        return T

    C = tabela("celula").reset_index()
    C.insert(1, "linha", C.celula // g[1]); C.insert(2, "coluna", C.celula % g[1])
    C.insert(3, "regiao", reg_cel_ren[C.celula])
    C.to_csv(out / "celulas.csv", sep=";", index=False)

    R = tabela("regiao").reset_index()
    # permanência: comprimento médio das sequências contínuas na mesma região
    fut1 = cache.futuro(10)[val]
    pos = pd.Series(np.arange(len(val)), index=val)
    j1 = pos.reindex(fut1).to_numpy()
    ok1 = ~np.isnan(j1)
    reg = A.regiao.to_numpy()
    fica = np.zeros(len(val), bool)
    fica[ok1] = reg[ok1] == reg[j1[ok1].astype(int)]
    R["prob_permanecer_10min"] = R.regiao.map(pd.Series(fica[ok1]).groupby(reg[ok1]).mean()).round(4)
    R["permanencia_media_min"] = (10 / (1 - R.prob_permanecer_10min.clip(upper=0.999))).round(1)
    R["nome"] = [f"R{int(r.regiao)} {NOMES_AGORA[int(np.argmax([r.frac_agora_seco, r.frac_agora_chuva_fraca, r.frac_agora_chuva_forte]))]}"
                 f", {100*r.taxa_forte_prox:.0f}% forte em {h} min" for r in R.itertuples()]
    R.to_csv(out / "regioes.csv", sep=";", index=False)
    log.info("REGIÕES\n" + R[["regiao", "nome", "n", "frac_tempo", "taxa_forte_prox", "taxa_chuva_prox",
                             "permanencia_media_min", "transicao_dominante"]].to_string(index=False))

    # matrizes de transição entre regiões
    for nome, f in (("10min", fut1), ("horizonte", cache.futuro(h)[val])):
        j = pos.reindex(f).to_numpy(); ok = ~np.isnan(j)
        M = pd.crosstab(pd.Series(reg[ok], name="de"), pd.Series(reg[j[ok].astype(int)], name="para"))
        (M.div(M.sum(1), axis=0).round(4)).to_csv(out / f"transicao_{nome}.csv", sep=";")
        M.to_csv(out / f"transicao_{nome}_contagem.csv", sep=";")

    # perfis padronizados
    Z = A[VARS]
    mu, sd = Z.mean(), Z.std().replace(0, 1)
    ((A.groupby("regiao")[VARS].mean() - mu) / sd).round(3).to_csv(out / "perfil_regioes.csv", sep=";")

    # o que distingue quem muda de região de quem permanece (horizonte h)
    jh = pos.reindex(cache.futuro(h)[val]).to_numpy(); okh = ~np.isnan(jh)
    dest = np.full(len(val), -1); dest[okh] = reg[jh[okh].astype(int)]
    linhas = []
    for r in np.unique(reg):
        m = (reg == r) & (dest >= 0)
        if m.sum() < 50:
            continue
        sr = Z[m].std().replace(0, 1)
        fica_ = m & (dest == r)
        for d in np.unique(dest[m]):
            if d == r:
                continue
            vai = m & (dest == d)
            if vai.sum() < 30 or fica_.sum() < 30:
                continue
            dif = (Z[vai].mean() - Z[fica_].mean()) / sr
            for v, x in dif.items():
                linhas.append(dict(regiao=r, destino=d, n_mudam=int(vai.sum()), n_permanecem=int(fica_.sum()),
                                   variavel=v, dif_padronizada=round(float(x), 3),
                                   media_mudam=round(float(Z[vai][v].mean()), 4),
                                   media_permanecem=round(float(Z[fica_][v].mean()), 4)))
    pd.DataFrame(linhas).to_csv(out / "perfil_mudanca.csv", sep=";", index=False)

    A.groupby(["regiao", "temporada"]).agg(n=("forte_prox", "size"), taxa_forte_prox=("forte_prox", "mean"))\
        .round(4).reset_index().to_csv(out / "temporadas.csv", sep=";", index=False)
    pd.crosstab(A.regiao, A.hora, normalize="columns").round(4).to_csv(out / "ciclo_diurno.csv", sep=";")

    # campo GOES médio por região (até 3000 horários por região)
    linhas = []
    for r, d in A.groupby("regiao"):
        ix = np.sort(val[d.sample(min(3000, len(d)), random_state=0).index.to_numpy()])
        F = cache.campo[ix].astype(np.float32).mean(0)              # (C, R, R)
        Cn, Rr, _ = F.shape
        ii, jj = np.meshgrid(np.arange(Rr), np.arange(Rr), indexing="ij")
        for c in range(Cn):
            linhas.append(pd.DataFrame({"regiao": r, "canal": c, "i": ii.ravel(), "j": jj.ravel(),
                                        "valor": F[c].ravel().round(3)}))
    pd.concat(linhas).to_csv(out / "goes_composto.csv", sep=";", index=False)
    log.info(f"cenário {chuva} h{h} concluído -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    a = ap.parse_args()
    setup_log()
    prm = dict(epocas=cfg.EPOCAS, lotes=cfg.LOTES_EPOCA, max=None) if not a.rapido else \
        dict(epocas=cfg.RAPIDO["epocas"], lotes=cfg.RAPIDO["lotes_epoca"], max=cfg.RAPIDO["max_amostras"])
    cache = nu.Cache()
    for chuva, h in cfg.ESTADOS_CENARIOS:
        rodar(cache, chuva, h, prm)


if __name__ == "__main__":
    main()
