#!/usr/bin/env python
"""
modelos_tese.py — comparação de classificadores e teste em dois estágios com o SOM.

  B1  HistGradientBoosting, Random Forest e CatBoost com as mesmas entradas resumidas
      (estatísticas das EMs, do GOES, da chuva nos anéis e na área central), nos três
      conjuntos de entrada (sem chuva, só anéis, anéis e área central), a 30 e 60 min
  B2  dois estágios com o SOM já treinado de cada temporada:
        a) o classificador recebe também o nó do SOM, a frequência de chuva forte do nó
           e a região do mapa
        b) um classificador por região do SOM (com o global como reserva)
  B4  variáveis de iniciação calculadas pixel a pixel no campo do GOES (resfriamento do topo
      em 10 e 30 min, topos frios e RRQPE num raio de 20 e 40 km), para testar se os inícios
      locais, sem chuva no entorno, passam a ser antecipados
  B3  avaliação por evento (inícios de chuva forte e alarmes falsos) do melhor modelo,
      com as mesmas regras do analise_tese.py

Validação: uma temporada fora por vez, igual ao SOM. Usa o cache, os modelos do SOM em
resultados_somgoes/cenarios e as previsões fora da temporada da síntese.

Saídas em modelos_tese/ e um log mestre modelos_tese/modelos_tese.log (é o que deve ser enviado).
Uso:   python -m tese.modelos_tese              (tudo; 1 a 2 h)
       RAPIDO=1 python -m tese.modelos_tese     (sem Random Forest e com menos árvores)
CatBoost: pip install catboost  (se não estiver instalado, é pulado e o log avisa)
"""
from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from core import config_somgoes as cfg
from tese import analise_tese as at

OUT = cfg.ROOT / "modelos_tese"
RAPIDO = os.environ.get("RAPIDO", "0") == "1"
N_REGIOES = getattr(cfg, "N_REGIOES_PADRAO", 8)
log = logging.getLogger("modelos")

try:
    from catboost import CatBoostClassifier
except Exception:                       # noqa: BLE001
    CatBoostClassifier = None


def configurar_log():
    OUT.mkdir(parents=True, exist_ok=True)
    at.OUT = OUT                         # reaproveita salvar() do analise_tese
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    r = logging.getLogger(); r.handlers.clear()
    for hd in (logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "modelos_tese.log", mode="w")):
        hd.setFormatter(fmt); r.addHandler(hd)
    r.setLevel(logging.INFO)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 400)


# ═══════════════════════════════ classificadores ═════════════════════════════
def novo(nome, n_pos, n):
    if nome == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_iter=150 if RAPIDO else 400, learning_rate=0.06,
                                              max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
                                              early_stopping=True, validation_fraction=0.1,
                                              n_iter_no_change=20, random_state=cfg.SEED)
    if nome == "RandomForest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                                      n_jobs=-1, random_state=cfg.SEED)
    if nome == "CatBoost":
        return CatBoostClassifier(iterations=300 if RAPIDO else 800, depth=6, learning_rate=0.05,
                                  loss_function="Logloss", random_seed=cfg.SEED, verbose=False,
                                  thread_count=-1, allow_writing_files=False)
    raise ValueError(nome)


def ajustar(nome, Xtr, ytr, Xte, cat_idx=None):
    m = novo(nome, int(ytr.sum()), len(ytr))
    if nome == "CatBoost":
        Xtr = pd.DataFrame(Xtr); Xte = pd.DataFrame(Xte)
        if cat_idx:
            for j in cat_idx:
                Xtr[j] = Xtr[j].fillna(-1).astype(int); Xte[j] = Xte[j].fillna(-1).astype(int)
        m.fit(Xtr, ytr, cat_features=cat_idx or None)
    elif nome == "HistGradientBoosting" and cat_idx:
        m.set_params(categorical_features=cat_idx)
        Xtr = Xtr.copy(); Xte = Xte.copy()
        for j in cat_idx:
            Xtr[:, j] = np.nan_to_num(Xtr[:, j], nan=-1) + 1; Xte[:, j] = np.nan_to_num(Xte[:, j], nan=-1) + 1
        m.fit(Xtr, ytr)
    else:
        m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def modelos_disponiveis():
    ms = ["HistGradientBoosting"]
    if not RAPIDO:
        ms.append("RandomForest")
    if CatBoostClassifier is not None:
        ms.append("CatBoost")
    else:
        log.info("CatBoost não instalado (pip install catboost); seguindo sem ele")
    return ms


def loso(E, X, nome, cat_idx=None, sel=None):
    y = E.forte.to_numpy(); f = E.fold.to_numpy()
    s = np.full(len(E), np.nan)
    sel = np.ones(len(E), bool) if sel is None else sel
    for k in sorted(np.unique(f)):
        tr = sel & (f != k); te = sel & (f == k)
        if y[tr].sum() < 10 or te.sum() == 0:
            continue
        t0 = time.time()
        s[te] = ajustar(nome, X[tr], y[tr], X[te], cat_idx)
        log.info(f"    {nome}: temporada {k} pronta em {time.time() - t0:.0f} s")
    return s


# ═══════════════════════════════ nós do SOM fora da temporada ════════════════
def nos_som(chuva, h):
    """Para cada temporada, carrega o SOM treinado sem ela e atribui nó, frequência do nó
    e região a todos os horários (treino e teste daquele fold)."""
    import torch
    from core import somgoes_nucleo as nu
    from analise import sintese_somgoes as si
    cache = cache_som()
    base = cfg.RESULTS_DIR / "cenarios" / f"cenarios_{chuva}_h{h}_{si.PERDA_OP}_10x10"
    folds = sorted(int(p.name[4:]) for p in base.glob("fold*") if (p / "melhor.pt").exists())
    out = {}
    for fold in folds:
        tr, va = cache.split(fold, None, h)
        norm = cache.normalizador(tr)
        m = nu.SOMGoes(cache, cfg.GOES_CENARIOS, "ae", (10, 10), chuva=chuva)
        m.load_state_dict(torch.load(base / f"fold{fold}" / "melhor.pt", map_location=nu.DEV,
                                     weights_only=False)["model"]); m.to(nu.DEV)
        a_tr = si.bmus_mod(m, cache, tr, norm); a_va = si.bmus_mod(m, cache, va, norm)
        f_tr = cache.rotulos(tr, h)["prox"] >= 3
        n = np.bincount(a_tr, minlength=100)
        tx = (np.bincount(a_tr[f_tr], minlength=100) + 5 * f_tr.mean()) / (n + 5)
        P = m.prot.detach().cpu().numpy()
        reg = fcluster(linkage(P, "ward"), N_REGIOES, "maxclust")
        idx = np.r_[tr, va]; a = np.r_[a_tr, a_va]
        out[fold] = pd.DataFrame({"ts": cache.A.ts.to_numpy()[idx], "no_som": a, "tx_no_som": tx[a],
                                  "regiao_som": reg[a], "treino": np.r_[np.ones(len(tr), bool), np.zeros(len(va), bool)]})
        log.info(f"  SOM {chuva} h{h}: temporada {fold}, {len(tr):,} horários de treino e {len(va):,} de teste")
    return out



def dois_estagios(E, X, nos):
    s_a = np.full(len(E), np.nan); s_b = np.full(len(E), np.nan)
    f = E.fold.to_numpy(); y = E.forte.to_numpy()
    for k, D in nos.items():
        J = E[["ts"]].merge(D, on="ts", how="left")
        Xs = np.column_stack([X, J.no_som.to_numpy(float), J.tx_no_som.to_numpy(float), J.regiao_som.to_numpy(float)])
        tr = (f != k) & np.isfinite(Xs[:, -1]); te = (f == k) & np.isfinite(Xs[:, -1])
        if y[tr].sum() < 10 or te.sum() == 0:
            continue
        cat = [Xs.shape[1] - 3, Xs.shape[1] - 1]
        s_a[te] = ajustar("HistGradientBoosting", Xs[tr], y[tr], Xs[te], cat)
        reg = J.regiao_som.to_numpy()
        s_b[te] = ajustar("HistGradientBoosting", X[tr], y[tr], X[te])
        for r in np.unique(reg[te]):
            trr = tr & (reg == r); ter = te & (reg == r)
            if y[trr].sum() >= 20 and (~y[trr]).sum() >= 20:
                s_b[ter] = ajustar("HistGradientBoosting", X[trr], y[trr], X[ter])
        log.info(f"    dois estágios: temporada {k} pronta")
    return s_a, s_b


# ═══════════════════════════════ variáveis de iniciação por pixel (B4) ═══════
_CACHE = None


def cache_som():
    global _CACHE
    if _CACHE is None:
        from core import somgoes_nucleo as nu
        _CACHE = nu.Cache()
    return _CACHE


def variaveis_ic(lote=4000):
    """Variáveis de iniciação calculadas pixel a pixel no campo do GOES (32 x 32 células de 4 km,
    centrado em BH), no estilo dos campos de interesse de Mecikalski e Bedka (2006): resfriamento
    do canal C13 em 10 e 30 min, fração de células com topo frio, mínimo do C13 e células com
    RRQPE positiva, num raio de 20 e de 40 km. Radiância menor = topo mais frio."""
    cache = cache_som()
    G = cache.campo; J = cache.jan
    N, C, R, _ = G.shape
    yy, xx = np.mgrid[0:R, 0:R]
    d = np.hypot(xx + .5 - R / 2, yy + .5 - R / 2) * (128.0 / R)
    mascaras = {20: d <= 20, 40: d <= 40}
    i13, i08, irr = 5, 0, 7
    amostra = np.asarray(G[np.linspace(0, N - 1, 3000).astype(int), i13], np.float32)
    frio = np.nanpercentile(amostra[amostra > 0], 5)
    cols = {}
    for r in mascaras:
        for nome in ("c13_min", "c13_frac_frio", "resfr10_min", "resfr30_min", "resfr30_p10",
                     "c08_resfr30_min", "rrqpe_frac", "rrqpe_max"):
            cols[f"ic{r}_{nome}"] = np.full(N, np.nan, np.float32)
    t10 = J[:, -2]; t30 = J[:, -4]
    for a in range(0, N, lote):
        b = min(N, a + lote); ix = np.arange(a, b)
        g0 = np.asarray(G[ix], np.float32)
        g10 = np.asarray(G[np.clip(t10[ix], 0, None)], np.float32)
        g30 = np.asarray(G[np.clip(t30[ix], 0, None)], np.float32)
        # quadros sem GOES válido foram gravados como zero no cache
        v0 = g0[:, i13].max((1, 2)) > 0
        ok10 = (t10[ix] >= 0) & v0 & (g10[:, i13].max((1, 2)) > 0)
        ok30 = (t30[ix] >= 0) & v0 & (g30[:, i13].max((1, 2)) > 0)
        for r, m in mascaras.items():
            c13 = g0[:, i13][:, m]
            r10 = (g0[:, i13] - g10[:, i13])[:, m]; r30 = (g0[:, i13] - g30[:, i13])[:, m]
            w30 = (g0[:, i08] - g30[:, i08])[:, m]; rr = g0[:, irr][:, m]
            cols[f"ic{r}_c13_min"][ix] = c13.min(1)
            cols[f"ic{r}_c13_frac_frio"][ix] = (c13 < frio).mean(1)
            cols[f"ic{r}_resfr10_min"][ix] = np.where(ok10, r10.min(1), np.nan)
            cols[f"ic{r}_resfr30_min"][ix] = np.where(ok30, r30.min(1), np.nan)
            cols[f"ic{r}_resfr30_p10"][ix] = np.where(ok30, np.percentile(r30, 10, axis=1), np.nan)
            cols[f"ic{r}_c08_resfr30_min"][ix] = np.where(ok30, w30.min(1), np.nan)
            cols[f"ic{r}_rrqpe_frac"][ix] = (rr > 0).mean(1)
            cols[f"ic{r}_rrqpe_max"][ix] = rr.max(1)
    log.info(f"variáveis de iniciação por pixel: {len(cols)} colunas; limiar de topo frio {frio:.2f}")
    return pd.DataFrame(cols)


def tipo_inicio_seco(E, ini, h):
    k = np.clip(ini.pos_tc.to_numpy() - h // 10, 0, None)
    c1 = np.nan_to_num(at.col(E, "in_c1_tot_max60"))[k]
    return np.where(c1 >= cfg.BASCULA, "com chuva no anel de 13 a 60 km", "sem chuva no anel (iniciação local)")

# ═══════════════════════════════ execução ════════════════════════════════════
CONJUNTOS = {"sem chuva": ("EMs", "EMs_tend", "GOES", "GOES_tend", "hora"),
             "anéis": ("EMs", "EMs_tend", "GOES", "GOES_tend", "hora", "colar1", "colar2"),
             "anéis e área central": ("EMs", "EMs_tend", "GOES", "GOES_tend", "hora", "colar1", "colar2", "nucleo")}
SOM_DO_CONJUNTO = {"sem chuva": ("nenhuma", "som_nenhuma"), "anéis": ("colar", "som_colar"),
                   "anéis e área central": ("colar_nucleo", "som_colar_nucleo")}


def avaliar(E, s, nome, conj, h, M, C, G):
    y = E.forte.to_numpy(); ok = np.isfinite(s)
    por = [at.metricas_continuas(y[ok & (E.fold == f).to_numpy()], s[ok & (E.fold == f).to_numpy()])
           for f in sorted(E.fold.unique())]
    P = pd.DataFrame([p for p in por if p])
    linha = dict(horizonte=h, entradas=conj, modelo=nome, n_temporadas=len(P))
    for k in P.columns:
        linha[k] = P[k].mean(); linha[f"{k}_dp"] = P[k].std()
    M.append(linha)
    for f in (0.02, 0.05, 0.10):
        thr = np.quantile(s[ok], 1 - f)
        C.append(dict(horizonte=h, entradas=conj, modelo=nome, ponto=f"{int(f*100)}% do tempo",
                      **at.categoricas(y[ok], s[ok] > thr if (s[ok] > thr).mean() > 0 else s[ok] >= thr)))
    for lim, nv in (() if nome.startswith("persist") else ((at.L_ALERTA, "≥0,10"), (at.L_PROVAVEL, "≥0,40"))):
        C.append(dict(horizonte=h, entradas=conj, modelo=nome, ponto=nv, **at.categoricas(y[ok], s[ok] >= lim)))
    gx, gy = at.curva_ganho(y[ok], s[ok])
    G.append(pd.DataFrame(dict(horizonte=h, entradas=conj, modelo=nome, frac_tempo=gx, recall=gy)))
    seco = (E.agora.to_numpy() == 0) & ok
    por = [at.metricas_continuas(y[seco & (E.fold == f).to_numpy()], s[seco & (E.fold == f).to_numpy()])
           for f in sorted(E.fold.unique())]
    P = pd.DataFrame([p for p in por if p])
    if len(P):
        M.append(dict(horizonte=h, entradas=conj, modelo=nome + " [só horários secos]", n_temporadas=len(P),
                      **P.mean().to_dict()))


def main():
    configurar_log()
    t00 = time.time()
    A = at.carregar(); B = at.blocos(A)
    try:
        IC = variaveis_ic()
        for c in IC.columns:
            A[c] = IC[c].to_numpy()[A.pos.to_numpy()]
        B["GOES_IC"] = list(IC.columns)
    except Exception as e:                    # noqa: BLE001
        log.info(f"variáveis de iniciação indisponíveis ({e})"); B["GOES_IC"] = []
    ms = modelos_disponiveis()
    M, C, G, EV = [], [], [], []
    for h in (30, 60):
        E = at.conjunto(A, h)
        for conj, blocos in CONJUNTOS.items():
            cols = [c for b in blocos for c in B[b]]
            X = E[cols].to_numpy(float)
            log.info(f"h{h} {conj}: {len(cols)} variáveis, {len(E):,} horários")
            ch, col_som = SOM_DO_CONJUNTO[conj]
            S = {"SOM profundo": E[col_som].to_numpy()}
            if conj == "anéis e área central":
                S["persistência"] = E.agora.to_numpy(float) + 1e-3 * np.nan_to_num(at.col(E, "in_nuc_max10"))
            for nome in ms:
                S[nome] = loso(E, X, nome)
            if B["GOES_IC"]:
                Xic = np.column_stack([X, E[B["GOES_IC"]].to_numpy(float)])
                S["HistGradientBoosting + iniciação por pixel do GOES"] = loso(E, Xic, "HistGradientBoosting")
            # dois estágios com o SOM do mesmo conjunto
            if conj in ("sem chuva", "anéis e área central"):
                try:
                    nos = nos_som(ch, h)
                except Exception as e:        # noqa: BLE001
                    log.info(f"  nós do SOM indisponíveis ({e}); dois estágios pulados"); nos = {}
                if nos:
                    s_a, s_b = dois_estagios(E, X, nos)
                    S["dois estágios a (SOM como variável)"] = s_a
                    S["dois estágios b (um modelo por região do SOM)"] = s_b
            for nome, s in S.items():
                avaliar(E, s, nome, conj, h, M, C, G)
            # avaliação por evento do melhor modelo de árvore e do SOM
            if conj in ("sem chuva", "anéis e área central"):
                ini = at.inicios(E, h)
                AP = {m["modelo"]: m.get("AP", 0) for m in M if m["horizonte"] == h and m["entradas"] == conj
                      and "[só" not in m["modelo"] and m["modelo"] not in ("SOM profundo", "persistência")}
                melhor = max(AP, key=AP.get) if AP else None
                tipos = tipo_inicio_seco(E, ini, h)
                for lin in at.resumo_inicios(E, ini, S, h, conj, tipos):
                    lin["grupo"] = f"inícios do seco, {lin['grupo']}"
                    EV.append(lin)
                for nome in [n for n in ("SOM profundo", melhor) if n]:
                    s = np.nan_to_num(S[nome], nan=-1)
                    c = at.classificar_inicio(E, ini, s, at.L_ALERTA, h)
                    d = pd.concat([ini.reset_index(drop=True), c], axis=1)
                    for grupo, g in (("todos", d), ("partiu do seco", d[d.partiu_do_seco])):
                        lin = dict(horizonte=h, entradas=conj, modelo=nome, grupo=grupo, inicios=len(g))
                        lin.update(g.categoria.value_counts(normalize=True).round(3).to_dict())
                        lin["antecedencia_limiar_mediana_min"] = g.antecedencia_limiar_min.median()
                        EV.append(lin)
    M = pd.DataFrame(M)
    at.salvar("B1_metricas_continuas", M.round(4))
    at.salvar("B1_pontos_de_operacao", pd.DataFrame(C).round(4))
    at.salvar("B1_curvas_de_ganho", pd.concat(G).round(4), mostrar=False)
    at.salvar("B3_inicios_por_modelo", pd.DataFrame(EV))
    resumo = M[~M.modelo.str.contains(r"\[só")].pivot_table(index=["entradas", "modelo"], columns="horizonte",
                                                             values="AP").round(3)
    log.info(f"\n### resumo AP por modelo\n{resumo.to_string()}\n")
    log.info(f"fim em {(time.time() - t00) / 60:.1f} min; saídas em {OUT}")


if __name__ == "__main__":
    main()
