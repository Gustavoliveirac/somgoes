#!/usr/bin/env python
"""
analise_tese.py — análises complementares do capítulo da tese (somgoes).

Usa o cache (cache_somgoes/) e as previsões fora da temporada já geradas pela síntese
(analise_somgoes/previsoes_fora_da_temporada.csv.gz). Não treina o SOM de novo.

  A1  frequência das transições de estado no núcleo (agora -> próximos 30/60 min),
      por temporada e por hora do dia
  A2  referências simples avaliadas no mesmo conjunto do SOM: regras de chuva presente
      no núcleo ("alerta sempre que chove"), chuva no entorno (colar 1, colar a barlavento),
      RRQPE do GOES sobre o núcleo, regressão logística e gradient boosting com as mesmas
      entradas; métricas AP, AUC ROC, recall e lift por fração do tempo, POD, FAR, CSI, HSS
  A3  inícios de chuva forte: quando o alerta dispara em relação ao início da chuva e ao
      momento em que a chuva passa do limiar; o que acontece nos inícios não previstos
  A4  anatomia dos episódios de alerta sem chuva forte (choveu? choveu no entorno?
      o alerta reagiu a chuva presente?)
  A5  fatores associados a cada transição (diferença padronizada e AUC univariada)
  A6  inícios a partir do seco: modelo específico para horários secos, importância por
      bloco de variáveis e separação entre iniciação local e chuva que chega do entorno

Saídas em analise_tese/ (CSVs) e um log mestre analise_tese/analise_tese.log com todas
as tabelas. Para revisão basta enviar o analise_tese.log.

Uso:  python -m tese.analise_tese            (tudo)
      python -m tese.analise_tese A1 A3      (só as partes indicadas)
"""
from __future__ import annotations

import logging
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from core import config_somgoes as cfg

OUT = cfg.ROOT / "analise_tese"
PREV = cfg.ROOT / "analise_somgoes" / "previsoes_fora_da_temporada.csv.gz"
HORIZONTES = (30, 60)
L_ATENCAO, L_ALERTA, L_PROVAVEL = (lim for lim, _ in cfg.NIVEIS_ALERTA[:3])   # 0.01, 0.10, 0.40
FRACS = (0.01, 0.02, 0.05, 0.10)
GOES_NOMES = getattr(cfg, "GOES_NOMES", ["C08", "C09", "C10", "C11", "C12", "C13", "C14", "RRQPE"])
ESTADOS = cfg.ESTADOS
RNG = np.random.default_rng(cfg.SEED)
import os
SEM_MODELOS = os.environ.get("SEM_MODELOS", "0") == "1"   # pula logística e gradient boosting (rodada rápida)
log = logging.getLogger("tese")


# ═══════════════════════════════ utilidades ══════════════════════════════════
def configurar_log():
    OUT.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    r = logging.getLogger(); r.handlers.clear()
    for hd in (logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "analise_tese.log", mode="w")):
        hd.setFormatter(fmt); r.addHandler(hd)
    r.setLevel(logging.INFO)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 400); pd.set_option("display.max_colwidth", 60)


def salvar(nome, df, mostrar=True, max_linhas=120):
    df.to_csv(OUT / f"{nome}.csv", sep=";", index=False)
    if mostrar:
        txt = df.head(max_linhas).to_string(index=False)
        log.info(f"\n### {nome} ({len(df)} linhas)\n{txt}\n")
    else:
        log.info(f"### {nome}: {len(df)} linhas (só no CSV)")


def sufixo(h):
    return "" if h == 60 else f"_{h}"


def recall_em(y, s, f):
    o = np.argsort(-s, kind="stable")
    k = max(1, int(round(f * len(s))))
    return y[o][:k].sum() / max(1, y.sum())


def tempo_para(y, s, alvo):
    o = np.argsort(-s, kind="stable")
    c = np.cumsum(y[o]) / max(1, y.sum())
    k = int(np.searchsorted(c, alvo))
    return (k + 1) / len(s)


def categoricas(y, a):
    y, a = y.astype(bool), a.astype(bool)
    h = int((a & y).sum()); f = int((a & ~y).sum()); m = int((~a & y).sum()); n = int((~a & ~y).sum())
    N = h + f + m + n
    pod = h / max(1, h + m); far = f / max(1, h + f); csi = h / max(1, h + f + m)
    pofd = f / max(1, f + n)
    esp = ((h + f) * (h + m) + (n + f) * (n + m)) / max(1, N)
    hss = (h + n - esp) / max(1e-9, N - esp)
    return dict(frac_tempo=a.mean(), POD=pod, FAR=far, CSI=csi, HSS=hss, PSS=pod - pofd)


def metricas_continuas(y, s):
    y = y.astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return {}
    d = dict(base=y.mean(), AP=average_precision_score(y, s), AUC=roc_auc_score(y, s))
    for f in FRACS:
        r = recall_em(y, s, f)
        d[f"recall_{int(f*100)}pct"] = r; d[f"lift_{int(f*100)}pct"] = r / f
    d["tempo_para_50pct"] = tempo_para(y, s, 0.5); d["tempo_para_80pct"] = tempo_para(y, s, 0.8)
    return d


def curva_ganho(y, s, grade=np.r_[0, np.linspace(0.0025, 0.30, 120)]):
    o = np.argsort(-s, kind="stable")
    c = np.r_[0, np.cumsum(y[o])] / max(1, y.sum())
    k = np.clip(np.round(grade * len(s)).astype(int), 0, len(s))
    return grade, c[k]


def episodios(ativo, cont, gap_max=3):
    """ativo: bool por horário (ordenado); cont: True se o horário segue o anterior em 10 min.
    Interrupções de até gap_max horários (30 min) não encerram o episódio.
    Retorna lista de (ini, fim) em posições."""
    ep, ini, ult = [], None, None
    for k in range(len(ativo)):
        if not cont[k] and ini is not None:
            ep.append((ini, ult)); ini = None
        if ativo[k]:
            if ini is None:
                ini = k
            elif k - ult > gap_max + 1:
                ep.append((ini, ult)); ini = k
            ult = k
    if ini is not None:
        ep.append((ini, ult))
    return ep


# ═══════════════════════════════ dados ═══════════════════════════════════════
def carregar():
    from core import somgoes_nucleo as nu
    cache = nu.Cache()
    A = cache.A.copy()
    A["pos"] = np.arange(len(A))
    A["hora"] = A.ts.dt.hour
    if "temporada" not in A:
        A["temporada"] = A.fold.map(dict(enumerate(cache.meta["temporadas"])))
    # tendências de 30 min (t0 menos t-30), com a janela do cache
    j30 = cache.jan[:, -4]
    ok = j30 >= 0
    for c in ["em_temperature", "em_umidity", "em_pression", "em_radiation",
              "g5_nucleo_min", "g5_min", "g5_desvio", "g5_media", "g3_nucleo_min", "g7_nucleo_max", "g7_max", "g7_media"]:
        if c in A:
            v = A[c].to_numpy(float)
            A[f"tend30_{c}"] = np.where(ok, v - v[np.clip(j30, 0, None)], np.nan)
    A["cont"] = np.r_[False, (A.ts.values[1:] - A.ts.values[:-1]) == np.timedelta64(10, "m")]
    # colar 1 a barlavento: quadrante de onde vem o vento médio das EMs
    if {"em_vento_sin", "em_vento_cos"} <= set(A.columns):
        ang = (np.degrees(np.arctan2(A.em_vento_sin, A.em_vento_cos)) + 360) % 360
        q = np.select([(ang >= 315) | (ang < 45), ang < 135, ang < 225], ["N", "L", "S"], "O")
        q = np.where(np.isfinite(ang), q, "")
        for v in ("max60", "max10", "frac60"):
            cols = {k: f"in_c1_{k}_{v}" for k in "NLSO"}
            if all(c in A for c in cols.values()):
                M = np.column_stack([A[cols[k]].to_numpy(float) for k in "NLSO"])
                idx = np.array([{"N": 0, "L": 1, "S": 2, "O": 3}.get(x, -1) for x in q])
                A[f"in_c1_barl_{v}"] = np.where(idx >= 0, M[np.arange(len(M)), np.clip(idx, 0, 3)], np.nan)
    log.info(f"cache: {len(A):,} horários; colunas {A.shape[1]}")
    return A


def blocos(A):
    em = [c for c in ["em_temperature", "em_umidity", "em_pression", "em_wind_speed", "em_radiation",
                      "em_vento_sin", "em_vento_cos"] if c in A]
    em_t = [c for c in A if c.startswith("tend30_em_")]
    goes = [c for c in A if c.startswith("g") and c[1:2].isdigit() and "_" in c and c.split("_", 1)[1] in
            ("media", "desvio", "min", "p10", "nucleo_min", "nucleo_max")]
    goes_t = [c for c in A if c.startswith("tend30_g")]
    c1 = [c for c in A if c.startswith("in_c1_")]
    c2 = [c for c in A if c.startswith("in_c2_")]
    nuc = [c for c in A if c.startswith("in_nuc_")]
    hora = [c for c in ("hora_sin", "hora_cos") if c in A]
    return dict(EMs=em, EMs_tend=em_t, GOES=goes, GOES_tend=goes_t, colar1=c1, colar2=c2, nucleo=nuc, hora=hora)


def nome_var(c):
    for k, n in enumerate(GOES_NOMES):
        c = c.replace(f"g{k}_", f"{n}_")
    return c


def conjunto(A, h):
    """Linhas do SOM fora da temporada (os três cenários de chuva) unidas às features."""
    P = pd.read_csv(PREV, sep=";", parse_dates=["ts"])
    P = P[P.horizonte == h]
    base = P[P.chuva == "colar_nucleo"][["ts", "fold", "forte", "agora", "transicao", "escore"]] \
        .rename(columns={"escore": "som_colar_nucleo"})
    for ch in ("colar", "nenhuma"):
        x = P[P.chuva == ch][["ts", "escore"]].rename(columns={"escore": f"som_{ch}"})
        base = base.merge(x, on="ts", how="left")
    A2_ = A.rename(columns={"agora": "intens_60", "agora_30": "intens_30", "prox": "intens_prox_60",
                            "prox_30": "intens_prox_30"})
    E = base.merge(A2_, on="ts", how="left", suffixes=("", "_A"))
    E["intens"] = E[f"intens_{h}"]
    E = E.sort_values("ts").reset_index(drop=True)
    E["forte"] = E.forte.astype(bool)
    E["cont"] = np.r_[False, (E.ts.values[1:] - E.ts.values[:-1]) == np.timedelta64(10, "m")]
    return E


# ═══════════════════════════════ A1 transições ═══════════════════════════════
def a1(A):
    log.info("A1 frequência das transições")
    L, T, H = [], [], []
    for h in HORIZONTES:
        sf = sufixo(h)
        v = A[A[f"alvo_ok{sf}"].astype(bool) & A.goes_ok.astype(bool)]
        a, p = v[f"estado_agora{sf}"].astype(int), v[f"estado_prox{sf}"].astype(int)
        ct = pd.crosstab(a, p)
        for i in ct.index:
            for j in ct.columns:
                L.append(dict(horizonte=h, agora=ESTADOS[i], depois=ESTADOS[j], n=int(ct.loc[i, j]),
                              frac_do_tempo=ct.loc[i, j] / ct.values.sum(),
                              prob_condicional=ct.loc[i, j] / ct.loc[i].sum()))
        tr = v[f"transicao{sf}"].astype(int).map(dict(enumerate(cfg.TRANSICOES)))
        t = pd.crosstab(v.temporada, tr)
        t["total"] = t.sum(1)
        t = t.reset_index(); t.insert(0, "horizonte", h); T.append(t)
        # por hora: probabilidade de mudança a partir do seco e da chuva fraca/moderada
        seco = a == 0; leve = a.isin([1, 2])
        g = pd.DataFrame({"hora": v.hora, "seco": seco, "leve": leve,
                          "seco_chuva": seco & (p >= 1), "seco_forte": seco & (p >= 3),
                          "leve_forte": leve & (p >= 3), "leve_seco": leve & (p == 0)})
        r = g.groupby("hora").sum()
        H.append(pd.DataFrame({"horizonte": h, "hora": r.index,
                               "p_seco_para_chuva": (r.seco_chuva / r.seco).round(4),
                               "p_seco_para_forte": (r.seco_forte / r.seco).round(5),
                               "p_fraca_moderada_para_forte": (r.leve_forte / r.leve.replace(0, np.nan)).round(4),
                               "p_fraca_moderada_para_seco": (r.leve_seco / r.leve.replace(0, np.nan)).round(4),
                               "n_seco": r.seco, "n_fraca_moderada": r.leve}))
    salvar("A1_matriz_transicoes", pd.DataFrame(L).round(5))
    salvar("A1_transicoes_por_temporada", pd.concat(T))
    salvar("A1_transicoes_por_hora", pd.concat(H))


# ═══════════════════════════════ A2 referências ══════════════════════════════
def ajustar_modelos(E, cols, nome, dry_only=False):
    """LOSO por temporada; devolve escore fora da temporada para cada linha de E."""
    s_log = np.full(len(E), np.nan); s_gb = np.full(len(E), np.nan)
    X = E[cols].to_numpy(float); y = E.forte.to_numpy()
    sel = np.ones(len(E), bool) if not dry_only else (E.agora.to_numpy() == 0)
    if SEM_MODELOS:
        log.info(f"  {nome}: modelos aprendidos pulados (SEM_MODELOS=1)")
        return s_log, s_gb
    for f in sorted(E.fold.unique()):
        t0 = time.time()
        tr = sel & (E.fold.to_numpy() != f); te = sel & (E.fold.to_numpy() == f)
        if y[tr].sum() < 10 or te.sum() == 0:
            continue
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                                            min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
                                            validation_fraction=0.1, n_iter_no_change=15, random_state=cfg.SEED)
        gb.fit(X[tr], y[tr]); s_gb[te] = gb.predict_proba(X[te])[:, 1]
        med = np.nanmedian(X[tr], 0); med[~np.isfinite(med)] = 0
        Xt = np.where(np.isfinite(X[tr]), X[tr], med); Xe = np.where(np.isfinite(X[te]), X[te], med)
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        lr = LogisticRegression(C=1.0, max_iter=2000)
        lr.fit((Xt - mu) / sd, y[tr]); s_log[te] = lr.predict_proba((Xe - mu) / sd)[:, 1]
        log.info(f"    {nome}: temporada {f} pronta em {time.time() - t0:.0f} s")
    log.info(f"  ajustados {nome}: {len(cols)} variáveis")
    return s_log, s_gb


def col(E, nome, padrao=np.nan):
    return E[nome].to_numpy(float) if nome in E else np.full(len(E), padrao)


def escores_referencia(E, B, h):
    S = {}
    S["SOM (colar e núcleo)"] = E.som_colar_nucleo.to_numpy()
    S["SOM (colar)"] = E.som_colar.to_numpy()
    S["SOM (sem chuva)"] = E.som_nenhuma.to_numpy()
    # chuva presente no núcleo: estado atual e intensidade dos últimos 10 min
    est = E.agora.to_numpy(float)
    S["persistência: estado atual do núcleo"] = est + 1e-3 * np.nan_to_num(col(E, "in_nuc_max10"))
    S["chuva no colar 1 (máx. 60 min)"] = np.nan_to_num(col(E, "in_c1_tot_max60"))
    if "in_c1_barl_max60" in E:
        S["chuva no colar 1 a barlavento"] = np.nan_to_num(E.in_c1_barl_max60.to_numpy(float))
    ent = np.nanmax(np.column_stack([np.nan_to_num(E[c].to_numpy(float)) for c in
                                     ("in_nuc_max60", "in_c1_tot_max60", "in_c2_tot_max60") if c in E]), 1)
    S["chuva máxima até 100 km"] = ent
    if "g7_nucleo_max" in E:
        S["RRQPE máximo sobre o núcleo"] = np.nan_to_num(E.g7_nucleo_max.to_numpy(float))
    sem = B["EMs"] + B["EMs_tend"] + B["GOES"] + B["GOES_tend"] + B["hora"]
    tudo = sem + B["colar1"] + B["colar2"] + B["nucleo"]
    lo, gb = ajustar_modelos(E, sem, f"sem chuva h{h}")
    S["logística (sem chuva)"], S["gradient boosting (sem chuva)"] = lo, gb
    lo, gb = ajustar_modelos(E, tudo, f"colar e núcleo h{h}")
    S["logística (colar e núcleo)"], S["gradient boosting (colar e núcleo)"] = lo, gb
    return S


REGRAS = [("chove no núcleo agora (≥0,2 mm/h)", 1), ("moderada no núcleo agora (≥2,5 mm/h)", 2),
          ("forte no núcleo agora (≥10 mm/h)", 3)]


def a2(A):
    log.info("A2 referências e métricas consagradas")
    B = blocos(A)
    M, C, G, K = [], [], [], {}
    for h in HORIZONTES:
        E = conjunto(A, h)
        S = escores_referencia(E, B, h)
        K[h] = (E, S)
        y = E.forte.to_numpy()
        for nome, s in S.items():
            ok = np.isfinite(s)
            por = [dict(fold=f, **metricas_continuas(y[ok & (E.fold == f).to_numpy()], s[ok & (E.fold == f).to_numpy()]))
                   for f in sorted(E.fold.unique())]
            por = pd.DataFrame(por)
            if por.empty or "AP" not in por:
                continue
            m = por.drop(columns="fold").agg(["mean", "std"]).T
            linha = dict(horizonte=h, escore=nome, n_temporadas=len(por))
            for k, r in m.iterrows():
                linha[k] = r["mean"]; linha[f"{k}_dp"] = r["std"]
            M.append(linha)
            gx, gy = curva_ganho(y[ok], s[ok])
            G.append(pd.DataFrame(dict(horizonte=h, escore=nome, frac_tempo=gx, recall=gy)))
            # pontos de operação: mesma fração do tempo e limiares de probabilidade
            for f in (0.02, 0.05, 0.10):
                thr = np.quantile(s[ok], 1 - f)
                C.append(dict(horizonte=h, escore=nome, ponto=f"{int(f*100)}% do tempo",
                              **categoricas(y[ok], s[ok] > thr if (s[ok] > thr).mean() > 0 else s[ok] >= thr)))
            if nome.startswith(("SOM", "logística", "gradient")):
                for lim, nv in ((L_ALERTA, "alerta ou acima (≥0,10)"), (L_PROVAVEL, "chuva forte provável (≥0,40)")):
                    C.append(dict(horizonte=h, escore=nome, ponto=nv, **categoricas(y[ok], s[ok] >= lim)))
        for nm, k in REGRAS:
            C.append(dict(horizonte=h, escore=nm, ponto="regra binária", **categoricas(y, E.agora.to_numpy() >= k)))
        for nm, col, lim in (("chuva forte no colar 1 (≥10 mm em 60 min)", "in_c1_tot_max60", 10.0),
                             ("chuva no colar 1 a barlavento (≥2,5 mm em 60 min)", "in_c1_barl_max60", 2.5)):
            if col in E:
                C.append(dict(horizonte=h, escore=nm, ponto="regra binária",
                              **categoricas(y, np.nan_to_num(E[col].to_numpy(float)) >= lim)))
    salvar("A2_metricas_continuas", pd.DataFrame(M).round(4))
    salvar("A2_pontos_de_operacao", pd.DataFrame(C).round(4))
    salvar("A2_curvas_de_ganho", pd.concat(G).round(4), mostrar=False)
    return K


# ═══════════════════════════════ A3 inícios ══════════════════════════════════
def inicios(E, h):
    """Inícios de chuva forte no núcleo dentro do conjunto avaliado.
    tc = horário em que a intensidade observada no núcleo (acumulado da janela do horizonte,
         em mm/h, máximo entre as estações) passa a >= 10 mm/h;
    tr = primeiro horário da sequência contínua com chuva registrada no núcleo (estado >= fraca)
         que termina em tc, ou seja, o início da chuva que virou forte."""
    est = E.agora.to_numpy()
    cont = E.cont.to_numpy()
    passo = h // 10
    out = []
    for i in range(1, len(E)):
        if not (est[i] >= 3 and est[i - 1] < 3 and cont[i]):
            continue
        j = i
        while j - 1 >= 0 and cont[j] and est[j - 1] >= 1 and i - j < 72:
            j -= 1
        k = i - passo
        ant = est[k] if k >= 0 and all(cont[k + 1:i + 1]) else np.nan
        out.append(dict(pos_tc=i, pos_tr=j, ts_limiar=E.ts.iat[i], ts_inicio_chuva=E.ts.iat[j],
                        fold=int(E.fold.iat[i]), estado_um_horizonte_antes=ant,
                        partiu_do_seco=bool(ant == 0), chuva_antes_do_limiar_min=10 * (i - j)))
    return pd.DataFrame(out)


def classificar_inicio(E, ini, s, lim, h, max_antes=18):
    cont = E.cont.to_numpy(); passo = h // 10
    lin = []
    for r in ini.itertuples():
        i, j = r.pos_tc, r.pos_tr
        # cadeia de alerta que chega a tc (tolera 30 min abaixo do limiar)
        ta, gap = None, 0
        for k in range(i, max(-1, i - max_antes - 1), -1):
            if k < 0 or (k < i and not cont[k + 1]):
                break
            if s[k] >= lim:
                ta, gap = k, 0
            else:
                gap += 1
                if gap > 3 and ta is not None:
                    break
                if gap > passo + 3 and ta is None:
                    break
        depois = any(s[k] >= lim for k in range(i, min(len(E), i + passo + 1)) if all(cont[i + 1:k + 1]))
        if ta is not None and ta < i:
            cat = "alerta antes do início da chuva" if ta < j else "alerta com chuva, antes do limiar"
        elif ta == i or depois:
            cat = "alerta só depois do limiar"
        else:
            cat = "sem alerta"
        lin.append(dict(categoria=cat, antecedencia_limiar_min=10 * (i - ta) if ta is not None and ta <= i else np.nan,
                        antecedencia_inicio_chuva_min=10 * (j - ta) if ta is not None and ta < j else np.nan,
                        escore_max_antes=float(np.nanmax(s[max(0, i - passo):i + 1]))))
    return pd.DataFrame(lin)


def a3(K):
    log.info("A3 inícios de chuva forte: antecedência em relação ao início da chuva e ao limiar")
    R, D = [], []
    for h, (E, S) in K.items():
        ini = inicios(E, h)
        if ini.empty:
            continue
        ctx = pd.DataFrame({
            "estado_nucleo_1h_antes": ini.estado_um_horizonte_antes,
            "colar1_max60_no_limiar_menos_h": [np.nan_to_num(E.in_c1_tot_max60.iat[max(0, p - h // 10)]) if "in_c1_tot_max60" in E else np.nan for p in ini.pos_tc],
            "colar1_barl_max60_no_inicio_chuva": [E.in_c1_barl_max60.iat[p] if "in_c1_barl_max60" in E else np.nan for p in ini.pos_tr],
            "rrqpe_nucleo_no_inicio_chuva": [E.g7_nucleo_max.iat[p] if "g7_nucleo_max" in E else np.nan for p in ini.pos_tr],
            "hora": [E.ts.iat[p].hour for p in ini.pos_tc]})
        base = pd.concat([ini, ctx], axis=1)
        for nome in ("SOM (colar e núcleo)", "SOM (sem chuva)", "gradient boosting (colar e núcleo)",
                     "persistência: estado atual do núcleo"):
            if nome not in S:
                continue
            s = np.nan_to_num(S[nome], nan=-1)
            limiares = [(L_ALERTA, "alerta (≥0,10)"), (L_PROVAVEL, "provável (≥0,40)")] if not nome.startswith("pers") \
                else [(1, "chove agora"), (3, "forte agora")]
            for lim, nv in limiares:
                c = classificar_inicio(E, ini, s, lim, h)
                d = pd.concat([base.reset_index(drop=True), c], axis=1)
                d.insert(0, "nivel", nv); d.insert(0, "escore", nome); d.insert(0, "horizonte", h)
                D.append(d)
                for grupo, g in (("todos", d), ("partiu do seco", d[d.partiu_do_seco]),
                                 ("já chovia no núcleo", d[~d.partiu_do_seco])):
                    if len(g) == 0:
                        continue
                    linha = dict(horizonte=h, escore=nome, nivel=nv, grupo=grupo, inicios=len(g))
                    for k, v in g.categoria.value_counts(normalize=True).items():
                        linha[k] = v
                    linha["antecedencia_limiar_mediana_min"] = g.antecedencia_limiar_min.median()
                    linha["antecedencia_inicio_chuva_mediana_min"] = g.antecedencia_inicio_chuva_min.median()
                    linha["chuva_antes_do_limiar_mediana_min"] = g.chuva_antes_do_limiar_min.median()
                    R.append(linha)
    D = pd.concat(D) if D else pd.DataFrame()
    salvar("A3_inicios_por_evento", D.drop(columns=["pos_tc", "pos_tr"], errors="ignore"), mostrar=False)
    R = pd.DataFrame(R)
    cols = ["horizonte", "escore", "nivel", "grupo", "inicios", "alerta antes do início da chuva",
            "alerta com chuva, antes do limiar", "alerta só depois do limiar", "sem alerta",
            "antecedencia_limiar_mediana_min", "antecedencia_inicio_chuva_mediana_min", "chuva_antes_do_limiar_mediana_min"]
    salvar("A3_inicios_resumo", R.reindex(columns=cols).fillna({c: 0 for c in cols[5:9]}).round(3))
    # inícios não previstos x previstos: contexto médio
    if len(D):
        X = D[D.escore.eq("SOM (colar e núcleo)") & D.nivel.str.startswith("alerta")]
        X = X.assign(previsto=X.categoria.str.startswith("alerta antes") | X.categoria.str.startswith("alerta com"))
        ctx = X.groupby(["horizonte", "partiu_do_seco", "previsto"])[
            ["colar1_max60_no_limiar_menos_h", "colar1_barl_max60_no_inicio_chuva", "rrqpe_nucleo_no_inicio_chuva",
             "chuva_antes_do_limiar_min", "hora"]].agg(["median", "count"]).round(2)
        ctx.columns = ["_".join(c) for c in ctx.columns]
        salvar("A3_contexto_previstos_x_nao_previstos", ctx.reset_index())
    return D


# ═══════════════════════════════ A4 falsos positivos ═════════════════════════
def a4(K):
    log.info("A4 episódios de alerta sem chuva forte")
    R, D = [], []
    for h, (E, S) in K.items():
        est = E.agora.to_numpy(); cont = E.cont.to_numpy(); passo = h // 10
        acum = np.nan_to_num(col(E, "intens"))
        c1 = np.nan_to_num(col(E, "in_c1_tot_max60"))
        r10 = np.nan_to_num(col(E, "in_nuc_max10"))
        for nome in ("SOM (colar e núcleo)", "SOM (sem chuva)", "gradient boosting (colar e núcleo)"):
            if nome not in S:
                continue
            s = np.nan_to_num(S[nome], nan=-1)
            for lim, nv in ((L_ALERTA, "alerta (≥0,10)"), (L_PROVAVEL, "provável (≥0,40)")):
                eps = episodios(s >= lim, cont)
                for a0, a1_ in eps:
                    fim = a1_
                    while fim + 1 < len(E) and cont[fim + 1] and fim < a1_ + passo:
                        fim += 1
                    W = slice(a0, fim + 1)
                    correto = bool((est[W] >= 3).any())
                    tarde = a1_ + passo
                    fim2 = tarde
                    while fim2 + 1 < len(E) and cont[fim2 + 1] and fim2 < tarde + 6:
                        fim2 += 1
                    if correto:
                        tipo = "correto"
                    elif fim2 > fim and (est[fim + 1:fim2 + 1] >= 3).any():
                        tipo = "forte no núcleo logo depois da janela"
                    elif (c1[W] >= 10).any():
                        tipo = "forte no colar 1, não no núcleo"
                    elif (acum[W] >= 5).any():
                        tipo = "quase forte no núcleo (5 a 10 mm/h)"
                    elif (est[W] >= 1).any():
                        tipo = "chuva fraca ou moderada no núcleo"
                    elif (c1[W] >= cfg.BASCULA).any():
                        tipo = "seco no núcleo, chuva no colar 1"
                    else:
                        tipo = "seco no núcleo e no colar 1"
                    D.append(dict(horizonte=h, escore=nome, nivel=nv, fold=int(E.fold.iat[a0]),
                                  inicio=E.ts.iat[a0], fim=E.ts.iat[a1_], duracao_min=10 * (a1_ - a0 + 1),
                                  escore_max=float(s[a0:a1_ + 1].max()), tipo=tipo,
                                  estado_nucleo_no_disparo=ESTADOS[int(est[a0])] if est[a0] >= 0 else "",
                                  chovia_no_nucleo_no_disparo=bool(est[a0] >= 1),
                                  colar1_max60_no_disparo=float(c1[a0]),
                                  max_10min_nucleo_na_janela=float(r10[W].max()),
                                  max_intensidade_nucleo_na_janela=float(acum[W].max()),
                                  hora=E.ts.iat[a0].hour))
    D = pd.DataFrame(D)
    salvar("A4_episodios_de_alerta", D, mostrar=False)
    if len(D):
        for (h, nome, nv), g in D.groupby(["horizonte", "escore", "nivel"]):
            f = g[g.tipo != "correto"]
            linha = dict(horizonte=h, escore=nome, nivel=nv, episodios=len(g), por_temporada=len(g) / g.fold.nunique(),
                         frac_falsos=len(f) / len(g),
                         falsos_choveu_no_nucleo=(f.max_10min_nucleo_na_janela >= cfg.BASCULA).mean() if len(f) else np.nan,
                         falsos_disparados_com_chuva_no_nucleo=f.chovia_no_nucleo_no_disparo.mean() if len(f) else np.nan,
                         corretos_disparados_com_chuva_no_nucleo=g[g.tipo == "correto"].chovia_no_nucleo_no_disparo.mean(),
                         duracao_mediana_falsos_min=f.duracao_min.median() if len(f) else np.nan)
            for k, v in f.tipo.value_counts(normalize=True).items():
                linha[f"falsos: {k}"] = v
            R.append(linha)
        salvar("A4_falsos_resumo", pd.DataFrame(R).round(3))
    return D


# ═══════════════════════════════ A5 fatores ══════════════════════════════════
def a5(A):
    log.info("A5 fatores associados às transições")
    B = blocos(A)
    feats = B["EMs"] + B["EMs_tend"] + B["GOES"] + B["GOES_tend"] + B["colar1"] + B["colar2"] + ["hora_sin", "hora_cos"]
    feats = [c for c in feats if c in A and not c.endswith(("_N_max10", "_L_max10", "_S_max10", "_O_max10"))]
    L = []
    for h in HORIZONTES:
        sf = sufixo(h)
        v = A[A[f"alvo_ok{sf}"].astype(bool) & A.goes_ok.astype(bool)]
        tr = v[f"transicao{sf}"].astype(int).map(dict(enumerate(cfg.TRANSICOES)))
        for alvo, ref, rotulo in (("inicio_forte", "SS", "seco -> forte  x  seco -> seco"),
                                  ("inicio", "SS", "seco -> fraca/moderada  x  seco -> seco"),
                                  ("intensifica", "CC", "fraca/moderada -> forte  x  fraca/moderada persiste"),
                                  ("termina", "CC", "fraca/moderada -> seco  x  fraca/moderada persiste")):
            a, b = v[tr == alvo], v[tr == ref]
            if len(a) < 20:
                continue
            for c in feats:
                xa, xb = a[c].to_numpy(float), b[c].to_numpy(float)
                xa, xb = xa[np.isfinite(xa)], xb[np.isfinite(xb)]
                if len(xa) < 20 or len(xb) < 20:
                    continue
                sd = np.sqrt((xa.var() + xb.var()) / 2) + 1e-12
                y = np.r_[np.ones(len(xa)), np.zeros(len(xb))]
                auc = roc_auc_score(y, np.r_[xa, xb])
                L.append(dict(horizonte=h, contraste=rotulo, variavel=nome_var(c), n_alvo=len(xa), n_ref=len(xb),
                              media_alvo=xa.mean(), media_ref=xb.mean(), diferenca_padronizada=(xa.mean() - xb.mean()) / sd,
                              auc_univariada=max(auc, 1 - auc), sentido="maior" if auc >= 0.5 else "menor"))
    D = pd.DataFrame(L)
    if len(D):
        D["ordem"] = D.groupby(["horizonte", "contraste"]).diferenca_padronizada.transform(lambda x: (-x.abs()).rank())
        salvar("A5_fatores_transicoes", D.sort_values(["horizonte", "contraste", "ordem"]).round(4), mostrar=False)
        salvar("A5_fatores_top10", D[D.ordem <= 10].sort_values(["horizonte", "contraste", "ordem"]).round(3))


# ═══════════════════════════════ A6 início a partir do seco ══════════════════
def a6(A, K):
    log.info("A6 inícios a partir do seco: modelo específico e importância por bloco")
    B = blocos(A)
    R, I, Cu = [], [], []
    for h, (E, S) in K.items():
        cols_bloco = {k: B[k] for k in ("EMs", "EMs_tend", "GOES", "GOES_tend", "colar1", "colar2", "hora") if B[k]}
        cols = sum(cols_bloco.values(), [])
        seco = E.agora.to_numpy() == 0
        X = E[cols].to_numpy(float); y = E.forte.to_numpy()
        s_seco = np.full(len(E), np.nan)
        imp = []
        for f in sorted(E.fold.unique()):
            trm = seco & (E.fold.to_numpy() != f); tem = seco & (E.fold.to_numpy() == f)
            if y[trm].sum() < 10 or y[tem].sum() == 0:
                continue
            t0 = time.time()
            gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
                                                min_samples_leaf=60, l2_regularization=1.0, early_stopping=True,
                                                validation_fraction=0.1, n_iter_no_change=15,
                                                random_state=cfg.SEED)
            gb.fit(X[trm], y[trm])
            pe = gb.predict_proba(X[tem])[:, 1]; s_seco[tem] = pe
            ref = average_precision_score(y[tem], pe)
            for bl, cc in cols_bloco.items():
                ix = [cols.index(c) for c in cc]
                quedas = []
                for rep in range(3):
                    Xp = X[tem].copy()
                    Xp[:, ix] = Xp[RNG.permutation(len(Xp))][:, ix]
                    quedas.append(ref - average_precision_score(y[tem], gb.predict_proba(Xp)[:, 1]))
                imp.append(dict(horizonte=h, fold=f, bloco=bl, queda_AP=np.mean(quedas)))
            log.info(f"    modelo do seco h{h}: temporada {f} pronta em {time.time() - t0:.0f} s")
        I.append(pd.DataFrame(imp))
        # avaliação só nos horários secos
        cand = {"modelo do seco (gradient boosting)": s_seco, **{k: S[k] for k in
                ("SOM (colar e núcleo)", "SOM (sem chuva)", "chuva no colar 1 (máx. 60 min)",
                 "chuva no colar 1 a barlavento", "RRQPE máximo sobre o núcleo",
                 "gradient boosting (colar e núcleo)") if k in S}}
        for nome, s in cand.items():
            ok = seco & np.isfinite(s)
            por = []
            for f in sorted(E.fold.unique()):
                m = ok & (E.fold.to_numpy() == f)
                if y[m].sum() > 0:
                    por.append(metricas_continuas(y[m], s[m]))
            if not por:
                continue
            P = pd.DataFrame(por)
            R.append(dict(horizonte=h, escore=nome, grupo="horários secos no núcleo", n=int(ok.sum()),
                          eventos=int(y[ok].sum()), **P.mean().to_dict()))
            gx, gy = curva_ganho(y[ok], s[ok])
            Cu.append(pd.DataFrame(dict(horizonte=h, escore=nome, frac_tempo=gx, recall=gy)))
        # híbrido: SOM quando chove, modelo do seco quando está seco (no conjunto todo)
        som = S["SOM (colar e núcleo)"]
        hib = np.where(seco & np.isfinite(s_seco), s_seco, som)
        for nome, s in (("SOM (colar e núcleo)", som), ("híbrido: SOM com chuva, modelo do seco sem chuva", hib)):
            P = pd.DataFrame([metricas_continuas(y[(E.fold == f).to_numpy()], s[(E.fold == f).to_numpy()])
                              for f in sorted(E.fold.unique())])
            R.append(dict(horizonte=h, escore=nome, grupo="todos os horários", n=len(E), eventos=int(y.sum()),
                          **P.mean().to_dict()))
        # iniciação local x chuva que chega do entorno, nos inícios a partir do seco
        ini = inicios(E, h)
        if len(ini):
            ini = ini[ini.partiu_do_seco]
            k = np.clip(ini.pos_tc.to_numpy() - h // 10, 0, None)
            c1 = np.nan_to_num(E.in_c1_tot_max60.to_numpy(float))[k] if "in_c1_tot_max60" in E else np.zeros(len(k))
            c1b = np.nan_to_num(E.in_c1_barl_max60.to_numpy(float))[k] if "in_c1_barl_max60" in E else np.zeros(len(k))
            tipo = np.where(c1b >= 2.5, "chuva moderada ou mais no colar 1 a barlavento",
                            np.where(c1 >= cfg.BASCULA, "chuva no colar 1, fora do barlavento ou fraca",
                                     "sem chuva no colar 1 (iniciação local)"))
            prev = s_seco[k]; som_k = som[k]
            T = pd.DataFrame(dict(tipo=tipo, prob_modelo_seco=prev, prob_som=som_k))
            T = T.groupby("tipo").agg(inicios=("tipo", "size"), prob_modelo_seco_mediana=("prob_modelo_seco", "median"),
                                      prob_som_mediana=("prob_som", "median"),
                                      frac_som_alerta=("prob_som", lambda x: (x >= L_ALERTA).mean()),
                                      frac_modelo_seco_acima_p95=("prob_modelo_seco", lambda x: (x >= np.nanquantile(s_seco[seco], 0.95)).mean())).reset_index()
            T.insert(0, "horizonte", h)
            salvar(f"A6_inicios_do_seco_por_tipo_h{h}", T.round(4))
    salvar("A6_desempenho_no_seco", pd.DataFrame(R).round(4))
    I = pd.concat(I) if I else pd.DataFrame()
    if len(I):
        salvar("A6_importancia_por_bloco", I.groupby(["horizonte", "bloco"]).queda_AP.agg(["mean", "std"]).round(4).reset_index())
    if Cu:
        salvar("A6_curvas_de_ganho_seco", pd.concat(Cu).round(4), mostrar=False)


# ═══════════════════════════════ execução ════════════════════════════════════
def main():
    configurar_log()
    partes = [a.upper() for a in sys.argv[1:]] or ["A1", "A2", "A3", "A4", "A5", "A6"]
    t0 = time.time()
    A = carregar()
    if "A1" in partes:
        a1(A)
    K = None
    if any(p in partes for p in ("A2", "A3", "A4", "A6")):
        K = a2(A)
    if "A3" in partes:
        a3(K)
    if "A4" in partes:
        a4(K)
    if "A5" in partes:
        a5(A)
    if "A6" in partes:
        a6(A, K)
    log.info(f"fim em {(time.time() - t0) / 60:.1f} min; saídas em {OUT}")


if __name__ == "__main__":
    main()
