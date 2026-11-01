#!/usr/bin/env python
"""
fase1_som.py — fase 1: elevar o SOM com os modelos já treinados e testar variáveis novas.

  C1  SOM: probabilidade dura (frequência do nó, como antes), probabilidade suave (mistura
      das frequências dos nós ponderada pela distância no espaço latente, com a temperatura
      escolhida por validação cruzada dentro das temporadas de treino) e vizinhos mais
      próximos no espaço latente
  C2  dois estágios corrigido: a frequência de chuva forte do nó dos horários de treino é
      calculada com as outras temporadas de treino (sem usar o próprio horário), e uma
      variante recebe o vetor latente do SOM
  C3  variáveis novas
        GOES em temperatura de brilho (constantes de Planck de goes_planck.json) e distâncias
        reais (goes_geometria.npz): topos abaixo de 0, -20 e -40 °C, resfriamento em 10 e
        30 min, diferença entre vapor de água e janela (C08 - C13), diferença 8,4 - 11,2 µm
        (C11 - C14) e RRQPE, em raios de 15, 30 e 60 km
        deslocamento das nuvens frias entre t-30 e t (correlação de fase) e o que está a
        montante: topos e RRQPE no ponto de onde a nuvem chega em 30 ou 60 min, e chuva no
        quadrante do anel de 13 a 60 km de onde as nuvens vêm
        pluviômetros com defasagem de 10, 20 e 30 min e variação em 30 min
  C4  avaliação: AP, recall e lift por fração do tempo, POD, FAR, CSI, HSS, informação mútua
      entre o nó e a transição seguinte (qualidade dos estados) e inícios a partir do seco,
      separando os locais

Precisa de goes_geometria.npz e goes_planck.json na pasta do somgoes (gerados pelo
extrai_goes_meta.py) e dos mesmos arquivos do modelos_tese.py.
Saídas em fase1_som/ e log mestre fase1_som/fase1_som.log (é o que deve ser enviado).
Uso:  python fase1_som.py        (cerca de 1 h)
      SEM_CATBOOST=1 python fase1_som.py
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors

import warnings

import config_somgoes as cfg
import modelos_tese as mt

at = mt.at
OUT = cfg.ROOT / "fase1_som"
SEM_CATBOOST = os.environ.get("SEM_CATBOOST", "0") == "1"
log = logging.getLogger("fase1")
warnings.filterwarnings("ignore", category=RuntimeWarning)
TAUS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
K_VIZ = (100, 300)


def configurar_log():
    OUT.mkdir(parents=True, exist_ok=True)
    at.OUT = OUT
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    r = logging.getLogger(); r.handlers.clear()
    for hd in (logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "fase1_som.log", mode="w")):
        hd.setFormatter(fmt); r.addHandler(hd)
    r.setLevel(logging.INFO)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 400)


# ═══════════════════════════════ GOES físico ═════════════════════════════════
BANDA = {"C08": 0, "C09": 1, "C10": 2, "C11": 3, "C12": 4, "C13": 5, "C14": 6, "RRQPE": 7}


def temperatura_brilho(L, p):
    L = np.where(L > 0, L, np.nan)
    return (p["planck_fk2"] / np.log(p["planck_fk1"] / L + 1.0) - p["planck_bc1"]) / p["planck_bc2"]


def geometria():
    g = np.load(cfg.ROOT / "goes_geometria.npz")
    lat, lon = g["lat32"], g["lon32"]
    lat0, lon0 = -19.9068, -43.9765
    E = (lon - lon0) * 111.32 * math.cos(math.radians(lat0))
    N = (lat - lat0) * 110.57
    # jacobiano local: km por linha e por coluna da grade 32 x 32
    dE_col, dN_col = np.gradient(E, axis=1).mean(), np.gradient(N, axis=1).mean()
    dE_lin, dN_lin = np.gradient(E, axis=0).mean(), np.gradient(N, axis=0).mean()
    log.info(f"grade do GOES 32 x 32: coluna = ({dE_col:.2f} km E, {dN_col:.2f} km N), "
             f"linha = ({dE_lin:.2f} km E, {dN_lin:.2f} km N)")
    return E, N, np.array([[dE_lin, dE_col], [dN_lin, dN_col]])


def deslocamento(a, b):
    """Deslocamento (linhas, colunas) do campo a para o campo b por correlação de fase,
    em lote, com ajuste parabólico em torno do pico."""
    Fa, Fb = np.fft.fft2(a), np.fft.fft2(b)
    R = Fb * np.conj(Fa); R /= np.abs(R) + 1e-9
    r = np.fft.ifft2(R).real
    n, H, W = r.shape
    k = r.reshape(n, -1).argmax(1); iy, ix = np.unravel_index(k, (H, W))
    def sub(c, cm, cp):
        den = cm - 2 * c + cp
        return np.where(np.abs(den) > 1e-9, 0.5 * (cm - cp) / den, 0.0)
    ii = np.arange(n)
    dy = iy + sub(r[ii, iy, ix], r[ii, (iy - 1) % H, ix], r[ii, (iy + 1) % H, ix])
    dx = ix + sub(r[ii, iy, ix], r[ii, iy, (ix - 1) % W], r[ii, iy, (ix + 1) % W])
    dy = np.where(dy > H / 2, dy - H, dy); dx = np.where(dx > W / 2, dx - W, dx)
    return dy, dx


def variaveis_goes_fisicas(cache, A, lote=3000):
    PL = json.load(open(cfg.ROOT / "goes_planck.json"))
    E, N, Jac = geometria()
    D = np.hypot(E, N)
    raios = (15, 30, 60)
    masc = {r: D <= r for r in raios}
    G, J = cache.campo, cache.jan
    n = G.shape[0]
    out = {}
    def col(nome):
        if nome not in out:
            out[nome] = np.full(n, np.nan, np.float32)
        return out[nome]
    ok_frame = lambda g: np.nanmax(g, axis=(1, 2)) > 0
    for a in range(0, n, lote):
        ix = np.arange(a, min(n, a + lote))
        g0 = np.asarray(G[ix], np.float32)
        def quadro(k):
            j = J[ix, k]
            g = np.asarray(G[np.clip(j, 0, None)], np.float32)
            v = (j >= 0) & ok_frame(g[:, BANDA["C13"]])
            return g, v
        g10, v10 = quadro(-2); g30, v30 = quadro(-4)
        v0 = ok_frame(g0[:, BANDA["C13"]])
        T = {b: temperatura_brilho(g0[:, BANDA[b]], PL[b]) for b in ("C08", "C11", "C13", "C14")}
        T13_10 = temperatura_brilho(g10[:, BANDA["C13"]], PL["C13"])
        T13_30 = temperatura_brilho(g30[:, BANDA["C13"]], PL["C13"])
        T08_30 = temperatura_brilho(g30[:, BANDA["C08"]], PL["C08"])
        rr = g0[:, BANDA["RRQPE"]]
        for r, m in masc.items():
            t13 = T["C13"][:, m]
            col(f"g{r}_t13_min")[ix] = np.where(v0, np.nanmin(t13, 1), np.nan)
            for lim in (273.15, 253.15, 233.15):
                col(f"g{r}_frac_abaixo_{round(lim - 273.15)}C")[ix] = np.where(v0, np.nanmean(t13 < lim, 1), np.nan)
            r10 = (T["C13"] - T13_10)[:, m]; r30 = (T["C13"] - T13_30)[:, m]
            col(f"g{r}_resfr10_min")[ix] = np.where(v0 & v10, np.nanmin(r10, 1), np.nan)
            col(f"g{r}_resfr30_min")[ix] = np.where(v0 & v30, np.nanmin(r30, 1), np.nan)
            col(f"g{r}_frac_resfr30_8K")[ix] = np.where(v0 & v30, np.nanmean(r30 <= -8, 1), np.nan)
            wv = (T["C08"] - T["C13"])[:, m]
            wv30 = ((T["C08"] - T["C13"]) - (T08_30 - T13_30))[:, m]
            col(f"g{r}_c08_c13_max")[ix] = np.where(v0, np.nanmax(wv, 1), np.nan)
            col(f"g{r}_c08_c13_var30_max")[ix] = np.where(v0 & v30, np.nanmax(wv30, 1), np.nan)
            col(f"g{r}_c11_c14_max")[ix] = np.where(v0, np.nanmax((T["C11"] - T["C14"])[:, m], 1), np.nan)
            col(f"g{r}_area_fria_var30")[ix] = np.where(v0 & v30, np.nanmean(t13 < 253.15, 1)
                                                        - np.nanmean(T13_30[:, m] < 253.15, 1), np.nan)
            col(f"g{r}_rrqpe_frac")[ix] = np.where(v0, np.mean(rr[:, m] > 0, 1), np.nan)
            col(f"g{r}_rrqpe_max")[ix] = np.where(v0, np.max(rr[:, m], 1), np.nan)
        # deslocamento das nuvens frias entre t-30 e t
        f0 = np.nan_to_num(np.clip(260 - T["C13"], 0, None)); f30 = np.nan_to_num(np.clip(260 - T13_30, 0, None))
        ok = v0 & v30 & (f0.sum((1, 2)) > 50) & (f30.sum((1, 2)) > 50)
        dy, dx = deslocamento(f30, f0)
        vE = Jac[0, 0] * dy + Jac[0, 1] * dx; vN = Jac[1, 0] * dy + Jac[1, 1] * dx      # km em 30 min
        vE = np.where(ok, vE, np.nan); vN = np.where(ok, vN, np.nan)
        col("nuvens_vel_kmh")[ix] = 2 * np.hypot(vE, vN)
        az = (np.degrees(np.arctan2(-vE, -vN)) + 360) % 360           # direção de onde vêm
        col("nuvens_dir_de_onde_vem")[ix] = az
        for h in (30, 60):
            pE = -vE * h / 30; pN = -vN * h / 30                       # ponto a montante
            dd = np.hypot(E[None] - pE[:, None, None], N[None] - pN[:, None, None])
            mm = dd <= 15
            t13 = np.where(mm, T["C13"], np.nan)
            with np.errstate(all="ignore"):
                col(f"mont{h}_t13_min")[ix] = np.where(ok, np.nanmin(t13, (1, 2)), np.nan)
                col(f"mont{h}_frac_abaixo_-20C")[ix] = np.where(ok, (np.nan_to_num(t13, nan=999) < 253.15).sum((1, 2)) / mm.sum((1, 2)).clip(1), np.nan)
                col(f"mont{h}_rrqpe_max")[ix] = np.where(ok, np.where(mm, rr, 0).max((1, 2)), np.nan)
        if a % (lote * 10) == 0:
            log.info(f"  GOES físico: {min(n, a + lote):,} de {n:,} horários")
    V = pd.DataFrame(out)
    # anel de 13 a 60 km no quadrante de onde vêm as nuvens
    az = V.nuvens_dir_de_onde_vem.to_numpy()
    q = np.select([(az >= 315) | (az < 45), az < 135, az < 225], ["N", "L", "S"], "O")
    q = np.where(np.isfinite(az), q, "")
    for v in ("max60", "max10", "frac60"):
        cols = [f"in_c1_{k}_{v}" for k in "NLSO"]
        if all(c in cache.A for c in cols):
            Mq = cache.A[cols].to_numpy(float)
            idx = np.array([{"N": 0, "L": 1, "S": 2, "O": 3}.get(x, -1) for x in q])
            V[f"in_c1_mont_{v}"] = np.where(idx >= 0, Mq[np.arange(n), np.clip(idx, 0, 3)], np.nan)
    log.info(f"GOES físico e deslocamento: {V.shape[1]} variáveis; deslocamento válido em "
             f"{np.isfinite(V.nuvens_vel_kmh).mean():.0%} dos horários; velocidade mediana "
             f"{np.nanmedian(V.nuvens_vel_kmh):.0f} km/h")
    return V


def variaveis_pluvio(cache):
    J = cache.jan; A = cache.A
    out = {}
    for c in ("in_c1_tot_max10", "in_c1_tot_max60", "in_c1_tot_frac60", "in_nuc_max10", "in_nuc_frac60"):
        if c not in A:
            continue
        v = A[c].to_numpy(float)
        for k, m in ((-2, 10), (-3, 20), (-4, 30)):
            j = J[:, k]
            out[f"{c}_lag{m}"] = np.where(j >= 0, v[np.clip(j, 0, None)], np.nan)
        out[f"{c}_var30"] = v - out[f"{c}_lag30"]
    return pd.DataFrame(out)


# ═══════════════════════════════ SOM: latente, dura, suave, vizinhos ═════════
def latentes(chuva, h):
    import torch
    import somgoes_nucleo as nu
    import sintese_somgoes as si
    cache = mt.cache_som()
    base = cfg.RESULTS_DIR / "cenarios" / f"cenarios_{chuva}_h{h}_{si.PERDA_OP}_10x10"
    folds = sorted(int(p.name[4:]) for p in base.glob("fold*") if (p / "melhor.pt").exists())
    res = {}
    for fold in folds:
        tr, va = cache.split(fold, None, h)
        norm = cache.normalizador(tr)
        m = nu.SOMGoes(cache, cfg.GOES_CENARIOS, "ae", (10, 10), chuva=chuva)
        m.load_state_dict(torch.load(base / f"fold{fold}" / "melhor.pt", map_location=nu.DEV,
                                     weights_only=False)["model"]); m.to(nu.DEV); m.eval()
        idx = np.r_[tr, va]; Z, Dd = [], []
        with torch.no_grad():
            for s in range(0, len(idx), 2048):
                b = {k: v.to(nu.DEV) for k, v in cache.lote(idx[s:s + 2048], norm, m.goes_modo, m.chuva).items()}
                z, _, _ = m.codificar(b)
                Z.append(z.cpu().numpy()); Dd.append(m.dist(z).cpu().numpy())
        res[fold] = dict(idx=idx, ntr=len(tr), z=np.concatenate(Z).astype(np.float32),
                         d=np.concatenate(Dd).astype(np.float32))
        log.info(f"  latente {chuva} h{h}: temporada {fold} pronta")
    return res


def prob_nos(bmu, y, K=100):
    n = np.bincount(bmu, minlength=K); f = np.bincount(bmu[y], minlength=K)
    return (f + 5 * y.mean()) / (n + 5)


def suave(d, p, tau, escala):
    """Mistura das frequências dos nós com pesos exp(-(d - d_min) / (tau * escala)); escala é a
    mediana da distância ao nó mais próximo nos horários de treino da temporada."""
    w = np.exp(-(d - d.min(1, keepdims=True)) / (tau * escala))
    w /= w.sum(1, keepdims=True)
    return w @ p


def saidas_som(chuva, h, E):
    """Para cada horário do conjunto E: probabilidade dura, suave e por vizinhos, nó, frequência
    do nó com validação cruzada interna e vetor latente, sempre do modelo que não viu a temporada."""
    cache = mt.cache_som()
    temp = cache.A.fold.to_numpy()
    yfull = cache.A[f"estado_prox{at.sufixo(h)}"].to_numpy() >= 3
    pos = E.pos.to_numpy(); fE = E.fold.to_numpy()
    out = {k: np.full(len(E), np.nan) for k in ("dura", "suave", "viz100", "viz300", "no", "tx_cv")}
    Zout = np.full((len(E), 32), np.nan, np.float32)
    taus = []
    for fold, R in latentes(chuva, h).items():
        idx, ntr, z, d = R["idx"], R["ntr"], R["z"], R["d"]
        bmu = d.argmin(1); y = yfull[idx]; s = temp[idx]
        itr = np.arange(ntr); ite = np.arange(ntr, len(idx))
        escala = float(np.median(d[itr].min(1)))
        # temperatura da probabilidade suave por validação cruzada interna
        notas = {t: [] for t in TAUS}; tx_cv = np.full(len(idx), np.nan)
        for si_ in np.unique(s[itr]):
            a = itr[s[itr] != si_]; b = itr[s[itr] == si_]
            p = prob_nos(bmu[a], y[a])
            tx_cv[b] = p[bmu[b]]
            if y[b].sum() > 0:
                for t in TAUS:
                    notas[t].append(average_precision_score(y[b], suave(d[b], p, t, escala)))
        tau = max(TAUS, key=lambda t: np.mean(notas[t]))
        taus.append(tau)
        p = prob_nos(bmu[itr], y[itr])
        tx_cv[ite] = p[bmu[ite]]
        ps = suave(d[ite], p, tau, escala)
        nn_ = NearestNeighbors(n_neighbors=max(K_VIZ)).fit(z[itr])
        _, viz = nn_.kneighbors(z[ite])
        yv = y[itr][viz]
        # linhas de E que são teste desta temporada
        mapa = pd.Series(np.arange(len(ite)), index=idx[ite])
        jE = np.where(fE == fold)[0]
        kk = mapa.reindex(pos[jE]).to_numpy()
        okk = np.isfinite(kk); jE = jE[okk]; kk = kk[okk].astype(int)
        out["dura"][jE] = p[bmu[ite][kk]]; out["suave"][jE] = ps[kk]
        out["viz100"][jE] = yv[kk, :100].mean(1); out["viz300"][jE] = yv[kk].mean(1)
        out["no"][jE] = bmu[ite][kk]; out["tx_cv"][jE] = tx_cv[ite][kk]; Zout[jE] = z[ite][kk]
        out.setdefault("_folds", {})
        out["_folds"][fold] = dict(pos=idx, bmu=bmu, tx_cv=tx_cv, z=z)
    log.info(f"  temperaturas escolhidas para a probabilidade suave: {taus}")
    return out, Zout


def dois_estagios_cv(E, X, folds, com_latente=False):
    """Classificador por temporada com o nó, a frequência do nó calculada sem a própria
    temporada de treino e, opcionalmente, o vetor latente, todos do SOM daquela temporada."""
    s = np.full(len(E), np.nan); y = E.forte.to_numpy(); f = E.fold.to_numpy(); pos = E.pos.to_numpy()
    for k, R in folds.items():
        mapa = pd.Series(np.arange(len(R["pos"])), index=R["pos"])
        j = mapa.reindex(pos).to_numpy()
        ok = np.isfinite(j); jj = np.nan_to_num(j).astype(int)
        extra = [np.where(ok, R["bmu"][jj], np.nan), np.where(ok, R["tx_cv"][jj], np.nan)]
        if com_latente:
            extra += [np.where(ok, R["z"][jj, c], np.nan) for c in range(R["z"].shape[1])]
        Xs = np.column_stack([X] + extra)
        tr = (f != k) & ok; te = (f == k) & ok
        if y[tr].sum() < 10 or te.sum() == 0:
            continue
        s[te] = mt.ajustar("HistGradientBoosting", Xs[tr], y[tr], Xs[te], [X.shape[1]])
    return s


def mi_estados(E, no, h):
    """Informação mútua normalizada entre o nó e a transição seguinte (8 classes), no teste."""
    tr = E.transicao.to_numpy(); ok = np.isfinite(no)
    a = no[ok].astype(int); b = tr[ok].astype(int)
    P = pd.crosstab(a, b).to_numpy().astype(float); P /= P.sum()
    pa, pb = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    with np.errstate(all="ignore"):
        mi = np.nansum(P * np.log(P / (pa @ pb)))
    hb = -np.nansum(pb * np.log(pb))
    return mi / hb


# ═══════════════════════════════ execução ════════════════════════════════════
def main():
    configurar_log()
    t00 = time.time()
    cache = mt.cache_som()
    A = at.carregar(); B = at.blocos(A)
    V = variaveis_goes_fisicas(cache, A); P = variaveis_pluvio(cache)
    for D in (V, P):
        for c in D.columns:
            A[c] = D[c].to_numpy()[A.pos.to_numpy()]
    GF = [c for c in V.columns if c.startswith("g") and c[1:3] in ("15", "30", "60")] + ["nuvens_vel_kmh"]
    PL_ = list(P.columns) + [c for c in V.columns if c.startswith("in_c1_mont")]
    M, C, G, EV, Q = [], [], [], [], []
    conjuntos = {"sem chuva": (("EMs", "EMs_tend", "GOES", "GOES_tend", "hora"), "nenhuma"),
                 "anéis e área central": (("EMs", "EMs_tend", "GOES", "GOES_tend", "hora", "colar1", "colar2", "nucleo"),
                                          "colar_nucleo")}
    for h in (30, 60):
        E = at.conjunto(A, h)
        MONT = [c for c in V.columns if c.startswith(f"mont{h}_")]
        for conj, (blocos, chuva) in conjuntos.items():
            base = [c for b in blocos for c in B[b]]
            novas = base + GF + MONT + (PL_ if chuva != "nenhuma" else [])
            Xb = E[base].to_numpy(float); Xn = E[novas].to_numpy(float)
            log.info(f"h{h} {conj}: {len(base)} variáveis de base, {len(novas)} com as novas")
            S = {}
            som, Z = saidas_som(chuva, h, E)
            S["SOM, probabilidade do nó (como antes)"] = som["dura"]
            S["SOM, probabilidade suave"] = som["suave"]
            S["SOM, 100 vizinhos no espaço latente"] = som["viz100"]
            S["SOM, 300 vizinhos no espaço latente"] = som["viz300"]
            Q.append(dict(horizonte=h, entradas=conj, mi_no_transicao=mi_estados(E, som["no"], h),
                          nos_usados=int(np.unique(som["no"][np.isfinite(som["no"])]).size)))
            S["HistGradientBoosting, variáveis de base"] = mt.loso(E, Xb, "HistGradientBoosting")
            S["HistGradientBoosting, com as variáveis novas"] = mt.loso(E, Xn, "HistGradientBoosting")
            S["dois estágios corrigido (nó do SOM e frequência sem vazamento)"] = dois_estagios_cv(E, Xn, som["_folds"])
            S["dois estágios corrigido + latente do SOM"] = dois_estagios_cv(E, Xn, som["_folds"], True)
            if mt.CatBoostClassifier is not None and not SEM_CATBOOST:
                S["CatBoost, com as variáveis novas (referência)"] = mt.loso(E, Xn, "CatBoost")
            if chuva != "nenhuma":
                S["persistência"] = E.agora.to_numpy(float) + 1e-3 * np.nan_to_num(at.col(E, "in_nuc_max10"))
            for nome, s in S.items():
                mt.avaliar(E, s, nome, conj, h, M, C, G)
            # inícios a partir do seco, locais e com chuva no anel
            ini = at.inicios(E, h)
            tipos = mt.tipo_inicio_seco(E, ini, h)
            seco = E.agora.to_numpy() == 0
            k = np.clip(ini.pos_tc.to_numpy() - h // 10, 0, None)
            for nome, s in S.items():
                if nome.startswith("persist"):
                    continue
                sc = np.nan_to_num(s, nan=-1); p95 = np.quantile(sc[seco], 0.95)
                for tp in np.unique(tipos):
                    mm = (tipos == tp) & ini.partiu_do_seco.to_numpy()
                    if mm.sum():
                        EV.append(dict(horizonte=h, entradas=conj, modelo=nome, grupo=tp, inicios=int(mm.sum()),
                                       frac_acima_p95_do_seco=round(float((sc[k][mm] >= p95).mean()), 3),
                                       frac_alerta_010=round(float((sc[k][mm] >= at.L_ALERTA).mean()), 3)))
    M = pd.DataFrame(M)
    at.salvar("C_metricas", M.round(4), mostrar=False)
    at.salvar("C_pontos_de_operacao", pd.DataFrame(C).round(4), mostrar=False)
    at.salvar("C_curvas_de_ganho", pd.concat(G).round(4), mostrar=False)
    at.salvar("C_inicios_do_seco", pd.DataFrame(EV))
    at.salvar("C_qualidade_dos_estados", pd.DataFrame(Q).round(4))
    tab = M[~M.modelo.str.contains(r"\[só")].pivot_table(index=["entradas", "modelo"], columns="horizonte",
                                                         values=["AP", "recall_5pct"]).round(3)
    log.info(f"\n### resumo (AP e recall a 5% do tempo)\n{tab.to_string()}\n")
    seco = M[M.modelo.str.contains(r"\[só")].pivot_table(index=["entradas", "modelo"], columns="horizonte",
                                                          values="AP").round(4)
    log.info(f"\n### AP só nos horários secos\n{seco.to_string()}\n")
    cpt = pd.DataFrame(C)
    cpt = cpt[cpt.ponto.isin(["5% do tempo", "≥0,10"])].pivot_table(index=["entradas", "modelo", "ponto"],
                                                                    columns="horizonte", values=["POD", "FAR", "CSI"]).round(3)
    log.info(f"\n### POD, FAR e CSI\n{cpt.to_string()}\n")
    log.info(f"fim em {(time.time() - t00) / 60:.1f} min; saídas em {OUT}")


if __name__ == "__main__":
    main()
