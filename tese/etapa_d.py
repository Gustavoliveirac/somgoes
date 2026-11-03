#!/usr/bin/env python
"""
etapa_d.py — duas formas de combinar o SOM com o gradient boosting.

  D1  SOM com as mesmas entradas do gradient boosting. As 174 variáveis da etapa A (evolução
      das nuvens, deslocamento, montante, histórico dos pluviômetros, hora do dia) entram no
      codificador por um ramo próprio e são unidas ao latente das imagens e das estações. O mapa
      se organiza sobre esse latente unido (pertinência t de Student, endurecimento e vizinhança
      suave, como no PSOM de Manduchi et al., 2021), com a reconstrução das entradas originais e
      a cabeça de previsão treinadas juntas.
  D2  SOM como correção do gradient boosting. A cabeça parte do logito da probabilidade do
      gradient boosting e aprende só o ajuste em cada estado do mapa. Nos horários de treino, a
      probabilidade do gradient boosting vem de modelos que não viram nem a temporada avaliada
      nem a temporada do próprio horário; na temporada avaliada, do modelo treinado com todas
      as outras.

Os dois usam a configuração escolhida na etapa C (entropia cruzada com pesos por classe, peso 5
da cabeça, taxa de aprendizado 1e-3), uma temporada de treino separada para a parada antecipada
e a calibração, e a cabeça dá ao mesmo tempo a probabilidade de passar de 0,2, 2,5, 10 e
20 mm/h. O gradient boosting desta etapa é o LightGBM, do mesmo tipo do HistGradientBoosting das
etapas anteriores e várias vezes mais rápido, o que torna viáveis os 24 ajustes por temporada.
A comparação é com ele, por limiar de chuva, nas seis temporadas, em POD,
FAR e CSI. Gradient boosting e probabilidades ficam salvos, e uma rodada interrompida continua de
onde parou.
Uso (da raiz):  python -m tese.etapa_d
                ETAPA_D_TESTE=1 python -m tese.etapa_d    (teste curto, saída em etapa_d/teste/)
"""
from __future__ import annotations

import logging
import os
import sys
import time

TESTE = os.environ.get("ETAPA_D_TESTE", "0") == "1"
if TESTE:
    os.environ.setdefault("RAPIDO", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from core import config_somgoes as cfg
from tese import etapa_c as ec

eb, at, mt, ea, rt = ec.eb, ec.at, ec.mt, ec.ea, ec.rt
OUT = cfg.ROOT / "etapa_d" / ("teste" if TESTE else "")
EPOCAS = 1 if TESTE else cfg.EPOCAS
LOTES = 10 if TESTE else cfg.LOTES_EPOCA
PACIENCIA = 5
NIVEIS = ec.NIVEIS
X_MM = ec.X_MM
VAR = ec.VARIANTES["bce_pesos"]
ROT = {"entradas": "SOM com cabeça e as mesmas entradas", "correcao": "SOM como correção do gradient boosting"}
GB = "gradient boosting (LightGBM), variáveis novas"
log = logging.getLogger("etapa_d")


def configurar_log():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gb").mkdir(exist_ok=True); (OUT / "probs").mkdir(exist_ok=True)
    at.OUT = OUT
    ea.CACHE_A.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    r = logging.getLogger(); r.handlers.clear()
    for hd in (logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "etapa_d.log", mode="a")):
        hd.setFormatter(fmt); r.addHandler(hd)
    r.setLevel(logging.INFO)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 500)


def logito(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# ═══════════════════════════════ gradient boosting fora da temporada ══════════
def gb(Xtr, ytr, Xte):
    """Gradient boosting por histogramas (LightGBM, mesmo tipo de modelo do HistGradientBoosting
    do scikit-learn, várias vezes mais rápido): 10% do treino separados para a parada antecipada."""
    try:
        import lightgbm as lgb
    except ImportError:
        return mt.ajustar("HistGradientBoosting", Xtr, ytr, Xte, None)
    rng = np.random.default_rng(cfg.SEED)
    va = rng.random(len(ytr)) < 0.1
    m = lgb.LGBMClassifier(n_estimators=200 if TESTE else 1000, learning_rate=0.05, num_leaves=31,
                           min_child_samples=40, reg_lambda=1.0, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, n_jobs=-1, verbose=-1, random_state=cfg.SEED)
    para = [lgb.early_stopping(50, verbose=False)]
    try:                                   # LightGBM 4.7 em diante
        m.fit(Xtr[~va], ytr[~va], eval_X=(Xtr[va],), eval_y=(ytr[va],), callbacks=para)
    except TypeError:
        m.fit(Xtr[~va], ytr[~va], eval_set=[(Xtr[va], ytr[va])], callbacks=para)
    return m.predict_proba(Xte)[:, 1]


def gb_aninhado(E, X, prox, h, chuva, folds_te):
    """Probabilidade do gradient boosting por limiar. Para cada temporada avaliada k: na própria k,
    modelo treinado com todas as outras; em cada temporada de treino s, modelo treinado sem k e
    sem s. Devolve {k: (posições no cache, matriz n x 4)} para teste e para treino."""
    fE, pE = E.fold.to_numpy(), E.pos.to_numpy()
    est = prox[pE]; ok = est >= 0
    folds = sorted(np.unique(fE))
    res = {}
    for k in folds_te:
        arq = OUT / "gb" / f"lgbm_h{h}_{chuva}_fold{k}.npz"
        if arq.exists():
            g = np.load(arq); res[k] = {c: g[c] for c in g.files}; continue
        t0 = time.time()
        te = fE == k
        P_te = np.column_stack([gb(X[~te & ok], est[~te & ok] >= nv, X[te])
                                for nv in NIVEIS])
        pos_tr, P_tr = [], []
        for s in folds:
            if s == k:
                continue
            ms = fE == s; fit = ~te & ~ms & ok
            P_tr.append(np.column_stack([gb(X[fit], est[fit] >= nv, X[ms])
                                         for nv in NIVEIS]))
            pos_tr.append(pE[ms])
        res[k] = dict(pos_te=pE[te], p_te=P_te, pos_tr=np.concatenate(pos_tr), p_tr=np.vstack(P_tr))
        np.savez(arq, **res[k])
        log.info(f"  gradient boosting h{h} {chuva}: temporada {k} e suas temporadas de treino em {(time.time() - t0) / 60:.1f} min")
    return res


# ═══════════════════════════════ treino ══════════════════════════════════════
def treinar(cache, chuva, h, tr_in, va_in, tr_all, modo, Xc, off, rotulo):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from core import somgoes_nucleo as nu
    from analise import sintese_somgoes as si
    torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED)
    norm = cache.normalizador(tr_all)
    m = nu.SOMGoes(cache, cfg.GOES_CENARIOS, "ae", (10, 10), chuva=chuva)
    K, lat = m.prot.shape
    if modo == "entradas":
        m.ramo_tab = nn.Sequential(nn.Linear(Xc.shape[1], 128), nn.ReLU(), nn.Linear(128, lat))
        m.fus_tab = nn.Linear(2 * lat, lat)     # nomes próprios; o SOMGoes já tem um atributo fus
    m.cab = nn.Sequential(nn.Linear(K, 64), nn.ReLU(), nn.Linear(64, len(NIVEIS)))
    m.to(nu.DEV)
    Xt = torch.from_numpy(Xc).to(nu.DEV) if Xc is not None else None
    Ot = torch.from_numpy(off).to(nu.DEV) if off is not None else None
    opt = torch.optim.Adam(m.parameters(), lr=VAR["lr"])
    yA = cache.rotulos(np.arange(len(cache.A)), h)
    amo = nu.Amostrador(tr_in, yA["transicao"][tr_in], cfg.LOTE)
    niv = torch.tensor(NIVEIS, device=nu.DEV)
    nu.pre_treino(m, cache, tr_in, norm, yA, amo, opt, LOTES)

    def lote(ix):
        return {k: v.to(nu.DEV) for k, v in cache.lote(ix, norm, m.goes_modo, m.chuva).items()}

    def latente(b, ix):
        z, _, _ = m.codificar(b)
        if modo == "entradas":
            z = m.fus_tab(torch.cat([z, m.ramo_tab(Xt[torch.as_tensor(ix, device=nu.DEV)])], 1))
        return z

    def saida(z, ix):
        q = ec.psom_termos(m, z)[0] if modo == "entradas" else nu.soft(m.dist(z))
        o = m.cab(q)
        if modo == "correcao":
            o = o + Ot[torch.as_tensor(ix, device=nu.DEV)]
        return o

    if modo == "entradas":     # protótipos reiniciados no latente unido
        with torch.no_grad():
            amostra = np.random.default_rng(cfg.SEED).choice(tr_in, min(4096, len(tr_in)), replace=False)
            zs = latente(lote(amostra), amostra)
            m.prot.data.copy_(zs[torch.randperm(len(zs), device=zs.device)[:K]])

    def prever(ix):
        m.eval(); out = []
        with torch.no_grad():
            for a in range(0, len(ix), 2048):
                ii = ix[a:a + 2048]
                out.append(torch.sigmoid(saida(latente(lote(ii), ii), ii)).cpu().numpy())
        return np.concatenate(out)

    prox_va = yA["prox"][va_in]; ok_va = prox_va >= 0; y_va = prox_va[ok_va] >= 3
    T0 = max(m.grade) / 2
    melhor, estado, sem, hist = -1.0, None, 0, []
    for ep in range(EPOCAS):
        T = max(T0 * (1 / T0) ** (ep / max(EPOCAS - 1, 1)), 0.5)
        m.train(); acc = [0.0, 0.0]
        for _ in range(LOTES):
            ix = amo(); b = lote(ix)
            yb = {k: torch.from_numpy(v[ix]).to(nu.DEV) for k, v in yA.items()}
            opt.zero_grad()
            if modo == "entradas":
                l_rec, _ = nu.perdas(m, b, yb, "pre", T, ep, EPOCAS)
                z = latente(b, ix)
                _, cah, ssom = ec.psom_termos(m, z)
                l_som = l_rec + 0.1 * cah + 0.1 * ssom
            else:
                l_som, _ = nu.perdas(m, b, yb, si.PERDA_OP, T, ep, EPOCAS)
                z = latente(b, ix)
            alvo = (yb["prox"].unsqueeze(1) >= niv.unsqueeze(0)).float()
            p = alvo.mean(0).clamp(1e-3, 1 - 1e-3)
            l_cab = F.binary_cross_entropy_with_logits(saida(z, ix), alvo, pos_weight=torch.sqrt((1 - p) / p))
            (l_som + VAR["peso"] * l_cab).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            acc[0] += float(l_som) / LOTES; acc[1] += float(l_cab) / LOTES
        pv = prever(va_in)[ok_va, 2]
        nota = ec.pod_fracoes(y_va, pv) if y_va.any() else 0.0
        hist.append(dict(epoca=ep, perda_som=round(acc[0], 4), perda_cabeca=round(acc[1], 4),
                         POD_medio_validacao_10mm=round(nota, 4)))
        log.info(f"    {rotulo} época {ep}: perda SOM {acc[0]:.4f}, cabeça {acc[1]:.4f}, POD médio na validação {nota:.3f}")
        if nota > melhor:
            melhor, sem = nota, 0
            estado = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            sem += 1
            if sem >= PACIENCIA:
                break
    m.load_state_dict(estado)
    m.prever = prever
    return m, pd.DataFrame(hist), melhor


def main():
    configurar_log()
    t00 = time.time()
    import torch
    log.info(f"etapa D {'(TESTE) ' if TESTE else ''}| torch {torch.__version__} | dispositivo {ea.dispositivo()} | "
             f"configuração {VAR}")
    cache, conj, dados = eb.preparar()
    N = len(cache.A)
    PF, PAR, HIST = [], [], []
    combos = [(30, "sem chuva")] if TESTE else [(h, c) for h in (30, 60) for c in conj]
    for h, cj in combos:
        E, X, chuva = dados(h, cj)
        prox = cache.A[f"estado_prox{at.sufixo(h)}"].to_numpy()
        fE, pE = E.fold.to_numpy(), E.pos.to_numpy()
        folds = sorted(np.unique(fE))[:1] if TESTE else sorted(np.unique(fE))
        G = gb_aninhado(E, X, prox, h, chuva, folds)
        S = {nm: {nv: np.full(len(E), np.nan) for nv in NIVEIS} for nm in (ROT["entradas"], ROT["correcao"], GB)}
        for k in folds:
            t0 = time.time()
            tr_in, va_in, te, tr_all, sval = ec.divisao(cache, k, h)
            te_E = fE == k
            # gradient boosting na temporada avaliada
            Pg = pd.DataFrame(G[k]["p_te"], index=G[k]["pos_te"])
            for i, nv in enumerate(NIVEIS):
                S[GB][nv][te_E] = Pg[i].reindex(pE[te_E]).to_numpy()
            # variáveis do gradient boosting no formato do cache, padronizadas com o treino
            Xc = np.zeros((N, X.shape[1]), np.float32)
            trE = np.isin(pE, tr_all)
            med = np.nanmedian(X[trE], 0); med[~np.isfinite(med)] = 0
            Xi = np.where(np.isfinite(X), X, med)
            mu, sd = Xi[trE].mean(0), Xi[trE].std(0); sd[sd < 1e-9] = 1
            Xc[pE] = ((Xi - mu) / sd).astype(np.float32)
            # logito do gradient boosting no formato do cache (treino aninhado, teste com as demais)
            base = np.array([logito(((prox[tr_all] >= nv) & (prox[tr_all] >= 0)).mean()) for nv in NIVEIS], np.float32)
            off = np.tile(base, (N, 1)).astype(np.float32)
            off[G[k]["pos_tr"]] = logito(G[k]["p_tr"]); off[G[k]["pos_te"]] = logito(G[k]["p_te"])
            for modo in ("entradas", "correcao"):
                arq = OUT / "probs" / f"{modo}_h{h}_{chuva}_fold{k}.csv.gz"
                if arq.exists():
                    P = pd.read_csv(arq, sep=";", index_col=0)
                else:
                    m, hist, nota = treinar(cache, chuva, h, tr_in, va_in, tr_all, modo,
                                            Xc if modo == "entradas" else None,
                                            off if modo == "correcao" else None, f"{modo} h{h} {cj} temporada {k}")
                    P = pd.DataFrame(ec.calibrar_e_prever(m, cache, h, va_in, te), index=te,
                                     columns=[f"p{nv}" for nv in NIVEIS])
                    P.to_csv(arq, sep=";")
                    HIST.append(hist.assign(modelo=ROT[modo], horizonte=h, entradas=cj, temporada=k))
                    PAR.append(dict(modelo=ROT[modo], horizonte=h, entradas=cj, temporada=k, temporada_validacao=sval,
                                    epocas=len(hist), POD_medio_validacao_10mm=round(nota, 4)))
                    del m
                for nv in NIVEIS:
                    S[ROT[modo]][nv][te_E] = P[f"p{nv}"].reindex(pE[te_E]).to_numpy()
            log.info(f"  h{h} {cj}: temporada {k} pronta em {(time.time() - t0) / 60:.1f} min")
        y_est = prox[pE]; usadas = np.isin(fE, folds) & (y_est >= 0)
        for rot, por in S.items():
            for nv in NIVEIS:
                PF += [dict(x_mm_h=X_MM[nv], **d) for d in rt.pod_far(y_est[usadas] >= nv, por[nv][usadas], rot, h, cj)]
        for k in folds:
            mk = usadas & (fE == k)
            lin = dict(horizonte=h, entradas=cj, temporada=k, comparacao="10 mm/h, alerta em 5% do tempo")
            for rot, por in S.items():
                d = [x for x in rt.pod_far(y_est[mk] >= 3, por[3][mk], rot, h, cj) if x["ponto"] == "alerta em 5% do tempo"][0]
                lin[f"POD {rot}"] = d["POD"]; lin[f"FAR {rot}"] = d["FAR"]
            PAR.append(lin)
    if HIST:
        at.salvar("D_historico_do_treino", pd.concat(HIST, ignore_index=True), mostrar=False)
    at.salvar("D_por_temporada", pd.DataFrame(PAR))
    at.salvar("D_pod_far", pd.DataFrame(PF), mostrar=False)
    eb.tabela_pod_far(PF, "D, seis temporadas")
    log.info(f"fim em {(time.time() - t00) / 60:.1f} min; saídas em {OUT}")


if __name__ == "__main__":
    main()
