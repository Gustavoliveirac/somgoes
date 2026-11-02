#!/usr/bin/env python
"""
etapa_c.py — SOM treinado do zero junto com uma cabeça de previsão.

A cabeça é uma rede pequena (duas camadas) sobre a pertinência do horário aos 100 nós do mapa e
dá, de uma vez, a probabilidade de passar de 0,2, 2,5, 10 e 20 mm/h no horizonte. Ela é treinada
por retropropagação junto com o codificador e os protótipos, desde o início do treino.
Em todo o treino, a parada antecipada e a calibração usam uma temporada de treino separada
(a última antes da avaliada), nunca a temporada avaliada.

  C1  triagem de variantes na temporada FOLD_TRIAGEM (padrão 2), usando só as temporadas de
      treino dela: a escolha é feita pelo POD médio com alerta em 3, 5 e 10% do tempo para
      chuva acima de 10 mm/h na temporada de validação interna, em dois casos (60 min com
      anéis e área central, 30 min sem chuva). A temporada avaliada não participa da escolha.
        base           perda atual do SOM (reconstrução, organização do mapa, informação mútua
                       com o estado atual e com o seguinte) + perda focal da cabeça (Lin et
                       al., 2017), peso 5, taxa de aprendizado 1e-3
        sem_inf_mutua  sem os termos de informação mútua; o alvo entra só pela cabeça
        psom           organização do mapa como no PSOM (Manduchi et al., 2021): pertinência
                       pela distribuição t de Student, endurecimento das atribuições (CAH) e
                       perda de vizinhança suave, com reconstrução e cabeça
        peso_cabeca_1, peso_cabeca_20   peso da cabeça em relação à perda do SOM
        taxa_3e-4      taxa de aprendizado 3e-4
        bce_pesos      entropia cruzada com pesos por classe no lugar da perda focal
        mapa_e_latente cabeça sobre a pertinência aos nós e o latente
  C2  rodada final com a variante escolhida nas seis temporadas, 30 e 60 min, sem chuva e com
      anéis e área central, comparada com um gradient boosting por limiar de chuva treinado
      nas mesmas temporadas, em POD, FAR e CSI.

Saídas em etapa_c/ e log mestre etapa_c/etapa_c.log. As probabilidades de cada temporada ficam
salvas, e uma rodada interrompida continua de onde parou.
Uso (da raiz):  python -m tese.etapa_c            (C1 e C2)
                python -m tese.etapa_c c1 | c2
                VARIANTE=psom python -m tese.etapa_c c2     (força a variante da rodada final)
                ETAPA_C_TESTE=1 python -m tese.etapa_c      (teste curto, saída em etapa_c/teste/)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

TESTE = os.environ.get("ETAPA_C_TESTE", "0") == "1"
if TESTE:
    os.environ.setdefault("RAPIDO", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from core import config_somgoes as cfg
from tese import etapa_b as eb

at, mt, ea, rt = eb.at, eb.mt, eb.ea, eb.rt
OUT = cfg.ROOT / "etapa_c" / ("teste" if TESTE else "")
FOLD_TRIAGEM = int(os.environ.get("FOLD_TRIAGEM", 2))
EPOCAS_TRIAGEM = 1 if TESTE else 20
EPOCAS_FINAL = 1 if TESTE else cfg.EPOCAS
LOTES = 10 if TESTE else cfg.LOTES_EPOCA
PACIENCIA = 5
NIVEIS = eb.NIVEIS
X_MM = eb.X_MM
VARIANTES = {
    "base":           dict(som="atual", cabeca="mapa", peso=5.0, perda="focal", lr=1e-3),
    "sem_inf_mutua":  dict(som="recon", cabeca="mapa", peso=5.0, perda="focal", lr=1e-3),
    "psom":           dict(som="psom", cabeca="mapa", peso=5.0, perda="focal", lr=1e-3),
    "peso_cabeca_1":  dict(som="atual", cabeca="mapa", peso=1.0, perda="focal", lr=1e-3),
    "peso_cabeca_20": dict(som="atual", cabeca="mapa", peso=20.0, perda="focal", lr=1e-3),
    "taxa_3e-4":      dict(som="atual", cabeca="mapa", peso=5.0, perda="focal", lr=3e-4),
    "bce_pesos":      dict(som="atual", cabeca="mapa", peso=5.0, perda="bce_pesos", lr=1e-3),
    "mapa_e_latente": dict(som="atual", cabeca="mapa_latente", peso=5.0, perda="focal", lr=1e-3),
}
CASOS_TRIAGEM = [(60, "anéis e área central"), (30, "sem chuva")]
log = logging.getLogger("etapa_c")


def configurar_log():
    OUT.mkdir(parents=True, exist_ok=True)
    at.OUT = OUT
    ea.CACHE_A.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    r = logging.getLogger(); r.handlers.clear()
    for hd in (logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "etapa_c.log", mode="a")):
        hd.setFormatter(fmt); r.addHandler(hd)
    r.setLevel(logging.INFO)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 500)


def pod_fracoes(y, p, fracoes=(0.03, 0.05, 0.10)):
    """POD médio com o alerta ligado nas frações de tempo indicadas."""
    v = []
    for f in fracoes:
        thr = np.quantile(p, 1 - f)
        a = p > thr
        if not a.any():
            a = p >= thr
        v.append((a & y).sum() / max(1, y.sum()))
    return float(np.mean(v))


def divisao(cache, fold, h):
    """Treino, validação interna (última temporada de treino) e teste."""
    tr_all, te = cache.split(fold, None, h)
    temp = cache.A.fold.to_numpy()
    sval = max(t for t in np.unique(temp[tr_all]) if t != fold)
    return tr_all[temp[tr_all] != sval], tr_all[temp[tr_all] == sval], te, tr_all, int(sval)


# ═══════════════════════════════ modelo e treino ═════════════════════════════
def psom_termos(m, z, alfa=10.0):
    """Pertinência t de Student, endurecimento (CAH) e vizinhança suave (Manduchi et al., 2021)."""
    import torch
    d = m.dist(z)
    s = (1 + d / alfa) ** (-(alfa + 1) / 2)
    s = s / s.sum(1, keepdim=True)
    t = s ** 2 / s.sum(0, keepdim=True)
    t = (t / t.sum(1, keepdim=True)).detach()
    cah = (t * (torch.log(t + 1e-9) - torch.log(s + 1e-9))).sum(1).mean()
    A = (m.mdist == 1).float()
    ssom = -(s * (torch.log(s + 1e-9) @ A.T)).sum(1).mean() / A.sum(1).mean()
    return s, cah, ssom


def treinar(cache, chuva, h, tr_in, va_in, tr_all, var, epocas, rotulo):
    """Treina SOM e cabeça do zero e devolve (modelo, normalizador, histórico, melhor nota)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from core import somgoes_nucleo as nu
    from analise import sintese_somgoes as si
    torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED)
    norm = cache.normalizador(tr_all)
    m = nu.SOMGoes(cache, cfg.GOES_CENARIOS, "ae", (10, 10), chuva=chuva)
    K, lat = m.prot.shape
    m.cab = nn.Sequential(nn.Linear(K + (lat if var["cabeca"] == "mapa_latente" else 0), 64), nn.ReLU(),
                          nn.Linear(64, len(NIVEIS)))
    m.to(nu.DEV)
    opt = torch.optim.Adam(m.parameters(), lr=var["lr"])
    yA = cache.rotulos(np.arange(len(cache.A)), h)
    amo = nu.Amostrador(tr_in, yA["transicao"][tr_in], cfg.LOTE)
    niv = torch.tensor(NIVEIS, device=nu.DEV)
    nu.pre_treino(m, cache, tr_in, norm, yA, amo, opt, LOTES)

    def pertinencia(z):
        if var["som"] == "psom":
            return psom_termos(m, z)[0]
        return nu.soft(m.dist(z))

    def saida(z):
        q = pertinencia(z)
        return m.cab(torch.cat([q, z], 1) if var["cabeca"] == "mapa_latente" else q)

    def perda_cabeca(logit, alvo):
        if var["perda"] == "focal":
            return eb.focal(logit, alvo)
        p = alvo.mean(0).clamp(1e-3, 1 - 1e-3)
        return F.binary_cross_entropy_with_logits(logit, alvo, pos_weight=torch.sqrt((1 - p) / p))

    def lote(ix):
        return {k: v.to(nu.DEV) for k, v in cache.lote(ix, norm, m.goes_modo, m.chuva).items()}

    def prever(ix):
        m.eval(); out = []
        with torch.no_grad():
            for a in range(0, len(ix), 2048):
                z, _, _ = m.codificar(lote(ix[a:a + 2048]))
                out.append(torch.sigmoid(saida(z)).cpu().numpy())
        return np.concatenate(out)

    prox_va = yA["prox"][va_in]; ok_va = prox_va >= 0; y_va = prox_va[ok_va] >= 3
    T0 = max(m.grade) / 2
    melhor, estado, sem, hist = -1.0, None, 0, []
    for ep in range(epocas):
        T = max(T0 * (1 / T0) ** (ep / max(epocas - 1, 1)), 0.5)
        m.train(); acc = {"som": 0.0, "cabeca": 0.0}
        for _ in range(LOTES):
            ix = amo(); b = lote(ix)
            yb = {k: torch.from_numpy(v[ix]).to(nu.DEV) for k, v in yA.items()}
            opt.zero_grad()
            if var["som"] == "psom":
                l_rec, _ = nu.perdas(m, b, yb, "pre", T, ep, epocas)
                z, _, _ = m.codificar(b)
                _, cah, ssom = psom_termos(m, z)
                l_som = l_rec + 0.1 * cah + 0.1 * ssom
            else:
                l_som, _ = nu.perdas(m, b, yb, si.PERDA_OP if var["som"] == "atual" else "recon", T, ep, epocas)
                z, _, _ = m.codificar(b)
            alvo = (yb["prox"].unsqueeze(1) >= niv.unsqueeze(0)).float()
            l_cab = perda_cabeca(saida(z), alvo)
            (l_som + var["peso"] * l_cab).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            acc["som"] += float(l_som) / LOTES; acc["cabeca"] += float(l_cab) / LOTES
        pv = prever(va_in)[ok_va, 2]
        nota = pod_fracoes(y_va, pv) if y_va.any() else 0.0
        ap = average_precision_score(y_va, pv) if y_va.any() else 0.0
        hist.append(dict(epoca=ep, perda_som=round(acc["som"], 4), perda_cabeca=round(acc["cabeca"], 4),
                         POD_medio_validacao_10mm=round(nota, 4), AP_validacao_10mm=round(ap, 4)))
        log.info(f"    {rotulo} época {ep}: perda SOM {acc['som']:.4f}, cabeça {acc['cabeca']:.4f}, "
                 f"POD médio na validação (10 mm/h) {nota:.3f}")
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


def calibrar_e_prever(m, cache, h, va_in, te):
    """Calibração logística por limiar na temporada de validação interna (o lote balanceado
    distorce as probabilidades) e previsão na temporada avaliada."""
    prox_va = cache.rotulos(va_in, h)["prox"]; ok = prox_va >= 0
    pv, pt = m.prever(va_in), m.prever(te)
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, None))
    cal = np.empty_like(pt)
    for i, nv in enumerate(NIVEIS):
        yy = prox_va[ok] >= nv
        if yy.any() and (~yy).any():
            c = LogisticRegression(C=1e6).fit(lg(pv[ok, i])[:, None], yy)
            cal[:, i] = c.predict_proba(lg(pt[:, i])[:, None])[:, 1]
        else:
            cal[:, i] = pt[:, i]
    return cal


# ═══════════════════════════════ C1 triagem ══════════════════════════════════
def c1(cache, conj):
    arq = OUT / "variante_escolhida.json"
    linhas, HIST = [], []
    variantes = list(VARIANTES)[:2] if TESTE else list(VARIANTES)
    casos = CASOS_TRIAGEM[:1] if TESTE else CASOS_TRIAGEM
    for nome in variantes:
        notas = []
        for h, cj in casos:
            chuva = conj[cj][1]
            tr_in, va_in, te, tr_all, sval = divisao(cache, FOLD_TRIAGEM, h)
            t0 = time.time()
            # a triagem usa só as temporadas de treino da temporada FOLD_TRIAGEM
            _, hist, nota = treinar(cache, chuva, h, tr_in, va_in, tr_all, VARIANTES[nome], EPOCAS_TRIAGEM,
                                    f"C1 {nome} h{h} {cj}")
            HIST.append(hist.assign(variante=nome, horizonte=h, entradas=cj))
            notas.append(nota)
            linhas.append(dict(variante=nome, horizonte=h, entradas=cj, POD_medio_validacao_10mm=round(nota, 4),
                               epocas=len(hist), minutos=round((time.time() - t0) / 60, 1)))
            log.info(f"  C1 {nome} h{h} {cj}: POD médio na validação {nota:.3f} em {(time.time() - t0) / 60:.1f} min")
        linhas.append(dict(variante=nome, horizonte="média", entradas="", POD_medio_validacao_10mm=round(np.mean(notas), 4)))
    T = pd.DataFrame(linhas)
    at.salvar("C1_triagem_de_variantes", T)
    at.salvar("C1_historico_do_treino", pd.concat(HIST, ignore_index=True), mostrar=False)
    med = T[T.horizonte == "média"].sort_values("POD_medio_validacao_10mm", ascending=False)
    escolhida = med.variante.iloc[0]
    json.dump(dict(variante=escolhida, **VARIANTES[escolhida]), open(arq, "w"), ensure_ascii=False, indent=1)
    log.info(f"variante escolhida pela validação interna: {escolhida}")
    return escolhida


# ═══════════════════════════════ C2 rodada final ═════════════════════════════
def c2(cache, conj, dados, escolhida):
    var = VARIANTES[escolhida]
    PF, PAR, HIST = [], [], []
    combos = [(30, "sem chuva")] if TESTE else [(h, c) for h in (30, 60) for c in conj]
    for h, cj in combos:
        E, X, chuva = dados(h, cj)
        prox = cache.A[f"estado_prox{at.sufixo(h)}"].to_numpy()
        fE, pE = E.fold.to_numpy(), E.pos.to_numpy()
        folds = sorted(np.unique(fE))[:1] if TESTE else sorted(np.unique(fE))
        S = {"SOM com cabeça": {nv: np.full(len(E), np.nan) for nv in NIVEIS},
             "gradient boosting, variáveis novas": {nv: np.full(len(E), np.nan) for nv in NIVEIS}}
        for k in folds:
            arq = OUT / "probs" / f"som_cabeca_{escolhida}_h{h}_{chuva}_fold{k}.csv.gz"
            arq.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            if arq.exists():
                P = pd.read_csv(arq, sep=";", index_col=0)
            else:
                tr_in, va_in, te, tr_all, sval = divisao(cache, k, h)
                m, hist, nota = treinar(cache, chuva, h, tr_in, va_in, tr_all, var, EPOCAS_FINAL, f"C2 h{h} {cj} temporada {k}")
                P = pd.DataFrame(calibrar_e_prever(m, cache, h, va_in, te), index=te,
                                 columns=[f"p{nv}" for nv in NIVEIS])
                P.to_csv(arq, sep=";")
                HIST.append(hist.assign(horizonte=h, entradas=cj, temporada=k, temporada_validacao=sval))
                PAR.append(dict(horizonte=h, entradas=cj, temporada=k, temporada_validacao=sval, epocas=len(hist),
                                POD_medio_validacao_10mm=round(nota, 4)))
                del m
            te_E = fE == k
            for nv in NIVEIS:
                S["SOM com cabeça"][nv][te_E] = P[f"p{nv}"].reindex(pE[te_E]).to_numpy()
            tr_E = ~te_E; ytr = prox[pE[tr_E]]; ok = ytr >= 0
            for nv in NIVEIS:
                S["gradient boosting, variáveis novas"][nv][te_E] = mt.ajustar(
                    "HistGradientBoosting", X[tr_E][ok], ytr[ok] >= nv, X[te_E], None)
            log.info(f"  C2 h{h} {cj}: temporada {k} pronta em {(time.time() - t0) / 60:.1f} min")
        y_est = prox[pE]; usadas = np.isin(fE, folds) & (y_est >= 0)
        for rot, por in S.items():
            for nv in NIVEIS:
                yy = y_est >= nv
                PF += [dict(x_mm_h=X_MM[nv], **d) for d in rt.pod_far(yy[usadas], por[nv][usadas], rot, h, cj)]
        # comparação por temporada, 10 mm/h, alerta em 5% do tempo
        for k in folds:
            mk = usadas & (fE == k)
            lin = dict(horizonte=h, entradas=cj, temporada=k)
            for rot, por in S.items():
                d = [x for x in rt.pod_far(y_est[mk] >= 3, por[3][mk], rot, h, cj) if x["ponto"] == "alerta em 5% do tempo"][0]
                lin[f"POD {rot}"] = d["POD"]; lin[f"FAR {rot}"] = d["FAR"]
            PAR.append(lin)
    if HIST:
        at.salvar("C2_historico_do_treino", pd.concat(HIST, ignore_index=True), mostrar=False)
    at.salvar("C2_por_temporada", pd.DataFrame(PAR))
    at.salvar("C2_pod_far", pd.DataFrame(PF), mostrar=False)
    eb.tabela_pod_far(PF, f"C2, seis temporadas, variante {escolhida}")


def main():
    configurar_log()
    t00 = time.time()
    partes = sys.argv[1:] or ["c1", "c2"]
    import torch
    log.info(f"etapa C {'(TESTE) ' if TESTE else ''}| partes {partes} | torch {torch.__version__} | dispositivo "
             f"{ea.dispositivo()} | triagem na temporada {FOLD_TRIAGEM} | épocas {EPOCAS_TRIAGEM} e {EPOCAS_FINAL}")
    cache, conj, dados = eb.preparar()
    escolhida = os.environ.get("VARIANTE")
    if "c1" in partes and not escolhida:
        escolhida = c1(cache, conj)
    if "c2" in partes:
        if not escolhida:
            arq = OUT / "variante_escolhida.json"
            escolhida = json.load(open(arq))["variante"] if arq.exists() else "base"
        c2(cache, conj, dados, escolhida)
    log.info(f"fim em {(time.time() - t00) / 60:.1f} min; saídas em {OUT}")


if __name__ == "__main__":
    main()
