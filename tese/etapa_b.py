#!/usr/bin/env python
"""
etapa_b.py — dois testes para tentar superar o gradient boosting.

  B1  gradient boosting com o histórico de transições no SOM, nas seis temporadas. Além das
      variáveis novas da etapa A, recebe o nó agora, 10 e 30 min antes, as regiões agora e 30 min
      antes (numeradas de 1 a 8 pela frequência de chuva forte), o deslocamento no mapa e as
      frequências de chuva forte do nó, do par de regiões e do nó com a região anterior. As
      frequências dos horários de treino são calculadas sem a própria temporada.
      Também roda o gradient boosting só com essas informações do SOM.
  B2  SOM e cabeça de previsão treinados juntos, numa temporada fora (FOLD, padrão 2). Parte do
      SOM já treinado daquela temporada e continua o treino somando à perda do SOM uma perda
      focal da cabeça, que dá ao mesmo tempo a probabilidade de passar de 0,2, 2,5, 10 e 20 mm/h
      no horizonte. Duas versões, cabeça só no mapa (pertinência suave aos 100 nós) e cabeça no
      mapa e no latente. A parada antecipada e a calibração usam uma temporada de treino
      separada, nunca a temporada avaliada. A comparação, na mesma temporada, é com um gradient
      boosting por limiar de chuva e com a previsão pela chuva atual.

POD, FAR e CSI por limiar de probabilidade e com alerta em 2, 5 e 10% do tempo, por horário de
10 min. Saídas em etapa_b/ e log mestre etapa_b/etapa_b.log.
Uso (da raiz):  python -m tese.etapa_b                  (B1 e B2)
                python -m tese.etapa_b b1  |  b2
                ETAPA_B_TESTE=1 python -m tese.etapa_b   (teste curto, saída em etapa_b/teste/)
"""
from __future__ import annotations

import logging
import os
import sys
import time

TESTE = os.environ.get("ETAPA_B_TESTE", "0") == "1"
if TESTE:
    os.environ.setdefault("RAPIDO", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from core import config_somgoes as cfg
from tese import etapa_a as ea
from tese import regioes_transicoes as rt

at, mt, f1 = ea.at, ea.mt, ea.f1
OUT = cfg.ROOT / "etapa_b" / ("teste" if TESTE else "")
FOLD = int(os.environ.get("FOLD", 2))
EPOCAS_B = 1 if TESTE else int(os.environ.get("EPOCAS_B", 12))
LOTES_B = 10 if TESTE else cfg.LOTES_EPOCA
LAMBDA = 5.0                       # peso da perda da cabeça em relação à perda do SOM
NIVEIS = (1, 2, 3, 4)              # estado >= 1, 2, 3, 4  =  >= 0,2, 2,5, 10, 20 mm/h
X_MM = {1: "0,2", 2: "2,5", 3: "10", 4: "20"}
REF = "gradient boosting, variáveis novas (referência)"
log = logging.getLogger("etapa_b")


def configurar_log():
    OUT.mkdir(parents=True, exist_ok=True)
    at.OUT = OUT; ea.OUT = OUT
    ea.CACHE_A.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    r = logging.getLogger(); r.handlers.clear()
    for hd in (logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "etapa_b.log", mode="w")):
        hd.setFormatter(fmt); r.addHandler(hd)
    r.setLevel(logging.INFO)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 500)


def preparar():
    """Mesmo conjunto de dados e variáveis da etapa A."""
    cache = mt.cache_som()
    A = at.carregar(); B = at.blocos(A)
    V, P = ea.variaveis_novas(cache, A)
    for Dd in (V, P):
        for c in Dd.columns:
            A[c] = Dd[c].to_numpy()[A.pos.to_numpy()]
    GF = [c for c in V.columns if c.startswith("g") and c[1:3] in ("15", "30", "60")] + ["nuvens_vel_kmh"]
    PL_ = list(P.columns) + [c for c in V.columns if c.startswith("in_c1_mont")]
    conj = {"sem chuva": (("EMs", "EMs_tend", "GOES", "GOES_tend", "hora"), "nenhuma"),
            "anéis e área central": (("EMs", "EMs_tend", "GOES", "GOES_tend", "hora", "colar1", "colar2", "nucleo"),
                                     "colar_nucleo")}
    def dados(h, nome):
        E = at.conjunto(A, h)
        blocos, chuva = conj[nome]
        MONT = [c for c in V.columns if c.startswith(f"mont{h}_")]
        novas = [c for b in blocos for c in B[b]] + GF + MONT + (PL_ if chuva != "nenhuma" else [])
        X = E[novas].to_numpy(float); X[~np.isfinite(X)] = np.nan
        return E, X, chuva
    return cache, conj, dados


def escores_a(h, chuva, E):
    arq = cfg.ROOT / "etapa_a" / f"escores_h{h}_{chuva}.csv.gz"
    if not arq.exists():
        return None
    S = pd.read_csv(arq, sep=";", parse_dates=["ts"])
    return E[["ts"]].merge(S, on="ts", how="left")


# ═══════════════════════════════ B1  transições no gradient boosting ═════════
def transicoes_dobra(R, yfull, temp, jan, ncache):
    """Variáveis de transição no SOM da dobra, para as linhas de R['idx'] (treino e depois teste).
    Frequências do treino calculadas sem a própria temporada; as do teste, com todo o treino."""
    D = rt.dobra(R, yfull, temp, jan, ncache)        # regiões numeradas e nó 30 min antes
    idx, ntr = R["idx"], R["ntr"]; n = len(idx)
    bmu = R["d"].argmin(1); y = D["y"]; s = temp[idx]; posmap = D["posmap"]
    itr = np.arange(ntr); ite = np.arange(ntr, n)
    def anterior(passos):
        jp = jan[idx, -1 - passos]
        p_ = np.where(jp >= 0, posmap[np.clip(jp, 0, None)], -1)
        return np.where(p_ >= 0, bmu[np.clip(p_, 0, None)], -1)
    b10, b30 = anterior(1), anterior(3)
    rg, rgp = D["rg"], D["rgp"]
    K9 = ea.N_REG + 1
    chaves = {"tx_no": (bmu, 100), "tx_no_30": (np.where(b30 >= 0, b30, -1), 100),
              "tx_par_regioes": (np.where(rgp > 0, rgp * K9 + rg, -1), K9 * K9),
              "tx_no_regiao_30": (np.where(rgp > 0, bmu * K9 + rgp, -1), 100 * K9)}
    out = {}
    for nome, (ch, nc) in chaves.items():
        v = np.full(n, np.nan)
        for si_ in np.unique(s[itr]):
            a = itr[s[itr] != si_]; b = itr[s[itr] == si_]
            base = np.full(n, (y[a].sum() + 1) / (len(a) + 2))
            v[b] = rt.tabela(ch, base, y, a, b, 10, nc)
        base = np.full(n, (y[itr].sum() + 1) / (len(itr) + 2))
        v[ite] = rt.tabela(ch, base, y, itr, ite, 10, nc)
        v[ch < 0] = np.nan
        out[nome] = v
    dist = lambda a, b: np.where((a >= 0) & (b >= 0), np.abs(a // 10 - b // 10) + np.abs(a % 10 - b % 10), np.nan)
    T = np.column_stack([bmu, np.where(b10 >= 0, b10, np.nan), np.where(b30 >= 0, b30, np.nan),
                         rg, np.where(rgp > 0, rgp, np.nan), dist(bmu, b10), dist(bmu, b30),
                         out["tx_no"], out["tx_no_30"], out["tx_no"] - out["tx_no_30"],
                         out["tx_par_regioes"], out["tx_no_regiao_30"]]).astype(float)
    return T, posmap, ntr


NOMES_T = ["no", "no_10min_antes", "no_30min_antes", "regiao", "regiao_30min_antes", "desloc_10min",
           "desloc_30min", "tx_no", "tx_no_30min_antes", "variacao_tx_no", "tx_par_regioes", "tx_no_e_regiao_30min"]


def b1(cache, conj, dados):
    PF = []
    for h in ((30,) if TESTE else (30, 60)):
        yfull = cache.A[f"estado_prox{at.sufixo(h)}"].to_numpy() >= 3
        temp = cache.A.fold.to_numpy()
        for nome in conj:
            E, X, chuva = dados(h, nome)
            n = len(E); y = E.forte.to_numpy(); fE = E.fold.to_numpy(); pE = E.pos.to_numpy()
            L = ea.latentes(chuva, h, cache, 2 if TESTE else None)
            s_xt = np.full(n, np.nan); s_t = np.full(n, np.nan)
            for k, R in L.items():
                t0 = time.time()
                T, posmap, ntr = transicoes_dobra(R, yfull, temp, cache.jan, len(cache.A))
                j = posmap[pE]
                te = (fE == k) & (j >= ntr); tr = (fE != k) & (j >= 0) & (j < ntr)
                if not te.any() or y[tr].sum() < 10:
                    continue
                cat = [X.shape[1], X.shape[1] + 1, X.shape[1] + 2]
                s_xt[te] = mt.ajustar("HistGradientBoosting", np.column_stack([X[tr], T[j[tr]]]), y[tr],
                                      np.column_stack([X[te], T[j[te]]]), cat)
                s_t[te] = mt.ajustar("HistGradientBoosting", T[j[tr]], y[tr], T[j[te]], [0, 1, 2])
                log.info(f"  B1 h{h} {nome}: temporada {k} em {(time.time() - t0) / 60:.1f} min")
            Sa = escores_a(h, chuva, E)
            usadas = np.isfinite(s_xt)
            S = {"gradient boosting, variáveis novas + transições no SOM": s_xt,
                 "gradient boosting só com nó e transições no SOM": s_t}
            if Sa is not None:
                for c in (REF, "SOM, probabilidade suave", "SOM, trajetória (nó agora e 30 min antes)", "persistência"):
                    if c in Sa:
                        S[c] = Sa[c].to_numpy(float)
            for nm, s in S.items():
                PF += [dict(x_mm_h="10", **d) for d in rt.pod_far(y[usadas], s[usadas], nm, h, nome)]
            dif = ea.diferenca_pareada(E.assign(), {k: np.where(usadas, v, np.nan) for k, v in S.items()}, h, nome) \
                if REF in S else []
            if dif:
                at.salvar(f"B1_diferenca_pareada_AP_h{h}_{chuva}", pd.DataFrame(dif))
    return PF


# ═══════════════════════════════ B2  SOM e cabeça treinados juntos ════════════
def focal(logit, alvo, gamma=2.0):
    import torch.nn.functional as F
    import torch
    ce = F.binary_cross_entropy_with_logits(logit, alvo, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def b2_uma(cache, chuva, h, usa_z, X, E):
    """Treina a cabeça junto com o SOM da temporada FOLD e devolve as probabilidades calibradas
    (uma coluna por limiar) para os horários de teste, indexadas pela posição no cache."""
    import torch
    import torch.nn as nn
    from core import somgoes_nucleo as nu
    from analise import sintese_somgoes as si
    torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED)
    base = cfg.RESULTS_DIR / "cenarios" / f"cenarios_{chuva}_h{h}_{si.PERDA_OP}_10x10" / f"fold{FOLD}" / "melhor.pt"
    tr_all, te = cache.split(FOLD, None, h)
    temp = cache.A.fold.to_numpy()
    sval = max(t for t in np.unique(temp[tr_all]) if t != FOLD)
    va_in = tr_all[temp[tr_all] == sval]; tr_in = tr_all[temp[tr_all] != sval]
    norm = cache.normalizador(tr_all)
    m = nu.SOMGoes(cache, cfg.GOES_CENARIOS, "ae", (10, 10), chuva=chuva)
    m.load_state_dict(torch.load(base, map_location=nu.DEV, weights_only=False)["model"])
    K = m.prot.shape[0]; lat = m.prot.shape[1]
    m.cab = nn.Sequential(nn.Linear(K + (lat if usa_z else 0), 64), nn.ReLU(), nn.Linear(64, len(NIVEIS)))
    m.to(nu.DEV)
    opt = torch.optim.Adam([{"params": [p for nm_, p in m.named_parameters() if not nm_.startswith("cab.")], "lr": 3e-4},
                            {"params": m.cab.parameters(), "lr": 1e-3}])
    yA = cache.rotulos(np.arange(len(cache.A)), h)
    amo = nu.Amostrador(tr_in, yA["transicao"][tr_in], cfg.LOTE)
    niv = torch.tensor(NIVEIS, device=nu.DEV)

    def saida(b):
        z, _, _ = m.codificar(b)
        q = nu.soft(m.dist(z))
        return m.cab(torch.cat([q, z], 1) if usa_z else q)

    def prever(ix):
        m.eval(); out = []
        with torch.no_grad():
            for a in range(0, len(ix), 2048):
                b = {k: v.to(nu.DEV) for k, v in cache.lote(ix[a:a + 2048], norm, m.goes_modo, m.chuva).items()}
                out.append(torch.sigmoid(saida(b)).cpu().numpy())
        return np.concatenate(out)

    prox_va = yA["prox"][va_in]
    melhor, estado, sem, hist = -1.0, None, 0, []
    for ep in range(EPOCAS_B):
        m.train(); acc = [0.0, 0.0]
        for _ in range(LOTES_B):
            ix = amo()
            b = {k: v.to(nu.DEV) for k, v in cache.lote(ix, norm, m.goes_modo, m.chuva).items()}
            yb = {k: torch.from_numpy(v[ix]).to(nu.DEV) for k, v in yA.items()}
            opt.zero_grad()
            l_som, _ = nu.perdas(m, b, yb, si.PERDA_OP, 0.5, cfg.EPOCAS, cfg.EPOCAS)
            alvo = (yb["prox"].unsqueeze(1) >= niv.unsqueeze(0)).float()
            l_cab = focal(saida(b), alvo)
            (l_som + LAMBDA * l_cab).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            acc[0] += float(l_som) / LOTES_B; acc[1] += float(l_cab) / LOTES_B
        pv = prever(va_in)
        ok = prox_va >= 0
        ap = average_precision_score(prox_va[ok] >= 3, pv[ok, 2]) if (prox_va[ok] >= 3).any() else 0.0
        hist.append(dict(epoca=ep, perda_som=round(acc[0], 4), perda_cabeca=round(acc[1], 4), AP_validacao_10mm=round(ap, 4)))
        log.info(f"    época {ep}: perda SOM {acc[0]:.4f}, cabeça {acc[1]:.4f}, AP validação (10 mm/h) {ap:.4f}")
        if ap > melhor:
            melhor, sem = ap, 0
            estado = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            sem += 1
            if sem >= 3:
                break
    m.load_state_dict(estado)
    # calibração na temporada de validação (o lote balanceado distorce as probabilidades)
    pv = prever(va_in); pt = prever(te)
    ok = prox_va >= 0
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, None))
    cal = np.empty_like(pt)
    for i, nv in enumerate(NIVEIS):
        yy = prox_va[ok] >= nv
        if yy.any() and (~yy).any():
            c = LogisticRegression(C=1e6).fit(lg(pv[ok, i])[:, None], yy)
            cal[:, i] = c.predict_proba(lg(pt[:, i])[:, None])[:, 1]
        else:
            cal[:, i] = pt[:, i]
    return pd.DataFrame(cal, index=te, columns=[f"p{nv}" for nv in NIVEIS]), pd.DataFrame(hist), sval


def b2(cache, conj, dados):
    PF, HIST = [], []
    combos = [(30, "sem chuva")] if TESTE else [(h, c) for h in (30, 60) for c in conj]
    for h, nome in combos:
        E, X, chuva = dados(h, nome)
        sel = E.fold.to_numpy() == FOLD
        Et = E[sel].reset_index(drop=True); pos = Et.pos.to_numpy()
        prox = cache.A[f"estado_prox{at.sufixo(h)}"].to_numpy()
        agora = Et.agora.to_numpy(float)
        S = {}
        for usa_z, rot in ((False, "SOM e cabeça treinados juntos, cabeça só no mapa"),
                           (True, "SOM e cabeça treinados juntos, cabeça no mapa e no latente")):
            t0 = time.time()
            P, hist, sval = b2_uma(cache, chuva, h, usa_z, X, E)
            HIST.append(hist.assign(horizonte=h, entradas=nome, modelo=rot, temporada_validacao=sval))
            S[rot] = {nv: P[f"p{nv}"].reindex(pos).to_numpy() for nv in NIVEIS}
            log.info(f"  B2 h{h} {nome}: {rot} em {(time.time() - t0) / 60:.1f} min")
        # gradient boosting por limiar de chuva, mesma temporada
        Etr = E[~sel]; ytr_estado = prox[Etr.pos.to_numpy()]; ok = ytr_estado >= 0
        S["gradient boosting, variáveis novas"] = {
            nv: mt.ajustar("HistGradientBoosting", X[~sel][ok], ytr_estado[ok] >= nv, X[sel], None) for nv in NIVEIS}
        S["previsão pela chuva atual"] = {nv: agora + 1e-3 * np.nan_to_num(at.col(Et, "in_nuc_max10")) - nv + 1
                                          for nv in NIVEIS}
        S_media = {nv: 0.5 * (S["gradient boosting, variáveis novas"][nv] +
                              S["SOM e cabeça treinados juntos, cabeça só no mapa"][nv]) for nv in NIVEIS}
        S["média do gradient boosting e do SOM com cabeça (só no mapa)"] = S_media
        y_est = prox[pos]
        for rot, por_nivel in S.items():
            for nv in NIVEIS:
                yy = y_est >= nv; okk = y_est >= 0
                nm_ = "persistência" if rot.startswith("previsão pela chuva") else rot
                lin = rt.pod_far(yy[okk], por_nivel[nv][okk], nm_, h, nome)
                for d in lin:
                    d["modelo"] = rot
                PF += [dict(x_mm_h=X_MM[nv], temporada=FOLD, **d) for d in lin]
    if HIST:
        at.salvar("B2_historico_do_treino", pd.concat(HIST, ignore_index=True))
    return PF


def tabela_pod_far(PF, titulo):
    if not len(PF):
        return
    PF = pd.DataFrame(PF)
    pts = ["alerta em 5% do tempo", "probabilidade >= 0.10", "probabilidade >= 0.20", "probabilidade >= 0.30"]
    for (x, h, conj), g in PF[PF.ponto.isin(pts)].groupby(["x_mm_h", "horizonte", "entradas"], sort=False):
        w = g.pivot_table(index="modelo", columns="ponto", values=["POD", "FAR"], sort=False).round(2)
        w = w[[c for c in [("POD", p) for p in pts] + [("FAR", p) for p in pts] if c in w.columns]]
        log.info(f"\n### {titulo}: chuva acima de {x} mm/h, h{h}, {conj}\n{w.to_string()}\n")


def main():
    configurar_log()
    t00 = time.time()
    partes = sys.argv[1:] or ["b1", "b2"]
    import torch
    log.info(f"etapa B {'(TESTE) ' if TESTE else ''}| partes {partes} | torch {torch.__version__} | "
             f"dispositivo {ea.dispositivo()} | temporada do B2 {FOLD} | épocas {EPOCAS_B} | peso da cabeça {LAMBDA}")
    cache, conj, dados = preparar()
    if "b1" in partes:
        PF = b1(cache, conj, dados)
        if PF:
            at.salvar("B1_pod_far", pd.DataFrame(PF), mostrar=False)
            tabela_pod_far(PF, "B1, seis temporadas")
    if "b2" in partes:
        PF = b2(cache, conj, dados)
        if PF:
            at.salvar("B2_pod_far", pd.DataFrame(PF), mostrar=False)
            tabela_pod_far(PF, f"B2, temporada {FOLD}")
    log.info(f"fim em {(time.time() - t00) / 60:.1f} min; saídas em {OUT}")


if __name__ == "__main__":
    main()
