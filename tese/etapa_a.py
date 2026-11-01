#!/usr/bin/env python
"""
etapa_a.py — etapa A do capítulo, sem retreinar o SOM.

Usa o SOM profundo já treinado de cada temporada (resultados_somgoes/cenarios), sempre o modelo
que não viu a temporada avaliada, e as mesmas dobras (uma temporada fora por vez) da fase 1.

  A0  saídas do SOM por temporada: nó, frequência de chuva forte do nó (nos horários de treino,
      calculada sem a própria temporada), probabilidade suave com a grade de temperatura ampliada
      até 50 e regiões do mapa (agrupamento de Ward dos protótipos)
  A1  diagnóstico dos inícios a partir do seco: nó, posição no mapa, região, taxa do nó, nó
      30 min antes e posição do escore entre os horários secos da temporada
  A2  probabilidade pela trajetória: P(chuva forte | nó agora, nó 30 min antes), com o nó atual
      como valor de partida quando o par é raro (peso escolhido dentro do treino)
  A3  análogos no espaço latente: estado atual e sequência de 30 min (t-30, t-20, t-10, t)
  A4  segundos estágios, todos com as mesmas dobras, a mesma semente e hiperparâmetros fixos
        logística com o nó como ponto de partida (logito da taxa do nó como termo fixo)
        TabPFN v2 por região do SOM e um TabPFN único (amostra de até 10 mil horários de treino,
        com correção da proporção de chuva forte)
        especialistas por região (gradient boosting), com atribuição dura e com mistura pelos
        pesos suaves do SOM
        modelos de sequência sobre o latente (LSTM, GRU, TCN e Transformer pequeno), 6 passos
        de 10 min, parada antecipada numa temporada de treino separada
        gradient boosting com as variáveis novas e com o nó e o latente, como referência
  A5  avaliação: AP e demais métricas por temporada, diferença pareada de AP em relação ao
      gradient boosting de referência e inícios antecipados (posição na ordenação entre os
      horários secos da mesma temporada, sem contar empates, com intervalo de Wilson)

Saídas em etapa_a/ (CSVs) e log mestre etapa_a/etapa_a.log. Variáveis novas e latentes ficam em
etapa_a/cache/ e são reaproveitados numa nova rodada.
Uso (da raiz):  python -m tese.etapa_a
                ETAPA_A_TESTE=1 python -m tese.etapa_a     (h30, anéis e área central, 2 temporadas,
                                                           poucas épocas; saída em etapa_a/teste/)
                SEM_TABPFN=1, SEM_SEQ=1, COM_CATBOOST=1 ligam ou desligam partes
"""
from __future__ import annotations

import logging
import os
import sys
import time
import warnings

TESTE = os.environ.get("ETAPA_A_TESTE", "0") == "1"
if TESTE:
    os.environ.setdefault("RAPIDO", "1")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from core import config_somgoes as cfg
from tese import fase1_som as f1
from tese import modelos_tese as mt

at = mt.at
OUT = cfg.ROOT / "etapa_a" / ("teste" if TESTE else "")
CACHE_A = cfg.ROOT / "etapa_a" / "cache"
SEM_TABPFN = os.environ.get("SEM_TABPFN", "0") == "1"
SEM_SEQ = os.environ.get("SEM_SEQ", "0") == "1"
COM_CATBOOST = os.environ.get("COM_CATBOOST", "0") == "1"
TAUS = f1.TAUS
ALFAS = (1, 3, 10, 30, 100)
N_REG = mt.N_REGIOES
K = 100                                   # nós da grade 10 x 10
L_SEQ = 6                                 # passos de 10 min nos modelos de sequência (t-50 a t)
L_ANAL = 4                                # passos dos análogos de sequência (t-30 a t)
N_TABPFN = 2000 if TESTE else 10000
EPOCAS = 2 if TESTE else 30
PACIENCIA = 5
SEED = cfg.SEED
REF = "gradient boosting, variáveis novas (referência)"
log = logging.getLogger("etapa_a")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def configurar_log():
    OUT.mkdir(parents=True, exist_ok=True); CACHE_A.mkdir(parents=True, exist_ok=True)
    at.OUT = OUT
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    r = logging.getLogger(); r.handlers.clear()
    for hd in (logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "etapa_a.log", mode="w")):
        hd.setFormatter(fmt); r.addHandler(hd)
    r.setLevel(logging.INFO)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 500)


def dispositivo():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════ dados ═══════════════════════════════════════
def variaveis_novas(cache, A):
    """Variáveis físicas do GOES, deslocamento das nuvens e defasagens dos pluviômetros da fase 1,
    calculadas uma vez e guardadas em etapa_a/cache."""
    av, ap = CACHE_A / "variaveis_goes.parquet", CACHE_A / "variaveis_pluvio.parquet"
    if av.exists() and ap.exists():
        V, P = pd.read_parquet(av), pd.read_parquet(ap)
        log.info(f"variáveis novas lidas de {CACHE_A}")
    else:
        V = f1.variaveis_goes_fisicas(cache, A); P = f1.variaveis_pluvio(cache)
        V.to_parquet(av); P.to_parquet(ap)
    return V, P


def latentes(chuva, h, cache, folds_max=None):
    """Latente, distâncias aos protótipos e protótipos do SOM de cada temporada, para os horários
    de treino e de teste daquela dobra. Um arquivo por dobra em etapa_a/cache."""
    import torch
    from core import somgoes_nucleo as nu
    from analise import sintese_somgoes as si
    base = cfg.RESULTS_DIR / "cenarios" / f"cenarios_{chuva}_h{h}_{si.PERDA_OP}_10x10"
    folds = sorted(int(p.name[4:]) for p in base.glob("fold*") if (p / "melhor.pt").exists())
    if folds_max:
        folds = folds[:folds_max]
    res = {}
    for fold in folds:
        arq = CACHE_A / f"latentes_{chuva}_h{h}_fold{fold}.npz"
        if arq.exists():
            g = np.load(arq)
            res[fold] = {k: g[k] for k in g.files}; res[fold]["ntr"] = int(res[fold]["ntr"])
            continue
        tr, va = cache.split(fold, None, h)
        norm = cache.normalizador(tr)
        m = nu.SOMGoes(cache, cfg.GOES_CENARIOS, "ae", (10, 10), chuva=chuva)
        m.load_state_dict(torch.load(base / f"fold{fold}" / "melhor.pt", map_location=nu.DEV,
                                     weights_only=False)["model"]); m.to(nu.DEV); m.eval()
        idx = np.r_[tr, va]; Z, D = [], []
        with torch.no_grad():
            for s in range(0, len(idx), 2048):
                b = {k: v.to(nu.DEV) for k, v in cache.lote(idx[s:s + 2048], norm, m.goes_modo, m.chuva).items()}
                z, _, _ = m.codificar(b)
                Z.append(z.cpu().numpy()); D.append(m.dist(z).cpu().numpy())
        R = dict(idx=idx, ntr=len(tr), z=np.concatenate(Z).astype(np.float32),
                 d=np.concatenate(D).astype(np.float32), prot=m.prot.detach().cpu().numpy().astype(np.float32))
        np.savez(arq, **R)
        res[fold] = R
        log.info(f"  latente {chuva} h{h}: temporada {fold} calculada")
    return res


# ═══════════════════════════════ saídas do SOM por dobra (A0, A2, A3) ════════
def logito(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def prob_traj(bmu, bprev, y, ajuste, pred, alfa):
    """P(forte | nó agora, nó 30 min antes) estimada nos horários `ajuste`, com o nó atual como
    valor de partida de peso alfa; sem nó anterior válido, usa só o nó atual."""
    p_now = f1.prob_nos(bmu[ajuste], y[ajuste])
    v = ajuste[bprev[ajuste] >= 0]
    chave = bmu * K + np.clip(bprev, 0, None)
    n = np.bincount(chave[v], minlength=K * K); f = np.bincount(chave[v][y[v]], minlength=K * K)
    kp = chave[pred]; pn = p_now[bmu[pred]]
    return np.where(bprev[pred] >= 0, (f[kp] + alfa * pn) / (n[kp] + alfa), pn)


def knn_media(Ztr, ytr, Zte, ks, lote=1024):
    """Fração de chuva forte entre os k horários de treino mais próximos (distância euclidiana)."""
    import torch
    dev = dispositivo()
    A = torch.from_numpy(np.ascontiguousarray(Ztr, np.float32)).to(dev)
    yt = torch.from_numpy(ytr.astype(np.float32)).to(dev)
    out = {k: np.empty(len(Zte)) for k in ks}; kmax = max(ks)
    with torch.no_grad():
        for s in range(0, len(Zte), lote):
            B = torch.from_numpy(np.ascontiguousarray(Zte[s:s + lote], np.float32)).to(dev)
            nn_ = torch.cdist(B, A).topk(kmax, largest=False).indices
            yy = yt[nn_]
            for k in ks:
                out[k][s:s + lote] = yy[:, :k].mean(1).cpu().numpy()
    return out


def som_dobra(R, yfull, temp, jan, ncache):
    """Tudo o que vem do SOM de uma dobra, indexado pelas linhas de R['idx'] (treino e depois teste)."""
    idx, ntr, z, d, prot = R["idx"], R["ntr"], R["z"], R["d"], R["prot"]
    n = len(idx); itr = np.arange(ntr); ite = np.arange(ntr, n)
    bmu = d.argmin(1); y = yfull[idx]; s = temp[idx]
    posmap = np.full(ncache, -1, np.int64); posmap[idx] = np.arange(n)
    escala = float(np.median(d[itr].min(1)))
    jp = jan[idx, -4]
    prev = np.where(jp >= 0, posmap[np.clip(jp, 0, None)], -1)
    bprev = np.where(prev >= 0, bmu[np.clip(prev, 0, None)], -1)
    # validação interna (uma temporada de treino fora por vez): taxa do nó, temperatura e alfa
    tx_cv = np.full(n, np.nan)
    notas_t = {t: [] for t in TAUS}; notas_a = {a: [] for a in ALFAS}
    for si_ in np.unique(s[itr]):
        a = itr[s[itr] != si_]; b = itr[s[itr] == si_]
        p = f1.prob_nos(bmu[a], y[a]); tx_cv[b] = p[bmu[b]]
        if y[b].sum() > 0:
            for t in TAUS:
                notas_t[t].append(average_precision_score(y[b], f1.suave(d[b], p, t, escala)))
            for al in ALFAS:
                notas_a[al].append(average_precision_score(y[b], prob_traj(bmu, bprev, y, a, b, al)))
    tau = max(TAUS, key=lambda t: np.mean(notas_t[t]) if notas_t[t] else -1)
    alfa = max(ALFAS, key=lambda a: np.mean(notas_a[a]) if notas_a[a] else -1)
    p = f1.prob_nos(bmu[itr], y[itr]); tx_cv[ite] = p[bmu[ite]]
    # pesos suaves sobre os nós e regiões do mapa
    w = np.exp(-(d - d.min(1, keepdims=True)) / (tau * escala)); w /= w.sum(1, keepdims=True)
    reg_no = fcluster(linkage(prot.astype(np.float64), "ward"), N_REG, "maxclust") - 1
    W = np.zeros((n, N_REG), np.float32)
    for r in range(N_REG):
        W[:, r] = w[:, reg_no == r].sum(1)
    # sequência de latentes (t-50 a t), passos ausentes marcados
    J = jan[idx]
    jj = np.where(J >= 0, posmap[np.clip(J, 0, None)], -1)
    msk = (jj >= 0).astype(np.float32)
    Zs = z[np.clip(jj, 0, None)] * msk[..., None]
    out = dict(idx=idx, ntr=ntr, itr=itr, ite=ite, bmu=bmu, bprev=bprev, y=y, posmap=posmap, z=z,
               p=p, tx_cv=tx_cv, tau=tau, alfa=alfa, W=W, reg_no=reg_no, reg=reg_no[bmu],
               Zs=Zs.astype(np.float32), msk=msk)
    out["dura"] = p[bmu]
    out["suave"] = w @ p
    out["traj"] = np.full(n, np.nan); out["traj"][ite] = prob_traj(bmu, bprev, y, itr, ite, alfa)
    # análogos: estado atual e sequência de 30 min (passo ausente recebe o passo seguinte)
    Za = Zs[:, -L_ANAL:].copy(); ma = msk[:, -L_ANAL:]
    for j in range(L_ANAL - 2, -1, -1):
        Za[:, j] = np.where(ma[:, j:j + 1] > 0, Za[:, j], Za[:, j + 1])
    Za = Za.reshape(n, -1)
    a1_ = knn_media(z[itr], y[itr], z[ite], (300,))
    a4_ = knn_media(Za[itr], y[itr], Za[ite], (100, 300))
    for nome, v in (("anal_atual300", a1_[300]), ("anal_seq100", a4_[100]), ("anal_seq300", a4_[300])):
        out[nome] = np.full(n, np.nan); out[nome][ite] = v
    return out


# ═══════════════════════════════ segundos estágios (A4) ══════════════════════
def logistica_offset(Xtr, ytr, off_tr, Xte, off_te, lam=1e-2):
    """Regressão logística com o logito da taxa do nó como termo fixo; as variáveis só corrigem
    a probabilidade de partida. Penalização L2 fixa, variáveis padronizadas com o treino."""
    med = np.nanmedian(Xtr, 0); med[~np.isfinite(med)] = 0
    Xt = np.where(np.isfinite(Xtr), Xtr, med); Xe = np.where(np.isfinite(Xte), Xte, med)
    mu, sd = Xt.mean(0), Xt.std(0); sd[sd < 1e-9] = 1
    Xt = (Xt - mu) / sd; Xe = (Xe - mu) / sd
    yt = ytr.astype(float); n = len(yt)

    def f(wv):
        eta = off_tr + wv[0] + Xt @ wv[1:]
        g = expit(eta) - yt
        loss = np.mean(np.logaddexp(0, eta) - yt * eta) + 0.5 * lam * wv[1:] @ wv[1:]
        return loss, np.r_[g.mean(), Xt.T @ g / n + lam * wv[1:]]

    r = minimize(f, np.zeros(Xt.shape[1] + 1), jac=True, method="L-BFGS-B", options=dict(maxiter=500))
    return expit(off_te + r.x[0] + Xe @ r.x[1:])


def gb_novo():
    return HistGradientBoostingClassifier(max_iter=150 if TESTE else 400, learning_rate=0.06, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)


def tabpfn_novo():
    """TabPFN com os pesos da versão 2 (Hollmann et al., 2025), que não exigem token."""
    import torch
    from tabpfn import TabPFNClassifier
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from tabpfn.constants import ModelVersion
        for nm in ("create_default_for_version", "get_default_for_version"):
            fn = getattr(TabPFNClassifier, nm, None)
            if fn is not None:
                try:
                    return fn(ModelVersion.V2, device=dev)
                except TypeError:
                    m = fn(ModelVersion.V2)
                    try:
                        m.set_params(device=dev)
                    except Exception:          # noqa: BLE001
                        pass
                    return m
    except ImportError:
        pass
    return TabPFNClassifier(device=dev)       # versões 2.0 a 2.2 do pacote usam os pesos v2


def tabpfn_ajustar(Xtr, ytr, Xte, rng, nmax=N_TABPFN):
    """Ajusta com todos os positivos (até metade da amostra) e negativos sorteados, e devolve a
    probabilidade com a razão de chances corrigida para a proporção original."""
    pos = np.where(ytr)[0]; neg = np.where(~ytr)[0]
    npos = min(len(pos), nmax // 2); nneg = min(len(neg), nmax - npos)
    sel = np.r_[rng.choice(pos, npos, replace=False), rng.choice(neg, nneg, replace=False)]
    m = tabpfn_novo()
    m.fit(Xtr[sel], ytr[sel].astype(int))
    p = np.concatenate([m.predict_proba(Xte[a:a + 4000])[:, 1] for a in range(0, len(Xte), 4000)])
    fator = (nneg / len(neg)) / (npos / len(pos))
    odds = np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, None) * fator
    return odds / (1 + odds)


# ── modelos de sequência sobre o latente
def seq_modelo(tipo, d_in, hd=64):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.tipo = tipo
            if tipo in ("LSTM", "GRU"):
                self.rnn = getattr(nn, tipo)(d_in, hd, batch_first=True)
            elif tipo == "TCN":
                self.ent = nn.Conv1d(d_in, hd, 1)
                self.convs = nn.ModuleList([nn.Conv1d(hd, hd, 2, dilation=dl) for dl in (1, 2, 4)])
            else:
                self.ent = nn.Linear(d_in, hd)
                self.pos = nn.Parameter(torch.zeros(1, L_SEQ, hd))
                cam = nn.TransformerEncoderLayer(hd, 4, 2 * hd, dropout=0.1, batch_first=True)
                self.enc = nn.TransformerEncoder(cam, 2)
            self.cab = nn.Sequential(nn.Linear(hd, hd), nn.ReLU(), nn.Linear(hd, 1))

        def forward(self, x, m):
            if self.tipo in ("LSTM", "GRU"):
                o, _ = self.rnn(x); h = o[:, -1]
            elif self.tipo == "TCN":
                h = self.ent(x.transpose(1, 2))
                for c, dl in zip(self.convs, (1, 2, 4)):
                    h = h + F.relu(c(F.pad(h, (dl, 0))))     # convolução causal com resíduo
                h = h[:, :, -1]
            else:
                h = self.enc(self.ent(x) + self.pos, src_key_padding_mask=m < 0.5)[:, -1]
            return self.cab(h).squeeze(-1)

    return Seq()


def seq_treinar(tipo, Xtr, Mtr, ytr, Xva, Mva, yva, Xte, Mte):
    import torch
    dev = dispositivo()
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    m = seq_modelo(tipo, Xtr.shape[2]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    perda = torch.nn.BCEWithLogitsLoss()
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(dev)

    def prever(X, M):
        m.eval(); out = []
        with torch.no_grad():
            for a in range(0, len(X), 4096):
                out.append(torch.sigmoid(m(T(X[a:a + 4096]), T(M[a:a + 4096]))).cpu().numpy())
        return np.concatenate(out)

    melhor, estado, sem = -1.0, None, 0
    for ep in range(EPOCAS):
        m.train()
        for a in np.array_split(rng.permutation(len(Xtr)), max(1, len(Xtr) // 512)):
            opt.zero_grad()
            l = perda(m(T(Xtr[a]), T(Mtr[a])), T(ytr[a]))
            l.backward(); opt.step()
        ap = average_precision_score(yva, prever(Xva, Mva)) if yva.sum() else 0.0
        if ap > melhor:
            melhor, sem = ap, 0
            estado = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            sem += 1
            if sem >= PACIENCIA:
                break
    m.load_state_dict(estado)
    return prever(Xte, Mte), melhor, ep + 1


# ═══════════════════════════════ execução por horizonte e conjunto ══════════
def rodar(E, X, cache, chuva, h, conj, rng):
    """Escores fora da temporada para cada linha de E, com X = variáveis novas."""
    n = len(E); y = E.forte.to_numpy(); fE = E.fold.to_numpy(); pE = E.pos.to_numpy()
    yfull = cache.A[f"estado_prox{at.sufixo(h)}"].to_numpy() >= 3
    temp = cache.A.fold.to_numpy()
    L = latentes(chuva, h, cache, 2 if TESTE else None)
    nomes = ["SOM, probabilidade do nó", "SOM, probabilidade suave", "SOM, trajetória (nó agora e 30 min antes)",
             "SOM, análogos do estado atual (300)", "SOM, análogos da sequência de 30 min (100)",
             "SOM, análogos da sequência de 30 min (300)", REF,
             "gradient boosting, variáveis novas + nó e latente (referência)",
             "logística com o nó como ponto de partida", "especialistas por região (atribuição dura)",
             "mistura de especialistas guiada pelo SOM"]
    if not SEM_TABPFN:
        nomes += ["TabPFN único (amostra)", "TabPFN por região do SOM"]
    if not SEM_SEQ:
        nomes += [f"{t} sobre o latente" for t in ("LSTM", "GRU", "TCN", "Transformer")]
    if COM_CATBOOST and mt.CatBoostClassifier is not None:
        nomes += ["CatBoost, variáveis novas (referência)"]
    S = {k: np.full(n, np.nan) for k in nomes}
    info = {k: np.full(n, np.nan) for k in ("no", "no_30min_antes", "regiao", "taxa_no", "pct_taxa_no",
                                            "dist_no_maior_taxa")}
    par = []
    for k, R in L.items():
        t0 = time.time()
        D = som_dobra(R, yfull, temp, cache.jan, len(cache.A))
        j = D["posmap"][pE]
        te = (fE == k) & (j >= D["ntr"]); tr = (fE != k) & (j >= 0) & (j < D["ntr"])
        if y[tr].sum() < 10 or te.sum() == 0:
            continue
        jt, jr = j[te], j[tr]
        for nome, ch in (("SOM, probabilidade do nó", "dura"), ("SOM, probabilidade suave", "suave"),
                         ("SOM, trajetória (nó agora e 30 min antes)", "traj"),
                         ("SOM, análogos do estado atual (300)", "anal_atual300"),
                         ("SOM, análogos da sequência de 30 min (100)", "anal_seq100"),
                         ("SOM, análogos da sequência de 30 min (300)", "anal_seq300")):
            S[nome][te] = D[ch][jt]
        # informação do mapa para o diagnóstico
        b = D["bmu"]; topo = int(D["p"].argmax())
        info["no"][te] = b[jt]; info["no_30min_antes"][te] = D["bprev"][jt]; info["regiao"][te] = D["reg"][jt]
        info["taxa_no"][te] = D["p"][b[jt]]
        info["pct_taxa_no"][te] = (D["p"][None, :] < D["p"][b[jt]][:, None]).mean(1)
        info["dist_no_maior_taxa"][te] = np.abs(b[jt] // 10 - topo // 10) + np.abs(b[jt] % 10 - topo % 10)
        par.append(dict(horizonte=h, entradas=conj, temporada=k, tau=D["tau"], alfa=D["alfa"],
                        horarios_treino=int(tr.sum()), horarios_teste=int(te.sum()), fortes_teste=int(y[te].sum())))
        # referência com o nó e o latente
        extra_tr = np.column_stack([b[jr], D["tx_cv"][jr], D["z"][jr]])
        extra_te = np.column_stack([b[jt], D["tx_cv"][jt], D["z"][jt]])
        S["gradient boosting, variáveis novas + nó e latente (referência)"][te] = mt.ajustar(
            "HistGradientBoosting", np.column_stack([X[tr], extra_tr]), y[tr],
            np.column_stack([X[te], extra_te]), [X.shape[1]])
        # logística com o nó como ponto de partida
        S["logística com o nó como ponto de partida"][te] = logistica_offset(
            X[tr], y[tr], logito(D["tx_cv"][jr]), X[te], logito(D["p"][b[jt]]))
        # especialistas por região
        g = gb_novo().fit(X[tr], y[tr]); pg = g.predict_proba(X[te])[:, 1]
        reg_tr, reg_te = D["reg"][jr], D["reg"][jt]
        Pr = np.repeat(pg[:, None], N_REG, 1)
        for r in range(N_REG):
            mr = reg_tr == r
            if y[tr][mr].sum() >= 20 and (~y[tr][mr]).sum() >= 20:
                Pr[:, r] = gb_novo().fit(X[tr][mr], y[tr][mr]).predict_proba(X[te])[:, 1]
        S["especialistas por região (atribuição dura)"][te] = Pr[np.arange(len(jt)), reg_te]
        S["mistura de especialistas guiada pelo SOM"][te] = (D["W"][jt] * Pr).sum(1)
        # TabPFN
        if not SEM_TABPFN:
            Xtf = np.where(np.isfinite(X), X, np.nan)
            pglob = tabpfn_ajustar(Xtf[tr], y[tr], Xtf[te], rng)
            S["TabPFN único (amostra)"][te] = pglob
            pr = pglob.copy()
            regs = range(2) if TESTE else range(N_REG)
            for r in regs:
                mr, mt_ = reg_tr == r, reg_te == r
                if mt_.any() and y[tr][mr].sum() >= 20 and (~y[tr][mr]).sum() >= 20:
                    pr[mt_] = tabpfn_ajustar(Xtf[tr][mr], y[tr][mr], Xtf[te][mt_], rng)
            S["TabPFN por região do SOM"][te] = pr
        # sequência sobre o latente
        if not SEM_SEQ:
            Zs, M = D["Zs"], D["msk"]
            v = M[D["itr"]] > 0
            mu = D["Zs"][D["itr"]][v].mean(0); sd = D["Zs"][D["itr"]][v].std(0) + 1e-6
            Xs = np.concatenate([((Zs - mu) / sd) * M[..., None], M[..., None]], -1).astype(np.float32)
            temporadas_tr = sorted(np.unique(fE[tr]))
            val = fE[tr] == temporadas_tr[-1]
            for tipo in ("LSTM", "GRU", "TCN", "Transformer"):
                s_, ap_val, ep = seq_treinar(tipo, Xs[jr][~val], M[jr][~val], y[tr][~val].astype(np.float32),
                                             Xs[jr][val], M[jr][val], y[tr][val], Xs[jt], M[jt])
                S[f"{tipo} sobre o latente"][te] = s_
                par.append(dict(horizonte=h, entradas=conj, temporada=k, modelo=tipo, epocas=ep,
                                AP_validacao=round(ap_val, 4), temporada_validacao=int(temporadas_tr[-1])))
        log.info(f"  {conj} h{h}: temporada {k} pronta em {(time.time() - t0) / 60:.1f} min "
                 f"(tau {D['tau']}, alfa {D['alfa']})")
    folds = sorted(L)
    sel = np.isin(fE, folds)
    S[REF] = mt.loso(E, X, "HistGradientBoosting", sel=sel)
    if "CatBoost, variáveis novas (referência)" in S:
        S["CatBoost, variáveis novas (referência)"] = mt.loso(E, X, "CatBoost", sel=sel)
    if chuva != "nenhuma":
        S["persistência"] = np.where(sel, E.agora.to_numpy(float) + 1e-3 * np.nan_to_num(at.col(E, "in_nuc_max10")), np.nan)
    return S, info, par


def diferenca_pareada(E, S, h, conj):
    """AP por temporada de cada modelo menos a do gradient boosting de referência."""
    y = E.forte.to_numpy(); f = E.fold.to_numpy()
    ref = {}
    lin = []
    for k in sorted(np.unique(f)):
        m = (f == k) & np.isfinite(S[REF])
        if m.any() and y[m].any():
            ref[k] = average_precision_score(y[m], S[REF][m])
    for nome, s in S.items():
        dif = []
        for k, r in ref.items():
            m = (f == k) & np.isfinite(s)
            if m.sum() and y[m].any():
                dif.append(average_precision_score(y[m], s[m]) - r)
        if dif:
            d = np.array(dif)
            lin.append(dict(horizonte=h, entradas=conj, modelo=nome, temporadas=len(d),
                            dif_AP_media=round(d.mean(), 4), dif_AP_dp=round(d.std(ddof=1), 4) if len(d) > 1 else np.nan,
                            dif_AP_min=round(d.min(), 4), dif_AP_max=round(d.max(), 4),
                            temporadas_acima_da_ref=int((d > 0).sum())))
    return lin


def diagnostico(E, ini, tipos, S, info, h, conj):
    """Uma linha por início a partir do seco: onde o horário de previsão cai no mapa e como os
    principais escores se posicionam entre os horários secos da temporada."""
    k = np.clip(ini.pos_tc.to_numpy() - h // 10, 0, None)
    m = ini.partiu_do_seco.to_numpy()
    if not m.any():
        return pd.DataFrame()
    k = k[m]
    D = pd.DataFrame(dict(horizonte=h, entradas=conj, ts_previsao=E.ts.to_numpy()[k],
                          ts_limiar=ini.ts_limiar.to_numpy()[m], temporada=E.fold.to_numpy()[k],
                          tipo=np.asarray(tipos)[m], chuva_antes_do_limiar_min=ini.chuva_antes_do_limiar_min.to_numpy()[m]))
    for c in ("no", "no_30min_antes", "regiao", "taxa_no", "pct_taxa_no", "dist_no_maior_taxa"):
        D[c] = info[c][k]
    D["linha_mapa"] = D.no // 10; D["coluna_mapa"] = D.no % 10
    D["mesmo_no_30min_antes"] = D.no == D.no_30min_antes
    # fração dos horários secos da temporada em nós de taxa menor ou igual
    seco = E.agora.to_numpy() == 0; fE = E.fold.to_numpy()
    D["frac_secos_em_nos_de_taxa_menor_ou_igual"] = [
        float(np.mean(info["taxa_no"][seco & (fE == t) & np.isfinite(info["taxa_no"])] <= tx)) if np.isfinite(tx) else np.nan
        for t, tx in zip(D.temporada, D.taxa_no)]
    for nome in ("SOM, probabilidade suave", "SOM, trajetória (nó agora e 30 min antes)", REF,
                 "logística com o nó como ponto de partida", "mistura de especialistas guiada pelo SOM"):
        if nome in S:
            D[f"posicao: {nome}"] = np.round(at.posicao_entre_secos(E, k, S[nome]), 3)
    for c in ("in_c1_tot_max60", "in_c2_tot_max60", "g30_t13_min", "g60_frac_abaixo_-20C", "g30_resfr30_min",
              "nuvens_vel_kmh", f"mont{h}_t13_min", "em_umidity", "hora"):
        if c in E:
            D[c] = np.round(E[c].to_numpy(float)[k], 3)
    return D


def main():
    global SEM_TABPFN
    configurar_log()
    t00 = time.time()
    import torch
    log.info(f"etapa A {'(TESTE) ' if TESTE else ''}| torch {torch.__version__} | dispositivo {dispositivo()} | "
             f"TabPFN {'desligado' if SEM_TABPFN else 'ligado'} | sequência {'desligada' if SEM_SEQ else 'ligada'}")
    if not SEM_TABPFN:
        try:
            import tabpfn
            log.info(f"tabpfn {getattr(tabpfn, '__version__', '?')}")
        except ImportError:
            log.info("tabpfn não instalado (pip install tabpfn); TabPFN pulado"); SEM_TABPFN = True
    log.info(f"protocolo: uma temporada fora por vez, semente {SEED}, hiperparâmetros fixos; temperatura da "
             f"probabilidade suave em {TAUS} e peso da trajetória em {ALFAS}, escolhidos dentro do treino")
    cache = mt.cache_som()
    A = at.carregar(); B = at.blocos(A)
    V, P = variaveis_novas(cache, A)
    for Dd in (V, P):
        for c in Dd.columns:
            A[c] = Dd[c].to_numpy()[A.pos.to_numpy()]
    GF = [c for c in V.columns if c.startswith("g") and c[1:3] in ("15", "30", "60")] + ["nuvens_vel_kmh"]
    PL_ = list(P.columns) + [c for c in V.columns if c.startswith("in_c1_mont")]
    conjuntos = {"sem chuva": (("EMs", "EMs_tend", "GOES", "GOES_tend", "hora"), "nenhuma"),
                 "anéis e área central": (("EMs", "EMs_tend", "GOES", "GOES_tend", "hora", "colar1", "colar2", "nucleo"),
                                          "colar_nucleo")}
    if TESTE:
        conjuntos = {"anéis e área central": conjuntos["anéis e área central"]}
    rng = np.random.default_rng(SEED)
    M, C, G, EV, PAR, DIF, DG = [], [], [], [], [], [], []
    for h in ((30,) if TESTE else (30, 60)):
        E = at.conjunto(A, h)
        MONT = [c for c in V.columns if c.startswith(f"mont{h}_")]
        for conj, (blocos, chuva) in conjuntos.items():
            base = [c for b in blocos for c in B[b]]
            novas = base + GF + MONT + (PL_ if chuva != "nenhuma" else [])
            X = E[novas].to_numpy(float); X[~np.isfinite(X)] = np.nan
            log.info(f"h{h} {conj}: {len(novas)} variáveis, {len(E):,} horários")
            S, info, par = rodar(E, X, cache, chuva, h, conj, rng)
            PAR += par
            usadas = np.isfinite(S[REF])
            for nome, s in S.items():
                s = np.where(usadas, s, np.nan)
                if np.isfinite(s).any():
                    mt.avaliar(E, s, nome, conj, h, M, C, G)
            DIF += diferenca_pareada(E, S, h, conj)
            ini = at.inicios(E, h)
            if len(ini):
                ini = ini[np.isin(ini.fold.to_numpy(), np.unique(E.fold.to_numpy()[usadas]))].reset_index(drop=True)
                tipos = mt.tipo_inicio_seco(E, ini, h)
                EV += at.resumo_inicios(E, ini, S, h, conj, tipos)
                DG.append(diagnostico(E, ini, tipos, S, info, h, conj))
            pd.DataFrame({"ts": E.ts, "fold": E.fold, "forte": E.forte, **{k: v for k, v in S.items()}}) \
                .to_csv(OUT / f"escores_h{h}_{chuva}.csv.gz", sep=";", index=False)
    M = pd.DataFrame(M)
    at.salvar("A_parametros_por_temporada", pd.DataFrame(PAR))
    at.salvar("A_metricas", M.round(4), mostrar=False)
    at.salvar("A_pontos_de_operacao", pd.DataFrame(C).round(4), mostrar=False)
    at.salvar("A_curvas_de_ganho", pd.concat(G).round(4), mostrar=False)
    at.salvar("A_diferenca_pareada_AP", pd.DataFrame(DIF))
    at.salvar("A_inicios_do_seco", pd.DataFrame(EV), max_linhas=300)
    if DG:
        at.salvar("A_diagnostico_inicios_no_mapa", pd.concat(DG, ignore_index=True), max_linhas=400)
    tab = M[~M.modelo.str.contains(r"\[só")].pivot_table(index=["entradas", "modelo"], columns="horizonte",
                                                         values=["AP", "AP_dp", "recall_5pct"]).round(3)
    log.info(f"\n### resumo (AP média e desvio padrão entre temporadas, recall a 5% do tempo)\n{tab.to_string()}\n")
    seco = M[M.modelo.str.contains(r"\[só")].pivot_table(index=["entradas", "modelo"], columns="horizonte",
                                                          values="AP").round(4)
    log.info(f"\n### AP só nos horários secos\n{seco.to_string()}\n")
    cpt = pd.DataFrame(C)
    cpt = cpt[cpt.ponto.isin(["5% do tempo", "≥0,10"])].pivot_table(index=["entradas", "modelo", "ponto"],
                                                                    columns="horizonte", values=["POD", "FAR", "CSI"]).round(3)
    log.info(f"\n### POD, FAR e CSI\n{cpt.to_string()}\n")
    lim = [p["tau"] for p in PAR if "tau" in p]
    if lim and max(lim) >= max(TAUS):
        log.info(f"AVISO: a temperatura escolhida chegou ao limite da grade ({max(TAUS)}) em "
                 f"{sum(t >= max(TAUS) for t in lim)} de {len(lim)} dobras")
    log.info(f"fim em {(time.time() - t00) / 60:.1f} min; saídas em {OUT}")


if __name__ == "__main__":
    main()
