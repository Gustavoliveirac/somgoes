"""
somgoes_nucleo.py — dados, modelos, perdas e métricas do somgoes.

Dados    Cache (amostras + EMs + GOES), janelas, normalização por fold,
         amostragem balanceada por transição
Modelo   EMs como tokens (posição, tipo, valores, máscara de falha) com agregação
         por atenção; GOES por nenhum | t0 | seq_cnn | convlstm; camada SOM; AE ou VAE
Perdas   recon | mi | mi_conc | mi_conc_sep | completa (a chuva só entra aqui)
Métricas por condição, sem depender do tamanho da grade: AP e skill, cobertura e
         lift em frações do tempo, nó forte, avaliação por estação, NMI, acerto de
         cantos (melhor simetria), separação entre pares, tabela por célula
"""
from __future__ import annotations

import json
import logging
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, normalized_mutual_info_score

from core import config_somgoes as cfg

warnings.filterwarnings("ignore")
log = logging.getLogger("nucleo")
if cfg.DEVICE == "cpu":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    DEV = torch.device("cuda" if (cfg.DEVICE in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
NT = len(cfg.TRANSICOES)


# ═══════════════════════════════════ dados ═══════════════════════════════════
class Cache:
    def __init__(self, d: Path = None):
        d = Path(d or cfg.CACHE_DIR)
        self.meta = json.load(open(d / "meta_cache.json"))
        self.A = pd.read_parquet(d / "amostras.parquet")
        self.em10 = np.load(d / "em10.npy")
        self.em1h = np.load(d / "em1h.npy")
        self.idade = np.load(d / "em1h_idade.npy")
        self.n_agora = np.load(d / "nucleo_agora.npy").astype(np.float32)
        self.n_prox = np.load(d / "nucleo_prox.npy").astype(np.float32)
        self.campo = np.load(d / "goes_campo.npy", mmap_mode="r")
        self.est = pd.read_csv(d / "estacoes_em.csv", sep=";")
        ts = self.A.ts.values
        cont = np.r_[False, (ts[1:] - ts[:-1]) == np.timedelta64(10, "m")]
        prev = np.where(cont, np.arange(len(ts)) - 1, -1)
        # janela de 6 horários (t-50 .. t0); -1 onde a sequência é interrompida
        J = np.full((len(ts), cfg.N_QUADROS), -1, np.int64)
        J[:, -1] = np.arange(len(ts))
        for j in range(cfg.N_QUADROS - 2, -1, -1):
            nxt = J[:, j + 1]
            J[:, j] = np.where(nxt >= 0, prev[np.clip(nxt, 0, None)], -1)
        self.jan = J
        # quadros GOES: faltante repete o mais recente válido (até LACUNA_GOES_MAX)
        ok = self.A.goes_ok.to_numpy()
        G = J.copy()
        falt = np.zeros(len(ts), np.int64)
        for j in range(cfg.N_QUADROS - 2, -1, -1):
            bom = (G[:, j] >= 0) & ok[np.clip(G[:, j], 0, None)]
            falt = np.where(bom, 0, falt + 1)
            G[:, j] = np.where(bom, G[:, j], np.where(falt <= cfg.LACUNA_GOES_MAX, G[:, j + 1], -1))
        self.quadros = G
        self.valida = (self.A.alvo_ok & self.A.goes_ok).to_numpy()
        self.valida30 = (self.A.get("alvo_ok_30", self.A.alvo_ok) & self.A.goes_ok).to_numpy()
        self.prox_idx = np.where(np.r_[cont[1:], False], np.arange(len(ts)) + 1, -1)
        self.cols_chuva = {"nenhuma": [], "colar": list(self.meta.get("entrada_colar", [])),
                           "colar_nucleo": list(self.meta.get("entrada_colar", [])) +
                                           list(self.meta.get("entrada_nucleo", []))}
        # médias por amostra das EMs (só para interpretar células)
        V = self.meta["variaveis"]
        tudo = np.concatenate([self.em10, self.em1h], 1)
        for k, v in enumerate(V):
            self.A[f"em_{v}"] = np.nanmean(tudo[:, :, k], 1)
        # estáticos das estações
        alt = self.est.altitude.to_numpy(float)
        s = np.stack([self.est.dx_km / 50, self.est.dy_km / 50, (alt - 850) / 300,
                      (self.est.tipo == "10min").astype(float)], 1).astype(np.float32)
        self.s10 = s[(self.est.tipo == "10min").to_numpy()]
        self.s1h = s[(self.est.tipo == "1h").to_numpy()]
        log.info(f"cache: {len(self.A):,} horários | válidos {self.valida.sum():,} | "
                 f"EMs {self.em10.shape[1]} de 10 min e {self.em1h.shape[1]} horárias | "
                 f"GOES {self.campo.shape}")

    def _val(self, h):
        return self.valida if h == 60 else self.valida30

    def split(self, fold, max_amostras=None, h=60):
        val = self._val(h)
        f = self.A.fold.to_numpy()
        if max_amostras:
            sel = np.where(val)[0][:max_amostras]
            c = int(0.8 * len(sel))
            return sel[:c], sel[c:]
        return np.where(val & (f != fold))[0], np.where(val & (f == fold))[0]

    def rotulos(self, idx, h=60):
        a = self.A.iloc[idx]
        sf = "" if h == 60 else f"_{h}"
        return dict(transicao=a[f"transicao{sf}"].fillna(-1).to_numpy(np.int64),
                    basico=a[f"basico{sf}"].fillna(-1).to_numpy(np.int64),
                    agora=a[f"estado_agora{sf}"].fillna(-1).to_numpy(np.int64),
                    prox=a[f"estado_prox{sf}"].fillna(-1).to_numpy(np.int64))

    def futuro(self, h=60):
        """índice do horário t+h (mesmo par usado no alvo); -1 se não existir"""
        f = np.arange(len(self.A))
        for _ in range(h // 10):
            f = np.where(f >= 0, self.prox_idx[np.clip(f, 0, None)], -1)
        return f

    # normalização com o treino do fold
    def normalizador(self, tr):
        amos = tr[:: max(1, len(tr) // 20000)]
        V = np.concatenate([self.em10[amos].reshape(-1, self.em10.shape[2]),
                            self.em1h[amos].reshape(-1, self.em1h.shape[2])], 0)
        mu, sd = np.nanmean(V, 0), np.nanstd(V, 0)
        mu[~np.isfinite(mu)] = 0; sd[~np.isfinite(sd) | (sd < 1e-6)] = 1
        g = self.campo[np.sort(amos[:2000])].astype(np.float32)
        gmu = g.mean((0, 2, 3)).reshape(1, 1, -1, 1, 1)
        gsd = g.std((0, 2, 3)).reshape(1, 1, -1, 1, 1) + 1e-6
        cc = self.cols_chuva["colar_nucleo"]
        C = np.log1p(np.clip(self.A.iloc[amos][cc].to_numpy(np.float32), 0, None)) if cc else np.zeros((1, 0))
        cmu, csd = np.nanmean(C, 0), np.nanstd(C, 0)
        cmu[~np.isfinite(cmu)] = 0; csd[~np.isfinite(csd) | (csd < 1e-6)] = 1
        return dict(mu=mu.astype(np.float32), sd=sd.astype(np.float32), gmu=gmu, gsd=gsd,
                    chuva=dict(zip(cc, zip(cmu, csd))))

    def lote(self, idx, norm, goes_modo, chuva="nenhuma"):
        mu, sd = norm["mu"], norm["sd"]
        J = self.jan[idx]
        x10 = np.full((len(idx), self.em10.shape[1], cfg.N_QUADROS, self.em10.shape[2]), np.nan, np.float32)
        okj = J >= 0
        x10[okj.any(1)] = np.nan
        for j in range(cfg.N_QUADROS):
            v = okj[:, j]
            x10[v, :, j, :] = self.em10[J[v, j]]
        m10 = np.isfinite(x10)
        x10 = np.where(m10, (x10 - mu) / sd, 0).astype(np.float32)
        x1h = self.em1h[idx]
        m1h = np.isfinite(x1h)
        x1h = np.where(m1h, (x1h - mu) / sd, 0).astype(np.float32)
        ida = np.nan_to_num(self.idade[idx] / 60.0, nan=1.0).astype(np.float32)
        out = dict(x10=x10, m10=m10.astype(np.float32), x1h=x1h, m1h=m1h.astype(np.float32), idade=ida)
        if goes_modo != "nenhum":
            Q = self.quadros[idx] if goes_modo != "t0" else self.quadros[idx][:, -1:]
            g = self.campo[np.clip(Q, 0, None).reshape(-1)].astype(np.float32)
            g = g.reshape(len(idx), Q.shape[1], *self.campo.shape[1:])
            g = (g - norm["gmu"]) / norm["gsd"]
            g[Q < 0] = 0
            out["goes"] = g.astype(np.float32)
        cc = self.cols_chuva[chuva]
        if cc:
            C = np.log1p(np.clip(self.A.iloc[idx][cc].to_numpy(np.float32), 0, None))
            mu_c = np.array([norm["chuva"][c][0] for c in cc], np.float32)
            sd_c = np.array([norm["chuva"][c][1] for c in cc], np.float32)
            mc = np.isfinite(C)
            out["chuva"] = np.where(mc, (C - mu_c) / sd_c, 0).astype(np.float32)
            out["mchuva"] = mc.astype(np.float32)
        return {k: torch.from_numpy(v) for k, v in out.items()}

    def planas(self, idx, norm):
        """Features planas para SOM clássico e k-means: EMs em t0 + estatísticas GOES."""
        x = np.concatenate([self.em10[idx].reshape(len(idx), -1), self.em1h[idx].reshape(len(idx), -1)], 1)
        mu = np.tile(norm["mu"], x.shape[1] // len(norm["mu"]))
        sd = np.tile(norm["sd"], x.shape[1] // len(norm["sd"]))
        x = (x - mu) / sd
        g = self.A.iloc[idx][self.meta["goes_stat_cols"]].to_numpy(np.float32)
        return np.nan_to_num(np.concatenate([x, g], 1), nan=0.0).astype(np.float32)


def normalizar_planas(Xtr, *Xs):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-6] = 1
    return [((X - mu) / sd).astype(np.float32) for X in (Xtr, *Xs)]


class Amostrador:
    """Lotes balanceados por transição: cota ~ sqrt(freq), mínimo COTA_MIN_CLASSE."""
    def __init__(self, idx, trans, lote, seed=cfg.SEED):
        self.rng = np.random.default_rng(seed)
        self.pools = [idx[trans == c] for c in range(NT)]
        fr = np.array([len(p) for p in self.pools], float)
        w = np.sqrt(fr) / np.sqrt(fr).sum()
        self.cota = [max(cfg.COTA_MIN_CLASSE, int(round(lote * x))) if len(p) else 0
                     for x, p in zip(w, self.pools)]

    def __call__(self):
        return np.concatenate([self.rng.choice(p, c, replace=True)
                               for p, c in zip(self.pools, self.cota) if c > 0])


# ═══════════════════════════════════ modelo ══════════════════════════════════
class Tokens(nn.Module):
    """EMs como tokens com máscara; agregação por atenção + média."""
    def __init__(self, n_var, s10, s1h, d=cfg.D_TOKEN):
        super().__init__()
        self.register_buffer("s10", torch.from_numpy(s10))
        self.register_buffer("s1h", torch.from_numpy(s1h))
        self.c10 = nn.Sequential(nn.Conv1d(2 * n_var, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(32, 32, 3, padding=1), nn.ReLU())
        self.p10 = nn.Sequential(nn.Linear(32 + 4, d), nn.ReLU(), nn.Linear(d, d))
        self.p1h = nn.Sequential(nn.Linear(2 * n_var + 1 + 4, d), nn.ReLU(), nn.Linear(d, d))
        self.q = nn.Parameter(torch.randn(d) * 0.1)
        self.d = d

    def forward(self, x10, m10, x1h, m1h, idade):
        B = x10.shape[0] if x10.numel() else x1h.shape[0]
        toks, pres = [], []
        if x10.shape[1]:
            S, T, V = x10.shape[1:]
            h = torch.cat([x10 * m10, m10], -1).reshape(B * S, T, 2 * V).transpose(1, 2)
            h = self.c10(h).mean(-1).reshape(B, S, -1)
            h = torch.cat([h, self.s10.unsqueeze(0).expand(B, -1, -1)], -1)
            toks.append(self.p10(h)); pres.append(m10.amax((2, 3)) > 0)
        if x1h.shape[1]:
            h = torch.cat([x1h * m1h, m1h, idade.unsqueeze(-1), self.s1h.unsqueeze(0).expand(B, -1, -1)], -1)
            toks.append(self.p1h(h)); pres.append(m1h.amax(2) > 0)
        T_ = torch.cat(toks, 1); P = torch.cat(pres, 1)
        sc = (T_ @ self.q / math.sqrt(self.d)).masked_fill(~P, -1e9)
        a = torch.softmax(sc, 1) * P.float()
        att = (a.unsqueeze(-1) * T_).sum(1)
        med = (T_ * P.unsqueeze(-1)).sum(1) / P.sum(1, keepdim=True).clamp(min=1)
        return torch.cat([att, med], -1)


class ConvLSTMCell(nn.Module):
    def __init__(self, i, h):
        super().__init__()
        self.h = h
        self.conv = nn.Conv2d(i + h, 4 * h, 3, padding=1)

    def forward(self, x, h, c):
        i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class EncGOES(nn.Module):
    def __init__(self, modo, C):
        super().__init__()
        self.modo = modo
        cin = C * cfg.N_QUADROS if modo == "seq_cnn" else C
        if modo in ("t0", "seq_cnn"):
            self.net = nn.Sequential(nn.Conv2d(cin, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        else:
            self.red = nn.Conv2d(C, 16, 3, 2, 1)
            self.cell = ConvLSTMCell(16, 32)
            self.fim = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, g):                      # g (B, Q, C, R, R)
        if self.modo == "t0":
            return self.net(g[:, -1])
        if self.modo == "seq_cnn":
            return self.net(g.flatten(1, 2))
        h = c = None
        for t in range(g.shape[1]):
            x = F.relu(self.red(g[:, t]))
            if h is None:
                h = torch.zeros(x.shape[0], 32, *x.shape[2:], device=x.device); c = torch.zeros_like(h)
            h, c = self.cell(x, h, c)
        return self.fim(h)


class SOMGoes(nn.Module):
    def __init__(self, cache: Cache, goes_modo, ae, grade, latente=cfg.LATENTE, chuva="nenhuma"):
        super().__init__()
        self.chuva = chuva
        nc = len(cache.cols_chuva[chuva])
        self.n_c = nc
        self.enc_c = nn.Sequential(nn.Linear(2 * nc, 64), nn.ReLU(), nn.Linear(64, 32)) if nc else None
        V = len(cache.meta["variaveis"])
        C = cache.campo.shape[1]
        self.goes_modo, self.vae, self.grade = goes_modo, ae == "vae", grade
        self.tok = Tokens(V, cache.s10, cache.s1h)
        dg = 64 if goes_modo != "nenhum" else 0
        self.enc_g = EncGOES(goes_modo, C) if dg else None
        self.fus = nn.Sequential(nn.Linear(2 * cfg.D_TOKEN + dg + (32 if nc else 0), 128), nn.ReLU(),
                                 nn.Linear(128, latente * (2 if self.vae else 1)))
        self.n_em = (cache.em10.shape[1] + cache.em1h.shape[1]) * V
        self.n_g = C * 64 if dg else 0          # GOES t0 reduzido a 8x8 na reconstrução
        self.dec = nn.Sequential(nn.Linear(latente, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(),
                                 nn.Linear(256, self.n_em + self.n_g + self.n_c))
        K = grade[0] * grade[1]
        self.trans = nn.Parameter(torch.zeros(K, K))     # modelo de Markov entre células
        self.prot = nn.Parameter(torch.randn(K, latente) * 0.1)
        ii = torch.arange(K)
        co = torch.stack([(ii // grade[1]).float(), (ii % grade[1]).float()], 1)
        self.register_buffer("coords", co)
        self.register_buffer("mdist", (co[None] - co[:, None]).abs().sum(-1))

    def codificar(self, b):
        h = self.tok(b["x10"], b["m10"], b["x1h"], b["m1h"], b["idade"])
        if self.enc_g is not None:
            h = torch.cat([h, self.enc_g(b["goes"])], -1)
        if self.enc_c is not None:
            h = torch.cat([h, self.enc_c(torch.cat([b["chuva"], b["mchuva"]], -1))], -1)
        h = self.fus(h)
        if self.vae:
            mu, lv = h.chunk(2, -1)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
            return z, mu, lv
        return h, None, None

    def alvo_rec(self, b):
        x = torch.cat([b["x10"][:, :, -1].flatten(1), b["x1h"].flatten(1)], 1)
        m = torch.cat([b["m10"][:, :, -1].flatten(1), b["m1h"].flatten(1)], 1)
        if self.n_g:
            g = F.adaptive_avg_pool2d(b["goes"][:, -1], 8).flatten(1)
            x = torch.cat([x, g], 1); m = torch.cat([m, torch.ones_like(g)], 1)
        if self.n_c:
            x = torch.cat([x, b["chuva"]], 1); m = torch.cat([m, b["mchuva"]], 1)
        return x, m

    def dist(self, z):
        return torch.cdist(z, self.prot) ** 2


# ═══════════════════════════════════ perdas ══════════════════════════════════
def soft(d):
    return torch.softmax(-d / (cfg.TAU_SOFT * d.detach().mean(1, keepdim=True).clamp(min=1e-6)) * 10, 1)


def dist_por_classe(q, y, n):
    oh = F.one_hot(y, n).float()
    cnt = oh.sum(0)
    return (oh.T @ q) / cnt.clamp(min=1).unsqueeze(1), cnt > 0     # (n,K), presentes


def entropia(p):
    return -(p * torch.log(p + 1e-9)).sum(-1)


def grupo3(e):
    """estado 0..4 -> 0 seco, 1 fraca/moderada, 2 forte/muito forte"""
    return torch.where(e >= 3, 2, torch.where(e >= 1, 1, 0))


def mi_rotulo(q, r, n):
    pkc, pres = dist_por_classe(q, r, n)
    pk = q.mean(0) + 1e-9
    Pc = F.one_hot(r, n).float().mean(0)
    return (Pc.unsqueeze(1) * pkc * torch.log((pkc + 1e-9) / pk)).sum() / math.log(n)


def mi_condicional(q, agora3, prox3):
    """média, entre os estados atuais presentes, de I(célula; estado próximo | estado atual)"""
    vals = []
    for g in range(3):
        m = agora3 == g
        if m.sum() >= 8 and prox3[m].unique().numel() > 1:
            vals.append(mi_rotulo(q[m], prox3[m], 3))
    return torch.stack(vals).mean() if vals else q.new_zeros(())


def perdas(model, b, y, estrategia, T, epoca, epocas=1, bf=None, vf=None):
    z, mu, lv = model.codificar(b)
    x, m = model.alvo_rec(b)
    err = ((model.dec(z) - x) ** 2 * m)
    rec = err.sum() / m.sum().clamp(min=1)
    tot = rec
    if model.vae:   # KL por variável reconstruída, com aquecimento (evita colapso)
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1).mean() / m.shape[1]
        tot = tot + cfg.BETA_VAE * min(1.0, (epoca + 1) / cfg.AQUEC_KL) * kl
    d = model.dist(z)
    bmu = d.argmin(1)
    w = torch.exp(-(model.mdist[bmu] ** 2) / (T ** 2))
    tot = tot + cfg.GAMMA_SOM * (w * d).sum(1).mean() / z.shape[1]
    termos = {"rec": float(rec)}
    if estrategia == "pre":
        return tot - cfg.GAMMA_SOM * (w * d).sum(1).mean() / z.shape[1], termos
    P = cfg.PERDAS_DEF[estrategia]
    K = d.shape[1]
    q = soft(d)
    tr = y["transicao"]
    pkc, pres = dist_por_classe(q, tr, NT)
    pk = q.mean(0) + 1e-9
    forte_ids = [2, 3, 6]                           # inicio_forte, intensifica, forte_persiste
    a3, p3 = grupo3(y["agora"]), grupo3(y["prox"])
    fase2 = epoca >= int(P.get("fase", 0) * epocas)
    if P.get("mi_agora"):
        w = P["mi_agora"] * (0.3 if (P.get("fase") and fase2) else 1.0)
        mia = mi_rotulo(q, a3, 3)
        tot = tot - w * mia
        termos["mi_agora"] = float(mia)
    if P.get("cmi") and fase2:
        cmi = mi_condicional(q, a3, p3)
        tot = tot - P["cmi"] * cmi
        termos["cmi"] = float(cmi)
    if P.get("markov") and bf is not None and vf is not None and vf.sum() > 1:
        zf, _, _ = model.codificar(bf)
        qf = soft(model.dist(zf))
        qt = q[vf]
        logP = torch.log_softmax(model.trans, 1)
        ltr = -(qt @ logP * qf).sum(1).mean()
        Pm = logP.exp()
        suave = (qt @ (Pm * model.mdist / model.mdist.max().clamp(min=1))).sum(1).mean()
        tot = tot + P["markov"] * ltr / math.log(K) + P.get("suave", 0) * suave
        termos.update(markov=float(ltr), suave=float(suave))
    if P.get("mi"):
        wc = torch.ones(NT, device=z.device)
        wc[forte_ids] = P.get("peso_forte", 1.0)
        Pc = F.one_hot(tr, NT).float().mean(0) * wc
        Pc = Pc / Pc.sum()
        mi = (Pc.unsqueeze(1) * pkc * torch.log((pkc + 1e-9) / pk)).sum() / math.log(NT)
        tot = tot - P["mi"] * mi
        termos["mi"] = float(mi)
    if P.get("lift"):
        kl = [(pkc[c] * torch.log((pkc[c] + 1e-9) / pk)).sum() for c in forte_ids if pres[c]]
        forte = y["prox"] >= 3
        if forte.sum() > 1:
            pf = q[forte].mean(0) + 1e-9
            kl.append((pf * torch.log(pf / pk)).sum())
        if kl:
            lift = torch.stack(kl).mean() / math.log(K)
            tot = tot - P["lift"] * lift
            termos["lift"] = float(lift)
    if P.get("conc"):
        pesos = torch.ones(NT, device=z.device); pesos[forte_ids] = 2.0
        conc = (entropia(pkc) / math.log(K) * pesos * pres).sum() / (pesos * pres).sum()
        forte = (y["prox"] >= 3)
        if forte.sum() > 1:
            conc = conc + 2.0 * entropia(q[forte].mean(0)) / math.log(K)
        tot = tot + P["conc"] * conc
        termos["conc"] = float(conc)
    if P.get("uso"):
        uso = 1 - entropia(pk) / math.log(K)
        tot = tot + P["uso"] * uso
        termos["uso"] = float(uso)
    if P.get("sep"):
        nome = {n: i for i, n in enumerate(cfg.TRANSICOES)}
        bc = [torch.sqrt(pkc[nome[a]] * pkc[nome[c]] + 1e-12).sum()
              for a, c in cfg.PARES_SEPARACAO if pres[nome[a]] and pres[nome[c]]]
        if bc:
            sep = torch.stack(bc).mean()
            tot = tot + P["sep"] * sep
            termos["sep"] = float(sep)
    if P.get("canto"):
        H, W = model.grade
        pos = q @ model.coords / torch.tensor([max(H - 1, 1), max(W - 1, 1)], device=z.device)
        cantos = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], device=z.device)
        canto = ((pos - cantos[y["basico"]]) ** 2).sum(1).mean()
        tot = tot + P["canto"] * canto
        termos["canto"] = float(canto)
    return tot, termos


def treinar(model, cache, tr, va, norm, estrategia, epocas, lotes_epoca, ckpt_dir: Path, h=60):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    ini, best, sem, hist = 0, float("inf"), 0, []
    ck = ckpt_dir / "checkpoint.pt"
    if ck.exists():
        st = torch.load(ck, map_location=DEV, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        ini, best, sem, hist = st["epoca"], st["best"], st["sem"], st["hist"]
        log.info(f"    retomando da época {ini}")
    yA = cache.rotulos(np.arange(len(cache.A)), h)
    amo = Amostrador(tr, yA["transicao"][tr], cfg.LOTE)
    P = cfg.PERDAS_DEF.get(estrategia, {})
    fut, valh = cache.futuro(h), cache._val(h)
    fase_ep = int(P.get("fase", 0) * epocas)
    if ini == 0 and not (ckpt_dir / "pre.pt").exists():
        pre_treino(model, cache, tr, norm, yA, amo, opt, lotes_epoca)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, ckpt_dir / "pre.pt")
    elif ini == 0:
        st = torch.load(ckpt_dir / "pre.pt", map_location=DEV, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
    amo_va = Amostrador(va, yA["transicao"][va], cfg.LOTE, seed=cfg.SEED + 1)
    lotes_va = [amo_va() for _ in range(max(2, lotes_epoca // 10))]
    T0 = max(model.grade) / 2

    def ylote(ix):
        return {k: torch.from_numpy(v[ix]).to(DEV) for k, v in yA.items()}

    def blote(ix):
        return {k: v.to(DEV) for k, v in cache.lote(ix, norm, model.goes_modo, model.chuva).items()}

    def calc(ix, ep):
        bf = vf = None
        if P.get("markov"):
            f = fut[ix]
            ok = (f >= 0) & valh[np.clip(f, 0, None)]
            if ok.sum() > 1:
                bf, vf = blote(f[ok]), torch.from_numpy(ok).to(DEV)
        return perdas(model, blote(ix), ylote(ix), estrategia, T, ep, epocas, bf, vf)

    for ep in range(ini, epocas):
        T = max(T0 * (1 / T0) ** (ep / max(epocas - 1, 1)), 0.5)
        model.train()
        acc = {}
        for _ in range(lotes_epoca):
            ix = amo()
            opt.zero_grad()
            l, termos = calc(ix, ep)
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in termos.items():
                acc[k] = acc.get(k, 0) + v / lotes_epoca
        model.eval()
        with torch.no_grad():
            val = np.mean([float(calc(ix, ep)[0]) for ix in lotes_va])
        hist.append(dict(epoca=ep, T=T, val=val, **acc))
        if fase_ep and ep == fase_ep:          # perda mudou: reinicia a referência do early stopping
            best, sem = float("inf"), 0
        if val < best:
            best, sem = val, 0
            torch.save({"model": model.state_dict()}, ckpt_dir / "melhor.pt")
        else:
            sem += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoca=ep + 1,
                        best=best, sem=sem, hist=hist), ck)
        if sem >= cfg.PACIENCIA and ep > fase_ep:
            break
    pd.DataFrame(hist).to_csv(ckpt_dir / "historia.csv", sep=";", index=False)
    if (ckpt_dir / "melhor.pt").exists():
        model.load_state_dict(torch.load(ckpt_dir / "melhor.pt", map_location=DEV, weights_only=False)["model"])
    return hist


def pre_treino(model, cache, tr, norm, yA, amo, opt, lotes_epoca):
    """Pré-treino só com reconstrução e inicialização linear dos protótipos no
    plano dos dois primeiros componentes principais do latente."""
    for ep in range(cfg.PRE_EPOCAS):
        model.train()
        for _ in range(lotes_epoca):
            ix = amo()
            b = {k: v.to(DEV) for k, v in cache.lote(ix, norm, model.goes_modo, model.chuva).items()}
            y = {k: torch.from_numpy(v[ix]).to(DEV) for k, v in yA.items()}
            opt.zero_grad()
            l, _ = perdas(model, b, y, "pre", 1.0, ep)
            l.backward(); opt.step()
    model.eval()
    amostra = tr[np.random.default_rng(cfg.SEED).choice(len(tr), min(4000, len(tr)), replace=False)]
    zs = []
    with torch.no_grad():
        for s in range(0, len(amostra), 1024):
            b = {k: v.to(DEV) for k, v in cache.lote(amostra[s:s + 1024], norm, model.goes_modo, model.chuva).items()}
            zs.append(model.codificar(b)[0].cpu())
    Z = torch.cat(zs)
    mu = Z.mean(0)
    U, S, Vt = torch.linalg.svd(Z - mu, full_matrices=False)
    sd = S[:2] / math.sqrt(len(Z))
    H, W = model.grade
    co = model.coords.cpu()
    a = (co[:, 0] / max(H - 1, 1) - 0.5) * 2 * 2 * sd[0]
    b_ = (co[:, 1] / max(W - 1, 1) - 0.5) * 2 * 2 * sd[1]
    with torch.no_grad():
        model.prot.copy_((mu + a[:, None] * Vt[0] + b_[:, None] * Vt[1]).to(model.prot.device))
    log.info(f"    pré-treino ok; protótipos iniciados no plano PCA (desvios {sd[0]:.2f}, {sd[1]:.2f})")


def bmus(model, cache, idx, norm, lote=1024):
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(idx), lote):
            b = {k: v.to(DEV) for k, v in cache.lote(idx[s:s + lote], norm, model.goes_modo, model.chuva).items()}
            z, _, _ = model.codificar(b)
            out.append(model.dist(z).argmin(1).cpu().numpy())
    return np.concatenate(out)


# ═══════════════════════════════════ SOM online ══════════════════════════════
class SOM:
    """SOM clássico; com n_x < n_features, as colunas extras são a parte
    supervisionada (XY-fused) que só existe no treino."""
    def __init__(self, rows, cols, n_feat, n_x=None, seed=cfg.SEED):
        self.rows, self.cols, self.n = rows, cols, rows * cols
        self.n_x = n_x or n_feat
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(self.n, n_feat, generator=g) * 0.1
        rr, cc = torch.meshgrid(torch.arange(rows).float(), torch.arange(cols).float(), indexing="ij")
        self.grid = torch.stack([rr.flatten(), cc.flatten()], 1)
        self.t = 0

    def fit(self, X, n_iter, ck: Path):
        rng = np.random.default_rng(cfg.SEED + self.t)
        s0 = max(self.rows, self.cols) / 2
        for t in range(self.t, n_iter):
            x = torch.from_numpy(X[rng.integers(len(X))])
            b = int(((self.W - x) ** 2).sum(1).argmin())
            sig = max(s0 * math.exp(-t / (n_iter / math.log(s0 + 1.01))), 0.5)
            lr = 0.5 * math.exp(-t / n_iter)
            h = torch.exp(-((self.grid - self.grid[b]) ** 2).sum(1) / (2 * sig ** 2))
            self.W += lr * h.unsqueeze(1) * (x - self.W)
            self.t = t + 1
            if self.t % cfg.SOM_CKPT == 0:
                self.salvar(ck)
        self.salvar(ck)

    def bmu(self, X):
        W = self.W[:, :X.shape[1]]
        out = []
        for s in range(0, len(X), 8192):
            xb = torch.from_numpy(X[s:s + 8192])
            out.append(((W * W).sum(1)[None] - 2 * xb @ W.T).argmin(1).numpy())
        return np.concatenate(out)

    def salvar(self, p):
        np.savez(p, W=self.W.numpy(), t=self.t, rows=self.rows, cols=self.cols, n_x=self.n_x)

    @classmethod
    def carregar(cls, p):
        z = np.load(p)
        s = cls(int(z["rows"]), int(z["cols"]), z["W"].shape[1], int(z["n_x"]))
        s.W, s.t = torch.from_numpy(z["W"]), int(z["t"])
        return s


def y_supervisionado(y):
    oh = np.eye(NT, dtype=np.float32)[y["transicao"]]
    cantos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32)[y["basico"]]
    return np.hstack([oh, cantos])


# ═══════════════════════════════════ métricas ════════════════════════════════
def condicoes(y):
    t, p = y["transicao"], y["prox"]
    c = {"forte_prox": p >= 3, "muito_forte_prox": p == 4, "chuva_prox": p >= 1}
    for i, n in enumerate(cfg.TRANSICOES):
        c[n] = t == i
    return c


def _curva(score_tr_cel, a_va, ev_va, K):
    """recall acumulado vs fração do tempo, ordenando células pelo escore do treino"""
    ordem = np.argsort(-score_tr_cel, kind="stable")
    sz = np.bincount(a_va, minlength=K)[ordem].astype(float)
    ev = np.bincount(a_va[ev_va], minlength=K)[ordem].astype(float)
    fs = np.r_[0, np.cumsum(sz)] / max(sz.sum(), 1)
    rc = np.r_[0, np.cumsum(ev)] / max(ev.sum(), 1)
    return fs, rc, ordem


COND = {  # nome: (estado atual 0 seco/1 fraca-moderada/2 forte, condição no estado próximo)
    "cond_seco_chuva":     (0, lambda p: p >= 1),
    "cond_seco_forte":     (0, lambda p: p == 2),
    "cond_fraca_forte":    (1, lambda p: p == 2),
    "cond_fraca_seco":     (1, lambda p: p == 0),
    "cond_forte_persiste": (2, lambda p: p == 2),
}


def _g3(e):
    return np.where(e >= 3, 2, np.where(e >= 1, 1, 0))


def avaliar(a_tr, y_tr, a_va, y_va, K, grade, cache=None, idx_va=None, h=60):
    met = {}
    # ── métricas condicionais ao estado atual (o que o mapa antecipa além do agora)
    g_tr, g_va = _g3(y_tr["agora"]), _g3(y_va["agora"])
    p_tr, p_va = _g3(y_tr["prox"]), _g3(y_va["prox"])
    from sklearn.metrics import mutual_info_score
    for nome, (g, f) in COND.items():
        mt, mv = g_tr == g, g_va == g
        et, ev = f(p_tr[mt]), f(p_va[mv])
        if mv.sum() < 20 or ev.sum() == 0 or ev.all() or et.sum() == 0:
            continue
        nt = np.bincount(a_tr[mt], minlength=K).astype(float)
        base_t = et.mean()
        taxa = (np.bincount(a_tr[mt][et], minlength=K) + 5 * base_t) / (nt + 5)
        sc = taxa[a_va[mv]]
        b = ev.mean()
        ap = average_precision_score(ev, sc)
        met[f"{nome}.base"] = round(float(b), 5)
        met[f"{nome}.ap_skill"] = round(float((ap - b) / (1 - b)), 5)
        fs, rc, _ = _curva(taxa, a_va[mv], ev, K)
        met[f"{nome}.cob_10pct"] = round(float(np.interp(0.10, fs, rc)), 4)
    ic = []
    for g in range(3):
        mv = g_va == g
        if mv.sum() > 20 and len(np.unique(p_va[mv])) > 1:
            Hp = -sum(pp * np.log(pp) for pp in np.bincount(p_va[mv]) / mv.sum() if pp > 0)
            ic.append(mutual_info_score(p_va[mv], a_va[mv]) / Hp)
    met["mi_condicional_norm"] = round(float(np.mean(ic)), 5) if ic else np.nan
    ctr, cva = condicoes(y_tr), condicoes(y_va)
    n_tr = np.bincount(a_tr, minlength=K).astype(float)
    escores = {}
    for nome, etr in ctr.items():
        eva = cva[nome]
        base = etr.mean()
        ev_tr = np.bincount(a_tr[etr], minlength=K)
        taxa = (ev_tr + 5 * base) / (n_tr + 5)              # suavização para células pequenas
        escores[nome] = taxa
        if eva.sum() == 0:
            continue
        s = taxa[a_va]
        bva = eva.mean()
        ap = average_precision_score(eva, s) if 0 < bva < 1 else np.nan
        met[f"{nome}.base"] = round(float(bva), 6)
        met[f"{nome}.ap"] = round(float(ap), 5)
        met[f"{nome}.ap_skill"] = round(float((ap - bva) / (1 - bva)), 5)
        fs, rc, ordem = _curva(taxa, a_va, eva, K)
        for f in cfg.FRACS_TEMPO:
            cob = float(np.interp(f, fs, rc))
            met[f"{nome}.cob_{int(f*100)}pct"] = round(cob, 4)
            met[f"{nome}.lift_{int(f*100)}pct"] = round(cob / f, 3)
        for r in (0.5, 0.8, 0.9):
            met[f"{nome}.tempo_para_{int(r*100)}pct"] = round(float(np.interp(r, rc, fs)), 4)
    # nó forte: célula(s) do treino com maior taxa de forte na próxima hora
    tx = escores["forte_prox"].copy(); tx[n_tr < 20] = -1
    top = np.argsort(-tx)
    ev = cva["forte_prox"]
    for k in (1, 3):
        cel = top[:k]
        dentro = np.isin(a_va, cel)
        met[f"no_forte_top{k}.precisao"] = round(float(ev[dentro].mean()) if dentro.any() else np.nan, 4)
        met[f"no_forte_top{k}.recall"] = round(float(ev[dentro].sum() / max(1, ev.sum())), 4)
        met[f"no_forte_top{k}.frac_tempo"] = round(float(dentro.mean()), 4)
    # por estação: cada (horário, estação do núcleo) com >= 10 mm na próxima hora
    if cache is not None and idx_va is not None and h == 60:
        P = cache.n_prox[idx_va]
        evs = (P >= cfg.LIMITES[2]).sum(1)
        fs, rc, _ = _curva(escores["forte_prox"], a_va, np.zeros(len(a_va), bool), K)
        ordem = np.argsort(-escores["forte_prox"], kind="stable")
        sz = np.bincount(a_va, minlength=K)[ordem].astype(float)
        es = np.bincount(a_va, weights=evs, minlength=K)[ordem]
        fs = np.r_[0, np.cumsum(sz)] / max(sz.sum(), 1)
        rc = np.r_[0, np.cumsum(es)] / max(es.sum(), 1)
        for f in (0.02, 0.05, 0.10):
            met[f"forte_por_estacao.cob_{int(f*100)}pct"] = round(float(np.interp(f, fs, rc)), 4)
    met["nmi_transicoes"] = round(float(normalized_mutual_info_score(y_va["transicao"], a_va)), 5)
    met["nmi_basicos"] = round(float(normalized_mutual_info_score(y_va["basico"], a_va)), 5)
    # acerto dos cantos: quadrante esperado de cada estado básico, melhor das 8 simetrias
    H, W = grade
    r, c = a_va // W, a_va % W
    melhor = 0.0
    for flip_r in (False, True):
        for flip_c in (False, True):
            for transp in ((False, True) if H == W else (False,)):
                rr, cc = (H - 1 - r if flip_r else r), (W - 1 - c if flip_c else c)
                if transp:
                    rr, cc = cc, rr
                lin = np.where(rr < (H - 1) / 2, 0, np.where(rr > (H - 1) / 2, 1, -1))
                col = np.where(cc < (W - 1) / 2, 0, np.where(cc > (W - 1) / 2, 1, -1))
                quad = np.where((lin >= 0) & (col >= 0), lin * 2 + col, -1)
                esp = np.array([0, 1, 2, 3])[y_va["basico"]]
                # mapeamento quadrante: SS(0,0)=0 SC(0,1)=1 CS(1,0)=2 CC(1,1)=3
                acc = np.mean([np.mean(quad[esp == k] == k) for k in range(4) if (esp == k).any()])
                melhor = max(melhor, acc)
    met["acerto_cantos"] = round(float(melhor), 4)
    # separação entre pares (Bhattacharyya; 0 = sem sobreposição)
    nm = {n: i for i, n in enumerate(cfg.TRANSICOES)}
    for a, b in cfg.PARES_SEPARACAO:
        ma, mb = y_va["transicao"] == nm[a], y_va["transicao"] == nm[b]
        if ma.any() and mb.any():
            pa = np.bincount(a_va[ma], minlength=K) / ma.sum()
            pb = np.bincount(a_va[mb], minlength=K) / mb.sum()
            met[f"sobreposicao_{a}_{b}"] = round(float(np.sqrt(pa * pb).sum()), 4)
    met["celulas_usadas"] = int((np.bincount(a_va, minlength=K) > 0).sum())
    p = np.bincount(a_va, minlength=K) / len(a_va); p = p[p > 0]
    met["celulas_efetivas"] = round(float(np.exp(-(p * np.log(p)).sum())), 2)
    # tabela por célula (validação) para interpretação
    tab = None
    if cache is not None and idx_va is not None:
        A = cache.A.iloc[idx_va].reset_index(drop=True)
        A["celula"] = a_va
        for i, n in enumerate(cfg.TRANSICOES):
            A[f"n_{n}"] = (y_va["transicao"] == i).astype(int)
        A["n_forte_prox"] = (y_va["prox"] >= 3).astype(int)
        A["hora"] = np.degrees(np.arctan2(A.hora_sin, A.hora_cos)) / 15 % 24
        cols_m = [c for c in A.columns if c.startswith("em_")] + \
                 [c for c in cache.meta["goes_stat_cols"] if c.endswith(("nucleo_min", "nucleo_max", "media", "p10"))] + \
                 cache.meta["colar_cols"][:2] + ["agora", "prox"]
        g = A.groupby("celula")
        tab = g[[f"n_{n}" for n in cfg.TRANSICOES] + ["n_forte_prox"]].sum()
        tab.insert(0, "n", g.size())
        tab["taxa_forte_prox"] = (tab.n_forte_prox / tab.n).round(4)
        tab["transicao_dominante"] = tab[[f"n_{n}" for n in cfg.TRANSICOES]].idxmax(1).str[2:]
        tab = tab.join(g[cols_m].mean().round(3))
        tab["hora_circular"] = g.apply(lambda d: (np.degrees(np.arctan2(d.hora_sin.mean(), d.hora_cos.mean())) / 15) % 24).round(1)
        tab = tab.reset_index()
        tab.insert(1, "linha", tab.celula // W); tab.insert(2, "coluna", tab.celula % W)
    # transições entre células (t -> t+h) na validação
    trans = None
    if cache is not None and idx_va is not None:
        pos = pd.Series(a_va, index=idx_va)
        fut = cache.futuro(h)[idx_va]
        ok = np.isin(fut, idx_va)
        de, para = a_va[ok], pos.reindex(fut[ok]).to_numpy()
        if len(de):
            H_, W_ = grade
            dmap = np.abs(de // W_ - para // W_) + np.abs(de % W_ - para % W_)
            met["frac_mesma_celula"] = round(float((de == para).mean()), 4)
            met["dist_media_transicao"] = round(float(dmap.mean()), 3)
            trans = pd.DataFrame({"de": de, "para": para}).value_counts().rename("n").reset_index()
            trans = trans[trans.n >= 3]
    return met, tab, trans
