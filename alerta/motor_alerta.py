#!/usr/bin/env python
"""
motor_alerta.py — motor de alerta de chuva forte no centro de BH (somgoes).

O motor usa um COMITÊ: os 6 modelos da validação cruzada (cada um treinado sem uma
temporada). Para um horário t, cada modelo coloca t numa célula e devolve a taxa de chuva
forte daquela célula no seu treino. A probabilidade é a média das 6 taxas e os votos são
quantos modelos estão em "alerta" ou acima. O estado (região) e a explicação vêm do modelo
final de estados. Se a chuva de colar e núcleo não estiver disponível, o motor passa
sozinho para o modelo sem chuva na entrada.

Comandos
  python -m alerta.motor_alerta exportar             empacota modelos, normalizações e taxas em motor/
  python -m alerta.motor_alerta replay INICIO FIM    roda o motor em um período do histórico (cache)
  python -m alerta.motor_alerta eventos N            replay dos N dias de maior chuva no núcleo
  python -m alerta.motor_alerta agora [AAAA-MM-DD HH:MM]   alerta do horário (dados brutos recentes)

No replay usa-se SÓ o modelo que não viu aquela temporada (prob_honesta e o nível dela).
O comitê só vale para horários futuros, que nenhum modelo viu; o seu ganho em relação ao
modelo isolado é medido sem vazamento pela validação aninhada (rodada_publicacao.py, S12).
Saídas em motor/: replay_*.csv, replay_*.png, alerta_atual.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from core import config_somgoes as cfg
from core import somgoes_nucleo as nu

PASTA = cfg.ROOT / "motor"
CENARIOS = [("colar_nucleo", 30), ("colar_nucleo", 60), ("nenhuma", 30), ("nenhuma", 60)]
NIVEIS = cfg.NIVEIS_ALERTA
EXPLICAR = None  # preenchido no exportar


def nivel_de(p):
    for i, (lim, nome) in enumerate(NIVEIS):
        if p < lim:
            return i, nome
    return len(NIVEIS) - 1, NIVEIS[-1][1]


# ═══════════════════════════════ exportação ══════════════════════════════════
def exportar():
    PASTA.mkdir(exist_ok=True)
    cache = nu.Cache()
    perda = cfg.ESTADOS_PERDA
    for ch, h in CENARIOS:
        base = cfg.RESULTS_DIR / "cenarios" / f"cenarios_{ch}_h{h}_{perda}_10x10"
        folds = sorted(int(p.name[4:]) for p in base.glob("fold*") if (p / "melhor.pt").exists())
        if not folds:
            print(f"sem modelos para {ch} h{h}"); continue
        comite = []
        for f in folds:
            tr, _ = cache.split(f, None if f >= 0 else cfg.RAPIDO["max_amostras"], h)
            norm = cache.normalizador(tr)
            m = nu.SOMGoes(cache, cfg.GOES_CENARIOS, "ae", (10, 10), chuva=ch)
            st = torch.load(base / f"fold{f}" / "melhor.pt", map_location="cpu", weights_only=False)["model"]
            m.load_state_dict(st)
            a = nu.bmus(m.to(nu.DEV), cache, tr, norm)
            fo = cache.rotulos(tr, h)["prox"] >= 3
            n = np.bincount(a, minlength=100)
            tx = (np.bincount(a[fo], minlength=100) + 5 * fo.mean()) / (n + 5)
            comite.append(dict(fold=f, estado=st, norm=norm, taxa=tx,
                               temporada=cache.meta["temporadas"][f] if f >= 0 else "rapido"))
        final = None
        de = cfg.RESULTS_DIR / "estados" / f"{ch}_h{h}"
        if (de / "modelo" / "melhor.pt").exists() and (de / "celulas.csv").exists():
            val = np.where(cache._val(h))[0]
            para = np.random.default_rng(cfg.SEED).random(len(val)) < 0.10
            R = pd.read_csv(de / "regioes.csv", sep=";")
            final = dict(estado=torch.load(de / "modelo" / "melhor.pt", map_location="cpu", weights_only=False)["model"],
                         norm=cache.normalizador(val[~para]),
                         regiao_celula=pd.read_csv(de / "celulas.csv", sep=";").set_index("celula").regiao.to_dict(),
                         nomes=R.set_index("regiao").nome.to_dict(),
                         taxa_regiao=R.set_index("regiao").taxa_forte_prox.to_dict())
        # climatologia das variáveis interpretáveis, para a explicação
        vv = [c for c in cache.A.columns if c.startswith("em_")] + \
             [c for c in cache.meta["goes_stat_cols"] if c.endswith(("nucleo_min", "nucleo_max"))] + cache.cols_chuva["colar_nucleo"]
        clim = cache.A.loc[cache._val(h), vv].agg(["mean", "std"]).to_dict()
        torch.save(dict(chuva=ch, horizonte=h, comite=comite, final=final, clima=clim,
                        cols_chuva=cache.cols_chuva[ch], meta=cache.meta,
                        s10=cache.s10, s1h=cache.s1h, n_var=len(cache.meta["variaveis"]),
                        n_em10=cache.em10.shape[1], n_em1h=cache.em1h.shape[1], C=cache.campo.shape[1]),
                   PASTA / f"motor_{ch}_h{h}.pt")
        print(f"exportado motor_{ch}_h{h}.pt ({len(comite)} modelos)")


# ═══════════════════════════════ execução ════════════════════════════════════
class _CacheMin:
    """o mínimo que o SOMGoes precisa para ser construído fora do cache"""
    def __init__(self, P):
        self.meta, self.s10, self.s1h = P["meta"], P["s10"], P["s1h"]
        self.em10 = np.zeros((1, P["n_em10"], P["n_var"])); self.em1h = np.zeros((1, P["n_em1h"], P["n_var"]))
        self.campo = np.zeros((1, P["C"], cfg.GOES_RES, cfg.GOES_RES))
        self.cols_chuva = {P["chuva"]: P["cols_chuva"], "nenhuma": []}


class Motor:
    def __init__(self, pasta=PASTA):
        self.P = {}
        for f in sorted(Path(pasta).glob("motor_*.pt")):
            P = torch.load(f, map_location="cpu", weights_only=False)
            cm = _CacheMin(P)
            mods = []
            for c in P["comite"]:
                m = nu.SOMGoes(cm, cfg.GOES_CENARIOS, "ae", (10, 10), chuva=P["chuva"])
                m.load_state_dict(c["estado"]); m.eval(); mods.append(m)
            P["modelos"] = mods
            if P["final"]:
                m = nu.SOMGoes(cm, cfg.GOES_CENARIOS, "ae", cfg.ESTADOS_GRADE, chuva=P["chuva"])
                m.load_state_dict(P["final"]["estado"]); m.eval(); P["final"]["modelo"] = m
            self.P[(P["chuva"], P["horizonte"])] = P

    @staticmethod
    def _norm(bruto, norm, cols):
        """bruto: dict com em10 (S,6,V), em1h (S,V), idade (S), goes (C,R,R), chuva (n) em mm"""
        mu, sd = norm["mu"], norm["sd"]
        x10 = bruto["em10"][None]; m10 = np.isfinite(x10)
        x1h = bruto["em1h"][None]; m1h = np.isfinite(x1h)
        b = dict(x10=np.where(m10, (x10 - mu) / sd, 0), m10=m10, x1h=np.where(m1h, (x1h - mu) / sd, 0), m1h=m1h,
                 idade=np.nan_to_num(bruto["idade"][None] / 60, nan=1.0),
                 goes=((bruto["goes"][None, None] - norm["gmu"]) / norm["gsd"]))
        if cols:
            C = np.log1p(np.clip(np.array([bruto["chuva"].get(c, np.nan) for c in cols], np.float32), 0, None))[None]
            mc = np.isfinite(C)
            mu_c = np.array([norm["chuva"][c][0] for c in cols], np.float32)
            sd_c = np.array([norm["chuva"][c][1] for c in cols], np.float32)
            b["chuva"], b["mchuva"] = np.where(mc, (C - mu_c) / sd_c, 0), mc
        return {k: torch.from_numpy(np.asarray(v, np.float32)) for k, v in b.items()}

    @staticmethod
    def _celula(m, b):
        with torch.no_grad():
            return int(m.dist(m.codificar(b)[0]).argmin(1)[0])

    def avaliar(self, bruto, h, fold_excluir=None):
        disp = np.mean([np.isfinite(bruto["chuva"].get(c, np.nan)) for c in self.P.get(("colar_nucleo", h), {}).get("cols_chuva", [])]) \
            if ("colar_nucleo", h) in self.P else 0.0
        ch = "colar_nucleo" if disp >= 0.5 and ("colar_nucleo", h) in self.P else "nenhuma"
        P = self.P[(ch, h)]
        taxas, honesta = [], np.nan
        for c, m in zip(P["comite"], P["modelos"]):
            t = float(c["taxa"][self._celula(m, self._norm(bruto, c["norm"], P["cols_chuva"]))])
            taxas.append(t)
            if fold_excluir is not None and c["fold"] == fold_excluir:
                honesta = t
        prob = float(np.mean(taxas))
        votos = int(sum(t >= NIVEIS[1][0] for t in taxas))
        out = dict(horizonte_min=h, modelo=ch, fontes_chuva_disponiveis=round(float(disp), 2),
                   prob_comite=round(prob, 4), prob_honesta=None if np.isnan(honesta) else round(honesta, 4),
                   votos_alerta=votos, n_modelos=len(taxas), nivel=nivel_de(prob)[1])
        F = P["final"]
        if F:
            cel = self._celula(F["modelo"], self._norm(bruto, F["norm"], P["cols_chuva"]))
            r = F["regiao_celula"].get(cel)
            out.update(estado=f"R{r}", estado_nome=F["nomes"].get(r), estado_taxa_forte=F["taxa_regiao"].get(r))
        # explicação: variáveis mais afastadas da climatologia agora
        z = {}
        for v, s in P["clima"].items():
            x = bruto["interp"].get(v, np.nan)
            if np.isfinite(x) and s["std"] > 0:
                z[v] = (x - s["mean"]) / s["std"]
        out["destaques"] = [dict(variavel=k, desvios=round(v, 2), valor=round(float(bruto["interp"][k]), 3))
                            for k, v in sorted(z.items(), key=lambda kv: -abs(kv[1]))[:5]]
        return out


# ═══════════════════════════════ entradas ════════════════════════════════════
def bruto_do_cache(cache, i):
    """monta a entrada bruta de um horário a partir do cache (replay histórico)"""
    J = cache.jan[i]
    em10 = np.full((cache.em10.shape[1], cfg.N_QUADROS, cache.em10.shape[2]), np.nan, np.float32)
    for j, k in enumerate(J):
        if k >= 0:
            em10[:, j] = cache.em10[k]
    q = cache.quadros[i, -1]
    goes = cache.campo[max(q, 0)].astype(np.float32) if q >= 0 else np.zeros(cache.campo.shape[1:], np.float32)
    row = cache.A.iloc[i]
    return dict(em10=em10, em1h=cache.em1h[i], idade=cache.idade[i], goes=goes,
                chuva={c: row[c] for c in cache.cols_chuva["colar_nucleo"]}, interp=row.to_dict())


def replay(inicio, fim, nome=None):
    motor, cache = Motor(), nu.Cache()
    A = cache.A
    sel = np.where((A.ts >= pd.Timestamp(inicio)) & (A.ts <= pd.Timestamp(fim)) & cache.valida)[0]
    linhas = []
    for i in sel:
        b = bruto_do_cache(cache, i)
        fold = int(A.fold.iat[i])
        lin = dict(ts=A.ts.iat[i], chuva_nucleo_60min=A.agora.iat[i], chuva_nucleo_prox60=A.prox.iat[i],
                   chuva_nucleo_prox30=A.get("prox_30", A.prox).iat[i])
        for h in cfg.HORIZONTES:
            r = motor.avaliar(b, h, fold_excluir=fold)
            lin[f"prob_honesta_{h}"] = r.get("prob_honesta")
            lin[f"nivel_{h}"] = nivel_de(r["prob_honesta"])[1] if r.get("prob_honesta") is not None else None
            for k in ("estado", "estado_nome", "modelo"):
                lin[f"{k}_{h}"] = r.get(k)
        linhas.append(lin)
    D = pd.DataFrame(linhas)
    nome = nome or f"replay_{pd.Timestamp(inicio):%Y%m%d%H%M}_{pd.Timestamp(fim):%Y%m%d%H%M}"
    D.to_csv(PASTA / f"{nome}.csv", sep=";", index=False)
    if len(D):
        fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
        ax[0].bar(D.ts, D.chuva_nucleo_60min, width=0.006, color="C0")
        ax[0].axhline(10, ls="--", color="C3", lw=0.8); ax[0].set_ylabel("máx. 60 min núcleo (mm)")
        for h, c in ((30, "C1"), (60, "C3")):
            ax[1].plot(D.ts, D[f"prob_honesta_{h}"], color=c, label=f"{h} min (modelo que não viu a temporada)")
        for lim, _ in NIVEIS[:-1]:
            ax[1].axhline(lim, color="k", lw=0.5, ls="--")
        ax[1].set(ylabel="prob. chuva forte", ylim=(0, 1)); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(PASTA / f"{nome}.png", bbox_inches="tight"); plt.close(fig)
    print(f"replay -> {PASTA / (nome + '.csv')} ({len(D)} horários)")
    return D


def eventos(n):
    cache = nu.Cache()
    A = cache.A[cache.valida]
    dias = A.groupby(A.ts.dt.date).agora.max().sort_values(ascending=False).head(n)
    for d in dias.index:
        d0 = pd.Timestamp(d)
        replay(d0 - pd.Timedelta(hours=4), d0 + pd.Timedelta(hours=23, minutes=50), nome=f"evento_{d:%Y%m%d}")


def agora(ts=None):
    """alerta do horário a partir dos dados brutos recentes. Na plataforma hidromet,
    substitua montar_ao_vivo por uma consulta ao banco com o mesmo formato de saída."""
    from alerta.motor_entrada import montar_ao_vivo
    ts = pd.Timestamp(ts) if ts else pd.Timestamp.now().floor("10min")
    b = montar_ao_vivo(ts)
    motor = Motor()
    res = dict(ts=str(ts), horizontes=[motor.avaliar(b, h) for h in cfg.HORIZONTES])
    (PASTA / "alerta_atual.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return res


if __name__ == "__main__":
    PASTA.mkdir(exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "agora"
    if cmd == "exportar":
        exportar()
    elif cmd == "replay":
        replay(sys.argv[2], sys.argv[3])
    elif cmd == "eventos":
        eventos(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    else:
        agora(" ".join(sys.argv[2:]) or None)
