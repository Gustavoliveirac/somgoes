#!/usr/bin/env python
"""
figuras_tese.py — todas as figuras do capítulo, em português e em inglês.

Lê:  analise_somgoes/dados_figuras (síntese), resultados_somgoes/metricas_consolidadas.csv,
     analise_somgoes/previsoes_fora_da_temporada.csv.gz, cache_somgoes/amostras.parquet
     e analise_tese/ (saídas do analise_tese.py).
Grava: figuras_tese/pt/*.png, figuras_tese/en/*.png e, para cada figura, o CSV com os dados
     plotados em figuras_tese/dados/. Log em figuras_tese/figuras_tese.log.

Uso:  python -m tese.figuras_tese                (todas)
      python -m tese.figuras_tese fig08 fig09    (só as indicadas)
      python -m tese.figuras_tese fig08 --evento "2023-03-12 16:00"   (replay de um horário extra)
"""
from __future__ import annotations

import io
import logging
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import config_somgoes as cfg

ROOT = cfg.ROOT
DF = ROOT / "analise_somgoes" / "dados_figuras"
AT = ROOT / "analise_tese"
PREV = ROOT / "analise_somgoes" / "previsoes_fora_da_temporada.csv.gz"
MET = cfg.RESULTS_DIR / "metricas_consolidadas.csv"
OUT = ROOT / "figuras_tese"
TILES = OUT / "cache_tiles"
L_ALERTA, L_PROVAVEL = 0.10, 0.40
log = logging.getLogger("fig")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "figure.dpi": 110})

# ═══════════════════════════════ textos PT / EN ══════════════════════════════
TXT = {
    "frac_tempo": ("fração do tempo em alerta", "fraction of time under warning"),
    "frac_eventos": ("fração dos horários com chuva forte capturados", "fraction of heavy-rain times captured"),
    "ap": ("precisão média (AP)", "average precision (AP)"),
    "nos": ("número de nós (ou grupos)", "number of nodes (or clusters)"),
    "som_cn": ("SOM, anéis e área central", "SOM, rings and central area"),
    "som_c": ("SOM, anéis", "SOM, rings only"),
    "som_n": ("SOM, sem chuva", "SOM, no rainfall input"),
    "kmeans": ("k-means", "k-means"),
    "som_cl": ("SOM clássico", "classical SOM"),
    "acaso": ("ordenação ao acaso", "random ranking"),
    "chuva_nuc": ("chuva na área central (máx. em 10 min, mm)", "central-area rainfall (10-min max, mm)"),
    "chuva_c1": ("anel de 13 a 60 km (máx. em 10 min, mm)", "13–60 km ring (10-min max, mm)"),
    "prob": ("probabilidade de chuva forte", "probability of heavy rain"),
    "inicio": ("início da chuva na área central", "rain onset in the central area"),
    "limiar": ("chuva passa de 10 mm/h", "rain exceeds 10 mm/h"),
    "alerta": ("alerta (0,10)", "warning (0.10)"),
    "provavel": ("chuva forte provável (0,40)", "heavy rain likely (0.40)"),
    "h30": ("próximos 30 min", "next 30 min"), "h60": ("próximos 60 min", "next 60 min"),
    "agora": ("estado agora", "current state"), "depois": ("estado no fim do horizonte", "state at the end of the horizon"),
    "estados": (["seco", "fraca", "moderada", "forte", "muito forte"], ["dry", "light", "moderate", "heavy", "very heavy"]),
    "cat": ({"alerta antes do início da chuva": "alerta antes do início da chuva",
             "alerta com chuva, antes do limiar": "alerta com chuva, antes do limiar",
             "alerta só depois do limiar": "alerta só depois do limiar", "sem alerta": "sem alerta"},
            {"alerta antes do início da chuva": "warning before rain onset",
             "alerta com chuva, antes do limiar": "warning during rain, before threshold",
             "alerta só depois do limiar": "warning only after threshold", "sem alerta": "no warning"}),
    "falsos": ({}, {"forte no núcleo logo depois da janela": "heavy rain in the core just after the window",
                    "forte no colar 1, não no núcleo": "heavy rain in ring 1, not in the core",
                    "quase forte no núcleo (5 a 10 mm/h)": "near-heavy rain in the core (5 to 10 mm/h)",
                    "chuva fraca ou moderada no núcleo": "light or moderate rain in the core",
                    "seco no núcleo, chuva no colar 1": "dry core, rain in ring 1",
                    "seco no núcleo e no colar 1": "dry core and ring 1"}),
    "grupo": ({"todos": "todos", "partiu do seco": "partiu do seco", "já chovia no núcleo": "já chovia no núcleo"},
              {"todos": "all", "partiu do seco": "from dry", "já chovia no núcleo": "already raining in the core"}),
    "dif_pad": ("diferença padronizada (alvo − referência)", "standardized difference (target − reference)"),
    "queda_ap": ("queda da AP ao embaralhar o bloco", "AP drop when the block is permuted"),
    "prev": ("probabilidade prevista", "forecast probability"), "obs": ("frequência observada", "observed frequency"),
    "nucleo": ("área central (até 12,7 km)", "central area (within 12.7 km)"), "colar1": ("anel de 13 a 60 km", "13–60 km ring"),
    "colar2": ("anel de 60 a 100 km", "60–100 km ring"), "em": ("estação meteorológica", "weather station"),
}
LANG = 0


def t(k):
    v = TXT[k][LANG]
    return v


def rot(dic_key, x):
    d = TXT[dic_key][LANG]
    return d.get(x, x)


def nome_escore(s):
    if LANG == 0:
        return s
    m = {"SOM (colar e núcleo)": "SOM (ring and core)", "SOM (colar)": "SOM (ring only)",
         "SOM (sem chuva)": "SOM (no rainfall input)", "persistência: estado atual do núcleo": "persistence: current core state",
         "chuva no colar 1 (máx. 60 min)": "rain in ring 1 (60-min max)", "chuva no colar 1 a barlavento": "rain in upwind ring 1",
         "chuva máxima até 100 km": "max rain within 100 km", "RRQPE máximo sobre o núcleo": "max RRQPE over the core",
         "logística (sem chuva)": "logistic (no rainfall)", "gradient boosting (sem chuva)": "gradient boosting (no rainfall)",
         "logística (colar e núcleo)": "logistic (ring and core)", "gradient boosting (colar e núcleo)": "gradient boosting (ring and core)",
         "modelo do seco (gradient boosting)": "dry-hours model (gradient boosting)",
         "híbrido: SOM com chuva, modelo do seco sem chuva": "hybrid: SOM when raining, dry model when dry"}
    return m.get(s, s)


# ═══════════════════════════════ utilidades ══════════════════════════════════
def ler(p, **kw):
    p = Path(p)
    if not p.exists():
        log.info(f"  falta {p}; figura pulada")
        return None
    return pd.read_csv(p, sep=";", **kw)


def salvar(nome, fig, dados=None):
    sub = "pt" if LANG == 0 else "en"
    (OUT / sub).mkdir(parents=True, exist_ok=True); (OUT / "dados").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / sub / f"{nome}.png", bbox_inches="tight"); plt.close(fig)
    if dados is not None and LANG == 0:
        dados.to_csv(OUT / "dados" / f"{nome}.csv", sep=";", index=False)
    log.info(f"  {sub}/{nome}.png")


def metricas():
    M = ler(MET, low_memory=False)
    if M is None:
        return None
    M = M[M.fold >= 0].copy()
    M["chuva"] = M.chuva.fillna("nenhuma"); M["horizonte"] = M.horizonte.fillna(60)
    M["nos"] = M.grade.str.split("x").apply(lambda v: int(v[0]) * int(v[1]))
    return M


# ═══════════════════════════════ fig01 mapa ══════════════════════════════════
def tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def mapa_base(lat0, lat1, lon0, lon1, z=9):
    import requests
    from PIL import Image
    TILES.mkdir(parents=True, exist_ok=True)
    x0, y0 = tile_xy(lat1, lon0, z); x1, y1 = tile_xy(lat0, lon1, z)
    X = range(int(x0), int(x1) + 1); Y = range(int(y0), int(y1) + 1)
    img = Image.new("RGB", (256 * len(X), 256 * len(Y)))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            f = TILES / f"{z}_{x}_{y}.png"
            if not f.exists():
                r = requests.get(f"https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                                 headers={"User-Agent": "somgoes-tese/1.0 (pesquisa UFMG)"}, timeout=30)
                r.raise_for_status(); f.write_bytes(r.content); time.sleep(0.2)
            img.paste(Image.open(f).convert("RGB"), (256 * i, 256 * j))
    def inv(px, py):
        n = 2 ** z
        lon = px / n * 360 - 180
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * py / n))))
        return lat, lon
    la_top, lo_left = inv(X[0], Y[0]); la_bot, lo_right = inv(X[-1] + 1, Y[-1] + 1)
    return np.asarray(img), (lo_left, lo_right, la_bot, la_top)


def recorte_goes():
    """Contorno do recorte do GOES em lat/lon. Procura, nesta ordem, goes_recorte.json na pasta
    goes/ do somgoes ({"lat": [[...]], "lon": [[...]]} ou "cantos": [[lat, lon], ...]) e arquivos
    *lat*.npy e *lon*.npy na pasta do GOES. Sem eles, a figura sai sem o contorno e o log avisa."""
    import json
    j = cfg.GOES_META / "goes_recorte.json"
    lat = lon = None
    if j.exists():
        d = json.load(open(j))
        if "cantos" in d:
            c = np.asarray(d["cantos"], float)
            return np.vstack([c, c[:1]])
        lat, lon = np.asarray(d["lat"], float), np.asarray(d["lon"], float)
    else:
        try:
            fl = sorted(cfg.GOES_DIR.glob("*lat*.npy")); fo = sorted(cfg.GOES_DIR.glob("*lon*.npy"))
            if fl and fo:
                lat, lon = np.load(fl[0]), np.load(fo[0])
        except Exception:                     # noqa: BLE001
            pass
    if lat is None or lat.ndim != 2:
        log.info("  contorno do GOES não encontrado (crie goes_recorte.json com os quatro cantos); mapa sem o recorte")
        return None
    borda = np.r_[np.c_[lat[0, :], lon[0, :]], np.c_[lat[:, -1], lon[:, -1]],
                  np.c_[lat[-1, ::-1], lon[-1, ::-1]], np.c_[lat[::-1, 0], lon[::-1, 0]]]
    return borda


def fig01():
    E = ler(DF / "F13_mapa_estacoes.csv")
    if E is None:
        return
    E = E[E.papel != "excluída"].copy()
    lat_c = float(np.mean(E.Latitude - E.y_km / 111.32))
    lon_c = float(np.mean(E.Longitude - E.x_km / (111.32 * np.cos(np.radians(E.Latitude)))))
    d = 1.05
    g = recorte_goes()
    if g is not None:
        d = max(d, 1.05 * np.nanmax(np.abs(g[:, 0] - lat_c)), 1.05 * np.nanmax(np.abs(g[:, 1] - lon_c)) * np.cos(np.radians(lat_c)))
    img, ext = mapa_base(lat_c - d, lat_c + d, lon_c - d / np.cos(np.radians(lat_c)), lon_c + d / np.cos(np.radians(lat_c)))
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(img, extent=ext, alpha=0.55)
    estilo = {"núcleo": ("o", "C3", 14, t("nucleo")), "colar 1 (13-60 km)": ("o", "C0", 12, t("colar1")),
              "colar 2 (60-100 km)": ("o", "C2", 12, t("colar2"))}
    for papel, (mk, cor, ms, lab) in estilo.items():
        g = E[E.papel.str.startswith(papel.split(" (")[0]) & ~E.Tipo.eq("meteorologica")]
        ax.scatter(g.Longitude, g.Latitude, s=ms, c=cor, marker=mk, edgecolor="k", lw=0.3, label=lab, zorder=3)
    g = E[E.Tipo.eq("meteorologica")]
    ax.scatter(g.Longitude, g.Latitude, s=40, c="gold", marker="^", edgecolor="k", lw=0.5, label=t("em"), zorder=4)
    for r in (12.7, 60, 100):
        a = np.linspace(0, 2 * np.pi, 361)
        ax.plot(lon_c + r * np.sin(a) / (111.32 * np.cos(np.radians(lat_c))), lat_c + r * np.cos(a) / 111.32,
                "k--", lw=0.7)
    goes = recorte_goes()
    if goes is not None:
        ax.plot(goes[:, 1], goes[:, 0], color="purple", lw=1.4,
                label="recorte do GOES-16 (32 x 32 células)" if LANG == 0 else "GOES-16 crop (32 x 32 cells)")
    ax.set(xlim=ext[:2], ylim=ext[2:], xlabel="longitude", ylabel="latitude"); ax.set_aspect(1 / np.cos(np.radians(lat_c)))
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.8)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
    ax.text(0.99, 0.01, "© OpenStreetMap", transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
    salvar("fig01_area_e_estacoes", fig, E[["Codigo", "Nome", "Fonte", "Tipo", "Latitude", "Longitude", "dist_km", "papel"]])


# ═══════════════════════════════ fig02 transições ════════════════════════════
def fig02():
    D = ler(AT / "A1_matriz_transicoes.csv")
    if D is None:
        return
    est = cfg.ESTADOS
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for a, h in zip(ax, (30, 60)):
        d = D[D.horizonte == h]
        M = d.pivot(index="agora", columns="depois", values="prob_condicional").reindex(index=est, columns=est)
        N = d.pivot(index="agora", columns="depois", values="n").reindex(index=est, columns=est)
        im = a.imshow(M.values, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                if np.isfinite(M.values[i, j]):
                    a.text(j, i, f"{M.values[i, j]:.2f}\n({int(N.values[i, j]):,})".replace(",", "." if LANG == 0 else ","),
                           ha="center", va="center", fontsize=6.5, color="w" if M.values[i, j] > 0.5 else "k")
        a.set(xticks=range(5), yticks=range(5), xticklabels=t("estados"), yticklabels=t("estados"),
              xlabel=t("depois"), ylabel=t("agora"), title=t(f"h{h}"))
        a.tick_params(axis="x", rotation=30)
    fig.colorbar(im, ax=ax, shrink=0.8)
    salvar("fig02_matriz_de_transicoes", fig, D)


# ═══════════════════════════════ fig03 concentração ══════════════════════════
def fig03():
    G = ler(AT / "A2_curvas_de_ganho.csv"); M = metricas()
    if G is None:
        return
    mostrar = ["SOM (colar e núcleo)", "SOM (sem chuva)", "persistência: estado atual do núcleo",
               "chuva no colar 1 (máx. 60 min)", "RRQPE máximo sobre o núcleo", "gradient boosting (colar e núcleo)"]
    estilo = dict(zip(mostrar, ["C3-", "C0-", "k--", "C2-.", "C1:", "C4-"]))
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)
    pts = []
    for a, h in zip(ax, (30, 60)):
        for nm in mostrar:
            g = G[(G.horizonte == h) & (G.escore == nm)]
            if len(g):
                a.plot(g.frac_tempo, g.recall, estilo[nm], lw=1.3, label=nome_escore(nm))
        a.plot([0, 0.3], [0, 0.3], color="0.6", lw=0.8, label=t("acaso"))
        if M is not None:
            for fam, ch, mk, lab in (("kmeans_cenarios", "colar_nucleo", "s", t("kmeans")),):
                r = M[(M.familia == fam) & (M.chuva == ch) & (M.horizonte == h) & (M.nos == 100)]
                if len(r):
                    xs = [0.01, 0.02, 0.05, 0.10]
                    ys = [r[f"forte_prox.cob_{int(x*100)}pct"].mean() for x in xs]
                    a.plot(xs, ys, mk, color="C5", ms=5, label=f"{lab}, 100 {'grupos' if LANG == 0 else 'clusters'}")
                    pts += [dict(horizonte=h, modelo="k-means 100", frac_tempo=x, recall=y) for x, y in zip(xs, ys)]
        a.set(xlim=(0, 0.3), ylim=(0, 1), xlabel=t("frac_tempo"), title=t(f"h{h}"))
        a.axvline(0.05, color="0.8", lw=0.6)
    ax[0].set_ylabel(t("frac_eventos")); ax[1].legend(fontsize=6.5, loc="lower right")
    salvar("fig03_concentracao_dos_eventos", fig, pd.concat([G[G.escore.isin(mostrar)], pd.DataFrame(pts)]))


# ═══════════════════════════════ fig04 grade ═════════════════════════════════
def fig04():
    M = metricas()
    if M is None:
        return
    som = M[(M.familia == "grades") & ((M.perda == "mi") | ((M.chuva == "colar") & (M.perda == "duas_etapas")))]
    km = M[M.familia == "kmeans_cenarios"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    linhas = []
    for a, h in zip(ax, (30, 60)):
        for ch, cor in (("colar_nucleo", "C3"), ("colar", "C2"), ("nenhuma", "C0")):
            for df, ls, lab in ((som, "-o", "SOM"), (km, "--s", "k-means")):
                g = df[(df.chuva == ch) & (df.horizonte == h)].groupby("nos")["forte_prox.ap"].agg(["mean", "std"]).reset_index()
                if len(g):
                    a.errorbar(g.nos, g["mean"], yerr=g["std"] / np.sqrt(6), fmt=ls, color=cor, ms=3, lw=1, capsize=2,
                               label=f"{lab}, {t({'colar_nucleo': 'som_cn', 'colar': 'som_c', 'nenhuma': 'som_n'}[ch]).split(', ')[1]}")
                    linhas.append(g.assign(modelo=lab, chuva=ch, horizonte=h))
        a.set(xscale="log", xlabel=t("nos"), title=t(f"h{h}")); a.set_xticks([4, 9, 25, 100, 225, 625]); a.set_xticklabels([4, 9, 25, 100, 225, 625])
    ax[0].set_ylabel(t("ap")); ax[1].legend(fontsize=6.5, ncol=2)
    salvar("fig04_ap_por_tamanho_da_grade", fig, pd.concat(linhas))


# ═══════════════════════════════ fig05 perdas e entradas ═════════════════════
def fig05():
    M = metricas()
    if M is None:
        return
    nomes = {("etapa1", "t0", "recon"): ("reconstrução", "reconstruction"),
             ("etapa1", "t0", "mi"): ("informação mútua (mi)", "mutual information (mi)"),
             ("etapa1", "t0", "mi_forte"): ("mi, peso nas fortes", "mi, heavy weighted"),
             ("etapa1", "t0", "mi_lift"): ("mi + lift", "mi + lift"),
             ("etapa1", "t0", "mi_sep"): ("mi + separação", "mi + separation"),
             ("etapa1", "t0", "mi_canto"): ("mi + cantos", "mi + corners"),
             ("etapa1", "t0", "completa"): ("todos os termos", "all terms"),
             ("etapa1", "nenhum", "mi"): ("mi, sem GOES", "mi, no GOES"),
             ("etapa1", "seq_cnn", "mi"): ("mi, 6 quadros GOES (CNN)", "mi, 6 GOES frames (CNN)"),
             ("etapa1", "convlstm", "mi"): ("mi, 6 quadros GOES (ConvLSTM)", "mi, 6 GOES frames (ConvLSTM)")}
    linhas = []
    for (fam, goes, perda), (pt, en) in nomes.items():
        g = M[(M.familia == fam) & (M.goes == goes) & (M.perda == perda) & (M.grade == "10x10") & (M.ae == "ae")]
        if len(g):
            linhas.append(dict(nome=(pt, en)[LANG], AP=g["forte_prox.ap"].mean(), ep=g["forte_prox.ap"].std() / np.sqrt(len(g)),
                               recall5=g["forte_prox.cob_5pct"].mean(), grupo="SOM profundo"))
    for fam, perda, pt, en in (("som_classico", "baseline", "SOM clássico", "classical SOM"),
                               ("som_classico", "supervisionado", "SOM clássico supervisionado", "supervised classical SOM")):
        g = M[(M.familia == fam) & (M.perda == perda) & (M.grade == "10x10")]
        if len(g):
            linhas.append(dict(nome=(pt, en)[LANG], AP=g["forte_prox.ap"].mean(), ep=g["forte_prox.ap"].std() / np.sqrt(len(g)),
                               recall5=g["forte_prox.cob_5pct"].mean(), grupo="referência"))
    g = M[M.exp_id.eq("kmeans_k100")]
    if len(g):
        linhas.append(dict(nome="k-means (100)", AP=g["forte_prox.ap"].mean(), ep=g["forte_prox.ap"].std() / np.sqrt(len(g)),
                           recall5=g["forte_prox.cob_5pct"].mean(), grupo="referência"))
    D = pd.DataFrame(linhas).sort_values("AP")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(D.nome, D.AP, xerr=D.ep, color=np.where(D.grupo == "referência", "0.65", "C0"), capsize=2)
    ax.axvline(0.040, color="k", ls=":", lw=0.8)
    ax.set(xlabel=t("ap") + (" (sem chuva na entrada, 60 min, grade 10x10)" if LANG == 0 else " (no rainfall input, 60 min, 10x10 grid)"))
    salvar("fig05_perdas_e_entradas_goes", fig, D)


# ═══════════════════════════════ fig06 fontes ════════════════════════════════
def fig06():
    S = ler(DF / "S05_valor_de_cada_fonte.csv")
    if S is None:
        return
    base = {30: 0.0458, 60: 0.0412}   # taxa base média fora da temporada (AP = skill x (1 - base) + base)
    S["AP_completo"] = S.skill_completo * (1 - S.horizonte.map(base)) + S.horizonte.map(base)
    S["AP_sem_fonte"] = S.skill_sem_fonte * (1 - S.horizonte.map(base)) + S.horizonte.map(base)
    S["queda_AP"] = S.AP_completo - S.AP_sem_fonte
    tr = {"EMs": "weather stations", "GOES": "GOES", "colar 1 (13-60 km)": "ring 1 (13-60 km)",
          "colar 2 (60-100 km)": "ring 2 (60-100 km)", "núcleo": "core"}
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for a, col, xl in zip(ax, ("queda_AP", "perda_recall5"), (t("ap"), "recall a 5% do tempo" if LANG == 0 else "recall at 5% of time")):
        for k, h in enumerate((30, 60)):
            g = S[S.horizonte == h]
            y = np.arange(len(g)) + (k - 0.5) * 0.35
            a.barh(y, g[col], height=0.35, label=t(f"h{h}"), color=("C1", "C3")[k])
        a.set_yticks(np.arange(len(g))); a.set_yticklabels([g_ if LANG == 0 else tr.get(g_, g_) for g_ in g.fonte_removida])
        a.axvline(0, color="k", lw=0.6)
        a.set_xlabel(("queda de " if LANG == 0 else "drop in ") + xl + (" ao retirar a fonte" if LANG == 0 else " when the source is removed"))
    ax[1].legend(fontsize=7)
    salvar("fig06_contribuicao_de_cada_fonte", fig, S)


# ═══════════════════════════════ fig07 regiões ═══════════════════════════════
def fig07():
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    dados = []
    for a, h in zip(ax, (30, 60)):
        G = ler(DF / f"F07_mapa_grade_colar_nucleo_h{h}.csv")
        if G is None:
            plt.close(fig); return
        R = G.pivot(index="linha", columns="coluna", values="regiao").values
        T = G.pivot(index="linha", columns="coluna", values="taxa_forte_prox").values
        a.imshow(R, cmap="tab10", vmin=0.5, vmax=10.5, alpha=0.35)
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                a.text(j, i, f"{100*T[i, j]:.0f}", ha="center", va="center", fontsize=6)
                for di, dj in ((0, 1), (1, 0)):
                    if i + di < R.shape[0] and j + dj < R.shape[1] and R[i, j] != R[i + di, j + dj]:
                        if di:
                            a.plot([j - .5, j + .5], [i + .5, i + .5], "k", lw=1)
                        else:
                            a.plot([j + .5, j + .5], [i - .5, i + .5], "k", lw=1)
        for r in np.unique(R):
            ii, jj = np.where(R == r)
            a.text(jj.mean(), ii.mean() - 0.35, f"R{int(r)}", ha="center", fontsize=8, weight="bold", color="0.2")
        a.set(xticks=[], yticks=[], title=t(f"h{h}") + (" (% de chuva forte por nó)" if LANG == 0 else " (% heavy rain per node)"))
        dados.append(G.assign(horizonte=h))
    salvar("fig07_regioes_do_mapa", fig, pd.concat(dados))


# ═══════════════════════════════ fig08 eventos (replay) ══════════════════════
def serie_obs():
    cols = ["ts", "in_nuc_max10", "in_c1_tot_max10", "estado_agora", "estado_agora_30", "agora", "agora_30"]
    A = pd.read_parquet(cfg.CACHE_DIR / "amostras.parquet", columns=cols)
    return A


def painel(ax, A, P, centro, antes=4, depois=2, titulo=""):
    t0, t1 = pd.Timestamp(centro) - pd.Timedelta(hours=antes), pd.Timestamp(centro) + pd.Timedelta(hours=depois)
    a = A[(A.ts >= t0) & (A.ts <= t1)]
    p = P[(P.ts >= t0) & (P.ts <= t1) & (P.chuva == "colar_nucleo")]
    w = 0.0065
    ax.bar(a.ts, a.in_nuc_max10, width=w, color="#4a7fb5", label=t("chuva_nuc"), zorder=2)
    ax.step(a.ts, a.in_c1_tot_max10, where="mid", color="0.55", lw=0.8, label=t("chuva_c1"), zorder=1)
    ax.set_ylabel("mm / 10 min")
    ax.set_ylim(0, max(2, np.nanmax(np.r_[a.in_nuc_max10.to_numpy(float), a.in_c1_tot_max10.to_numpy(float)]) * 1.15))
    b = ax.twinx()
    for h, cor in ((30, "C1"), (60, "C3")):
        q = p[p.horizonte == h].sort_values("ts")
        b.plot(q.ts, q.escore, color=cor, lw=1.4, label=f"{t('prob')}, {t(f'h{h}')}")
    b.axhline(L_ALERTA, color="k", ls="--", lw=0.6); b.axhline(L_PROVAVEL, color="k", ls="-.", lw=0.6)
    b.text(a.ts.iloc[0], L_ALERTA + 0.01, t("alerta"), fontsize=6); b.text(a.ts.iloc[0], L_PROVAVEL + 0.01, t("provavel"), fontsize=6)
    b.set(ylim=(0, 1), ylabel=t("prob")); b.spines["right"].set_visible(True)
    # marcos: início da chuva no núcleo e passagem do limiar (60 min)
    e = a.estado_agora.to_numpy()
    k = np.where((e[1:] >= 3) & (e[:-1] < 3))[0]
    marcos = []
    if len(k):
        i = k[0] + 1; tc = a.ts.iloc[i]
        j = i
        while j - 1 >= 0 and e[j - 1] >= 1:
            j -= 1
        tr = a.ts.iloc[j]
        ax.axvline(tr, color="k", ls=":", lw=1.1, label=t("inicio"))
        ax.axvline(tc, color="C3", lw=1.2, label=t("limiar"))
        marcos = [("inicio_chuva", tr), ("limiar_10mm_h", tc)]
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title(titulo, fontsize=8.5, loc="left")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
    return h1 + h2, l1 + l2, a.merge(p.pivot_table(index="ts", columns="horizonte", values="escore").add_prefix("prob_h").reset_index(),
                                     on="ts", how="left").assign(marcos=str(marcos))


def escolher_eventos():
    I = ler(AT / "A3_inicios_por_evento.csv", parse_dates=["ts_limiar", "ts_inicio_chuva"])
    F = ler(AT / "A4_episodios_de_alerta.csv", parse_dates=["inicio", "fim"])
    ev = []
    if I is not None:
        s = I[(I.escore == "SOM (colar e núcleo)") & (I.horizonte == 60) & I.nivel.str.startswith("alerta")]
        a = s[s.partiu_do_seco & (s.categoria == "alerta antes do início da chuva")].sort_values("antecedencia_limiar_min")
        if len(a):
            r = a.iloc[len(a) // 2]
            ev.append(("acerto_partindo_do_seco", r.ts_limiar, "(a) " + ("alerta antes do início da chuva" if LANG == 0 else "warning before rain onset")))
        b = s[(~s.partiu_do_seco) & (s.categoria == "alerta com chuva, antes do limiar")].sort_values("antecedencia_limiar_min")
        if len(b):
            r = b.iloc[len(b) // 2]
            ev.append(("acerto_com_chuva", r.ts_limiar, "(b) " + ("alerta com chuva, antes do limiar" if LANG == 0 else "warning during rain, before threshold")))
        c = s[s.categoria.isin(["sem alerta", "alerta só depois do limiar"])]
        if len(c):
            r = c.sort_values("chuva_antes_do_limiar_min").iloc[len(c) // 2]
            ev.append(("nao_previsto", r.ts_limiar, "(c) " + ("início não previsto" if LANG == 0 else "missed onset")))
    if F is not None:
        f = F[(F.escore == "SOM (colar e núcleo)") & (F.horizonte == 60) & F.nivel.str.startswith("provável") & (F.tipo != "correto")]
        if len(f):
            r = f.sort_values("escore_max").iloc[-1]
            ev.append(("falso_positivo", r.inicio + (r.fim - r.inicio) / 2, "(d) " + ("alerta sem chuva forte: " if LANG == 0 else "false alarm: ") + rot("falsos", r.tipo)))
    return ev


def fig08(extras=()):
    P = pd.read_csv(PREV, sep=";", parse_dates=["ts"])
    A = serie_obs()
    ev = escolher_eventos() + [(f"extra_{pd.Timestamp(x):%Y%m%d%H%M}", pd.Timestamp(x), str(x)) for x in extras]
    if not ev:
        return
    n = len(ev)
    fig, ax = plt.subplots(n, 1, figsize=(9, 2.6 * n))
    ax = np.atleast_1d(ax)
    dados, leg = [], {}
    for a, (nome, centro, tit) in zip(ax, ev):
        h, l, d = painel(a, A, P, centro, titulo=f"{tit}  ({pd.Timestamp(centro):%d/%m/%Y})")
        leg.update(dict(zip(l, h)))
        dados.append(d.assign(painel=nome))
    fig.legend(list(leg.values()), list(leg.keys()), loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    salvar("fig08_eventos_replay", fig, pd.concat(dados))
    # um painel por dia dos eventos do motor (para conferência)
    pasta = ROOT / "motor"
    for p in sorted(pasta.glob("evento_*.csv")):
        D = pd.read_csv(p, sep=";", parse_dates=["ts"])
        c = D.loc[D.chuva_nucleo_60min.idxmax(), "ts"]
        fig, a = plt.subplots(figsize=(9, 2.8))
        h, l, _ = painel(a, A, P, c, antes=6, depois=3, titulo=p.stem)
        a.legend(h, l, fontsize=6, loc="upper left", ncol=2)
        salvar(f"eventos/{p.stem}", fig)


# ═══════════════════════════════ fig09 antecedência ══════════════════════════
def fig09():
    R = ler(AT / "A3_inicios_resumo.csv")
    if R is None:
        return
    cats = ["alerta antes do início da chuva", "alerta com chuva, antes do limiar", "alerta só depois do limiar", "sem alerta"]
    cores = ["C2", "C0", "C1", "0.6"]
    R = R[R.escore.isin(["SOM (colar e núcleo)", "SOM (sem chuva)", "gradient boosting (colar e núcleo)"]) &
          R.nivel.str.startswith("alerta") & R.grupo.isin(["partiu do seco", "já chovia no núcleo"])]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for a, h in zip(ax, (30, 60)):
        g = R[R.horizonte == h].reset_index(drop=True)
        lab = [f"{nome_escore(e)}\n{rot('grupo', gr)} (n={n})" for e, gr, n in zip(g.escore, g.grupo, g.inicios)]
        esq = np.zeros(len(g))
        for c, cor in zip(cats, cores):
            v = g[c].fillna(0).to_numpy()
            a.barh(lab, v, left=esq, color=cor, label=rot("cat", c)); esq += v
        a.set(xlim=(0, 1), title=t(f"h{h}"), xlabel="fração dos inícios de chuva forte" if LANG == 0 else "fraction of heavy-rain onsets")
        a.tick_params(axis="y", labelsize=6.5)
    ax[1].legend(fontsize=6.5, loc="lower right")
    salvar("fig09_antecedencia_dos_alertas", fig, R)


# ═══════════════════════════════ fig10 falsos ════════════════════════════════
def fig10():
    R = ler(AT / "A4_falsos_resumo.csv")
    if R is None:
        return
    tipos = [c for c in R.columns if c.startswith("falsos: ")]
    R = R[R.escore.isin(["SOM (colar e núcleo)", "SOM (sem chuva)"])].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    lab = [f"{nome_escore(e)}, {n}, {h} min" for e, n, h in zip(R.escore, R.nivel, R.horizonte)]
    esq = np.zeros(len(R))
    for k, c in enumerate(tipos):
        v = R[c].fillna(0).to_numpy()
        ax.barh(lab, v, left=esq, color=plt.cm.tab20(k), label=rot("falsos", c.replace("falsos: ", ""))); esq += v
    ax.set(xlim=(0, 1), xlabel="fração dos episódios de alerta sem chuva forte" if LANG == 0 else "fraction of false-alarm episodes")
    ax.tick_params(axis="y", labelsize=6.5); ax.legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left")
    salvar("fig10_anatomia_dos_falsos_positivos", fig, R)


# ═══════════════════════════════ fig11 fatores ═══════════════════════════════
def fig11():
    D = ler(AT / "A5_fatores_top10.csv")
    if D is None:
        return
    D = D[D.horizonte == 60]
    cs = D.contraste.unique()
    fig, ax = plt.subplots(1, len(cs), figsize=(3.2 * len(cs), 3.6))
    ax = np.atleast_1d(ax)
    for a, c in zip(ax, cs):
        g = D[D.contraste == c].sort_values("diferenca_padronizada")
        a.barh(g.variavel, g.diferenca_padronizada, color=np.where(g.diferenca_padronizada > 0, "C3", "C0"))
        a.axvline(0, color="k", lw=0.6); a.set_title(c, fontsize=7); a.tick_params(axis="y", labelsize=6)
        a.set_xlabel(t("dif_pad"), fontsize=7)
    salvar("fig11_fatores_das_transicoes", fig, D)


# ═══════════════════════════════ fig12 seco ══════════════════════════════════
def fig12():
    C = ler(AT / "A6_curvas_de_ganho_seco.csv"); I = ler(AT / "A6_importancia_por_bloco.csv")
    if C is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, h in zip(ax[:2], (30, 60)):
        for nm, g in C[C.horizonte == h].groupby("escore"):
            a.plot(g.frac_tempo, g.recall, lw=1.2, label=nome_escore(nm))
        a.plot([0, .3], [0, .3], color="0.6", lw=0.8)
        a.set(xlim=(0, 0.3), ylim=(0, 1), xlabel=t("frac_tempo"), title=t(f"h{h}") + (" (só horários secos)" if LANG == 0 else " (dry times only)"))
    ax[0].set_ylabel(t("frac_eventos")); ax[1].legend(fontsize=6)
    if I is not None:
        for k, h in enumerate((30, 60)):
            g = I[I.horizonte == h]
            y = np.arange(len(g)) + (k - 0.5) * 0.35
            ax[2].barh(y, g["mean"], xerr=g["std"], height=0.35, color=("C1", "C3")[k], label=t(f"h{h}"), capsize=2)
        ax[2].set_yticks(np.arange(len(g))); ax[2].set_yticklabels(g.bloco); ax[2].set_xlabel(t("queda_ap"))
        ax[2].axvline(0, color="k", lw=0.6); ax[2].legend(fontsize=7)
    salvar("fig12_inicio_a_partir_do_seco", fig, C)


# ═══════════════════════════════ fig13 confiabilidade ════════════════════════
def fig13():
    S = ler(DF / "S06_confiabilidade.csv")
    if S is None:
        return
    fig, a = plt.subplots(figsize=(4.4, 4.2))
    for (ch, h), g in S[S.chuva.isin(["colar_nucleo", "nenhuma"])].groupby(["chuva", "horizonte"]):
        a.plot(g.prob_prevista, g.freq_observada, "o-", ms=3, label=f"{t('som_cn' if ch == 'colar_nucleo' else 'som_n')}, {h} min")
    a.plot([0, 1], [0, 1], "k:"); a.set(xlabel=t("prev"), ylabel=t("obs"), xlim=(0, 1), ylim=(0, 1)); a.legend(fontsize=7)
    salvar("fig13_confiabilidade", fig, S)



# ═══════════════════════════════ fig14 fluxograma de estados ═════════════════
NOMES_FATOR = {"C08": "vapor alto (C08)", "C09": "vapor médio (C09)", "C10": "vapor baixo (C10)",
               "C11": "topo (C11)", "C12": "topo (C12)", "C13": "topo (C13)", "C14": "topo (C14)",
               "RRQPE": "RRQPE"}


def nome_fator(v):
    if LANG == 1:
        return v
    if v.startswith("in_c1"):
        return "chuva no anel de 13 a 60 km" + (" (10 min)" if v.endswith("max10") else " (60 min)" if "max60" in v else " (fração de estações)")
    if v.startswith("in_c2"):
        return "chuva no anel de 60 a 100 km"
    if v.startswith("em_umidity"):
        return "umidade (EMs)"
    if v.startswith("em_temperature"):
        return "temperatura (EMs)"
    for k, n in NOMES_FATOR.items():
        if v.startswith(k):
            return n + (" mais frio na área central" if "nucleo_min" in v else " mais frio na região" if v.endswith(("_min", "_p10")) else "")
    return v


def fig14():
    """Fluxograma com três estados da área central (seco, chuva fraca ou moderada, chuva forte ou
    muito forte). Cada seta traz a probabilidade de passar de um estado ao outro em 60 min e as
    duas variáveis mais associadas à transição (A5), com a AUC de cada uma."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    D = ler(AT / "A1_matriz_transicoes.csv"); F = ler(AT / "A5_fatores_top10.csv")
    if D is None:
        return
    h = 60
    D = D[D.horizonte == h].copy()
    grupo = {"seco": "S", "fraca": "C", "moderada": "C", "forte": "F", "muito_forte": "F"}
    D["g0"] = D.agora.map(grupo); D["g1"] = D.depois.map(grupo)
    N = D.groupby(["g0", "g1"]).n.sum().unstack(fill_value=0)
    P = N.div(N.sum(1), axis=0); frac = N.sum(1) / N.values.sum()
    nomes = {"S": ("seco", "dry"), "C": ("chuva fraca\nou moderada", "light or\nmoderate rain"),
             "F": ("chuva forte\nou muito forte", "heavy or\nvery heavy rain")}
    pos = {"S": (0.0, 0.0), "C": (9.0, 0.0), "F": (4.5, 6.0)}
    cores = {"S": "#e0e0e0", "C": "#9ecae1", "F": "#08519c"}
    fig, ax = plt.subplots(figsize=(12, 8.5))
    for k, (x, y) in pos.items():
        ax.add_patch(FancyBboxPatch((x - 1.5, y - 0.75), 3.0, 1.5, boxstyle="round,pad=0.06", fc=cores[k], ec="k", lw=0.9))
        cor_t = "w" if k == "F" else "k"
        ax.text(x, y + 0.28, nomes[k][LANG], ha="center", va="center", fontsize=10, weight="bold", color=cor_t)
        ax.text(x, y - 0.42, (f"{100*frac[k]:.1f}% do tempo\npermanece {100*P.loc[k, k]:.0f}%" if LANG == 0 else
                              f"{100*frac[k]:.1f}% of time\nstays {100*P.loc[k, k]:.0f}%"),
                ha="center", va="center", fontsize=7.5, color=cor_t)

    def agrupa(v):
        if v[:3] in ("C08", "C09", "C10"):
            return ("vapor de água mais frio (C08 a C10)" if LANG == 0 else "colder water vapour (C08–C10)")
        if v[:3] in ("C11", "C12", "C13", "C14"):
            loc = "na área central" if "nucleo" in v else "na região"
            return (f"topos de nuvem mais frios {loc} (C11 a C14)" if LANG == 0 else
                    f"colder cloud tops {'over the central area' if 'nucleo' in v else 'in the region'} (C11–C14)")
        if v.startswith("RRQPE"):
            return "RRQPE"
        if v.startswith("in_c1"):
            return ("chuva no anel de 13 a 60 km" if LANG == 0 else "rain in the 13–60 km ring")
        if v.startswith("in_c2"):
            return ("chuva no anel de 60 a 100 km" if LANG == 0 else "rain in the 60–100 km ring")
        if v.startswith("em_umidity"):
            return ("umidade menor" if LANG == 0 else "lower humidity")
        if v.startswith("em_temperature"):
            return ("temperatura maior" if LANG == 0 else "higher temperature")
        return v

    def fatores(contraste):
        if F is None:
            return []
        g = F[(F.horizonte == h) & (F.contraste == contraste)].sort_values("ordem")
        vistos, out = set(), []
        for v in g.itertuples():
            n = agrupa(v.variavel)
            if n not in vistos:
                vistos.add(n); out.append(f"{n} (AUC {v.auc_univariada:.2f})")
            if len(out) == 2:
                break
        return out

    import textwrap
    setas = [("S", "C", -0.25, "seco -> fraca/moderada  x  seco -> seco"),
             ("C", "S", -0.25, "fraca/moderada -> seco  x  fraca/moderada persiste"),
             ("S", "F", -0.3, "seco -> forte  x  seco -> seco"),
             ("F", "S", -0.3, None),
             ("C", "F", 0.3, "fraca/moderada -> forte  x  fraca/moderada persiste"),
             ("F", "C", 0.3, None)]
    for a, b, rad, contraste in setas:
        (x0, y0), (x1, y1) = pos[a], pos[b]
        p = P.loc[a, b]
        arr = FancyArrowPatch((x0, y0), (x1, y1), connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                              mutation_scale=14, lw=0.8 + 18 * p, shrinkA=55, shrinkB=55,
                              color="C3" if b == "F" or (a == "S" and b == "C") else "C0", alpha=0.85)
        ax.add_patch(arr)
        # ponto médio da curva quadrática usada pelo arc3
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        cx, cy = xm + rad * dy, ym - rad * dx
        bx, by = 0.25 * x0 + 0.5 * cx + 0.25 * x1, 0.25 * y0 + 0.5 * cy + 0.25 * y1
        texto = f"{100*p:.1f}%"
        if contraste:
            texto += "\n" + "\n".join(textwrap.fill(f, 40) for f in fatores(contraste))
        ax.text(bx, by, texto, ha="center", va="center", fontsize=7.5,
                bbox=dict(fc="#fff7e6" if contraste else "w", ec="#d98c00" if contraste else "0.7", lw=0.7, pad=3))
    ax.set(xlim=(-2.2, 11.2), ylim=(-2.6, 7.2)); ax.axis("off")
    ax.set_title(("Transições da área central em 60 min, com a probabilidade e as variáveis mais associadas a cada uma"
                  if LANG == 0 else "Central-area transitions within 60 min, with probability and most associated variables"),
                 fontsize=10)
    salvar("fig14_fluxograma_de_estados", fig, P.reset_index())


# ═══════════════════════════════ fig15 comparação de modelos ═════════════════
def fig15():
    M = ler(ROOT / "modelos_tese" / "B1_metricas_continuas.csv")
    if M is None:
        return
    M = M[~M.modelo.str.contains(r"\[só")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    ordem = list(M.groupby("modelo").AP.mean().sort_values().index)
    for a, h in zip(ax, (30, 60)):
        g = M[M.horizonte == h]
        for k, (conj, cor) in enumerate((("sem chuva", "C0"), ("anéis", "C2"), ("anéis e área central", "C3"))):
            q = g[g.entradas == conj].set_index("modelo").reindex(ordem)
            yy = np.arange(len(ordem)) + (k - 1) * 0.27
            a.barh(yy, q.AP, xerr=q.AP_dp, height=0.27, color=cor, capsize=1.5, label=conj)
        a.set_yticks(np.arange(len(ordem))); a.set_yticklabels(ordem, fontsize=7)
        a.set(xlabel=t("ap"), title=t(f"h{h}"))
    ax[1].legend(fontsize=7)
    salvar("fig15_comparacao_de_modelos", fig, M)


# ═══════════════════════════════ execução ════════════════════════════════════
FIGS = dict(fig01=fig01, fig02=fig02, fig03=fig03, fig04=fig04, fig05=fig05, fig06=fig06, fig07=fig07,
            fig08=fig08, fig09=fig09, fig10=fig10, fig11=fig11, fig12=fig12, fig13=fig13,
            fig14=fig14, fig15=fig15)


def main():
    global LANG
    OUT.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    r = logging.getLogger(); r.handlers.clear()
    for hd in (logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "figuras_tese.log", mode="w")):
        hd.setFormatter(fmt); r.addHandler(hd)
    r.setLevel(logging.INFO)
    args = sys.argv[1:]
    extras = []
    if "--evento" in args:
        k = args.index("--evento"); extras = [args[k + 1]]; args = args[:k] + args[k + 2:]
    pedidas = args or list(FIGS)
    (OUT / "pt" / "eventos").mkdir(parents=True, exist_ok=True); (OUT / "en" / "eventos").mkdir(parents=True, exist_ok=True)
    for LANG in (0, 1):
        log.info(f"idioma: {'português' if LANG == 0 else 'inglês'}")
        for nome in pedidas:
            try:
                FIGS[nome](extras) if nome == "fig08" else FIGS[nome]()
            except Exception as e:
                log.exception(f"  {nome} falhou: {e}")
    log.info(f"figuras em {OUT}")


if __name__ == "__main__":
    main()
