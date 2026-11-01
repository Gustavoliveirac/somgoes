"""
motor_entrada.py — monta a entrada do motor para um horário a partir dos dados brutos,
com as MESMAS funções do prep_somgoes.py (filtro de báscula, conversão para a grade de
10 min, acumulados, colares e quadrantes).

Hoje lê os CSVs das pastas de estações e o cubo GOES do horário. Na plataforma hidromet,
troque `serie_recente` por uma consulta ao banco que devolva a mesma série (índice de
data em BRT e colunas rainfall e variáveis meteorológicas) e `cubo_goes` pela leitura do
buffer de satélite. O restante não muda.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from core import config_somgoes as cfg
from core import prep_somgoes as pp

JANELA_H = 3        # horas de histórico lidas por estação
_IDX = None


def _meta():
    m = pd.read_csv(cfg.META_FILE, sep=";")
    m["Codigo"] = m.Codigo.astype(str).str.strip()
    m["dx"], m["dy"], m["dist_km"], m["azim"] = pp.geo(m.Latitude, m.Longitude)
    m["dez"] = m.Frequencia_norm.astype(str).str.contains("10")
    exc = cfg.CACHE_DIR / "estacoes_excluidas.csv"
    excl = set(pd.read_csv(exc, sep=";").query("excluida").codigo.astype(str)) if exc.exists() else set()
    return m, excl


def serie_recente(row, ini, fim, cols):
    """série bruta da estação entre ini e fim (substituir por consulta ao banco)"""
    global _IDX
    if _IDX is None:
        _IDX = pp.indexar(_meta()[0].Codigo)
    s = pp.ler(row, _IDX, cols)
    if s is None:
        return None
    return s[(s.index > ini - pd.Timedelta(minutes=10)) & (s.index <= fim)]


def cubo_goes(ts):
    """cubo (8,64,64) do horário (ou até 30 min antes); stem em UTC como no preProcess.py"""
    for atraso in range(0, 40, 10):
        t = ts - pd.Timedelta(minutes=atraso) + pd.Timedelta(hours=3)
        for p in sorted(cfg.GOES_DIR.glob(f"{t:%Y%m%d_%H}{t.minute // 10}*.npy")):
            try:
                x = np.load(p).astype(np.float32)
                if x.shape == (8, cfg.IMG_SIZE, cfg.IMG_SIZE):
                    return x[cfg.GOES_CANAIS], atraso
            except Exception:
                pass
    return None, None


def montar_ao_vivo(ts):
    ts = pd.Timestamp(ts).floor("10min")
    ini = ts - pd.Timedelta(hours=JANELA_H)
    m, excl = _meta()
    meta_c = json.load(open(cfg.CACHE_DIR / "meta_cache.json"))
    R0 = meta_c["raio_nucleo_km"]
    V = meta_c["variaveis"]
    grade = pd.date_range(ini, ts, freq="10min")

    # ── EMs na mesma ordem do treino ─────────────────────────────────────────
    E = pd.read_csv(cfg.CACHE_DIR / "estacoes_em.csv", sep=";")
    E["codigo"] = E.codigo.astype(str)
    e10, e1h, ida = [], [], []
    for r in E.itertuples():
        row = m[m.Codigo == r.codigo].iloc[0]
        s = serie_recente(row, ini, ts, cfg.MET_VARS + ["wind_dir"])
        if s is None or s.empty:
            s = pd.DataFrame(columns=cfg.MET_VARS + ["wind_dir"], dtype=float)
        for v, (lo, hi) in cfg.BOUNDS.items():
            s[v] = s[v].where((s[v] >= lo) & (s[v] <= hi)) if v in s else np.nan
        rad = np.radians(s["wind_dir"]); s["vento_sin"], s["vento_cos"] = np.sin(rad), np.cos(rad)
        s = s[V]
        if r.tipo == "10min":
            s.index = s.index.round("10min")
            e10.append(s[~s.index.duplicated()].reindex(grade[-cfg.N_QUADROS:]).to_numpy(np.float32))
        else:
            s.index = s.index.round("1h"); s = s[~s.index.duplicated()]
            s = s[s.index <= ts]
            ok = len(s) and (ts - s.index[-1]) <= pd.Timedelta(minutes=50)
            e1h.append(s.iloc[-1].to_numpy(np.float32) if ok else np.full(len(V), np.nan, np.float32))
            ida.append((ts - s.index[-1]).total_seconds() / 60 if ok else np.nan)
    em10 = np.stack(e10, 0) if e10 else np.zeros((0, cfg.N_QUADROS, len(V)), np.float32)
    em1h = np.stack(e1h, 0) if e1h else np.zeros((0, len(V)), np.float32)

    # ── chuva (núcleo e colares) ─────────────────────────────────────────────
    chuva10, horaria = {}, {}
    for row in m.itertuples():
        if row.Codigo in excl or row.dist_km > 100:
            continue
        s = serie_recente(row, ini, ts, ["rainfall"])
        if s is None or "rainfall" not in s or s.empty:
            continue
        rr = s["rainfall"].where((s["rainfall"] >= 0) & (s["rainfall"] <= 500))
        if row.dez:
            chuva10[row.Codigo] = pp.para_grade(pp.nativa_filtrada(rr), grade)[-6:]
        else:
            rr.index = rr.index.round("1h"); rr = rr[rr.index <= ts]
            if len(rr) and (ts - rr.index[-1]) <= pd.Timedelta(minutes=50):
                v = float(rr.iloc[-1]); horaria[row.Codigo] = v if v >= cfg.HORARIA_MIN else 0.0
    dist = m.set_index("Codigo")
    cods = list(chuva10)
    R10 = np.column_stack([chuva10[c] for c in cods]) if cods else np.full((6, 0), np.nan)
    A60 = pp.acum(R10, 6)[-1] if cods else np.array([])
    A30 = pp.acum(R10, 3)[-1] if cods else np.array([])
    ult = R10[-1] if cods else np.array([])
    nuc = [i for i, c in enumerate(cods) if dist.loc[c, "dist_km"] <= R0]
    mx = lambda x, k=cfg.MIN_EST_NUCLEO: pp.max_nucleo(np.asarray(x, np.float32)[None], k)[0][0]
    An = A60[nuc]
    with np.errstate(all="ignore"):
        ch = {"in_nuc_max10": mx(ult[nuc]), "in_nuc_max30": mx(A30[nuc]), "in_nuc_max60": mx(An),
              "in_nuc_media60": float(np.nanmean(An)) if len(An) else np.nan,
              "in_nuc_frac60": float(np.nanmean(An[np.isfinite(An)] >= cfg.BASCULA)) if np.isfinite(An).any() else np.nan,
              "in_nuc_fracforte60": float(np.nanmean(An[np.isfinite(An)] >= cfg.LIMITES[2])) if np.isfinite(An).any() else np.nan}
    acc = {c: A60[i] for i, c in enumerate(cods)} | horaria
    u10 = {c: ult[i] for i, c in enumerate(cods)}
    for nome, (r1, r2) in {"c1": (R0, 60), "c2": (60, 100)}.items():
        for q, (a0, a1) in {"tot": (0, 360), "N": (315, 45), "L": (45, 135), "S": (135, 225), "O": (225, 315)}.items():
            sel = [c for c in acc if r1 < dist.loc[c, "dist_km"] <= r2 and
                   ((a0 <= dist.loc[c, "azim"] < a1) if a0 < a1 else (dist.loc[c, "azim"] >= a0 or dist.loc[c, "azim"] < a1))]
            v = np.array([acc[c] for c in sel], float); v = v[np.isfinite(v)]
            ch[f"in_{nome}_{q}_max60"] = float(v.max()) if len(v) else np.nan
            ch[f"in_{nome}_{q}_frac60"] = float((v >= cfg.BASCULA).mean()) if len(v) else np.nan
            w = np.array([u10[c] for c in sel if c in u10], float); w = w[np.isfinite(w)]
            ch[f"in_{nome}_{q}_max10"] = float(w.max()) if len(w) else np.nan

    # ── GOES ─────────────────────────────────────────────────────────────────
    x, atraso = cubo_goes(ts)
    C, R = len(cfg.GOES_CANAIS), cfg.GOES_RES
    interp = dict(ch)
    if x is not None:
        f = cfg.IMG_SIZE // R
        x[~np.isfinite(x)] = np.nan
        goes = np.nan_to_num(np.nanmean(x.reshape(C, R, f, R, f), axis=(2, 4)), nan=0.0).astype(np.float32)
        yy, xx = np.mgrid[0:cfg.IMG_SIZE, 0:cfg.IMG_SIZE]
        disco = np.hypot(xx + .5 - cfg.IMG_SIZE / 2, yy + .5 - cfg.IMG_SIZE / 2) * cfg.KM_PER_PX <= 13
        for c in range(C):
            interp[f"g{cfg.GOES_CANAIS[c]}_nucleo_min"] = float(np.nanmin(x[c][disco]))
            interp[f"g{cfg.GOES_CANAIS[c]}_nucleo_max"] = float(np.nanmax(x[c][disco]))
    else:
        goes = np.zeros((C, R, R), np.float32)
    tudo = np.concatenate([em10[:, -1], em1h], 0) if len(em10) or len(em1h) else np.zeros((0, len(V)))
    with np.errstate(all="ignore"):
        for k, v in enumerate(V):
            interp[f"em_{v}"] = float(np.nanmean(tudo[:, k])) if len(tudo) else np.nan
    interp["goes_atraso_min"] = atraso
    return dict(em10=em10, em1h=em1h, idade=np.array(ida, np.float32), goes=goes, chuva=ch, interp=interp)
