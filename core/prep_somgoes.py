#!/usr/bin/env python
"""
prep_somgoes.py — monta o cache do somgoes a partir das séries brutas e do GOES.

Saídas em cache_somgoes/
  amostras.parquet   1 linha por horário de 10 min (out-mar), com alvo (estado agora
                     e próxima hora no núcleo, transição, estado básico), temporada,
                     validade do GOES, estatísticas do GOES, chuva nos colares
                     (reservada para a segunda etapa) e colunas de horário
  em10.npy           (N, S10, V) EMs de 10 min no horário (NaN = falta)
  em1h.npy           (N, S1h, V) EMs horárias, última leitura até o horário
  em1h_idade.npy     (N, S1h) minutos desde essa leitura
  nucleo_agora.npy / nucleo_prox.npy  (N, S) acumulado 60 min por estação do núcleo
  goes_campo.npy     (N, C, R, R) float16, campo GOES reduzido
  estacoes_em.csv, estacoes_nucleo.csv, estacoes_excluidas.csv, meta_cache.json, prep.log

Uso:  python -m core.prep_somgoes              (tudo)
      python -m core.prep_somgoes --so-chuva   (refaz chuva e alvos, reaproveita o GOES já processado)
"""
from __future__ import annotations

import json
import logging
import sys
import warnings

import numpy as np
import pandas as pd

from core import config_somgoes as cfg

warnings.filterwarnings("ignore")
log = logging.getLogger("prep")


def setup_log():
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(cfg.CACHE_DIR / "prep.log", mode="w")):
        h.setFormatter(fmt); log.addHandler(h)
    log.setLevel(logging.INFO)


def geo(lat, lon):
    R = 6371.0
    dy = (np.asarray(lat, float) - cfg.BH_LAT) * np.pi / 180 * R
    dx = (np.asarray(lon, float) - cfg.BH_LON) * np.pi / 180 * R * np.cos(np.radians(cfg.BH_LAT))
    return dx, dy, np.hypot(dx, dy), (np.degrees(np.arctan2(dx, dy)) + 360) % 360


# ═══════════════════════════════ leitura das séries ══════════════════════════
def indexar(codigos):
    alvo, idx = {f"{c}.csv" for c in codigos}, {}
    for raiz in cfg.ESTACOES_DIR:
        if raiz.exists():
            for f in raiz.rglob("*.csv"):
                if f.name in alvo:
                    idx.setdefault(f.stem, []).append(f)
    return idx


def ler(row, idx, cols):
    c = idx.get(str(row.Codigo), [])
    if len(c) > 1:
        pasta = cfg.FONTE_DIR.get(row.Fonte, row.Fonte).upper()
        c = [p for p in c if p.parent.name.upper() == pasta] or c
    if not c:
        return None
    try:
        s = pd.read_csv(c[0], sep=";", usecols=lambda x: x in ["date"] + cols)
    except Exception:
        return None
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s = s.dropna(subset=["date"]).drop_duplicates("date").set_index("date").sort_index()
    for v in cols:
        if v in s:
            s[v] = pd.to_numeric(s[v], errors="coerce")
    return s


def nativa_filtrada(s):
    s = s.dropna().sort_index()
    w = s >= cfg.BASCULA - 1e-6
    t = s.index.to_series()
    lim = pd.Timedelta(minutes=cfg.VIZINHO_MAX_MIN)
    viz = (w.shift(1, fill_value=False) & ((t - t.shift(1)) <= lim)) | \
          (w.shift(-1, fill_value=False) & ((t.shift(-1) - t) <= lim))
    ok = (w & viz) | (s >= cfg.ISOLADA_MIN - 1e-6)
    return s.where(ok | ~w, 0.0)


def para_grade(s, grade):
    s = s.dropna().sort_index()
    if len(s) < 2:
        return np.full(len(grade), np.nan, np.float32)
    tr = s.index.values.astype("datetime64[ns]").astype(np.int64)
    C = np.cumsum(s.to_numpy(np.float64))
    tr = np.r_[tr[0] - np.median(np.diff(tr)), tr]; C = np.r_[0.0, C]
    tg = grade.values.astype("datetime64[ns]").astype(np.int64)
    dg = tg[1] - tg[0]
    inc = np.diff(np.r_[np.nan, np.interp(tg, tr, C, left=np.nan, right=np.nan)])
    gmax = cfg.GAP_MAX_MIN * 60e9
    for tt in (tg, tg - dg):
        j = np.clip(np.searchsorted(tr, tt, side="left"), 1, len(tr) - 1)
        inc[(tr[j] - tr[j - 1]) > gmax] = np.nan
    inc[(tg - dg < tr[0]) | (tg > tr[-1])] = np.nan
    inc[(inc < 0) & (inc > -1e-6)] = 0.0
    return inc.astype(np.float32)


def acum(M, passos=6):
    """M (T,S) chuva de 10 min -> acumulado nos últimos `passos` intervalos;
    válido se completo ou se já passa do limiar de forte proporcional."""
    df = pd.DataFrame(M)
    soma = df.rolling(passos, min_periods=1).sum()
    n = df.notna().rolling(passos, min_periods=1).sum()
    return soma.where((n == passos) | (soma >= cfg.LIMITES[2] * passos / 6)).to_numpy(np.float32)


def acum60(M):
    return acum(M, 6)


def max_nucleo(X, minimo=cfg.MIN_EST_NUCLEO):
    n = np.isfinite(X).sum(1)
    v = np.nanmax(np.where(np.isfinite(X), X, -1), 1)
    v = np.where((n >= minimo) & (v >= 0), v, np.nan)
    return v.astype(np.float32), n


def estado(x):
    e = np.digitize(np.nan_to_num(x, nan=-1), cfg.LIMITES).astype(float)
    e[np.isnan(x)] = np.nan
    return e


def transicao(a, p):
    t = np.full(len(a), -1, np.int64)
    ok = ~np.isnan(a) & ~np.isnan(p)
    a, p = np.nan_to_num(a, nan=-9), np.nan_to_num(p, nan=-9)
    seco_a, leve_a, forte_a = a == 0, (a == 1) | (a == 2), a >= 3
    seco_p, leve_p, forte_p = p == 0, (p == 1) | (p == 2), p >= 3
    t[seco_a & seco_p] = 0; t[seco_a & leve_p] = 1; t[seco_a & forte_p] = 2
    t[leve_a & forte_p] = 3; t[leve_a & leve_p] = 4
    t[(leve_a | forte_a) & seco_p] = 5
    t[forte_a & forte_p] = 6; t[forte_a & leve_p] = 7
    t[~ok] = -1
    b = np.full(len(a), -1, np.int64)
    b[ok & (a == 0) & (p == 0)] = 0; b[ok & (a == 0) & (p > 0)] = 1
    b[ok & (a > 0) & (p == 0)] = 2; b[ok & (a > 0) & (p > 0)] = 3
    return t, b


# ═══════════════════════════════ GOES ════════════════════════════════════════
def campo_goes(slots):
    C, R = len(cfg.GOES_CANAIS), cfg.GOES_RES
    f = cfg.IMG_SIZE // R
    arqs = {}
    for p in cfg.GOES_DIR.glob("*.npy"):
        try:
            ts = pd.Timestamp(pd.to_datetime(p.stem, format="%Y%m%d_%H%M%S")) - pd.Timedelta(hours=3)
        except Exception:
            continue
        arqs.setdefault(ts.round("10min"), p)
    pos = {t: i for i, t in enumerate(slots)}
    mm = np.lib.format.open_memmap(cfg.CACHE_DIR / "goes_campo.npy", mode="w+",
                                   dtype=np.float16, shape=(len(slots), C, R, R))
    yy, xx = np.mgrid[0:cfg.IMG_SIZE, 0:cfg.IMG_SIZE]
    disco = np.hypot(xx + .5 - cfg.IMG_SIZE / 2, yy + .5 - cfg.IMG_SIZE / 2) * cfg.KM_PER_PX <= 13
    cols = [f"g{c}_{s}" for c in cfg.GOES_CANAIS for s in ("media", "desvio", "min", "max", "p10", "nucleo_min", "nucleo_max")]
    stats = np.full((len(slots), len(cols)), np.nan, np.float32)
    ok = np.zeros(len(slots), bool)
    n_ruim = 0
    for k, (t, p) in enumerate(arqs.items()):
        i = pos.get(t)
        if i is None:
            continue
        try:
            x = np.load(p).astype(np.float32)
        except Exception:
            n_ruim += 1; continue
        if x.shape != (8, cfg.IMG_SIZE, cfg.IMG_SIZE):
            n_ruim += 1; continue
        x = x[cfg.GOES_CANAIS]; x[~np.isfinite(x)] = np.nan
        ir = x[:7]
        if (np.isfinite(ir).mean(axis=(1, 2)) < 0.9).any() or (np.nanstd(ir, axis=(1, 2)) < 1e-3).any():
            n_ruim += 1; continue
        ok[i] = True
        red = np.nanmean(x.reshape(C, R, f, R, f), axis=(2, 4))
        mm[i] = np.nan_to_num(red, nan=0.0).astype(np.float16)
        fl, cen = x.reshape(C, -1), x[:, disco]
        stats[i] = np.stack([np.nanmean(fl, 1), np.nanstd(fl, 1), np.nanmin(fl, 1), np.nanmax(fl, 1),
                             np.nanpercentile(fl, 10, 1), np.nanmin(cen, 1), np.nanmax(cen, 1)], 1).reshape(-1)
        if (k + 1) % 20000 == 0:
            log.info(f"  GOES {k+1:,}/{len(arqs):,}")
    mm.flush()
    log.info(f"GOES: {ok.sum():,}/{len(slots):,} horários válidos ({ok.mean():.1%}); "
             f"{n_ruim} arquivos descartados por conteúdo")
    return ok, pd.DataFrame(stats, columns=cols)


# ═══════════════════════════════ principal ═══════════════════════════════════
SO_CHUVA = "--so-chuva" in sys.argv


def main():
    setup_log()
    if SO_CHUVA:
        log.info("modo --so-chuva: refaz chuva, alvos e EMs; reaproveita o GOES do cache")
    m = pd.read_csv(cfg.META_FILE, sep=";")
    m["Codigo"] = m["Codigo"].astype(str).str.strip()
    m["dx"], m["dy"], m["dist_km"], m["azim"] = geo(m["Latitude"], m["Longitude"])
    m["dez"] = m["Frequencia_norm"].astype(str).str.contains("10")
    idx = indexar(m.Codigo)
    log.info(f"séries encontradas: {len(idx)}/{len(m)}")
    if not idx:
        log.info("ERRO: nenhuma série encontrada; ajuste ESTACOES_DIR"); sys.exit(1)

    sud = m[(m.Fonte == "SUDECAP") & (m.Tipo == "pluviometro")]
    R0 = float(np.ceil(sud.dist_km.max() * 10) / 10)
    log.info(f"NÚCLEO: raio {R0:.1f} km (pluviômetro SUDECAP mais afastado)")

    g10 = pd.date_range(cfg.INICIO, cfg.FIM, freq="10min")
    sel = g10.month.isin(cfg.MESES)
    slots = g10[sel]
    log.info(f"horários out-mar: {len(slots):,}")

    # ── chuva ────────────────────────────────────────────────────────────────
    chuva, horarias, diag = {}, {}, []
    for row in m.itertuples():
        s = ler(row, idx, ["rainfall"])
        if s is None or "rainfall" not in s:
            continue
        r = s["rainfall"].where((s["rainfall"] >= 0) & (s["rainfall"] <= 500))
        if row.dez:
            v = para_grade(nativa_filtrada(r), g10)
            chuva[row.Codigo] = v
            vv = v[sel]
            diag.append((row.Codigo, np.isfinite(vv).mean(), np.nansum(vv > 0) / max(1, np.isfinite(vv).sum())))
        else:
            r.index = r.index.round("1h")
            h = r[~r.index.duplicated()].reindex(g10)
            h = h.where((h >= cfg.HORARIA_MIN) | h.isna(), 0.0)
            horarias[row.Codigo] = h.to_numpy(np.float32)
            hv = h[sel]
            diag.append((row.Codigo, hv.notna().sum() / max(1, sel.sum() / 6), np.nan))
    D = pd.DataFrame(diag, columns=["codigo", "cobertura", "frac_chuva"])
    med = D.frac_chuva.median()
    D["excluida"] = (D.cobertura < cfg.COB_MIN_ESTACAO) | (D.frac_chuva > cfg.CHUVA_MAX_REL * med)
    D.to_csv(cfg.CACHE_DIR / "estacoes_excluidas.csv", sep=";", index=False)
    excl = set(D[D.excluida].codigo)
    log.info(f"chuva: {len(chuva)} estações de 10 min e {len(horarias)} horárias; "
             f"{len(excl)} excluídas (cobertura < {cfg.COB_MIN_ESTACAO:.0%} ou chuva > "
             f"{cfg.CHUVA_MAX_REL:.0f}x a mediana)")

    cods10 = [c for c in chuva if c not in excl]
    A = acum60(np.column_stack([chuva[c] for c in cods10]))
    dist = m.set_index("Codigo")
    nuc = [i for i, c in enumerate(cods10) if dist.loc[c, "dist_km"] <= R0]
    An = A[:, nuc]
    agora, nval = max_nucleo(An)
    prox = np.r_[agora[6:], np.full(6, np.nan)]
    ea, ep = estado(agora), estado(prox)
    tr, bas = transicao(ea, ep)
    ent = {}
    # horizonte de 30 min: acumulado de 30 min convertido em taxa horária (x2)
    R10 = np.column_stack([chuva[c] for c in cods10])
    A30 = acum(R10, 3)
    agora30 = max_nucleo(A30[:, nuc])[0] * 2
    prox30 = np.r_[agora30[3:], np.full(3, np.nan)]
    ea30, ep30 = estado(agora30), estado(prox30)
    tr30, bas30 = transicao(ea30, ep30)
    # chuva no núcleo como ENTRADA (cenário iii)
    with np.errstate(all="ignore"):
        ent |= {"in_nuc_max10": max_nucleo(R10[:, nuc])[0], "in_nuc_max30": max_nucleo(A30[:, nuc])[0],
               "in_nuc_max60": agora, "in_nuc_media60": np.nanmean(An, 1),
               "in_nuc_frac60": np.nanmean(np.where(np.isfinite(An), An >= cfg.BASCULA, np.nan), 1),
               "in_nuc_fracforte60": np.nanmean(np.where(np.isfinite(An), An >= cfg.LIMITES[2], np.nan), 1)}
    pd.DataFrame({"codigo": [cods10[i] for i in nuc],
                  "dist_km": [dist.loc[cods10[i], "dist_km"] for i in nuc],
                  "fonte": [dist.loc[cods10[i], "Fonte"] for i in nuc]})\
        .to_csv(cfg.CACHE_DIR / "estacoes_nucleo.csv", sep=";", index=False)
    np.save(cfg.CACHE_DIR / "nucleo_agora.npy", An[sel].astype(np.float16))
    np.save(cfg.CACHE_DIR / "nucleo_prox.npy", np.r_[An[6:], np.full((6, An.shape[1]), np.nan)][sel].astype(np.float16))
    log.info(f"núcleo: {len(nuc)} estações de 10 min")

    # colares (segunda etapa): 13-60 e 60-100 km, total e por quadrante
    Ah = pd.DataFrame({c: horarias[c] for c in horarias if c not in excl}, index=g10).ffill(limit=5)
    Afull = pd.concat([pd.DataFrame(A, index=g10, columns=cods10), Ah], axis=1)
    col = {}
    for nome, (r1, r2) in {"colar1": (R0, 60), "colar2": (60, 100)}.items():
        for q, (a0, a1) in {"": (0, 360), "_N": (315, 45), "_L": (45, 135), "_S": (135, 225), "_O": (225, 315)}.items():
            mz = dist.reindex(Afull.columns)
            az = mz.azim
            dentro = (mz.dist_km > r1) & (mz.dist_km <= r2) & \
                     ((az >= a0) & (az < a1) if a0 < a1 else (az >= a0) | (az < a1))
            X = Afull.loc[:, dentro.values]
            col[f"{nome}{q}_max"] = X.max(1).to_numpy(np.float32)[sel] if X.shape[1] else np.nan
            ab = "c1" if nome == "colar1" else "c2"
            q_ = q.strip("_") or "tot"
            ent[f"in_{ab}_{q_}_max60"] = X.max(1).to_numpy(np.float32) if X.shape[1] else np.full(len(g10), np.nan)
            with np.errstate(all="ignore"):
                ent[f"in_{ab}_{q_}_frac60"] = (X >= cfg.BASCULA).sum(1).to_numpy(np.float32) / \
                    X.notna().sum(1).replace(0, np.nan).to_numpy(np.float32) if X.shape[1] else np.full(len(g10), np.nan)
            c10 = [i for i, c in enumerate(cods10) if c in set(X.columns)]
            ent[f"in_{ab}_{q_}_max10"] = max_nucleo(R10[:, c10], 1)[0] if c10 else np.full(len(g10), np.nan)

    # ── EMs ──────────────────────────────────────────────────────────────────
    em = m[(m.Tipo == "meteorologica") & (m.dist_km <= 150)].copy()
    em["Altitude"] = em["Altitude"].fillna(em["Altitude"].median())
    V = cfg.MET_VARS + ["vento_sin", "vento_cos"]

    def variaveis(row):
        s = ler(row, idx, cfg.MET_VARS + ["wind_dir"])
        if s is None:
            return None
        for v, (lo, hi) in cfg.BOUNDS.items():
            if v in s:
                s[v] = s[v].where((s[v] >= lo) & (s[v] <= hi))
            else:
                s[v] = np.nan
        rad = np.radians(s["wind_dir"])
        s["vento_sin"], s["vento_cos"] = np.sin(rad), np.cos(rad)
        return s[V]

    e10, e1h, id1h, info = [], [], [], []
    for row in em.itertuples():
        s = variaveis(row)
        if s is None:
            continue
        if row.dez:
            s.index = s.index.round("10min")
            x = s[~s.index.duplicated()].reindex(slots).to_numpy(np.float32)
            e10.append(x); tipo = "10min"
        else:
            s.index = s.index.round("1h")
            s = s[~s.index.duplicated()]
            ult = pd.Series(s.index, index=s.index).reindex(g10, method="ffill", limit=5)
            x = s.reindex(g10, method="ffill", limit=5)[sel].to_numpy(np.float32)
            idade = ((g10 - pd.DatetimeIndex(ult.values)).total_seconds() / 60)[sel]
            e1h.append(x); id1h.append(np.asarray(idade, np.float32)); tipo = "1h"
        cob = np.isfinite(x).mean(0)
        info.append(dict(codigo=row.Codigo, tipo=tipo, dx_km=round(row.dx, 2), dy_km=round(row.dy, 2),
                         dist_km=round(row.dist_km, 1), altitude=row.Altitude,
                         **{f"cob_{v}": round(float(c), 3) for v, c in zip(V, cob)}))
    I = pd.DataFrame(info)
    I.to_csv(cfg.CACHE_DIR / "estacoes_em.csv", sep=";", index=False)
    np.save(cfg.CACHE_DIR / "em10.npy", np.stack(e10, 1) if e10 else np.zeros((len(slots), 0, len(V)), np.float32))
    np.save(cfg.CACHE_DIR / "em1h.npy", np.stack(e1h, 1) if e1h else np.zeros((len(slots), 0, len(V)), np.float32))
    np.save(cfg.CACHE_DIR / "em1h_idade.npy", np.stack(id1h, 1) if id1h else np.zeros((len(slots), 0), np.float32))
    log.info(f"EMs: {len(e10)} de 10 min e {len(e1h)} horárias (até 150 km)")

    # ── GOES ─────────────────────────────────────────────────────────────────
    antigo = cfg.CACHE_DIR / "amostras.parquet"
    if SO_CHUVA and antigo.exists() and (cfg.CACHE_DIR / "goes_campo.npy").exists():
        old = pd.read_parquet(antigo).set_index("ts").reindex(slots)
        gcols = json.load(open(cfg.CACHE_DIR / "meta_cache.json"))["goes_stat_cols"]
        goes_ok = old["goes_ok"].fillna(False).to_numpy(bool)
        gst = old[gcols].reset_index(drop=True)
        log.info(f"GOES reaproveitado do cache existente ({goes_ok.sum():,} horários válidos)")
    else:
        goes_ok, gst = campo_goes(slots)

    # ── tabela de amostras ───────────────────────────────────────────────────
    S = pd.DataFrame({"ts": slots, "agora": agora[sel], "prox": prox[sel],
                      "estado_agora": ea[sel], "estado_prox": ep[sel],
                      "transicao": tr[sel], "basico": bas[sel], "n_est_nucleo": nval[sel],
                      "goes_ok": goes_ok,
                      "agora_30": agora30[sel], "prox_30": prox30[sel], "estado_agora_30": ea30[sel],
                      "estado_prox_30": ep30[sel], "transicao_30": tr30[sel], "basico_30": bas30[sel]})
    E = pd.DataFrame({k: np.asarray(v)[sel] for k, v in ent.items()})
    S = pd.concat([S, gst, pd.DataFrame(col), E], axis=1)
    S["temporada"] = np.where(S.ts.dt.month >= 10, S.ts.dt.year, S.ts.dt.year - 1)
    S["temporada"] = S.temporada.astype(str) + "/" + (S.temporada + 1).astype(str)
    temps = sorted(S.temporada.unique())
    S["fold"] = S.temporada.map({t: i for i, t in enumerate(temps)})
    S["hora_sin"] = np.sin(2 * np.pi * S.ts.dt.hour / 24); S["hora_cos"] = np.cos(2 * np.pi * S.ts.dt.hour / 24)
    S["alvo_ok"] = S.transicao >= 0
    S["alvo_ok_30"] = S.transicao_30 >= 0
    S.to_parquet(cfg.CACHE_DIR / "amostras.parquet", index=False)

    v = S[S.alvo_ok & S.goes_ok]
    log.info(f"amostras com alvo e GOES: {len(v):,}")
    log.info("transições (instantes): " + str({cfg.TRANSICOES[k]: int(n) for k, n in
                                               v.transicao.value_counts().sort_index().items()}))
    log.info("por temporada: " + str(v.groupby("temporada").size().to_dict()))
    v30 = S[S.alvo_ok_30 & S.goes_ok]
    log.info("transições 30 min (instantes): " + str({cfg.TRANSICOES[k]: int(n) for k, n in
                                                      v30.transicao_30.value_counts().sort_index().items()}))
    log.info(f"entradas de chuva: {E.shape[1]} colunas (núcleo {sum(c.startswith('in_nuc') for c in E)}, "
             f"colares {sum(c.startswith(('in_c1','in_c2')) for c in E)})")
    json.dump(dict(raio_nucleo_km=R0, temporadas=temps, variaveis=V, n_slots=len(slots),
                   goes_res=cfg.GOES_RES, canais=cfg.GOES_CANAIS, goes_stat_cols=list(gst.columns),
                   colar_cols=list(col), entrada_nucleo=[c for c in E if c.startswith("in_nuc")],
                   entrada_colar=[c for c in E if c.startswith(("in_c1", "in_c2"))]), open(cfg.CACHE_DIR / "meta_cache.json", "w"), indent=1)
    log.info(f"cache pronto em {cfg.CACHE_DIR}")


if __name__ == "__main__":
    main()
