#!/usr/bin/env python
"""
train_somgoes.py — treina, avalia e consolida as alternativas do somgoes.

Ordem padrão (sem argumentos roda tudo, retomável):
  1. kmeans        referência, features planas, nas grades de cfg.GRADES
  2. som_classico  SOM online com features planas, baseline e supervisionado
  3. etapa1        modelo profundo, GOES (nenhum, t0, seq_cnn, convlstm) x perda,
                   grade 10x10, AE; + VAE na perda completa
  4. etapa2        varredura de grades com a melhor combinação da etapa 1

Uso
  python -m core.train_somgoes                 tudo
  python -m core.train_somgoes --rapido        conferência rápida (sem folds)
  python -m core.train_somgoes --familias etapa1 --folds 0 1
  python -m core.train_somgoes --so-zip        só consolida e zipa
Envie de volta: pacote_somgoes.zip
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans

from core import config_somgoes as cfg
from core import somgoes_nucleo as nu

log = logging.getLogger("treino")


def setup_log():
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(name)s %(message)s", "%Y-%m-%d %H:%M:%S")
    r = logging.getLogger(); r.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(cfg.LOG_FILE, mode="a")):
        h.setFormatter(fmt); r.addHandler(h)
    r.setLevel(logging.INFO)


def feito(out):
    return (out / "done.json").exists()


def gravar(out, ident, met, tab, trans=None):
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([ident | met]).to_csv(out / "metricas.csv", sep=";", index=False)
    if tab is not None:
        tab.to_csv(out / "celulas_val.csv", sep=";", index=False)
    if trans is not None:
        trans.to_csv(out / "transicoes_celulas.csv", sep=";", index=False)
    (out / "done.json").write_text(json.dumps(ident | {"fim": time.strftime("%Y-%m-%d %H:%M")}))
    log.info(f"  ok {ident['exp_id']} fold{ident['fold']} | skill forte={met.get('forte_prox.ap_skill')} "
             f"cob5%={met.get('forte_prox.cob_5pct')} nó forte top1 prec={met.get('no_forte_top1.precisao')} "
             f"rec={met.get('no_forte_top1.recall')} | cond seco->chuva={met.get('cond_seco_chuva.ap_skill')} "
             f"fraca->forte={met.get('cond_fraca_forte.ap_skill')} IMcond={met.get('mi_condicional_norm')}")


def params(rapido):
    if not rapido:
        return dict(grades=cfg.GRADES, goes=cfg.GOES_MODOS, perdas=cfg.PERDAS, epocas=cfg.EPOCAS,
                    lotes=cfg.LOTES_EPOCA, som_iter=cfg.SOM_N_ITER, max_amostras=None)
    r = cfg.RAPIDO
    return dict(grades=r["grades"], goes=r["goes_modos"], perdas=r["perdas"], epocas=r["epocas"],
                lotes=r["lotes_epoca"], som_iter=r["som_n_iter"], max_amostras=r["max_amostras"])


def dados(cache, fold, prm):
    tr, va = cache.split(fold, prm["max_amostras"])
    norm = cache.normalizador(tr)
    return tr, va, norm, cache.rotulos(tr), cache.rotulos(va)


# ═══════════════════════════════ famílias ════════════════════════════════════
def fam_kmeans(cache, prm, folds):
    for fold in folds:
        tr, va, norm, ytr, yva = dados(cache, fold, prm)
        Xtr, Xva = nu.normalizar_planas(cache.planas(tr, norm), cache.planas(va, norm))
        for g in prm["grades"]:
            K = g[0] * g[1]
            eid = f"kmeans_k{K}"
            out = cfg.RESULTS_DIR / "kmeans" / eid / f"fold{fold}"
            if feito(out):
                continue
            km = MiniBatchKMeans(K, n_init=cfg.KMEANS_N_INIT, random_state=cfg.SEED, batch_size=4096).fit(Xtr)
            met, tab, trans = nu.avaliar(km.predict(Xtr), ytr, km.predict(Xva), yva, K, g, cache, va)
            gravar(out, dict(familia="kmeans", exp_id=eid, fold=fold, grade=f"{g[0]}x{g[1]}",
                             goes="estatisticas", perda="-", ae="-"), met, tab, trans)


def fam_som(cache, prm, folds):
    for fold in folds:
        tr, va, norm, ytr, yva = dados(cache, fold, prm)
        Xtr, Xva = nu.normalizar_planas(cache.planas(tr, norm), cache.planas(va, norm))
        for g in prm["grades"]:
            for estr in ("baseline", "supervisionado"):
                eid = f"som_{g[0]}x{g[1]}_{estr}"
                out = cfg.RESULTS_DIR / "som_classico" / eid / f"fold{fold}"
                if feito(out):
                    continue
                out.mkdir(parents=True, exist_ok=True)
                Xf = Xtr if estr == "baseline" else \
                    np.hstack([Xtr, cfg.SOM_SUP_LAMBDA * nu.y_supervisionado(ytr)]).astype(np.float32)
                ck = out / "som.npz"
                som = nu.SOM.carregar(ck) if ck.exists() else nu.SOM(g[0], g[1], Xf.shape[1], n_x=Xtr.shape[1])
                som.fit(Xf, prm["som_iter"], ck)
                met, tab, trans = nu.avaliar(som.bmu(Xtr), ytr, som.bmu(Xva), yva, som.n, g, cache, va)
                gravar(out, dict(familia="som_classico", exp_id=eid, fold=fold, grade=f"{g[0]}x{g[1]}",
                                 goes="estatisticas", perda=estr, ae="-"), met, tab, trans)


def treinar_profundo(cache, prm, folds, combos, etapa):
    for fold in folds:
        tr, va, norm, ytr, yva = dados(cache, fold, prm)
        for goes, perda, ae, g in combos:
            eid = f"{etapa}_{goes}_{perda}_{ae}_{g[0]}x{g[1]}"
            out = cfg.RESULTS_DIR / etapa / eid / f"fold{fold}"
            if feito(out):
                continue
            log.info(f"[{etapa}] {eid} fold{fold}")
            torch.manual_seed(cfg.SEED)
            model = nu.SOMGoes(cache, goes, ae, g)
            nu.treinar(model, cache, tr, va, norm, perda, prm["epocas"], prm["lotes"], out)
            met, tab, trans = nu.avaliar(nu.bmus(model, cache, tr, norm), ytr, nu.bmus(model, cache, va, norm),
                                  yva, g[0] * g[1], g, cache, va)
            gravar(out, dict(familia=etapa, exp_id=eid, fold=fold, grade=f"{g[0]}x{g[1]}",
                             goes=goes, perda=perda, ae=ae), met, tab, trans)


def fam_etapa1(cache, prm, folds):
    g = cfg.GRADE_ETAPA1 if prm["max_amostras"] is None else prm["grades"][-1]
    combos = [(go, pe, "ae", g) for go in prm["goes"] for pe in prm["perdas"]]
    combos += [(go, pe, "vae", g) for go in prm["goes"] for pe in cfg.PERDAS_VAE if pe in prm["perdas"]]
    treinar_profundo(cache, prm, folds, combos, "etapa1")


def melhor_etapa1():
    if cfg.ETAPA2_FIXA:
        return cfg.ETAPA2_FIXA
    fs = list((cfg.RESULTS_DIR / "etapa1").rglob("metricas.csv"))
    if not fs:
        return None
    M = pd.concat([pd.read_csv(f, sep=";") for f in fs])
    M = M[M.goes.isin(cfg.GOES_MODOS) & M.perda.isin(cfg.PERDAS)]   # só as combinações em uso
    g = M.groupby(["goes", "perda", "ae"])
    r = g[[cfg.METRICA_PRINCIPAL, cfg.METRICA_SECUNDARIA]].mean()
    r["n_folds"] = g.fold.nunique()
    r = r[r.n_folds == r.n_folds.max()]                             # exige todas as temporadas
    r = r.sort_values([cfg.METRICA_PRINCIPAL, cfg.METRICA_SECUNDARIA], ascending=False)
    log.info("ETAPA 1 (médias entre folds)\n" + r.round(4).to_string())
    return r.index[0]


def fam_etapa2(cache, prm, folds):
    best = melhor_etapa1()
    if best is None:
        log.warning("etapa2 pulada: etapa1 sem resultados"); return
    goes, perda, ae = best
    log.info(f"ETAPA 2: varredura de grades com goes={goes} perda={perda} ae={ae}")
    treinar_profundo(cache, prm, folds, [(goes, perda, ae, g) for g in prm["grades"]], "etapa2")


def treinar_cenarios(cache, prm, folds, combos, familia):
    """combos: (chuva, h, perda, grade); GOES sempre no quadro atual."""
    goes = cfg.GOES_CENARIOS
    for fold in folds:
        prontos = {}
        for chuva, h, perda, g in combos:
            eid = f"{familia}_{chuva}_h{h}_{perda}_{g[0]}x{g[1]}"
            out = cfg.RESULTS_DIR / familia / eid / f"fold{fold}"
            if feito(out):
                continue
            if h not in prontos:
                tr, va = cache.split(fold, prm["max_amostras"], h)
                prontos[h] = (tr, va, cache.normalizador(tr), cache.rotulos(tr, h), cache.rotulos(va, h))
            tr, va, norm, ytr, yva = prontos[h]
            log.info(f"[{familia}] {eid} fold{fold}")
            torch.manual_seed(cfg.SEED)
            model = nu.SOMGoes(cache, goes, "ae", g, chuva=chuva)
            nu.treinar(model, cache, tr, va, norm, perda, prm["epocas"], prm["lotes"], out, h)
            met, tab, trans = nu.avaliar(nu.bmus(model, cache, tr, norm), ytr, nu.bmus(model, cache, va, norm),
                                         yva, g[0] * g[1], g, cache, va, h)
            gravar(out, dict(familia=familia, exp_id=eid, fold=fold, grade=f"{g[0]}x{g[1]}", goes=goes,
                             perda=perda, ae="ae", chuva=chuva, horizonte=h), met, tab, trans)


def fam_cenarios(cache, prm, folds):
    r = cfg.RAPIDO_CENARIOS if prm["max_amostras"] else None
    chs = r["chuva"] if r else cfg.CHUVA_ENTRADA
    hs = r["horizontes"] if r else cfg.HORIZONTES
    pes = r["perdas"] if r else cfg.PERDAS_CENARIOS
    g = cfg.GRADE_CENARIOS if not r else prm["grades"][-1]
    combos = [(c, h, p, g) for c in chs for h in hs for p in pes]
    treinar_cenarios(cache, prm, folds, combos, "cenarios")


def fam_grades(cache, prm, folds):
    fs = list((cfg.RESULTS_DIR / "cenarios").rglob("metricas.csv"))
    if not fs:
        log.warning("grades pulada: cenários sem resultados"); return
    M = pd.concat([pd.read_csv(f, sep=";") for f in fs])
    combos = []
    for (chuva, h), d in M.groupby(["chuva", "horizonte"]):
        if h not in cfg.HORIZONTES_GRADES and not prm["max_amostras"]:
            continue
        gp = d.groupby("perda")
        r = gp[cfg.METRICA_PRINCIPAL].mean().to_frame().assign(n=gp.fold.nunique())
        r = r[r.n == r.n.max()].sort_values(cfg.METRICA_PRINCIPAL, ascending=False)
        perda = r.index[0]
        log.info(f"GRADES: chuva={chuva} h={h} -> perda {perda} ({r.iloc[0, 0]:.3f})")
        combos += [(chuva, int(h), perda, g) for g in prm["grades"]]
    # a grade dos cenários já foi treinada: copia o resultado em vez de treinar de novo
    for chuva, h, perda, g in combos:
        if tuple(g) != tuple(cfg.GRADE_CENARIOS):
            continue
        for fold in folds:
            src = cfg.RESULTS_DIR / "cenarios" / f"cenarios_{chuva}_h{h}_{perda}_{g[0]}x{g[1]}" / f"fold{fold}"
            eid = f"grades_{chuva}_h{h}_{perda}_{g[0]}x{g[1]}"
            dst = cfg.RESULTS_DIR / "grades" / eid / f"fold{fold}"
            if feito(src) and not feito(dst):
                dst.mkdir(parents=True, exist_ok=True)
                for f in src.glob("*.csv"):
                    shutil.copy(f, dst / f.name)
                m = pd.read_csv(dst / "metricas.csv", sep=";")
                m["familia"], m["exp_id"] = "grades", eid
                m.to_csv(dst / "metricas.csv", sep=";", index=False)
                shutil.copy(src / "done.json", dst / "done.json")
    treinar_cenarios(cache, prm, folds, combos, "grades")


def fam_kmeans_cenarios(cache, prm, folds):
    """Referência k-means com as mesmas entradas de cada cenário (EMs em t0, estatísticas
    do GOES e, conforme o cenário, chuva no colar e no núcleo), nos dois horizontes e
    com K igual ao número de células de cada grade."""
    r = cfg.RAPIDO_CENARIOS if prm["max_amostras"] else None
    chs = r["chuva"] if r else cfg.CHUVA_ENTRADA
    hs = r["horizontes"] if r else cfg.HORIZONTES
    for fold in folds:
        for h in hs:
            tr, va = cache.split(fold, prm["max_amostras"], h)
            norm = cache.normalizador(tr)
            ytr, yva = cache.rotulos(tr, h), cache.rotulos(va, h)
            base_tr, base_va = cache.planas(tr, norm), cache.planas(va, norm)
            for chuva in chs:
                cc = cache.cols_chuva[chuva]
                def comchuva(ix, X):
                    if not cc:
                        return X
                    C = np.log1p(np.clip(cache.A.iloc[ix][cc].to_numpy(np.float32), 0, None))
                    return np.hstack([X, np.nan_to_num(C, nan=0.0)]).astype(np.float32)
                Xtr, Xva = nu.normalizar_planas(comchuva(tr, base_tr), comchuva(va, base_va))
                for g in prm["grades"]:
                    K = g[0] * g[1]
                    eid = f"kmeans_{chuva}_h{h}_k{K}"
                    out = cfg.RESULTS_DIR / "kmeans_cenarios" / eid / f"fold{fold}"
                    if feito(out):
                        continue
                    km = MiniBatchKMeans(K, n_init=cfg.KMEANS_N_INIT, random_state=cfg.SEED,
                                         batch_size=4096).fit(Xtr)
                    met, tab, trans = nu.avaliar(km.predict(Xtr), ytr, km.predict(Xva), yva, K, g, cache, va, h)
                    gravar(out, dict(familia="kmeans_cenarios", exp_id=eid, fold=fold, grade=f"{g[0]}x{g[1]}",
                                     goes="estatisticas", perda="-", ae="-", chuva=chuva, horizonte=h),
                           met, None, None)


FAMILIAS = {"kmeans": fam_kmeans, "som_classico": fam_som, "etapa1": fam_etapa1, "etapa2": fam_etapa2,
            "cenarios": fam_cenarios, "grades": fam_grades, "kmeans_cenarios": fam_kmeans_cenarios}
PADRAO = ["kmeans_cenarios", "cenarios", "grades"]


# ═══════════════════════════════ consolidação ════════════════════════════════
def consolidar_e_zipar():
    fs = list(cfg.RESULTS_DIR.rglob("metricas.csv"))
    if fs:
        M = pd.concat([pd.read_csv(f, sep=";") for f in fs], ignore_index=True)
        M.to_csv(cfg.RESULTS_DIR / "metricas_consolidadas.csv", sep=";", index=False)
        chaves = [cfg.METRICA_PRINCIPAL, "forte_prox.cob_5pct", "forte_prox.lift_5pct",
                  "muito_forte_prox.ap_skill", "inicio_forte.ap_skill", "intensifica.ap_skill",
                  "no_forte_top1.precisao", "no_forte_top1.recall", "no_forte_top3.recall",
                  "forte_por_estacao.cob_5pct", "nmi_transicoes", "acerto_cantos", "celulas_efetivas",
                  "cond_seco_chuva.ap_skill", "cond_seco_forte.ap_skill", "cond_fraca_forte.ap_skill",
                  "cond_fraca_seco.ap_skill", "cond_forte_persiste.ap_skill", "mi_condicional_norm",
                  "frac_mesma_celula", "dist_media_transicao"]
        chaves = [c for c in chaves if c in M]
        for c in ("chuva", "horizonte"):
            if c not in M:
                M[c] = np.nan
        M["chuva"] = M.chuva.fillna("nenhuma"); M["horizonte"] = M.horizonte.fillna(60)
        g = M.groupby(["familia", "exp_id", "goes", "perda", "ae", "grade", "chuva", "horizonte"])
        R = g[chaves].mean().round(4)
        R[f"{cfg.METRICA_PRINCIPAL}_ep"] = (g[cfg.METRICA_PRINCIPAL].std() / np.sqrt(g.size())).fillna(0).round(4)
        R["n_folds"] = g.size()
        R = R.sort_values(cfg.METRICA_PRINCIPAL, ascending=False)
        R.to_csv(cfg.RESULTS_DIR / "ranking.csv", sep=";")
        # regra do um erro padrão: menor grade dentro de 1 EP da melhor, por configuração
        E = []
        for (fam, goes, perda, ae, chuva, hz), d in R.reset_index().groupby(
                ["familia", "goes", "perda", "ae", "chuva", "horizonte"]):
            if d.grade.nunique() < 2:
                continue
            d = d.assign(K=d.grade.str.split("x").map(lambda x: int(x[0]) * int(x[1])))
            b = d.loc[d[cfg.METRICA_PRINCIPAL].idxmax()]
            lim = b[cfg.METRICA_PRINCIPAL] - b[f"{cfg.METRICA_PRINCIPAL}_ep"]
            esc = d[d[cfg.METRICA_PRINCIPAL] >= lim - 1e-9].sort_values("K").iloc[0]
            E.append(dict(familia=fam, goes=goes, perda=perda, ae=ae, chuva=chuva, horizonte=hz,
                          grade_maior_media=b.grade,
                          media=b[cfg.METRICA_PRINCIPAL], ep=b[f"{cfg.METRICA_PRINCIPAL}_ep"],
                          grade_escolhida_1ep=esc.grade, media_escolhida=esc[cfg.METRICA_PRINCIPAL]))
        pd.DataFrame(E).to_csv(cfg.RESULTS_DIR / "escolha_grade.csv", sep=";", index=False)
        log.info(f"consolidadas {len(M)} linhas | {M.exp_id.nunique()} experimentos")
    shutil.copy(Path(__file__).parent / "config_somgoes.py", cfg.RESULTS_DIR / "config_snapshot.py")
    with zipfile.ZipFile(cfg.ZIP_FILE, "w", zipfile.ZIP_DEFLATED) as z:
        for pat in ("*.csv", "treino.log", "config_snapshot.py", "**/metricas.csv",
                    "**/celulas_val.csv", "**/historia.csv", "**/transicoes_celulas.csv"):
            for f in cfg.RESULTS_DIR.glob(pat):
                z.write(f, f.relative_to(cfg.ROOT))
        for f in cfg.CACHE_DIR.glob("*"):
            if f.suffix in (".log", ".json", ".csv"):
                z.write(f, f.relative_to(cfg.ROOT))
    log.info(f"pacote: {cfg.ZIP_FILE} ({cfg.ZIP_FILE.stat().st_size/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--familias", nargs="*", default=PADRAO, choices=list(FAMILIAS))
    ap.add_argument("--folds", nargs="*", type=int)
    ap.add_argument("--so-zip", action="store_true")
    a = ap.parse_args()
    setup_log()
    log.info("=" * 78)
    log.info(f"train_somgoes | rapido={a.rapido} | familias={a.familias} | device={nu.DEV}")
    if a.so_zip:
        consolidar_e_zipar(); return
    cache = nu.Cache()
    prm = params(a.rapido)
    folds = a.folds if a.folds is not None else \
        ([-1] if a.rapido else sorted(cache.A.loc[cache.valida, "fold"].unique().tolist()))
    log.info(f"famílias: {a.familias}")
    log.info(f"folds (temporadas): {folds} = {[cache.meta['temporadas'][f] for f in folds if f >= 0]}")
    t0 = time.time()
    for fam in a.familias:
        try:
            FAMILIAS[fam](cache, prm, folds)
        except KeyboardInterrupt:
            log.warning("interrompido; estados salvos"); break
        except Exception:
            log.error(f"{fam} falhou:\n{traceback.format_exc()}")
    log.info(f"treino encerrado em {(time.time()-t0)/60:.1f} min")
    consolidar_e_zipar()


if __name__ == "__main__":
    main()
