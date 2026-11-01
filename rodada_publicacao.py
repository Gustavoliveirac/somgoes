#!/usr/bin/env python
"""
rodada_publicacao.py — completa nas 6 temporadas o que só rodou em uma (triagem) e treina o
comitê sem vazamento, para que todas as comparações do artigo usem a mesma base.

Parte 1, estratégias da etapa 1 que só rodaram em 2022/23 (GOES no quadro atual, salvo
indicado): mi_lift, mi_sep, mi_conc, mi_conc_sep, completa, VAE com mi e com completa,
convlstm com mi e com reconstrução, 6 quadros com reconstrução.
Parte 2, perdas dos cenários que só rodaram em 2022/23: cmi e a transição original, nas
3 entradas de chuva e nos 2 horizontes.
Parte 3, comitê aninhado: para cada temporada s e cada outra temporada t, um modelo treinado
sem s e sem t (t só para parar o treino). Os 5 modelos de cada s formam um comitê que nunca
viu s. Cenários em cfg.COMITE_CENARIOS, perda e grade do modelo escolhido.

Tudo é retomável e o que já existe é pulado. Estimativa na GPU: parte 1 cerca de 10 h,
parte 2 cerca de 7 h, parte 3 cerca de 6 h.
Uso:  python rodada_publicacao.py            (as 3 partes)
      python rodada_publicacao.py 3          (só a parte 3; aceita 1, 2, 3 ou combinações)
Depois: ./roda_analise.sh e envie o pacote_analise.zip
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

import config_somgoes as cfg
import somgoes_nucleo as nu
import train_somgoes as T


def parte1(cache, prm, folds):
    g = cfg.GRADE_ETAPA1
    combos = [("t0", p, "ae", g) for p in ("mi_lift", "mi_sep", "mi_conc", "mi_conc_sep", "completa")]
    combos += [("t0", "mi", "vae", g), ("t0", "completa", "vae", g)]
    combos += [("convlstm", "mi", "ae", g), ("convlstm", "recon", "ae", g), ("seq_cnn", "recon", "ae", g)]
    T.treinar_profundo(cache, prm, folds, combos, "etapa1")


def parte2(cache, prm, folds):
    combos = [(c, h, p, cfg.GRADE_CENARIOS) for c in cfg.CHUVA_ENTRADA for h in cfg.HORIZONTES
              for p in ("cmi", "duas_etapas_markov")]
    T.treinar_cenarios(cache, prm, folds, combos, "cenarios")


def parte3(cache, prm, folds):
    f = cache.A.fold.to_numpy()
    for ch, h in cfg.COMITE_CENARIOS:
        val = cache._val(h)
        for s in folds:
            for t in folds:
                if t == s:
                    continue
                out = cfg.RESULTS_DIR / "comite_aninhado" / f"{ch}_h{h}" / f"fora{s}_sem{t}"
                if (out / "done.json").exists():
                    continue
                T.log.info(f"[comitê aninhado] {ch} h{h} fora={s} sem={t}")
                tr = np.where(val & (f != s) & (f != t))[0]
                va = np.where(val & (f == t))[0]
                if len(tr) == 0 or len(va) == 0:
                    T.log.info("  sem dados suficientes, pulado"); continue
                norm = cache.normalizador(tr)
                torch.manual_seed(cfg.SEED)
                m = nu.SOMGoes(cache, cfg.GOES_CENARIOS, "ae", cfg.ESTADOS_GRADE, chuva=ch)
                nu.treinar(m, cache, tr, va, norm, cfg.ESTADOS_PERDA, prm["epocas"], prm["lotes"], out, h)
                (out / "done.json").write_text(json.dumps(dict(fora=s, sem=t, fim=time.strftime("%Y-%m-%d %H:%M"))))


def main():
    partes = sys.argv[1] if len(sys.argv) > 1 else "123"
    T.setup_log()
    cache = nu.Cache()
    prm = T.params(False)
    folds = sorted(cache.A.loc[cache.valida, "fold"].unique().tolist())
    T.log.info(f"rodada de publicação | partes {partes} | folds {folds} | device {nu.DEV}")
    for k, fn in (("1", parte1), ("2", parte2), ("3", parte3)):
        if k in partes:
            fn(cache, prm, folds)
    T.consolidar_e_zipar()


if __name__ == "__main__":
    main()
