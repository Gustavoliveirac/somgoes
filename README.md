# somgoes

Code for a chapter of a doctoral thesis at the Federal University of Minas Gerais (UFMG) on the classification of atmospheric states with self-organizing maps (SOM) for heavy rainfall detection and early warning in Belo Horizonte, Brazil.

A deep SOM receives, every 10 minutes, the surface weather stations of the metropolitan area, the infrared channels and the rain rate product (RRQPE) of the GOES-16 imager over a window of about 160 by 150 km, and, in some configurations, the rain gauges around the city. Each 10-minute time step is assigned to a node of a 10 by 10 map, and the nodes are used to describe the atmospheric states, the transitions between rain states and the factors associated with each transition. The warning target is heavy rain (at least 10 mm/h at any of the 45 stations of the central area of the city) 30 and 60 minutes ahead. Rain gauges are grouped into the central area, a ring from 13 to 60 km and a ring from 60 to 100 km. All results are obtained with one rainy season left out at a time. Gradient boosting models with the same inputs are used only as a reference for how much the inputs allow to predict.

## Repository layout

| Folder | Content |
|---|---|
| `core/` | configuration, cache preparation, model, losses and metrics (`somgoes_nucleo.py`), training and the full publication run |
| `analise/` | atmospheric states, out-of-season validation of states and warning levels, synthesis tables and figures |
| `alerta/` | warning engine (export, replay of past events and real-time run) |
| `tese/` | chapter analyses, comparison with other classifiers, phase 1 and stage A analyses, and chapter figures |
| `goes/` | GOES-16 preprocessing, crop geometry and Planck constants |
| `scripts/` | shell scripts that run each part in order |

## How to run

All modules run from the repository root as packages, for example `python -m tese.fase1_som`. The usual order is `scripts/roda_tudo.sh` (cache and SOM training), `scripts/roda_analise.sh` (states, validation, synthesis and warning engine), `scripts/roda_tese.sh` and `scripts/roda_modelos.sh` (chapter analyses), `scripts/roda_fase1.sh` and `scripts/roda_etapa_a.sh`. Paths to the station and GOES data are set in `core/config_somgoes.py` or through the environment variables `GFUSE_ROOT`, `ESTACOES_DIR` and `GOES_DIR`.

The code uses Python 3 with PyTorch (a CUDA GPU is recommended), NumPy, pandas, SciPy, scikit-learn and pyarrow. CatBoost and TabPFN (version 2 weights) are optional references.

## Data

Station records, GOES-16 data, caches, trained models and results are not part of this repository. Station data come from public Brazilian networks (ANA, CEMADEN, INMET) and from the Belo Horizonte city drainage agency (Sudecap). GOES-16 data are public and available from NOAA.

## Status

Work in progress. Methods and results will be described in the thesis chapter and in a journal article.
