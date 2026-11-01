"""
config_somgoes.py — configuração do estudo SOM sobre EMs + GOES (Grande BH).

Prioridades do estudo
  1. concentrar em poucos nós os horários em que a próxima hora terá chuva forte
     (>= 10 mm/h) ou muito forte (>= 20 mm/h) no núcleo de BH
  2. separar os estados e as transições entre eles (agora -> próxima hora), para
     entender pelas variáveis de entrada o que caracteriza cada transição

Entrada do modelo (nunca chuva de pluviômetro): todas as EMs (temperatura,
umidade, pressão, vento, radiação) e o campo GOES (7 IR + RRQPE).
Chuva das estações do núcleo entra só no ALVO, nas perdas e na avaliação.
"""
import os
from pathlib import Path

# ═══════════════════════════════ caminhos ════════════════════════════════════
ROOT       = Path(__file__).resolve().parent
GFUSE_ROOT = Path(os.environ.get("GFUSE_ROOT", Path.home() / "Documentos" / "gfuse"))
META_FILE  = GFUSE_ROOT / "processados" / "metadados.csv"
ESTACOES_DIR = [Path(p) for p in os.environ.get("ESTACOES_DIR", "").split(":") if p] or [
    Path("/media/gustavo/8e275a83-51ed-4fe7-8a40-b8801f6efed5/ufmg/gfuse/processados"),
    GFUSE_ROOT / "processados"]
GOES_DIR   = Path(os.environ.get(
    "GOES_DIR", "/media/gustavo/8e275a83-51ed-4fe7-8a40-b8801f6efed5/ufmg/canada/numpy_64px"))
CACHE_DIR   = ROOT / "cache_somgoes"
RESULTS_DIR = ROOT / "resultados_somgoes"
LOG_FILE    = RESULTS_DIR / "treino.log"
ZIP_FILE    = ROOT / "pacote_somgoes.zip"
FONTE_DIR = {"ANA": "ANA", "CEMADEN": "CEMADEM", "CEMADEM": "CEMADEM", "INMET": "INMET",
             "SUDECAP": "SUDECAP", "SUDECAP_METEO": "SUDECAP_METEO"}

# ═══════════════════════════════ período e geometria ═════════════════════════
INICIO, FIM = os.environ.get("INICIO", "2020-01-01 00:00"), os.environ.get("FIM", "2024-12-31 23:50")
MESES       = [10, 11, 12, 1, 2, 3]
BH_LAT, BH_LON = -19.9068, -43.9765
IMG_SIZE, KM_PER_PX = 64, 2.0
GOES_RES    = 32                      # campo reduzido por média 2x2 (4 km/px)
N_QUADROS   = 6                       # quadro atual + 5 anteriores (50 min)
LACUNA_GOES_MAX = 3                   # quadros faltantes até 30 min: repete o anterior

# ═══════════════════════════════ chuva (alvo) ════════════════════════════════
BASCULA, ISOLADA_MIN, HORARIA_MIN = 0.2, 0.4, 0.4
VIZINHO_MAX_MIN, GAP_MAX_MIN = 20, 70
ESTADOS   = ["seco", "fraca", "moderada", "forte", "muito_forte"]
LIMITES   = [0.2, 2.5, 10.0, 20.0]    # OMM (2,5 e 10 mm/h) e INMET aviso amarelo (20 mm/h)
MIN_EST_NUCLEO = 3
COB_MIN_ESTACAO = 0.30                # estação com menos dados válidos é excluída
CHUVA_MAX_REL   = 3.0                 # ou com fração de chuva > 3x a mediana

# Transições agrupadas (usadas nas perdas e na avaliação)
#   0 SS seco->seco          4 CC chuva fraca/moderada persiste
#   1 inicio seco->fraca/mod  5 termina chuva->seco
#   2 inicio_forte seco->forte+  6 forte_persiste forte+->forte+
#   3 intensifica fraca/mod->forte+  7 enfraquece forte+->fraca/mod
TRANSICOES = ["SS", "inicio", "inicio_forte", "intensifica", "CC",
              "termina", "forte_persiste", "enfraquece"]
# quatro estados básicos -> cantos da grade (linha = agora, coluna = próxima hora)
#   SS canto (0,0)  SC canto (0,W-1)  CS canto (H-1,0)  CC canto (H-1,W-1)
BASICOS = ["SS", "SC", "CS", "CC"]

# ═══════════════════════════════ entrada ═════════════════════════════════════
MET_VARS = ["temperature", "umidity", "pression", "wind_speed", "radiation"]  # + vento sin/cos
BOUNDS = {"temperature": (-20, 55), "umidity": (0, 100), "pression": (800, 1100),
          "wind_speed": (0, 100), "wind_dir": (0, 360), "radiation": (-10, 2000)}
JANELA_10MIN = 6                      # EMs de 10 min: t-50 .. t0
GOES_CANAIS  = [0, 1, 2, 3, 4, 5, 6, 7]   # 7 IR + RRQPE (canal 7)

# ═══════════════════════════════ validação ═══════════════════════════════════
# fold = temporada chuvosa (out a mar); temporadas parciais entram como estão
SEED   = 42
DEVICE = os.environ.get("DEVICE", "auto")   # "auto" usa GPU se o torch do ambiente funcionar

# ═══════════════════════════════ modelo profundo ═════════════════════════════
# entrada GOES: nenhum | t0 (só quadro atual) | seq_cnn (6 quadros como canais) | convlstm
GOES_MODOS   = ["nenhum", "t0"]   # seq_cnn e convlstm não superaram t0 no fold 3
# Perdas (a chuva nunca entra no encoder; só nestes termos). Cada perda é um
# dicionário de pesos: mi (informação mútua célula x transição), peso_forte
# (peso das transições para forte dentro da mi), lift (divergência entre a
# distribuição das transições para forte e a de todos os horários no mapa),
# canto (âncora dos 4 estados básicos), sep (separação entre pares),
# conc/uso (concentração por entropia e uso do mapa, versão da rodada anterior)
PERDAS_DEF = {
    "recon":       {},
    "mi":          {"mi": 1.0},
    "mi_x2":       {"mi": 2.0},
    "mi_forte":    {"mi": 1.0, "peso_forte": 3.0},
    "mi_lift":     {"mi": 1.0, "lift": 1.0},
    "mi_canto":    {"mi": 1.0, "canto": 2.0},
    "mi_sep":      {"mi": 1.0, "sep": 0.5},
    # versões anteriores (mantidas para reaproveitar resultados)
    "mi_conc":     {"mi": 1.0, "conc": 0.5, "uso": 0.5},
    "mi_conc_sep": {"mi": 1.0, "conc": 0.5, "uso": 0.5, "sep": 0.5},
    "completa":    {"mi": 1.0, "conc": 0.5, "uso": 0.5, "sep": 0.5, "canto": 0.5},
    # perdas condicionais ao estado atual (rodada de cenários)
    #   cmi                 informação mútua entre célula e estado da próxima hora, dentro de
    #                       cada estado atual (seco, fraca/moderada, forte/muito forte)
    #   duas_etapas         primeiro organiza pelo estado atual (mi_agora), a partir da fração
    #                       `fase` das épocas passa à cmi (mi_agora segue com 30% do peso)
    #   duas_etapas_markov  + modelo de Markov entre células (t -> t+h) com suavidade
    #                       topológica, como no SOM-VAE (Fortuin et al. 2019)
    "cmi":                {"cmi": 1.0},
    "duas_etapas":        {"mi_agora": 1.0, "cmi": 1.0, "fase": 0.4},
    "duas_etapas_markov": {"mi_agora": 1.0, "cmi": 1.0, "fase": 0.4, "markov": 0.5, "suave": 0.5},
    # versão corrigida: a anterior colapsou (3 a 6 células) porque a suavidade premiava
    # ficar na mesma célula. Aqui sem suavidade, peso de transição menor e termo de uso
    # do mapa, que penaliza concentrar todas as amostras em poucas células
    "duas_etapas_markov2": {"mi_agora": 1.0, "cmi": 1.0, "fase": 0.4, "markov": 0.1, "uso": 0.5},
}
PERDAS     = ["recon", "mi", "mi_x2", "mi_forte", "mi_canto"]   # finalistas do fold 3
PERDAS_VAE = []                           # VAE ficou abaixo do AE em todas as entradas
LATENTE      = 32
D_TOKEN      = 64
EPOCAS       = 30
LOTES_EPOCA  = 120                    # lotes por época (amostragem balanceada)
LOTE         = int(os.environ.get("LOTE", 512))   # reduza (ex. LOTE=256) se faltar memória
LR           = 1e-3
PACIENCIA    = 6
PRE_EPOCAS   = 3                      # pré-treino só com reconstrução, depois os
                                      # protótipos são inicializados no plano PCA do latente
GAMMA_SOM    = 0.05
BETA_VAE     = 1.0
AQUEC_KL     = 10                     # épocas de aquecimento do KL (evita colapso do VAE)
TAU_SOFT     = float(os.environ.get("TAU_SOFT", 0.2))   # menor = atribuição suave mais próxima do BMU
COTA_MIN_CLASSE = 24                  # mínimo de amostras por transição em cada lote
PARES_SEPARACAO = [("SS", "inicio"), ("inicio", "inicio_forte"), ("CC", "intensifica"),
                   ("forte_persiste", "enfraquece"), ("CC", "termina"), ("SS", "inicio_forte")]

# ═══════════════════════════════ cenários (rodada atual) ═════════════════════
# entrada de chuva: nenhuma | colar (13-60 e 60-100 km, por quadrante) | colar_nucleo
CHUVA_ENTRADA   = ["nenhuma", "colar", "colar_nucleo"]
HORIZONTES      = [60, 30]                # minutos; 30 min usa o acumulado de 30 min em mm/h
PERDAS_CENARIOS = ["mi", "duas_etapas", "duas_etapas_markov2"]   # cmi abaixo da mi; markov original colapsou
GOES_CENARIOS   = "t0"
GRADE_CENARIOS  = (10, 10)
# varredura de grades: melhor perda de cada entrada de chuva, nestes horizontes
HORIZONTES_GRADES = [60, 30]

# ═══════════════════════════════ grades e etapas ═════════════════════════════
GRADES = [(2, 2), (3, 3), (4, 4), (5, 5), (10, 10), (15, 15), (20, 20), (25, 25)]
GRADE_ETAPA1 = (10, 10)
# Etapa 1: GOES x perda na grade de 10x10 (+ VAE na perda completa)
# Etapa 2: varredura de grades com a melhor combinação da etapa 1
#          (ou a fixada aqui, ex. ("convlstm", "completa", "ae"))
ETAPA2_FIXA  = None
# Referências: SOM clássico (features planas) e k-means, nas mesmas grades
SOM_N_ITER   = 200_000
SOM_SUP_LAMBDA = 3.0
SOM_CKPT     = 20_000
KMEANS_N_INIT = 5

# métrica principal (prioridade 1) e secundária (prioridade 2)
METRICA_PRINCIPAL  = "forte_prox.ap_skill"
METRICA_SECUNDARIA = "mi_condicional_norm"
FRACS_TEMPO = [0.01, 0.02, 0.05, 0.10]

# ═══════════════════════════════ modo rápido ═════════════════════════════════
RAPIDO_CENARIOS = dict(chuva=["nenhuma", "colar_nucleo"], horizontes=[60, 30],
                       perdas=["mi", "duas_etapas_markov2"])
RAPIDO = dict(max_amostras=8000, epocas=4, lotes_epoca=15, som_n_iter=8000,
              grades=[(4, 4), (10, 10)], goes_modos=["nenhum", "convlstm"],
              perdas=["recon", "mi", "mi_lift", "completa"])

# ═══════════════════════════════ estados (modelo final) ══════════════════════
# Modelo escolhido na rodada completa: perda de duas etapas, grade 10x10, GOES t0.
# Cenário operacional (colar e núcleo) e referência sem chuva, nos dois horizontes.
ESTADOS_CENARIOS  = [("colar_nucleo", 60), ("colar_nucleo", 30), ("nenhuma", 60), ("nenhuma", 30)]
ESTADOS_PERDA     = "duas_etapas"
ESTADOS_GRADE     = (10, 10)
N_REGIOES         = [6, 8, 10, 12]
N_REGIOES_PADRAO  = 8
# níveis de alerta: limite superior da taxa de chuva forte (no treino) de cada nível
NIVEIS_ALERTA = [(0.01, "sem monitoramento"), (0.10, "atenção"), (0.40, "alerta"), (1.01, "chuva forte provável")]
# canais do cubo GOES (preProcess.py): radiância L1b dos canais IR e RRQPE em mm/h
GOES_NOMES = ["C08", "C09", "C10", "C11", "C12", "C13", "C14", "RRQPE"]

# comitê avaliado sem vazamento (validação aninhada) e rodada completa para publicação
COMITE_CENARIOS = [("colar_nucleo", 60), ("colar_nucleo", 30)]
