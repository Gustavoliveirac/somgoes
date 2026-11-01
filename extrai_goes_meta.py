#!/usr/bin/env python
"""
extrai_goes_meta.py — geometria do recorte do GOES-16 e constantes de Planck das bandas (Linux).

Precisa de um arquivo bruto (NetCDF) do RRQPE e um de cada banda C08 a C14. Procura primeiro
os arquivos brutos usados no preProcess.py do gfuse (pasta goes16bh, com ABI-L2-RRQPEF e
ABI-L1b-RadF/Cxx), na variável GOES_BRUTO ou em /media/<usuário>/*/goes16bh e
/media/<usuário>/*/ufmg/goes16bh. Se não achar, baixa um arquivo de cada do repositório
público da NOAA na AWS (noaa-goes16), sem senha, em torno de 400 MB no total, e apaga depois.

Grava na pasta do somgoes (a mesma de config_somgoes.py):
  goes_geometria.npz  lat e lon de cada pixel do recorte 64 x 64 e da grade reduzida 32 x 32
  goes_recorte.json   os quatro cantos do recorte (mapa da figura 1)
  goes_planck.json    constantes de Planck de C08 a C14 (radiância para temperatura de brilho)
Uso:   python extrai_goes_meta.py
Envie de volta o goes_extrai.log.
"""
import glob
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from pyproj import CRS, Geod, Transformer

OUT = Path(__file__).resolve().parent
BH_LAT, BH_LON = -19.9068, -43.9765
BANDAS = ["C08", "C09", "C10", "C11", "C12", "C13", "C14"]
AWS = "https://noaa-goes16.s3.amazonaws.com"
PREFIXO_HORA = "2023/015/15"          # 15/01/2023, 15 UTC
log = open(OUT / "goes_extrai.log", "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n"); log.flush()


def pasta_local():
    cands = [os.environ.get("GOES_BRUTO", "")]
    usuario = os.environ.get("USER", "*")
    cands += glob.glob(f"/media/{usuario}/*/goes16bh") + glob.glob(f"/media/{usuario}/*/ufmg/goes16bh")
    for c in cands:
        if c and Path(c).exists():
            dias = sorted(d for d in Path(c).iterdir() if (d / "ABI-L2-RRQPEF").exists())
            if dias:
                return dias[0]
    return None


def baixar(produto, filtro, tmp):
    r = requests.get(f"{AWS}/?list-type=2&prefix={produto}/{PREFIXO_HORA}/", timeout=60)
    chaves = [k for k in re.findall(r"<Key>([^<]+)</Key>", r.text) if filtro in k]
    if not chaves:
        raise RuntimeError(f"nada encontrado em {produto} com {filtro}")
    destino = Path(tmp) / Path(chaves[0]).name
    with requests.get(f"{AWS}/{chaves[0]}", stream=True, timeout=300) as rr:
        rr.raise_for_status()
        with open(destino, "wb") as f:
            for bloco in rr.iter_content(1 << 20):
                f.write(bloco)
    return destino


def arquivos():
    dia = pasta_local()
    if dia is not None:
        p(f"usando arquivos brutos locais em {dia}")
        rr = sorted((dia / "ABI-L2-RRQPEF").glob("*.nc"))[0]
        bandas = {b: sorted((dia / "ABI-L1b-RadF" / b).glob("*.nc"))[0] for b in BANDAS}
        return rr, bandas, None
    tmp = tempfile.mkdtemp()
    p("arquivos brutos locais não encontrados; baixando da AWS (noaa-goes16)")
    rr = baixar("ABI-L2-RRQPEF", "RRQPEF", tmp)
    bandas = {b: baixar("ABI-L1b-RadF", f"M6{b}", tmp) for b in BANDAS}
    return rr, bandas, tmp


rr, bandas, tmp = arquivos()
ds = xr.open_dataset(rr)
proj = ds.goes_imager_projection.attrs
h = proj["perspective_point_height"]
crs = CRS.from_cf(proj)
xm, ym = ds["x"].values * h, ds["y"].values * h
tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
bx, by = tr.transform(BH_LON, BH_LAT)
ix, iy = int(np.argmin(np.abs(xm - bx))), int(np.argmin(np.abs(ym - by)))
X, Y = np.meshgrid(xm[ix - 32:ix + 32], ym[iy - 32:iy + 32])
lon, lat = inv.transform(X, Y)
lat32 = lat.reshape(32, 2, 32, 2).mean((1, 3)); lon32 = lon.reshape(32, 2, 32, 2).mean((1, 3))
g = Geod(ellps="WGS84")
ew = g.inv(lon[32, 31], lat[32, 31], lon[32, 32], lat[32, 32])[2] / 1000
ns = g.inv(lon[31, 32], lat[31, 32], lon[32, 32], lat[32, 32])[2] / 1000
np.savez(OUT / "goes_geometria.npz", lat64=lat, lon64=lon, lat32=lat32, lon32=lon32, km_ew=ew, km_ns=ns)
dx, dy = (xm[1] - xm[0]) / 2, (ym[1] - ym[0]) / 2
cx = [xm[ix - 32] - dx, xm[ix + 31] + dx, xm[ix + 31] + dx, xm[ix - 32] - dx]
cy = [ym[iy - 32] - dy, ym[iy - 32] - dy, ym[iy + 31] + dy, ym[iy + 31] + dy]
cl = [inv.transform(a, b) for a, b in zip(cx, cy)]
json.dump({"cantos": [[la, lo] for lo, la in cl]}, open(OUT / "goes_recorte.json", "w"), indent=1)
p(f"RRQPE: {Path(rr).name}")
p(f"pixel central: x={ix}, y={iy}; centro em {lat[32, 32]:.4f}, {lon[32, 32]:.4f}")
p(f"pixel perto de BH: {ew:.2f} km (leste-oeste) x {ns:.2f} km (norte-sul)")
p("cantos (lat, lon):", [(round(la, 3), round(lo, 3)) for lo, la in cl])

planck = {}
for b, f in bandas.items():
    d = xr.open_dataset(f)
    mesma = d["x"].shape == ds["x"].shape and np.allclose(d["x"].values, ds["x"].values)
    planck[b] = {k: float(d[k].values) for k in ("planck_fk1", "planck_fk2", "planck_bc1", "planck_bc2")}
    planck[b]["comprimento_onda_um"] = float(np.ravel(d["band_wavelength"].values)[0])
    p(f"{b}: {planck[b]}; mesma grade do RRQPE: {mesma}")
    d.close()
json.dump(planck, open(OUT / "goes_planck.json", "w"), indent=1)
if tmp:
    for f in Path(tmp).glob("*"):
        f.unlink()
p("gravados goes_geometria.npz, goes_recorte.json e goes_planck.json")
