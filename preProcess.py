import os
import numpy as np
import xarray as xr
import rioxarray
from pathlib import Path
from pyproj import CRS, Transformer
from datetime import datetime
import time
import gc
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
BASE_DIR = r"F:\goes16bh"
OUTPUT_DIR = r"F:\ufmg\canada\goes16bh_processed"
BH_LAT = -19.9068
BH_LON = -43.9765
PIXEL_SIZES = [64, 128]
BANDAS_IR = ['C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14']
MAX_RETRY_SECONDS = 10
ERROR_WAIT_SECONDS = 20
LOG_FILE = Path(OUTPUT_DIR, "processing_log.txt")

# Criar estrutura de diretórios
for size in PIXEL_SIZES:
    Path(OUTPUT_DIR, f"tiff_{size}px").mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR, f"numpy_{size}px").mkdir(parents=True, exist_ok=True)

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def log_message(message, console=True):
    """Escreve mensagem no log e opcionalmente no console"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {message}\n"
    
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_line)
    
    if console:
        print(message)

def retry_file_access(file_path, max_seconds=10):
    """Tenta acessar arquivo com retry em caso de HD desconectado"""
    start_time = time.time()
    
    while (time.time() - start_time) < max_seconds:
        try:
            if Path(file_path).exists():
                with open(file_path, 'rb') as f:
                    f.read(1)
                return True
        except:
            time.sleep(0.5)
    
    return False

def file_already_processed(timestamp_str, size):
    """Verifica se arquivo já foi processado"""
    numpy_file = Path(OUTPUT_DIR, f"numpy_{size}px", f"{timestamp_str}.npy")
    return numpy_file.exists()

def get_goes_crs_and_coords(ds, sat_h):
    """Extrai CRS e coordenadas em metros do GOES"""
    proj_info = ds.goes_imager_projection.attrs
    goes_crs = CRS.from_cf(proj_info)
    
    x_rad = ds['x'].values
    y_rad = ds['y'].values
    x_m = x_rad * sat_h
    y_m = y_rad * sat_h
    
    return goes_crs, x_m, y_m

def find_center_indices(x_m, y_m, bh_x_m, bh_y_m):
    """Encontra índices do pixel central mais próximo de BH"""
    idx_x = np.argmin(np.abs(x_m - bh_x_m))
    idx_y = np.argmin(np.abs(y_m - bh_y_m))
    return idx_x, idx_y

def crop_data(data_array, x_m, y_m, idx_x, idx_y, size):
    """Recorta dados e retorna array numpy"""
    half = size // 2
    x_start = max(0, idx_x - half)
    x_end = min(len(x_m), idx_x + half)
    y_start = max(0, idx_y - half)
    y_end = min(len(y_m), idx_y + half)
    
    # Recortar dados
    cropped = data_array.values[y_start:y_end, x_start:x_end]
    
    # Garantir que tem o tamanho correto (padding se necessário)
    if cropped.shape[0] < size or cropped.shape[1] < size:
        padded = np.full((size, size), np.nan, dtype=np.float32)
        h, w = cropped.shape
        padded[:h, :w] = cropped
        return padded
    
    return cropped.astype(np.float32)

def save_tiff_multiband(data_stack, x_m, y_m, goes_crs, idx_x, idx_y, size, output_path):
    """Salva TIFF multi-banda"""
    half = size // 2
    x_start = max(0, idx_x - half)
    x_end = min(len(x_m), idx_x + half)
    y_start = max(0, idx_y - half)
    y_end = min(len(y_m), idx_y + half)
    
    # Coordenadas
    x_coords = x_m[x_start:x_end]
    y_coords = y_m[y_start:y_end]
    
    # Ajustar se necessário devido a padding
    actual_h, actual_w = data_stack.shape[1], data_stack.shape[2]
    if len(y_coords) < actual_h:
        y_coords = np.linspace(y_coords[0], y_coords[-1], actual_h)
    if len(x_coords) < actual_w:
        x_coords = np.linspace(x_coords[0], x_coords[-1], actual_w)
    
    # Criar DataArray
    multi_da = xr.DataArray(
        data_stack,
        coords={
            'band': range(1, data_stack.shape[0] + 1),
            'y': y_coords,
            'x': x_coords
        },
        dims=['band', 'y', 'x']
    )
    
    multi_da = multi_da.rio.write_crs(goes_crs)
    multi_da = multi_da.rio.set_spatial_dims(x_dim='x', y_dim='y')
    multi_da.rio.to_raster(output_path, compress='lzw', dtype='float32')

def cleanup_resources():
    """Limpa recursos e força garbage collection"""
    gc.collect()
    time.sleep(1)

def process_timestamp(dia_path, chuva_file, bh_x_m, bh_y_m):
    """Processa um timestamp específico"""
    ds_chuva = None
    ds_bandas = []
    
    try:
        # Extrair timestamp
        timestamp_str = chuva_file.stem.split('_')[3][1:]
        timestamp = datetime.strptime(timestamp_str[:13], '%Y%j%H%M%S')
        output_name = timestamp.strftime('%Y%m%d_%H%M%S')
        
        # Verificar se já foi processado para todos os tamanhos
        all_processed = all(file_already_processed(output_name, size) for size in PIXEL_SIZES)
        if all_processed:
            return True, f"Já processado: {output_name}"
        
        # Verificar acesso ao arquivo de chuva
        if not retry_file_access(chuva_file, MAX_RETRY_SECONDS):
            return False, f"Erro de acesso (chuva): {chuva_file.name}"
        
        # Abrir arquivo de chuva
        ds_chuva = xr.open_dataset(chuva_file)
        sat_h = ds_chuva.goes_imager_projection.attrs['perspective_point_height']
        
        # Obter CRS e coordenadas
        goes_crs, x_m, y_m = get_goes_crs_and_coords(ds_chuva, sat_h)
        
        # Encontrar índice central
        idx_x, idx_y = find_center_indices(x_m, y_m, bh_x_m, bh_y_m)
        
        # Ler dados de chuva
        chuva_data = ds_chuva['RRQPE']
        
        # Ler bandas IR
        bandas_data = {}
        for banda_name in BANDAS_IR:
            banda_path = dia_path / "ABI-L1b-RadF" / banda_name
            
            if not banda_path.exists():
                raise Exception(f"Pasta da banda {banda_name} não existe")
            
            banda_files = list(banda_path.glob(f"*{timestamp_str[:13]}*.nc"))
            if not banda_files:
                raise Exception(f"Arquivo da banda {banda_name} não encontrado")
            
            banda_file = banda_files[0]
            
            # Retry de acesso
            if not retry_file_access(banda_file, MAX_RETRY_SECONDS):
                raise Exception(f"Erro de acesso (banda {banda_name}): {banda_file.name}")
            
            ds_banda = xr.open_dataset(banda_file)
            ds_bandas.append(ds_banda)
            bandas_data[banda_name] = ds_banda['Rad']
        
        # Processar cada tamanho
        for size in PIXEL_SIZES:
            if file_already_processed(output_name, size):
                continue
            
            # Lista para armazenar todas as camadas
            all_layers = []
            
            # Processar bandas IR
            for banda_name in BANDAS_IR:
                layer_data = crop_data(bandas_data[banda_name], x_m, y_m, idx_x, idx_y, size)
                all_layers.append(layer_data)
            
            # Processar chuva
            layer_data = crop_data(chuva_data, x_m, y_m, idx_x, idx_y, size)
            all_layers.append(layer_data)
            
            # Stack: [7 bandas IR + 1 chuva, height, width]
            # Shape será (8, size, size)
            data_stack = np.stack(all_layers, axis=0).astype(np.float32)
            
            # Salvar TIFF multi-banda
            tiff_multi = Path(OUTPUT_DIR, f"tiff_{size}px", f"{output_name}.tif")
            save_tiff_multiband(data_stack, x_m, y_m, goes_crs, idx_x, idx_y, size, tiff_multi)
            
            # Salvar numpy
            # Shape: (8, height, width) - formato adequado para CNN/Transformer
            numpy_file = Path(OUTPUT_DIR, f"numpy_{size}px", f"{output_name}.npy")
            np.save(numpy_file, data_stack)
        
        # Fechar datasets explicitamente
        if ds_chuva is not None:
            ds_chuva.close()
        for ds in ds_bandas:
            ds.close()
        
        cleanup_resources()
        
        return True, f"Processado com sucesso: {output_name}"
        
    except Exception as e:
        # Fechar todos os datasets abertos
        if ds_chuva is not None:
            try:
                ds_chuva.close()
            except:
                pass
        
        for ds in ds_bandas:
            try:
                ds.close()
            except:
                pass
        
        # Limpar recursos
        cleanup_resources()
        
        # Retornar erro
        error_msg = f"Erro no processamento: {str(e)}"
        return False, error_msg

# ============================================================================
# PROCESSAMENTO PRINCIPAL
# ============================================================================

def main():
    log_message("="*80)
    log_message("INICIANDO PREPROCESSAMENTO GOES-16")
    log_message("="*80)
    
    # Calcular coordenadas de BH uma única vez
    goes_proj = f"+proj=geos +h=35786023 +lon_0=-75 +sweep=x +ellps=GRS80"
    goes_crs_temp = CRS.from_proj4(goes_proj)
    transformer = Transformer.from_crs("EPSG:4326", goes_crs_temp, always_xy=True)
    bh_x_m, bh_y_m = transformer.transform(BH_LON, BH_LAT)
    
    log_message(f"Coordenadas BH: {BH_LAT}°, {BH_LON}°")
    log_message(f"Coordenadas BH (GOES): {bh_x_m:.0f}m, {bh_y_m:.0f}m")
    log_message(f"Tamanhos de pixel: {PIXEL_SIZES}")
    log_message(f"Bandas IR: {BANDAS_IR}")
    
    # Listar todos os dias
    dias = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])
    log_message(f"Total de dias encontrados: {len(dias)}")
    
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    
    for dia in dias:
        log_message(f"\n{'='*60}")
        log_message(f"Processando dia: {dia}")
        log_message(f"{'='*60}")
        
        dia_path = Path(BASE_DIR, dia)
        
        # Verificar pasta de chuva
        chuva_path = dia_path / "ABI-L2-RRQPEF"
        if not chuva_path.exists():
            log_message(f"Pasta de chuva não existe: {chuva_path}")
            continue
        
        # Listar arquivos de chuva
        chuva_files = sorted(list(chuva_path.glob("*.nc")))
        log_message(f"Arquivos de chuva encontrados: {len(chuva_files)}")
        
        for i, chuva_file in enumerate(chuva_files, 1):
            success, message = process_timestamp(dia_path, chuva_file, bh_x_m, bh_y_m)
            
            if success:
                if "Já processado" in message:
                    total_skipped += 1
                else:
                    total_processed += 1
                    log_message(f"[{i}/{len(chuva_files)}] {message}")
            else:
                total_errors += 1
                log_message(f"[{i}/{len(chuva_files)}] ERRO: {message}")
                
                # Aguardar antes de continuar após erro
                log_message(f"Aguardando {ERROR_WAIT_SECONDS} segundos antes de continuar...")
                time.sleep(ERROR_WAIT_SECONDS)
                log_message("Retomando processamento...")
    
    log_message("\n" + "="*80)
    log_message("PROCESSAMENTO FINALIZADO")
    log_message("="*80)
    log_message(f"Total processado: {total_processed}")
    log_message(f"Total pulado (já processado): {total_skipped}")
    log_message(f"Total com erro: {total_errors}")
    log_message("="*80)

if __name__ == "__main__":
    main()