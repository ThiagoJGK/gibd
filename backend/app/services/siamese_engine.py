import io
import time
import hashlib
import random
from PIL import Image

# Base de datos semilla de marcas del GIBD para simulación One-Shot
MOCK_BRANDS = [
    {
        "id": "brand_01",
        "name": "Marca La Esperanza",
        "image_url": "https://supabase-mock-storage.gibd.utn.edu.ar/brands/la_esperanza.png"
    },
    {
        "id": "brand_02",
        "name": "Marca El Ceibo",
        "image_url": "https://supabase-mock-storage.gibd.utn.edu.ar/brands/el_ceibo.png"
    },
    {
        "id": "brand_03",
        "name": "Marca San Juan",
        "image_url": "https://supabase-mock-storage.gibd.utn.edu.ar/brands/san_juan.png"
    },
    {
        "id": "brand_04",
        "name": "Marca Don Pedro",
        "image_url": "https://supabase-mock-storage.gibd.utn.edu.ar/brands/don_pedro.png"
    },
    {
        "id": "brand_05",
        "name": "Marca La Estancia",
        "image_url": "https://supabase-mock-storage.gibd.utn.edu.ar/brands/la_estancia.png"
    },
    {
        "id": "brand_06",
        "name": "Marca Santa Catalina",
        "image_url": "https://supabase-mock-storage.gibd.utn.edu.ar/brands/santa_catalina.png"
    },
    {
        "id": "brand_07",
        "name": "Marca El Palmar",
        "image_url": "https://supabase-mock-storage.gibd.utn.edu.ar/brands/el_palmar.png"
    },
    {
        "id": "brand_08",
        "name": "Marca Los Pinos",
        "image_url": "https://supabase-mock-storage.gibd.utn.edu.ar/brands/los_pinos.png"
    },
    {
        "id": "brand_09",
        "name": "Marca San Francisco",
        "image_url": "https://supabase-mock-storage.gibd.utn.edu.ar/brands/san_francisco.png"
    },
    {
        "id": "brand_10",
        "name": "Marca La Querencia",
        "image_url": "https://supabase-mock-storage.gibd.utn.edu.ar/brands/la_querencia.png"
    }
]

def process_brand_inference(image_bytes: bytes, top_k: int = 10) -> list:
    """
    Simula la inferencia de marcas de ganado utilizando redes siamesas.
    Aplica preprocesamiento de imágenes real con Pillow y retorna resultados ordenados.
    """
    # 1. Preprocesamiento real de la imagen con Pillow
    try:
        img = Image.open(io.BytesIO(image_bytes))
        original_width, original_height = img.size
        
        # Simular Crop (recorte central de la marca al 90% para limpieza de bordes)
        left = int(original_width * 0.05)
        top = int(original_height * 0.05)
        right = int(original_width * 0.95)
        bottom = int(original_height * 0.95)
        img_cropped = img.crop((left, top, right, bottom))
        
        # Conversión a escala de grises (Flat Filtering para redes siamesas de marcas)
        img_gray = img_cropped.convert("L")
        
        # Resize a tamaño estándar de la red de inferencia (224x224)
        img_resized = img_gray.resize((224, 224))
        
        # Operación básica en memoria para forzar el flujo del buffer
        processed_buffer = io.BytesIO()
        img_resized.save(processed_buffer, format="PNG")
        processed_size = len(processed_buffer.getvalue())
        
        print(f"[SiameseEngine] Imagen procesada con éxito. "
              f"Tamaño original: {original_width}x{original_height}. "
              f"Tamaño normalizado: 224x224 (Gris). "
              f"Bytes de salida: {processed_size} bytes.")
    except Exception as e:
        print(f"[SiameseEngine Error] Error al preprocesar la imagen con Pillow: {str(e)}")
        # Si falla el preprocesamiento, seguimos con el flujo de simulación pero registramos la falla

    # 2. Inferencia Siamés mediante cálculo determinista por hash
    # Generamos un hash MD5 de los bytes recibidos
    img_hash = hashlib.md5(image_bytes).hexdigest()
    
    # Sembramos el generador aleatorio con el hash de la imagen para que sea estable/consistente
    seed_val = int(img_hash, 16) % (2**32)
    local_rng = random.Random(seed_val)
    
    results = []
    for brand in MOCK_BRANDS:
        # Generar un score de similitud determinista pero realista entre 40.0% y 98.5%
        similarity = round(local_rng.uniform(40.0, 98.5), 1)
        
        # La distancia métrica es inversamente proporcional a la similitud (de 0 a 1)
        # Distancia = (100 - Similitud) / 100.
        # Ej: 94.8% -> 0.052 de distancia
        distance = round((100.0 - similarity) / 100.0, 4)
        
        results.append({
            "id": brand["id"],
            "name": brand["name"],
            "similarity_score": similarity,
            "distance": distance,
            "image_url": brand["image_url"]
        })
        
    # Ordenar los resultados por similitud descendente (o distancia ascendente)
    sorted_results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)
    
    # Retornar el Top-K solicitado
    return sorted_results[:top_k]
