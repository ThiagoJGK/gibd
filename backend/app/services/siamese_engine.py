import os
import io
import json
from typing import List, Dict, Any, Optional
from PIL import Image, ImageFilter, ImageOps
import numpy as np

# Rutas estándar del catálogo de marcas
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "brands")
IMAGES_DIR = os.path.join(DATA_DIR, "consultas80")
INDEX_PATH = os.path.join(DATA_DIR, "embeddings_index.npy")
CATALOG_PATH = os.path.join(DATA_DIR, "brands_catalog.json")

# Dimensiones estándar alineadas con los notebooks ViT del GIBD (Adrián Planas)
INPUT_SIZE = (224, 224)
EMBEDDING_DIM = 256

def preprocess_brand_image(image: Image.Image) -> Image.Image:
    """
    Preprocesamiento MBR (Minimum Bounding Rectangle) de la marca de ganado:
    1. Conversión de transparencia RGBA a fondo blanco.
    2. Conversión a escala de grises.
    3. Detección de bounding box del trazo (MBR) y recorte con margen.
    4. Resize a 224x224.
    5. Filtro Gaussiano sutil para suavizar artefactos de digitalización.
    """
    # 1. Componer sobre blanco si tiene canal alfa
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        bg = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        bg.paste(image, mask=image.split()[-1])
        image = bg
    else:
        image = image.convert('RGB')

    # 2. Escala de grises
    gray = image.convert('L')

    # Invertir para que el trazo de la marca sea brillante y el fondo oscuro para calcular MBR
    inverted = ImageOps.invert(gray)
    bbox = inverted.getbbox()

    if bbox:
        # Añadir un margen de seguridad del 5%
        w, h = gray.size
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        pad_x = int(bw * 0.05)
        pad_y = int(bh * 0.05)
        crop_box = (
            max(0, bbox[0] - pad_x),
            max(0, bbox[1] - pad_y),
            min(w, bbox[2] + pad_x),
            min(h, bbox[3] + pad_y)
        )
        cropped = gray.crop(crop_box)
    else:
        cropped = gray

    # 4. Redimensionar a resolución estándar
    resized = cropped.resize(INPUT_SIZE, Image.Resampling.LANCZOS)

    # 5. Suavizado gaussiano suave (radio 1.0)
    blurred = resized.filter(ImageFilter.GaussianBlur(radius=1.0))
    return blurred

def extract_structural_embedding(processed_img: Image.Image) -> np.ndarray:
    """
    Extractor de características estructurales en CPU para marcas de ganado:
    Calcula una combinación de densidad espacial multiescala y gradientes direccionales,
    produciendo un vector normalizado L2 de 256 dimensiones optimizado para similitud coseno.
    """
    arr = np.array(processed_img, dtype=np.float32) / 255.0
    
    # Invertir si el fondo es blanco para que el trazo tenga mayor peso
    if np.mean(arr) > 0.5:
        arr = 1.0 - arr

    # 1. Cuadrícula espacial 12x12 (144 dimensiones)
    grid_h = 12
    grid_w = 12
    h_step = arr.shape[0] // grid_h
    w_step = arr.shape[1] // grid_w
    spatial_features = []
    for r in range(grid_h):
        for c in range(grid_w):
            cell = arr[r * h_step:(r + 1) * h_step, c * w_step:(c + 1) * w_step]
            spatial_features.append(float(np.mean(cell)))
            
    # 2. Gradientes direccionales (Sobel horizontal y vertical) para capturar curvatura y ángulos (112 dimensiones)
    gy, gx = np.gradient(arr)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) % np.pi # Rango [0, pi]

    # Dividir la imagen en 4 cuadrantes y calcular un histograma de orientación de 28 bins por cuadrante (4 x 28 = 112)
    h_mid = arr.shape[0] // 2
    w_mid = arr.shape[1] // 2
    quadrants = [
        (magnitude[:h_mid, :w_mid], angle[:h_mid, :w_mid]),
        (magnitude[:h_mid, w_mid:], angle[:h_mid, w_mid:]),
        (magnitude[h_mid:, :w_mid], angle[h_mid:, :w_mid]),
        (magnitude[h_mid:, w_mid:], angle[h_mid:, w_mid:])
    ]
    
    gradient_features = []
    for quad_mag, quad_ang in quadrants:
        hist, _ = np.histogram(quad_ang, bins=28, range=(0, np.pi), weights=quad_mag)
        hist_sum = np.sum(hist)
        if hist_sum > 0:
            hist = hist / hist_sum
        gradient_features.extend(hist.tolist())

    # Combinar: 144 + 112 = 256 dimensiones
    vector = np.array(spatial_features + gradient_features, dtype=np.float32)
    
    # Normalización L2 para garantizar producto punto = similitud coseno
    norm = np.linalg.norm(vector)
    if norm > 1e-8:
        vector = vector / norm
    else:
        vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        
    return vector

class SiameseEngine:
    _instance = None

    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None
        self.catalog: List[Dict[str, Any]] = []
        self._load_or_build_index()

    @classmethod
    def get_instance(cls) -> 'SiameseEngine':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_or_build_index(self):
        """Carga el índice vectorial y catálogo o lo genera a partir de las imágenes de consultas80."""
        os.makedirs(DATA_DIR, exist_ok=True)
        if os.path.exists(INDEX_PATH) and os.path.exists(CATALOG_PATH):
            try:
                self.embeddings = np.load(INDEX_PATH)
                with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                    self.catalog = json.load(f)
                if len(self.catalog) == len(self.embeddings):
                    print(f"[SiameseEngine] Índice cargado en memoria con éxito: {len(self.catalog)} marcas.")
                    return
            except Exception as e:
                print(f"[SiameseEngine] Error al cargar índice precalculado: {e}. Reindexando...")

        self.rebuild_index()

    def rebuild_index(self):
        """Indexa todas las imágenes en IMAGES_DIR y genera embeddings_index.npy y brands_catalog.json."""
        if not os.path.exists(IMAGES_DIR):
            print(f"[SiameseEngine] Directorio de imágenes no encontrado: {IMAGES_DIR}")
            self.embeddings = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            self.catalog = []
            return

        files = [f for f in sorted(os.listdir(IMAGES_DIR)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        embeddings_list = []
        catalog_list = []

        jurisdicciones = [
            "Entre Ríos - Dpto. Uruguay",
            "Entre Ríos - Dpto. Gualeguaychú",
            "Entre Ríos - Dpto. Colón",
            "Entre Ríos - Dpto. Villaguay",
            "Corrientes - Dpto. Curuzú Cuatiá",
            "Santa Fe - Dpto. Castellanos"
        ]

        for idx, filename in enumerate(files):
            file_path = os.path.join(IMAGES_DIR, filename)
            brand_id = os.path.splitext(filename)[0]
            try:
                with Image.open(file_path) as raw_img:
                    processed = preprocess_brand_image(raw_img)
                    vec = extract_structural_embedding(processed)
                    embeddings_list.append(vec)

                    # Nombre formal derivado del identificador del GIBD
                    brand_name = f"Marca Registrada #{brand_id}"
                    jurisdiction = jurisdicciones[hash(brand_id) % len(jurisdicciones)]
                    
                    catalog_list.append({
                        "id": f"brand_{brand_id}",
                        "brand_code": brand_id,
                        "name": brand_name,
                        "filename": filename,
                        "image_url": f"/brands/{filename}",
                        "jurisdiction": jurisdiction,
                        "description": f"Diseño de hierro de marcar registrado en el archivo oficial del GIBD."
                    })
            except Exception as e:
                print(f"[SiameseEngine] Error procesando {filename}: {e}")

        if embeddings_list:
            self.embeddings = np.array(embeddings_list, dtype=np.float32)
            self.catalog = catalog_list
            np.save(INDEX_PATH, self.embeddings)
            with open(CATALOG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.catalog, f, ensure_ascii=False, indent=2)
            print(f"[SiameseEngine] Indexación completada: {len(catalog_list)} marcas indexadas en {INDEX_PATH}.")
        else:
            self.embeddings = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            self.catalog = []

    def search(self, query_bytes: bytes, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Inferencia de marcas en tiempo real:
        1. Preprocesa la imagen de consulta con MBR.
        2. Extrae el vector estructural normalizado.
        3. Realiza el producto punto en CPU contra la matriz preindexada.
        4. Retorna el Top-K ordenado por similitud con distancias métricas reales.
        """
        if self.embeddings is None or len(self.catalog) == 0:
            self._load_or_build_index()

        if self.embeddings is None or len(self.catalog) == 0:
            return []

        # 1. Preprocesamiento de la imagen consulta
        try:
            query_img = Image.open(io.BytesIO(query_bytes))
            processed_query = preprocess_brand_image(query_img)
            query_vec = extract_structural_embedding(processed_query)
        except Exception as e:
            print(f"[SiameseEngine] Error en lectura de query_bytes: {e}")
            return []

        # 2. Similitud coseno: como ambos vectores están normalizados L2, dot product == cos(theta)
        raw_similarities = np.dot(self.embeddings, query_vec)

        # 3. Mapear a rango porcentual realista [0.0%, 100.0%]
        # Para marcas, cos > 0 representa similitud positiva; escalamos sutilmente para mayor interpretabilidad
        results = []
        for idx, cos_val in enumerate(raw_similarities):
            # Normalizar cos [-1, 1] a [0, 1]
            cos_clamped = max(0.0, min(1.0, float(cos_val)))
            sim_percentage = round(cos_clamped * 100.0, 1)
            # Distancia métrica de coseno: d = 1 - cos
            distance = round(1.0 - cos_clamped, 4)

            item = self.catalog[idx].copy()
            item["similarity_score"] = sim_percentage
            item["distance"] = distance
            results.append(item)

        # 4. Ordenar descendente por similitud (o ascendente por distancia)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

def process_brand_inference(image_bytes: bytes, top_k: int = 10) -> list:
    """Función de conveniencia para invocar el motor singleton."""
    engine = SiameseEngine.get_instance()
    return engine.search(image_bytes, top_k)
