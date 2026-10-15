from PIL import Image, ImageOps
import io
import os
import json
import logging
import math
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class CrestsEngine:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CrestsEngine, cls).__new__(cls, *args, **kwargs)
        return cls._instance
        
    def __init__(self):
        if hasattr(self, 'initialized') and self.initialized:
            return
            
        # Rutas a los datos
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.catalog_path = os.path.join(base_dir, 'data', 'crests', 'crests_catalog.json')
        self.embeddings_path = os.path.join(base_dir, 'data', 'crests', 'crests_embeddings.npy')
        
        self.catalog: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = np.array([])
        
        self.load_data()
        
        # Parámetros logísticos (Semana 1 spec)
        self.tau = 0.82
        self.T = 0.04
        
        self.initialized = True
        
    def load_data(self):
        try:
            with open(self.catalog_path, 'r', encoding='utf-8') as f:
                self.catalog = json.load(f)
                
            self.embeddings = np.load(self.embeddings_path)
            logger.info(f"CrestsEngine: Cargados {len(self.catalog)} items de catálogo y {self.embeddings.shape} embeddings.")
        except Exception as e:
            logger.error(f"Error inicializando CrestsEngine: {e}")
            raise RuntimeError(f"Error crítico de inicialización: {e}")
            
    def compute_euclidean_metric(self, raw_cosine: float) -> float:
        """ Calcula métrica euclidiana a partir de similitud coseno de vectores normalizados L2 """
        val = 2.0 * (1.0 - raw_cosine)
        if val < 0.0:
            val = 0.0
        return math.sqrt(val)
        
    def calibrate_similarity(self, raw_cosine: float) -> float:
        """ Aplica función logística inversa para estirar similitud y evitar el 'efecto cono' """
        try:
            exp_val = math.exp(-(raw_cosine - self.tau) / self.T)
            calibrated = 1.0 / (1.0 + exp_val)
        except OverflowError:
            # Si el exponente es muy grande (coseno muy bajo), el valor es prácticamente 0
            calibrated = 0.0
            
        return calibrated * 100.0
        



    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """ 
        1. Decodificar imagen y convertir a RGB.
        2. Aplicar padding centrado a 224x224.
        3. En este MVP (Zero-Cost backend sin GPU), como el modelo ONNX final 
           no está presente, generamos un embedding determinista basado en los píxeles
           para poder probar consistencia en las consultas.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert("RGB")
            # Resize a 224x224 con padding para mantener relación de aspecto (fondo blanco)
            image = ImageOps.pad(image, (224, 224), color=(255, 255, 255))
            
            # Hash rudimentario basado en la suma de píxeles para tener una semilla consistente
            arr = np.array(image)
            seed_val = int(np.sum(arr) % (2**32 - 1))
            np.random.seed(seed_val)
        except Exception as e:
            logger.warning(f"Error procesando con Pillow, usando fallback de bytes: {e}")
            np.random.seed(len(image_bytes))

        q = np.random.randn(256).astype(np.float32)
        q = q / np.linalg.norm(q)
        return q

    def search(self, query_bytes: bytes, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.embeddings.size == 0 or len(self.catalog) == 0:
            return []
            
        # Simular feature extraction
        query_vector = self.preprocess_image(query_bytes)
        
        # Producto punto matricial para coseno
        similarities = np.dot(self.embeddings, query_vector)
        
        # Obtener los índices de los top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            raw_cosine = float(similarities[idx])
            
            # Ajustar a rango [-1, 1] por errores de punto flotante
            raw_cosine = max(min(raw_cosine, 1.0), -1.0)
            
            sim_score = self.calibrate_similarity(raw_cosine)
            distance = self.compute_euclidean_metric(raw_cosine)
            
            item = self.catalog[idx].copy()
            item["similarity_score"] = sim_score
            item["distance"] = distance
            results.append(item)
            
        return results

crests_engine = CrestsEngine()
