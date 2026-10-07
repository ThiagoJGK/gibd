import os
import time
import pytest
from PIL import Image
import io
import numpy as np

from app.services.crests_engine import CrestsEngine, crests_engine
from app.services.cloudinary_service import upload_query_image

# Utilidad para crear una imagen sintética en memoria
def create_test_image(color=(255, 0, 0), size=(200, 200)) -> bytes:
    img = Image.new("RGB", size, color=color)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_health_check_crests_count():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["crests_indexed"] == 150

def test_python_compilation():
    """ Verifica que los módulos clave se puedan importar correctamente """
    assert crests_engine is not None
    assert callable(upload_query_image)

def test_inference_latency_cpu():
    """ 
    Latencia < 150 ms en CPU (promedio de 10 ejecuciones).
    """
    img_bytes = create_test_image()
    latencies = []
    
    # Warmup
    crests_engine.search(img_bytes, top_k=5)
    
    for _ in range(10):
        start = time.time()
        crests_engine.search(img_bytes, top_k=5)
        end = time.time()
        latencies.append((end - start) * 1000) # ms
        
    avg_latency = sum(latencies) / len(latencies)
    assert avg_latency < 150.0, f"Latencia promedio {avg_latency:.2f}ms excede 150ms"

def test_self_query_identity():
    """ 
    Consulta con una imagen idéntica retorna similarity_score >= 98.0 y distance <= 0.05
    En el backend real esto se hace evaluando el accuracy de la CNN.
    Aquí verificamos la rigurosidad matemática inyectando un target exacto.
    """
    img_bytes = create_test_image()
    
    # Guardamos estado
    orig_catalog = crests_engine.catalog.copy()
    orig_embeddings = crests_engine.embeddings.copy()
    
    # Inyectamos un target exacto
    q = crests_engine.preprocess_image(img_bytes)
    crests_engine.embeddings = np.vstack([crests_engine.embeddings, q])
    
    mock_item = {"id": "test_001", "name": "Exact Match"}
    crests_engine.catalog.append(mock_item)
    
    try:
        results = crests_engine.search(img_bytes, top_k=1)
        assert len(results) > 0
        top_match = results[0]
        
        assert top_match["similarity_score"] >= 98.0
        assert top_match["distance"] <= 0.05
    finally:
        # Restaurar
        crests_engine.catalog = orig_catalog
        crests_engine.embeddings = orig_embeddings

def test_heterogeneous_query_discrimination():
    """ 
    Consulta con imagen muy diferente retorna similarity_score <= 25.0
    Con vectores pseudo-ortogonales en 256D, el coseno tiende a ~0.
    Calibrando en base a tau=0.82, esto descarta el ruido.
    """
    img_bytes = create_test_image(color=(0, 255, 0)) 
    results = crests_engine.search(img_bytes, top_k=5)
    if len(results) > 0:
        assert results[0]["similarity_score"] <= 25.0

def test_cloudinary_graceful_fallback():
    """ operación correcta sin credenciales configuradas """
    img_bytes = create_test_image()
    res = upload_query_image(img_bytes)
    assert res is None or isinstance(res, str)
