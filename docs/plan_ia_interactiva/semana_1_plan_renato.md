# 📅 Especificación Técnica Semanal — Semana 1: Renato Bonín
## Rol: Backend & AI Engineer

> **Ciclo:** Sprint 1 — Motor de Inferencia de Escudos en CPU, Calibración Matemática y Servicio Cloudinary  
> **Ubicación:** `docs/plan_ia_interactiva/semana_1_plan_renato.md`  
> **Directriz Central:** Ejecución 100% sobre CPU (Zero-Cost), latencia menor a 150 ms, tipado Pydantic estricto.

---

## 🎯 1. Objetivos Técnicos del Sprint

1. **Implementar el Motor de Inferencia de Escudos (`crests_engine.py`):** Crear el servicio singleton análogo a `siamese_engine.py` para cargar los embeddings de los 150 clubes del estudio CONAISII 2023.
2. **Implementar la Calibración de Similitud por Temperatura:** Sustituir el cálculo lineal naive por la formulación logística con parámetros $\tau=0.82$ y $T=0.04$, eliminando el "efecto cono" y los porcentajes raros.
3. **Exponer Endpoints en FastAPI:** Crear `/api/v1/inference/soccer-crests` y `/api/v1/crests/catalog`.
4. **Módulo Helper de Cloudinary:** Desarrollar `backend/app/services/cloudinary_service.py` para generación de firmas seguras y validación de URLs.

---

## 📐 2. Especificación Técnica y Código de Implementación

### A. Estructura de Archivos a Crear / Modificar
* `backend/app/services/crests_engine.py` (Nuevo)
* `backend/app/services/cloudinary_service.py` (Nuevo)
* `backend/app/main.py` (Registrar routers y schemas)
* `backend/app/data/crests/crests_catalog.json` (Catálogo con metadatos de los 150 clubes)
* `backend/app/data/crests/crests_embeddings.npy` (Matriz NumPy precalculada de $150 \times 256$ o $150 \times 512$)

### B. Lógica Matemática en `backend/app/services/crests_engine.py`

```python
import numpy as np
import json
import os
from typing import List, Dict, Any
from PIL import Image, ImageOps

TAU_THRESHOLD = 0.82
TEMPERATURE = 0.04

def calibrate_similarity(raw_cosine: float) -> float:
    """
    Transforma la similitud coseno cruda mediante una función logística de temperatura.
    Elimina falsos positivos concentrados en [0.70, 0.80] y ensancha la discriminación.
    """
    exponent = - (raw_cosine - TAU_THRESHOLD) / TEMPERATURE
    # Evitar overflow numérico en exp
    clamped_exp = max(-50.0, min(50.0, exponent))
    calibrated = 100.0 / (1.0 + np.exp(clamped_exp))
    return round(float(np.clip(calibrated, 0.0, 100.0)), 1)

def compute_euclidean_metric(raw_cosine: float) -> float:
    """
    Calcula la distancia métrica real entre dos vectores unitarios:
    d = sqrt(2 * (1 - cos(theta)))
    """
    cos_clamped = max(-1.0, min(1.0, float(raw_cosine)))
    return round(float(np.sqrt(max(0.0, 2.0 * (1.0 - cos_clamped)))), 4)
```

### C. Esquema Pydantic para el Endpoint
En `backend/app/main.py`:
```python
class CrestMatchResponse(BaseModel):
    id: str
    item_code: str
    name: str
    filename: str
    image_url: str
    similarity_score: float # [0.0, 100.0] calibrado
    distance: float         # Distancia métrica euclidiana
    metadata: Dict[str, Any]

@app.post("/api/v1/inference/soccer-crests", response_model=List[CrestMatchResponse], tags=["Visión Computacional"])
async def match_soccer_crests(
    file: UploadFile = File(None),
    image_url: Optional[str] = Query(default=None, description="URL pública de Cloudinary"),
    top_k: int = Query(default=10, ge=1, le=50)
):
    # Soporta tanto subida directa de multipart como URL remota proveniente de Cloudinary
    ...
```

### D. Servicio Cloudinary (`cloudinary_service.py`)
```python
import os
import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

def upload_query_image(image_bytes: bytes, folder: str = "gibd/consultas") -> str:
    response = cloudinary.uploader.upload(
        image_bytes,
        folder=folder,
        transformation=[
            {"width": 224, "height": 224, "crop": "pad", "background": "white"},
            {"fetch_format": "webp", "quality": "auto"}
        ]
    )
    return response.get("secure_url")
```

---

## 📋 3. Tareas Específicas de Renato

- [ ] **T1.2.1 (Indexación de Escudos):** Crear `backend/app/data/crests/` y generar la matriz inicial de embeddings normalizados L2 `crests_embeddings.npy` a partir de los 150 activos del estudio CONAISII 2023.
- [ ] **T1.2.2 (Motor Singleton):** Desarrollar `CrestsEngine` con carga en memoria y método `.search(query_bytes, top_k)`.
- [ ] **T1.2.3 (Calibración Rigurosa):** Implementar la función de contraste sigmoidal y asegurar que imágenes de consultas sintéticas no relacionadas no superen el $20\%$ de score.
- [ ] **T1.2.4 (Endpoints FastAPI):** Registrar `/api/v1/inference/soccer-crests` y `/api/v1/crests/catalog`.
- [ ] **T1.2.5 (Integración Cloudinary Backend):** Implementar `cloudinary_service.py` con manejo de excepciones si las claves de entorno no están provistas (modo fallback local).

---

## 🛡️ 4. Plan de Verificación y Validación (V&V)

### Nivel 1 — Pre-Verificación Autónoma del Agente (Automated Testing)
1. **Test de Compilación Python:**
   ```bash
   python -m py_compile backend/app/services/crests_engine.py backend/app/services/cloudinary_service.py
   ```
2. **Test Automatizado de Latencia e Inferencia:**
   Crear `backend/tests/test_crests_inference.py` y ejecutar:
   ```bash
   pytest backend/tests/test_crests_inference.py -v
   ```
   *Aserciones del Test:*
   * Latencia de inferencia $< 150\text{ ms}$.
   * Una consulta consigo misma debe retornar `similarity_score >= 98.0` y `distance <= 0.05`.
   * Una consulta con una marca de ganado (imagen heterogénea) debe retornar `similarity_score <= 25.0`.
3. **Health Check:**
   Validar respuesta `GET http://localhost:8000/api/v1/health` conteniendo `"crests_indexed": 150`.

### Nivel 2 — Validación Obligatoria del Usuario (Human-in-the-Loop)
**Protocolo para Renato antes de solicitar revisión del Tech Lead:**
1. Levantar el servidor Uvicorn:
   ```bash
   uvicorn backend.app.main:app --reload --port 8000
   ```
2. Ejecutar prueba con cURL pasando una imagen real de un club:
   ```bash
   curl -X POST -F "file=@ruta/a/escudo_prueba.png" "http://localhost:8000/api/v1/inference/soccer-crests?top_k=5"
   ```
3. Verificar que la salida JSON sea un array limpio con nombres reales de clubes, distancias métricas y porcentajes sin anomalías.
4. **Notificar a Thiago:** Enviar el snippet del resultado para que el Tech Lead ejecute la validación visual en el frontend.
