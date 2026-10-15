# 📅 Especificación Técnica Semanal — Semana 2: Renato Bonín
## Rol: Backend & AI Engineer

> **Ciclo:** Sprint 2 — Motor de Reducción Dimensional G5 (PCA/UMAP) y Expansión de Catálogos  
> **Ubicación:** `docs/plan_ia_interactiva/semana_2_plan_renato.md`  
> **Directriz Central:** Cálculo matemático riguroso en CPU, proyecciones estables con scikit-learn, serialización compacta.

---

## 🎯 1. Objetivos Técnicos del Sprint

> ⚠️ **Deuda Técnica Reprogramada de Semana 1 (Regla de Oro GIBD):**
> Al no haberse completado en la Semana 1, Renato asume como prioridad inmediata el desarrollo del Motor de Inferencia de Escudos (`crests_engine.py`), la calibración logística ($\tau=0.82$, $T=0.04$), los endpoints `/api/v1/inference/soccer-crests` y `/api/v1/crests/catalog`, y `cloudinary_service.py`.

1. **Deuda Técnica S1 — Motor de Inferencia y Calibración de Escudos:** Implementar `crests_engine.py` en CPU con embeddings normalizados L2 (<120 ms) y calibración por temperatura para integración con el frontend.
2. **Crear el Motor de Reducción Dimensional (`projection_engine.py`):** Implementar PCA ajustado sobre los embeddings de los catálogos del GIBD para proyectar tensores de 256/512 dimensiones hacia $\mathbb{R}^3$ y $\mathbb{R}^2$.
3. **Proyección al Vuelo de Consultas:** Implementar el método `.project_query(query_vector)` para ubicar instantáneamente en el espacio tridimensional la imagen que el usuario acaba de subir sin necesidad de reentrenar el modelo PCA.
4. **Exponer Endpoints de Proyección:** Crear `/api/v1/embeddings/projection` y `/api/v1/embeddings/project-point`.
5. **Indexar Nuevos Datasets de Visión:** Incorporar los activos de Tatuajes y Pinturas disponibles en el Vault del GIBD, generando sus matrices `.npy` y catálogos `.json`.

---

## 📐 2. Especificación Técnica y Código de Implementación

### A. Estructura de Archivos a Crear / Modificar
* `backend/app/services/projection_engine.py` (Nuevo)
* `backend/app/data/projections/crests_pca_3d.pkl` (Modelo entrenado serializado)
* `backend/app/data/projections/brands_pca_3d.pkl`
* `backend/app/services/tattoo_engine.py` (Opcional o unificado en `VisionCatalogManager`)
* `backend/app/main.py` (Nuevas rutas bajo el tag "Reducción Dimensional")

### B. Implementación de `projection_engine.py`

```python
import numpy as np
import os
import joblib
from typing import Dict, Any, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

class ProjectionEngine:
    _instances: Dict[str, 'ProjectionEngine'] = {}

    def __init__(self, dataset_name: str, embeddings: np.ndarray, catalog: List[Dict[str, Any]]):
        self.dataset_name = dataset_name
        self.catalog = catalog
        self.embeddings = embeddings
        self.pca_3d = PCA(n_components=3, random_state=42)
        self.pca_2d = PCA(n_components=2, random_state=42)
        self.scaler_3d = MinMaxScaler(feature_range=(-1.0, 1.0))
        self.scaler_2d = MinMaxScaler(feature_range=(-1.0, 1.0))
        self._fit_models()

    def _fit_models(self):
        # Ajustar PCA 3D y 2D
        raw_3d = self.pca_3d.fit_transform(self.embeddings)
        self.coords_3d = self.scaler_3d.fit_transform(raw_3d)

        raw_2d = self.pca_2d.fit_transform(self.embeddings)
        self.coords_2d = self.scaler_2d.fit_transform(raw_2d)

    def get_full_projection_payload(self) -> Dict[str, Any]:
        nodes = []
        for idx, item in enumerate(self.catalog):
            nodes.append({
                "id": item.get("id"),
                "item_code": item.get("item_code", item.get("brand_code", str(idx))),
                "label": item.get("name"),
                "category": item.get("jurisdiction", "General"),
                "thumbnail_url": item.get("image_url"),
                "coordinates_3d": [round(float(c), 4) for c in self.coords_3d[idx]],
                "coordinates_2d": [round(float(c), 4) for c in self.coords_2d[idx]],
                "cluster_id": int(idx % 6) # O clustering K-Means real
            })

        return {
            "dataset": self.dataset_name,
            "reduction_method": "PCA",
            "total_nodes": len(nodes),
            "variance_explained_ratio": [round(float(v), 4) for v in self.pca_3d.explained_variance_ratio_],
            "nodes": nodes
        }

    def project_single_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        vec_2d = vector.reshape(1, -1)
        raw_3d = self.pca_3d.transform(vec_2d)
        norm_3d = self.scaler_3d.transform(raw_3d)[0]

        raw_2d = self.pca_2d.transform(vec_2d)
        norm_2d = self.scaler_2d.transform(raw_2d)[0]

        return {
            "coordinates_3d": [round(float(c), 4) for c in norm_3d],
            "coordinates_2d": [round(float(c), 4) for c in norm_2d]
        }
```

### C. Endpoints FastAPI en `backend/app/main.py`
```python
@app.get("/api/v1/embeddings/projection", tags=["Reducción Dimensional G5"])
async def get_dataset_projection(
    dataset: str = Query(default="soccer_crests", regex="^(soccer_crests|brands|tattoos)$")
):
    engine = get_projection_engine_for(dataset)
    return engine.get_full_projection_payload()
```

---

## 📋 3. Tareas Específicas de Renato

- [ ] **T2.2.1 (Módulo de Proyección):** Implementar `ProjectionEngine` con ajuste automático sobre las matrices `.npy` existentes.
- [ ] **T2.2.2 (Persistencia de Modelos):** Guardar los objetos `PCA` entrenados con `joblib` para acelerar el arranque en frío del servidor.
- [ ] **T2.2.3 (Proyección de Query en Tiempo Real):** Conectar el endpoint de inferencia de escudos para que, opcionalmente, retorne las coordenadas `coordinates_3d` de la imagen consultada por el usuario.
- [ ] **T2.2.4 (Incorporar Tatuajes/Pinturas):** Pre-indexar las imágenes disponibles en `Contexto/GIBD_Obsidian_Vault/50 - Archivo Histórico/G4_Imágenes/` para generar sus embeddings y catálogo.
- [ ] **T2.2.5 (Tests Unitarios):** Redactar pruebas con `pytest` asegurando que la varianza explicada acumulada sea mayor al $30\%$ y las coordenadas estén acotadas en $[-1.0, 1.0]$.

---

## 🛡️ 4. Plan de Verificación y Validación (V&V)

### Nivel 1 — Pre-Verificación Autónoma del Agente (Automated Testing)
1. **Compilación y Tests Pytest:**
   ```bash
   pytest backend/tests/test_projection.py -v
   ```
   *Validaciones obligatorias:*
   * `total_nodes == 150` para `soccer_crests` y `80` para `brands`.
   * Todos los valores de `coordinates_3d` satisfacen $-1.0 \le x_i \le 1.0$.
   * Tiempo de respuesta del endpoint de proyección completo $< 80\text{ ms}$.
2. **Prueba de Proyección de Vector Arbitrario:**
   * Enviar un vector aleatorio normalizado; el método `.project_single_vector()` debe retornar coordenadas válidas en $< 5\text{ ms}$.

### Nivel 2 — Validación Obligatoria del Usuario (Human-in-the-Loop)
**Protocolo para Renato antes del pase a Thiago:**
1. Ejecutar en terminal:
   ```bash
   curl "http://localhost:8000/api/v1/embeddings/projection?dataset=brands"
   ```
2. Inspeccionar visualmente que el JSON contenga el listado estructurado de nodos con sus miniaturas y coordenadas sin errores de tipo.
3. Notificar al Tech Lead para la integración con el visualizador 3D en el frontend.
