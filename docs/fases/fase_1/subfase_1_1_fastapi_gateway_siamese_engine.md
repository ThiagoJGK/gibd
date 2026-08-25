# Plan de Subfase 1.1: FastAPI Gateway y Engine de Inferencia de Marcas de Ganado

## 1. Información General
- **Fase Perteneciente:** FASE 1 - Core de IA de Marcas de Ganado, Gateway Python y Wrapper Embebido
- **Subfase:** 1.1
- **Desarrollador Asignado:** Renato Bonín

---

## 2. Objetivo de la Subfase
Crear el pipeline de inferencia en el backend Python (FastAPI Gateway) para recibir imágenes de marcas de ganado, aplicar la normalización necesaria (crop, resize, blur/flat filtering) y ejecutar el cálculo de similitud para retornar un JSON estructurado con el Top-K de marcas más similares y sus distancias métricas relativas.

---

## 3. Tareas Técnicas Detalladas

- [ ] **Tarea 1.1.1: Creación del Módulo Inferencia en Python**
  - Archivo: `backend/app/services/siamese_engine.py`
  - Implementar la función `process_brand_inference(image_bytes: bytes, top_k: int)` que realice el pre-procesamiento y extracción de vectores de similitud.
- [ ] **Tarea 1.1.2: Endpoint POST de Inferencia en FastAPI**
  - Archivo: `backend/app/main.py`
  - Crear la ruta `/api/v1/inference/siamese-brands` que reciba `UploadFile` (multipart/form-data) y parámetro opcional `top_k: int = 10`.
  - Retornar respuesta estructurada:
    ```json
    {
      "status": "success",
      "query_info": { "filename": "marca_test.png", "top_k": 10 },
      "results": [
        { "id": "brand_01", "name": "Marca La Esperanza", "similarity_score": 94.8, "distance": 0.052, "image_url": "https://..." }
      ]
    }
    ```
- [ ] **Tarea 1.1.3: Verificación de CORS y Dependencias**
  - Archivos: `backend/requirements.txt`, `backend/app/main.py`
  - Asegurar soporte de `Pillow`, `torch`/`torchvision` o numpy/scipy según corresponda.

---

## 4. Plan de Comprobación, Validación y Testing

### A. Ejecución del Servidor Backend
```powershell
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

### B. Pruebas de API
1. Enviar petición GET a `http://localhost:8000/api/v1/health` -> Verificar `status: healthy`.
2. Enviar petición POST a `http://localhost:8000/api/v1/inference/siamese-brands` adjuntando una imagen -> Verificar respuesta HTTP 200 con el JSON de resultados Top-K.

---

## 5. Criterios de Aceptación de la Subfase
- Endpoint operativo sin excepciones 500.
- Retorno de JSON estandarizado con campo `similarity_score` y `distance`.
- Logs de consola limpios reportando tiempo de inferencia en ms.
