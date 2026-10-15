# 📅 Especificación Técnica Semanal — Semana 4: Renato Bonín
## Rol: Backend & AI Engineer

> **Ciclo:** Sprint 4 — Motor G2 (Video Retrieval y Detección de Eventos por Keyframes) y Hardening de Producción  
> **Ubicación:** `docs/plan_ia_interactiva/semana_4_plan_renato.md`  
> **Directriz Central:** Muestreo temporal liviano en CPU, extracción de keyframes sin GPUs, blindaje de endpoints.

---

## 🎯 1. Objetivos Técnicos del Sprint

1. **Implementar el Motor de Video G2 (`video_engine.py`):** Crear un pipeline de análisis temporal en CPU que extraiga keyframes a razón de 1 fps, calcule vectores de características morfológicas por frame y detecte picos de dinamismo ("cambio abrupto de dinámica" para caídas, movimientos bruscos o eventos).
2. **Endpoint de Video Retrieval:** Exponer `/api/v1/inference/video-retrieval` que reciba un clip corto (MP4/WEBM de hasta 15 MB) y devuelva la línea temporal de dinamismo junto con clips o fragmentos similares del archivo del GIBD.
3. **Hardening y Documentación OpenAPI:** Completar la documentación interactiva en `/docs` con descripciones exhaustivas, códigos HTTP estándar (200, 400, 422, 500) y modelos Pydantic validados.
4. **Pruebas de Carga en CPU:** Asegurar que las peticiones concurrentes a los distintos motores no saturen los núcleos de la CPU.

---

## 📐 2. Especificación Técnica y Código de Implementación

### A. Estructura de Archivos a Crear / Modificar
* `backend/app/services/video_engine.py` (Nuevo)
* `backend/app/data/video/video_samples_catalog.json` (Clips de prueba del GIBD)
* `backend/tests/test_e2e_all_engines.py` (Suite completa de integración)
* `backend/app/main.py` (Registrar rutas de video y middlewares de producción)

### B. Implementación de `video_engine.py` (Extracción de Keyframes en CPU)

```python
import cv2
import numpy as np
import tempfile
import os
from typing import Dict, Any, List

class VideoEngine:
    _instance = None

    @classmethod
    def get_instance(cls) -> 'VideoEngine':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def analyze_video_stream(self, video_bytes: bytes, max_duration_sec: int = 30) -> Dict[str, Any]:
        """
        Procesa el archivo de video en CPU:
        1. Extrae frames a 1 fps.
        2. Calcula el gradiente de cambio inter-frame (diferencia de movimiento).
        3. Identifica timestamps de picos dinámicos (eventos abruptos).
        """
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            step = int(max(1, fps)) # Muestrear 1 frame por segundo
            prev_gray = None
            timeline = []
            frame_idx = 0
            
            while cap.isOpened() and frame_idx < (fps * max_duration_sec):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % step == 0:
                    timestamp_sec = round(frame_idx / fps, 1)
                    # Resize a 128x128 para cálculo ultrarrápido
                    small = cv2.resize(frame, (128, 128))
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    
                    if prev_gray is not None:
                        diff = cv2.absdiff(gray, prev_gray)
                        dynamism_score = float(np.mean(diff) / 255.0 * 100.0)
                    else:
                        dynamism_score = 0.0
                        
                    prev_gray = gray
                    is_peak = dynamism_score > 35.0 # Umbral de dinamismo abrupto
                    
                    timeline.append({
                        "timestamp": timestamp_sec,
                        "dynamism_score": round(dynamism_score, 1),
                        "event_flag": "Pico Dinámico" if is_peak else "Normal"
                    })
                    
                frame_idx += 1
                
            cap.release()
            return {
                "duration_analyzed_sec": round(frame_idx / fps, 1),
                "fps": round(fps, 1),
                "timeline": timeline,
                "peak_events_detected": sum(1 for t in timeline if t["event_flag"] == "Pico Dinámico")
            }
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
```

---

## 📋 3. Tareas Específicas de Renato

- [ ] **T4.2.1 (Motor de Video en CPU):** Desarrollar `VideoEngine` utilizando OpenCV en modo headless (`opencv-python-headless`) para evitar dependencias de GUI en el servidor.
- [ ] **T4.2.2 (Endpoint de Video):** Exponer `/api/v1/inference/video-retrieval` recibiendo el archivo de video y retornando la serie temporal de dinamismo.
- [ ] **T4.2.3 (Suite E2E Pytest):** Crear `backend/tests/test_e2e_all_engines.py` que compruebe la salud e inferencia de todos los motores (Marcas, Escudos, Espacio 3D, PTAH Texto, Audio, Video).
- [ ] **T4.2.4 (Hardening de Uvicorn):** Configurar límites de tamaño de subida (máx. 25 MB) y sanitización de nombres de archivo temporales.
- [ ] **T4.2.5 (Documentación Técnica):** Exportar el esquema OpenAPI y redactar la sección de backend para el reporte final de entrega.

---

## 🛡️ 4. Plan de Verificación y Validación (V&V)

### Nivel 1 — Pre-Verificación Autónoma del Agente (Automated Testing)
1. **Ejecución de Suite E2E Completa:**
   ```bash
   pytest backend/tests/ -v --durations=5
   ```
   *Criterio:* 100% de tests aprobados en menos de 15 segundos.
2. **Verificación de Limpieza de Archivos Temporales:**
   * Comprobar que tras procesar 5 videos consecutivamente, no queden archivos `.mp4` huérfanos en el directorio de temporales del sistema (`tempfile.gettempdir()`).

### Nivel 2 — Validación Obligatoria del Usuario (Human-in-the-Loop)
**Protocolo para Renato antes de entrega a Thiago:**
1. Ejecutar prueba cURL enviando un video MP4 de 5 segundos:
   ```bash
   curl -X POST -F "file=@ruta/video_prueba.mp4" "http://localhost:8000/api/v1/inference/video-retrieval"
   ```
2. Verificar que la respuesta contenga la lista ordenada de marcas de tiempo con sus valores de dinamismo.
3. Notificar a Thiago con el reporte de aprobación de tests para proceder a la auditoría final.
