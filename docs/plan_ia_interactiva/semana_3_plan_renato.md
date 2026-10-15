# 📅 Especificación Técnica Semanal — Semana 3: Renato Bonín
## Rol: Backend & AI Engineer

> **Ciclo:** Sprint 3 — Motores de Inferencia en CPU para PTAH-Jurídico (Texto) y Eventos Acústicos (Sonido)  
> **Ubicación:** `docs/plan_ia_interactiva/semana_3_plan_renato.md`  
> **Directriz Central:** Modelos compactos ONNX / NumPy en CPU, latencia $< 200\text{ ms}$, autenticidad de datasets UTN.

---

## 🎯 1. Objetivos Técnicos del Sprint

1. **Implementar el Motor Semántico PTAH (`ptah_engine.py`):** Cargar los artículos normativos y las 300 consultas del archivo de investigación del GIBD, vectorizándolos con un modelo compacto en español (BGE-M3 o MiniLM exportado a ONNX/NumPy para ejecución liviana en CPU).
2. **Implementar el Motor de Audio G1 (`audio_engine.py`):** Desarrollar extracción de características espectrales (Mel-Spectrogram y coeficientes MFCC) con scipy/numpy para comparar firmas de audio y clasificar eventos acústicos.
3. **Exponer Endpoints en FastAPI:** Crear `/api/v1/inference/ptah-query` y `/api/v1/inference/audio-match`.
4. **Optimización de Memoria:** Garantizar que ambos servicios coexistan en memoria dentro del proceso de Uvicorn consumiendo menos de 450 MB de RAM.

---

## 📐 2. Especificación Técnica y Código de Implementación

### A. Estructura de Archivos a Crear / Modificar
* `backend/app/services/ptah_engine.py` (Nuevo)
* `backend/app/services/audio_engine.py` (Nuevo)
* `backend/app/data/ptah/ptah_corpus.json` (Dataset normativo y artículos del GIBD)
* `backend/app/data/ptah/ptah_embeddings.npy` (Matriz de embeddings precalculados)
* `backend/app/data/audio/sound_events_catalog.json` (Catálogo de eventos sonoros de prueba)
* `backend/app/data/audio/sound_signatures.npy` (Matriz de firmas acústicas MFCC)

### B. Implementación de `ptah_engine.py`

```python
import numpy as np
import json
import os
from typing import List, Dict, Any

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ptah")
CORPUS_PATH = os.path.join(DATA_DIR, "ptah_corpus.json")
INDEX_PATH = os.path.join(DATA_DIR, "ptah_embeddings.npy")

class PtahEngine:
    _instance = None

    def __init__(self):
        self.corpus: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = None
        self._load_corpus()

    @classmethod
    def get_instance(cls) -> 'PtahEngine':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_corpus(self):
        if os.path.exists(CORPUS_PATH) and os.path.exists(INDEX_PATH):
            with open(CORPUS_PATH, "r", encoding="utf-8") as f:
                self.corpus = json.load(f)
            self.embeddings = np.load(INDEX_PATH)
            # Asegurar normalización L2
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / np.maximum(norms, 1e-8)

    def search_normative(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Vectorización ligera de consulta (bag of dense n-grams o ONNX embedding)
        query_vec = self._vectorize_query(query_text)
        
        # Similitud coseno rápida
        scores = np.dot(self.embeddings, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            raw_score = float(scores[idx])
            # Calibración para texto
            pct = round(max(0.0, min(100.0, (raw_score + 1.0) / 2.0 * 100.0)), 1)
            
            item = self.corpus[idx].copy()
            item["similarity_score"] = pct
            item["relevance_badge"] = "Alta" if pct >= 75 else ("Media" if pct >= 50 else "Baja")
            results.append(item)
            
        return results

    def _vectorize_query(self, text: str) -> np.ndarray:
        # Implementación con modelo ONNX liviano o hashing denso TF-IDF preentrenado
        ...
```

### C. Implementación de `audio_engine.py` (Firmas Acústicas MFCC)

```python
import numpy as np
import io
from scipy.io import wavfile
from scipy.fftpack import dct

def extract_mfcc_signature(wav_bytes: bytes, num_ceps: int = 24) -> np.ndarray:
    """
    Extrae un vector compacto de coeficientes cepstrales de frecuencia de Mel (MFCC)
    en CPU pura con Scipy/Numpy sin requerir librerías pesadas como librosa.
    """
    samplerate, signal = wavfile.read(io.BytesIO(wav_bytes))
    if len(signal.shape) > 1:
        signal = signal.mean(axis=1) # Estéreo a mono
    
    # Pre-énfasis
    emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    
    # Framing y Windowing
    frame_size, frame_stride = int(0.025 * samplerate), int(0.01 * samplerate)
    # FFT, Filtros Mel y DCT
    ...
    feature_vector = np.mean(mfcc_matrix, axis=0) # Vector medio de 24 dimensiones
    norm = np.linalg.norm(feature_vector)
    return feature_vector / max(norm, 1e-8)
```

---

## 📋 3. Tareas Específicas de Renato

- [ ] **T3.2.1 (Indexación de PTAH):** Estructurar `ptah_corpus.json` a partir de los documentos de texto del archivo histórico del GIBD y generar los embeddings iniciales `ptah_embeddings.npy`.
- [ ] **T3.2.2 (Servicio PTAH en CPU):** Desarrollar `PtahEngine` con búsqueda matricial optimizada y tiempo de respuesta inferior a $100\text{ ms}$.
- [ ] **T3.2.3 (Servicio de Audio G1):** Desarrollar `AudioEngine` capaz de recibir un archivo de audio, extraer su vector de características y comparar por distancia euclidiana/coseno contra el catálogo acústico.
- [ ] **T3.2.4 (Endpoints FastAPI):** Registrar `/api/v1/inference/ptah-query` y `/api/v1/inference/audio-match`.
- [ ] **T3.2.5 (Tests Automatizados):** Escribir tests unitarios en `pytest` validando que las consultas devuelvan respuestas estructuradas sin excepciones de decodificación.

---

## 🛡️ 4. Plan de Verificación y Validación (V&V)

### Nivel 1 — Pre-Verificación Autónoma del Agente (Automated Testing)
1. **Tests Pytest de Texto y Audio:**
   ```bash
   pytest backend/tests/test_ptah_audio.py -v
   ```
   *Aserciones obligatorias:*
   * Consulta "marca a fuego y señal" retorna en el Top 1 un artículo de la ley de marcas ganaderas con similitud $> 75\%$.
   * Archivo de audio de prueba devuelve evento acústico correspondiente en $< 150\text{ ms}$.
2. **Auditoría de Memoria RAM:**
   * Medir consumo de proceso con `psutil`; el incremento de RAM no debe exceder los 150 MB tras cargar ambos catálogos.

### Nivel 2 — Validación Obligatoria del Usuario (Human-in-the-Loop)
**Protocolo para Renato antes de entrega a Thiago:**
1. Ejecutar prueba cURL de PTAH:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"query": "registro oficial de marcas vacunas"}' "http://localhost:8000/api/v1/inference/ptah-query"
   ```
2. Ejecutar prueba cURL de Audio con un archivo `.wav` de prueba:
   ```bash
   curl -X POST -F "file=@ruta/audio_prueba.wav" "http://localhost:8000/api/v1/inference/audio-match"
   ```
3. Verificar que los resultados reflejen con fidelidad las investigaciones del GIBD y no contengan texto roto ni fallos de formato.
4. **Notificar al Tech Lead:** Enviar reporte de latencia y respuestas JSON para su validación en la interfaz gráfica.
