# 📅 Especificación Técnica Semanal — Semana 3: Thiago Gomez Kehler
## Rol: Tech Lead / Arquitecto de Software / Diseñador UI-UX Principal

> **Ciclo:** Sprint 3 — Núcleo de Texto G3 (PTAH-Jurídico) y Núcleo de Audio G1 (Eventos Acústicos)  
> **Ubicación:** `docs/plan_ia_interactiva/semana_3_plan_thiago.md`  

---

## 🎯 1. Objetivos Técnicos del Sprint

1. **Diseñar la Experiencia Interactiva de PTAH-Jurídico (G3 Texto):** Transformar el tab "Texto" del dial de `Laboratorio.tsx` en un buscador semántico de jurisprudencia y normativa del GIBD, con barra de consulta en lenguaje natural, sugerencias precargadas y tarjetas de artículos con similitud conceptual.
2. **Diseñar el Laboratorio de Audio G1 (Sonido):** Crear la interfaz del tab "Sonido", incorporando un reproductor de audio de alta fidelidad y un visualizador de espectrograma dinámico (ondas y frecuencias en naranja `#FF5500` sobre negro OLED `#0A0A0A`).
3. **Auditar la Inferencia en CPU para NLP y Audio:** Asegurar que tanto el modelo de embeddings textuales como la extracción de espectrogramas Mel corran en tiempo real (<200 ms) sin requerir GPUs ni APIs pagas.
4. **Coherencia del Dial Giratorio:** Verificar que al girar el selector físico a "Texto" o "Sonido", el feedback háptico y sonoro mecánico se ejecute con precisión.

---

## 📐 2. Contratos y Especificaciones Técnicas

### A. Contrato de Datos para PTAH-Jurídico (`frontend/src/types/ptah.ts`)
```typescript
export interface PtahArticleResult {
  id: string;
  article_number: string;
  statute_title: string;    // Ej: "Ley 14.346 - Protección Animal / Marcas de Señalamiento"
  jurisdiction: string;      // Ej: "Nacional - Argentina" o "Provincial - Entre Ríos"
  excerpt: string;          // Texto del artículo con fragmentos relevantes
  full_content: string;
  similarity_score: number; // Porcentaje semántico [0.0, 100.0]
  relevance_badge: 'Alta' | 'Media' | 'Baja';
  key_terms_matched: string[];
}

export interface PtahQueryPayload {
  query: string;
  top_k?: number;
}
```

### B. Contrato de Datos para Audio G1 (`frontend/src/types/audio.ts`)
```typescript
export interface AudioEventMatch {
  id: string;
  event_name: string;        // Ej: "Sirena de Emergencia", "Vidrio Roto", "Motor en Ralentí"
  category: 'Eventos Acústicos' | 'Música' | 'Urbano';
  audio_url: string;
  duration_seconds: number;
  similarity_score: number;  // Porcentaje de coincidencia espectral
  distance: number;          // Distancia de coseno en espacio de embeddings de audio
  spectrogram_url?: string;  // Miniatura de imagen del espectrograma Mel
}
```

### C. Directrices de Diseño Flat para Texto y Audio
* **Buscador Semántico PTAH:**
  * Input de consulta tipo cápsula alargada (`rounded-full bg-[#18181B] border border-white/15 px-6 py-4 text-white placeholder-zinc-500 focus:border-[#FF5500]`).
  * Pills de consultas sugeridas rápidas (`rounded-full bg-zinc-800 text-zinc-300 hover:bg-[#FF5500] hover:text-white transition-all text-xs px-4 py-1.5`).
* **Visualizador de Espectrograma:**
  * Contenedor orgánico `rounded-[2rem] bg-[#121214] border border-white/10 p-6 overflow-hidden`.
  * Barras de frecuencia renderizadas en Canvas 2D con gradiente lineal vertical (`#FF5500` a `#FFAA00`).
  * Botones de reproducción circulares (`rounded-full w-12 h-12 bg-[#FF5500] text-white flex items-center justify-center`).

---

## 📋 3. Tareas Específicas de Thiago

- [ ] **T3.1.1 (Especificación UI/UX de Texto):** Diseñar la vista de resultados de PTAH-Jurídico en Figma/código, con resaltado visual de términos clave y acordeón para expandir el texto completo del artículo.
- [ ] **T3.1.2 (Especificación UI/UX de Audio):** Diseñar el panel de audio con waveform y espectrograma sincronizados en tiempo real mediante `requestAnimationFrame`.
- [ ] **T3.1.3 (Auditoría de Inferencia CPU de Renato):** Comprobar que el motor de texto no consuma más de 400 MB de RAM y que el procesamiento de audio con librosa/scipy sea instantáneo.
- [ ] **T3.1.4 (Code Review de Emmanuel):** Revisar que el Canvas de audio gestione el ciclo de vida del `AudioContext` de la Web Audio API, reanudándolo automáticamente al primer toque del usuario.
- [ ] **T3.1.5 (Quality Gate Semanal):** Ejecutar el protocolo de Verificación y Validación en 2 niveles antes de autorizar la integración en la rama `dev`.

---

## 🛡️ 4. Plan de Verificación y Validación (V&V)

### Nivel 1 — Pre-Verificación Autónoma del Agente (Automated Testing)
1. **Compilación Limpia:**
   ```bash
   cd frontend && npm run build
   ```
2. **Validación de Consultas PTAH:**
   * Ejecutar consulta automatizada a `/api/v1/inference/ptah-query` con el texto "prohibición de marcar ganado ajeno".
   * *Aserción:* El resultado #1 debe tener `similarity_score >= 82.0%` y pertenecer a la normativa de marcas y señales del dataset del GIBD.
3. **Validación de Espectrogramas de Audio:**
   * Enviar un archivo `.wav` de prueba al endpoint `/api/v1/inference/audio-match`.
   * *Aserción:* Retornar lista de 5 eventos acústicos con distancias válidas y latencia $< 200\text{ ms}$.

### Nivel 2 — Validación Obligatoria del Usuario (Human-in-the-Loop)
**Protocolo para Thiago:**
1. En el Laboratorio, rotar el dial físico a la opción **"Texto"**:
   - Escribir una consulta jurídica en lenguaje natural (ej. "responsabilidad por daños de animales sueltos").
   - *Validación:* Verificar que se desplieguen los artículos pertinentes, con badges claros y fragmentos legibles.
2. Rotar el dial a la opción **"Sonido"**:
   - Cargar un audio de prueba o hacer clic en una muestra preconfigurada (ej. "Vidrio roto").
   - Clic en Reproducir.
   - *Validación:* La onda sonora y el espectrograma deben moverse en tiempo real al ritmo del sonido con colores naranja y negro OLED.
3. **Aprobación Formal:** Si la respuesta de texto y audio es científicamente auténtica y fluida, Thiago aprueba el Sprint 3.
