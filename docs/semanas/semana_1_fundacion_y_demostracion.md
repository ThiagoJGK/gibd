# 📅 Plan de Trabajo: Semana 1 — Fundación Real y Demostración

> **Ciclo:** Sprint 1  
> **Objetivo Central:** Construir la base real de IA de Marcas de Ganado sobre CPU (Zero-Cost), sanear el frontend y presentar un prototipo funcional y estético al equipo.  
> **Puesta en Común:** Viernes (Reunión Semanal del Equipo)  

---

## 1. Contexto de la Semana: Sprint de Liderazgo ("Show, Don't Tell")

En esta primera semana, **Thiago (Tech Lead & Diseñador) asume la ejecución completa** para resolver de raíz la arquitectura de inferencia, eliminar simulaciones por hash o patentes de autos y sentar el estándar estético del **Design System Flat** que el equipo deberá respetar en las semanas siguientes.

---

## 2. Distribución de Tareas por Integrante

### 👑 Thiago Gomez Kehler (Tech Lead & Diseñador) — Responsable Ejecutivo
- [ ] **Tarea 1.1: Motor de Inferencia Real sobre Marcas del GIBD (Zero-Cost)**
  - Archivo: `backend/app/services/siamese_engine.py` y `backend/app/main.py`.
  - Implementar el preprocesamiento MBR del notebook de investigación (recorte de bordes blancos/negros, transparencia a blanco, redimensionamiento a 224x224, blur gaussiano).
  - Pre-indexar el catálogo de marcas del GIBD (`Logos-consultas80` o las marcas de CACIC 2023) generando la matriz de embeddings (`embeddings_index.npy`) y el catálogo de metadatos (`brands_catalog.json`).
  - Calcular la similitud del coseno real en memoria en CPU (<150 ms de latencia).
- [ ] **Tarea 1.2: Saneamiento del Frontend y Componente Wrapper**
  - Archivos: `frontend/public/favicon.svg`, `frontend/src/pages/Laboratorio.tsx`, `backend/demo_python_app.py`.
  - Revertir las modificaciones accidentales de `favicon.svg`.
  - Limpiar `Laboratorio.tsx`: remover los botones hardcodeados a `localhost:8010` y `localhost:9999`.
  - Reubicar `backend/demo_python_app.py` en `backend/demos/`.
  - Integrar `PythonAppWrapper.tsx` dentro de un modal o contenedor limpio para aplicaciones externas de investigación (Gradio/Streamlit/docs).
- [ ] **Tarea 1.3: Construcción del Laboratorio Interactivo de Marcas**
  - Archivos: `frontend/src/pages/Laboratorio.tsx`, `frontend/src/components/BrandResultCard.tsx`.
  - Crear zona interactiva Drag & Drop con feedback táctil y visual.
  - Implementar slider reactivo de Top-K (1 a 20) y dial de umbral de similitud.
  - Diseñar tarjetas Flat con badges de porcentaje de coincidencia, distancia métrica y previsualización comparativa entre la imagen de consulta y el resultado del catálogo.
- [ ] **Tarea 1.4: Preparación de la Demo del Equipo**
  - Levantar los servidores locales y verificar el flujo end-to-end para la presentación del viernes.

---

### 🐍 Renato Bonín (Backend & IA) — Actividades de la Semana
- [ ] **Tarea R.1: Asimilación del Plan y Entorno Local**
  - Clonar la rama `dev` actualizada y verificar la ejecución de Uvicorn con `uv` o `pip`.
  - Revisar el notebook de investigación `Contexto/Archivos importantes/GIBD2025_ViT_Large...ipynb` y familiarizarse con el cálculo de similitud del coseno sobre embeddings normalizados.
  - Participar en la reunión del viernes para recibir el contrato de la Semana 2 (Análisis de Papers con Gemini AI).

---

### ⚛️ Emmanuel Davezac (Frontend & CMS) — Actividades de la Semana
- [ ] **Tarea E.1: Asimilación del Plan y Design System Flat**
  - Clonar la rama `dev` actualizada y ejecutar `npm run dev`.
  - Leer la guía de diseño en `Contexto/Declaracion_Sistema.md` y familiarizarse con los componentes orgánicos de cápsula (`rounded-full`, `rounded-[2rem]`, `#0A0A0A`, acento `#FF5500`).
  - Participar en la reunión del viernes para recibir el contrato de la Semana 2 (Formularios de Noticias y Papers en `Admin.tsx`).

---

## 3. Pruebas y Criterios de Aceptación

### A. Verificación Intermedia (Miércoles)
1. **Prueba de Inferencia con cURL:**
   ```bash
   curl -X POST -F "file=@frontend/src/assets/hero.png" "http://localhost:8000/api/v1/inference/siamese-brands?top_k=5"
   ```
   *Criterio:* Retorno HTTP 200 con array ordenado de marcas reales y distancias matemáticas del coseno.
2. **Prueba de Sintaxis Python:**
   ```bash
   python -m py_compile backend/app/main.py backend/app/services/siamese_engine.py
   ```

### B. Verificación de Entrega (Viernes)
1. **Compilación Limpia:**
   ```bash
   cd frontend && npm run build
   ```
   *Criterio:* Código 0 sin errores de TypeScript.
2. **Experiencia de Usuario en Laboratorio:**
   - Arrastrar una marca al recuadro -> previsualización inmediata.
   - Clic en "Buscar Similitudes" -> animación de carga fluida -> renderizado de tarjetas con marcas reales del GIBD.
   - Sin botones rotos de prueba en la interfaz.

---

## 4. Protocolo de Reprogramación
Si por imprevistos técnicos la inferencia en CPU o el Laboratorio requieren ajustes adicionales, la puesta en común se utiliza para mostrar el avance parcial y la tarea de integración se extiende hasta el martes de la Semana 2 sin bloquear la asignación de tareas de CMS para el equipo.
