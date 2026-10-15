# 📅 Plan de Trabajo: Semana 4 — Expansión Multimodal, Auditoría Flat y Despliegue Final

> **Ciclo:** Sprint 4  
> **Objetivo Central:** Integrar la búsqueda semántica de normativas (PTAH-Jurídico), auditar pixel a pixel el Design System Flat y realizar el despliegue final en producción en Vercel.  
> **Puesta en Común:** Viernes (Reunión Final de Cierre y Lanzamiento a Producción)  

---

## 1. Contratos de Integración (Definidos en el Kick-off del Lunes)

### Contrato API PTAH-Jurídico (Renato - FastAPI):
* **Endpoint:** `POST /api/v1/inference/ptah-nlp`
* **Input (JSON):**
  ```json
  {
    "consulta": "¿Qué requisitos se exigen para el traslado de ganado vacuno según la normativa?",
    "top_k": 3
  }
  ```
* **Output (JSON):**
  ```json
  {
    "status": "success",
    "query": "¿Qué requisitos se exigen...",
    "results": [
      {
        "articulo_id": "Art. 42 - Decreto Ley 22939",
        "normativa": "Régimen de Marcas y Señales",
        "texto_relevante": "El certificado de adquisición y guía de traslado...",
        "similarity_score": 94.2
      }
    ]
  }
  ```

---

## 2. Distribución de Tareas por Integrante

### 🐍 Renato Bonín (Backend & IA)
- [ ] **Tarea 4.1: Endpoint NLP PTAH-Jurídico**
  - Archivo: `backend/app/main.py`.
  - Conectar el indexador semántico sobre el archivo `Consultas.csv` (dataset de 300 consultas de normativa del GIBD) usando embeddings ligeros o TF-IDF/Sentence-Transformers.
  - Asegurar tiempos de respuesta inferiores a 200 ms en CPU.
- [ ] **Tarea 4.2: Health Check y Configuración de Producción**
  - Configuración final de CORS, compresión Gzip y endpoints de monitoreo para despliegue en servidor en la nube.

---

### ⚛️ Emmanuel Davezac (Frontend & CMS)
- [ ] **Tarea 4.3: Auditoría Estética Estricta Design System Flat**
  - Archivos: `frontend/src/index.css`, `frontend/src/pages/*.tsx`.
  - Eliminar cualquier propiedad `box-shadow` residual en todo el proyecto.
  - Verificar que todos los contenedores principales utilicen `rounded-[2rem]` o `rounded-full` y bordes finos de alto contraste.
- [ ] **Tarea 4.4: Auditoría de Responsividad Móvil**
  - Probar en emulador y dispositivos físicos (pantallas de 360px a 1440px): navegación, menús de cápsula y formularios.

---

### 👑 Thiago Gomez Kehler (Tech Lead & Diseñador)
- [ ] **Tarea 4.5: Consola Interactiva de PTAH-Jurídico en Laboratorio**
  - Archivo: `frontend/src/pages/Laboratorio.tsx`.
  - Diseñar la interfaz de búsqueda en lenguaje natural en la pestaña de Texto: input de búsqueda expandible, botón con animación de pulso y tarjetas de artículos normativos citados.
- [ ] **Tarea 4.6: Maquetado Conceptual de Audio y Video**
  - Dejar listos los visualizadores de espectrogramas de audio y análisis de cuadros de video como adelanto visual para las investigaciones en curso (policía y acústica).
- [ ] **Tarea 4.7: Quality Gate E2E Global, Merge y Despliegue en Vercel**
  - Ejecutar `npm run build` certificando cero errores de compilación.
  - Verificar variables de entorno en el panel de Vercel.
  - Solicitar autorización del usuario para aplicar la regla global y ejecutar el `git merge dev` hacia `main` y `git push origin main`.
  - Certificar el despliegue en vivo en `https://gibd.vercel.app`.

---

## 3. Pruebas y Criterios de Aceptación Finales

### A. Verificación Intermedia (Miércoles)
1. **Prueba de PTAH NLP:** Realizar consultas sobre normativas agrarias y comprobar la recuperación de artículos exactos con porcentajes de similitud coherentes.
2. **Auditoría de Sombras:** Correr búsqueda global en el código (`grep box-shadow`) y confirmar que no existan sombras fuera de los efectos luminosos autorizados.

### B. Verificación de Entrega Final (Viernes)
1. **Compilación y Rendimiento:**
   - `npm run build` finaliza con código 0.
   - Puntuación de Lighthouse en escritorio superior a 90 en Performance y Accesibilidad.
2. **Plataforma 100% Operativa en Producción:**
   - Landing institucional con navegación y miembros dinámicos.
   - Laboratorio con Búsqueda de Marcas por IA real en CPU + Búsqueda PTAH-Jurídico.
   - Panel de Administración protegido con Supabase Auth permitiendo CRUD de noticias, papers, grupos y equipo.

---

## 4. Protocolo de Reprogramación
Si alguna de las optimizaciones de rendimiento o el endpoint NLP requiere ajustes de última hora, se pospone el lanzamiento público en redes institucionales durante 48 horas sin comprometer la estabilidad de la rama `dev`.
