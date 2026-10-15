# 📅 Plan de Trabajo: Semana 2 — CMS No-Dev de Noticias y Papers Académicos

> **Ciclo:** Sprint 2  
> **Objetivo Central:** Permitir a los investigadores publicar artículos científicos con asistencia de IA (Gemini) y redactar noticias institucionales desde `/admin` sin modificar código.  
> **Puesta en Común:** Viernes (Reunión Semanal del Equipo)  

---

## 1. Contratos de Integración (Definidos en el Kick-off del Lunes)

Para trabajar en paralelo sin bloqueos, frontend y backend se alinean sobre estos contratos desde el día 1:

### Contrato API (Renato - FastAPI):
* **Endpoint:** `POST /api/v1/ai/analyze-paper`
* **Input (JSON):** `{ "paper_text": "string con el texto extraído del paper..." }`
* **Output (JSON):**
  ```json
  {
    "abstract": "Resumen ejecutivo en español de 2 párrafos...",
    "image_prompt": "Conceptual English prompt for DALL-E/Midjourney..."
  }
  ```

### Contrato de Datos (Emma & Thiago - TypeScript en `dbService.ts`):
```typescript
interface Paper {
  id: string;
  titulo: string;
  abstract: string;
  anio: number;
  pdf_url: string;
  autores_ids: string[]; // IDs de los 24 investigadores del Vault
  categoria: 'IA' | 'Legal' | 'Agro' | 'General';
}

interface Noticia {
  id: string;
  titulo: string;
  bajada: string;
  cuerpo: string;
  imagen_url: string;
  fecha: string;
}
```

---

## 2. Distribución de Tareas por Integrante

### 🐍 Renato Bonín (Backend & IA)
- [ ] **Tarea 2.1: Endpoint de Asistencia Gemini en FastAPI**
  - Archivo: `backend/app/main.py`.
  - Conectar el SDK de Google GenAI con fallback determinista si no hay API Key configurada.
  - Asegurar tiempo de respuesta ágil y validación Pydantic estricta.
- [ ] **Tarea 2.2: Prueba Unitaria y Documentación de Endpoint**
  - Probar el endpoint con texto real de uno de los papers del Vault (ej. *CACIC 2023*) y documentar el comando cURL de prueba.

---

### ⚛️ Emmanuel Davezac (Frontend & CMS)
- [ ] **Tarea 2.3: Formularios de Gestión en `Admin.tsx`**
  - Archivo: `frontend/src/pages/Admin.tsx`.
  - Crear la pestaña y formulario para **Noticias**: título, bajada, cuerpo, subida de foto y previsualización.
  - Crear la pestaña y formulario para **Papers**: título, abstract (con botón "Autocompletar con IA"), año, subida de archivo PDF y selector multi-autor interactivo (checkboxes/tags de los 24 investigadores).
- [ ] **Tarea 2.4: Consumo contra Mocks Tipados en Desarrollo**
  - Trabajar conectado a `dbService.ts` para que la UI funcione tanto con Supabase como con LocalStorage sin romperse.

---

### 👑 Thiago Gomez Kehler (Tech Lead & Diseñador)
- [ ] **Tarea 2.5: Arquitectura Supabase, RLS y Storage**
  - Archivo: `Contexto/database_schema.sql`.
  - Crear tablas relacionales en Supabase: `papers`, `noticias`, `miembros_equipo`, `paper_autores`.
  - Configurar Storage Buckets: `papers-pdf` y `noticias-imagenes` con políticas públicas de lectura y autenticadas de escritura.
- [ ] **Tarea 2.6: Auditoría Visual y Refactor Estético de `Admin.tsx`**
  - Revisar y perfeccionar los formularios de Emma para garantizar que respeten los radios orgánicos (`rounded-[2rem]`), fondos oscuros y microanimaciones de éxito/error.
- [ ] **Tarea 2.7: Integración en Landing y Papers**
  - Verificar que las noticias creadas se rendericen reactivamente en `Landing.tsx` y los papers con descarga de PDF en `Papers.tsx`.

---

## 3. Pruebas y Criterios de Aceptación

### A. Verificación Intermedia (Miércoles)
1. **Prueba de Gemini:** Enviar un fragmento de texto a `/api/v1/ai/analyze-paper` y comprobar que devuelve abstract y prompt coherentes.
2. **Prueba de Formulario:** Comprobar que el selector de autores de Emma no desborde la pantalla en vista móvil.

### B. Verificación de Entrega (Viernes)
1. **Compilación Limpia:** `npm run build` con código 0.
2. **Flujo E2E de Publicación:**
   - Iniciar sesión en `/admin`.
   - Crear una noticia con foto y publicar un paper PDF seleccionando 3 coautores.
   - Presionar F5 y verificar persistencia en la base de datos.
   - Descargar el PDF desde `Papers.tsx` y comprobar apertura correcta.

---

## 4. Protocolo de Reprogramación
Si la integración con Supabase Storage o el asistente Gemini presentan fallas de red o autenticación, la tarea no aprobada se reprograma para el sprint siguiente. Se mantiene la versión funcional con LocalStorage en `dev` para no frenar la Semana 3.
