# 📅 Plan de Trabajo: Semana 1 — Consolidación CMS, CRUD de Noticias y Desacoplamiento

> **Ciclo:** Sprint 1  
> **Objetivo Central:** Modularizar el monolito de `Admin.tsx`, completar el CRUD faltante de Noticias (listado, edición y eliminación) y testear el endpoint de Gemini con papers reales.  
> **Puesta en Común:** Jueves (Demo en Vivo en Entorno Local)  
> **Integración:** Jueves por Thiago (Merge a `dev`)  

---

## 1. Contexto de la Semana y Distribución de Roles

En este sprint, **Emma y Renato son los encargados del desarrollo de features en paralelo**.  
**Thiago (Tech Lead & Diseñador Principal)** no programa features en este ciclo; su responsabilidad radica en congelar los contratos de datos en `frontend/src/types/`, dejar preparado el Shell enrutador de `Admin.tsx`, auditar los Quality Gates y realizar el merge final el **Jueves**.

```
                           ┌───────────────────────────┐
                           │    THIAGO GOMEZ KEHLER    │
                           │         Tech Lead         │
                           │  • Lunes: Contratos Types │
                           │  • Jueves: Audita & Merge │
                           └─────────────┬─────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
  ┌─────────────────────────────┐                 ┌─────────────────────────────┐
  │        RENATO BONÍN         │                 │      EMMANUEL DAVEZAC       │
  │     Backend & AI Engineer   │                 │     Frontend & CMS Admin    │
  │  • Rama: feature/semana1-   │                 │  • Rama: feature/semana1-   │
  │    backend-gemini           │                 │    frontend-cms             │
  │  • Tests reales de Gemini   │                 │  • AdminNoticiasTab (CRUD)  │
  │  • Modularizar router paper │                 │  • AdminPapersTab           │
  │  • Archivo: backend/app/    │                 │  • Archivo: frontend/src/   │
  └─────────────────────────────┘                 └─────────────────────────────┘
```

---

## 2. Asignación de Tareas por Desarrollador

### 👑 Thiago Gomez Kehler (Tech Lead & Arquitecto) — Lunes Kick-off & Jueves Merge
- [ ] **Lunes (Contratos):**
  - Crear la carpeta `frontend/src/types/`:
    - `noticias.ts`: tipos `Noticia`, `NoticiaFormData`.
    - `papers.ts`: tipos `Paper`, `PaperFormData`, `Author`.
    - `api.ts`: tipos `PaperAnalysisRequest`, `PaperAnalysisResponse`.
  - Crear `frontend/src/config/env.ts` para centralizar `API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'`.
  - Modularizar `Admin.tsx` para reducirlo a un cascarón Shell de ~120 líneas que solo importe condicionalmente `<AdminNoticiasTab />` y `<AdminPapersTab />`.
- [ ] **Jueves (Puesta en Común & Merge):**
  - Auditar compilación de frontend (`npm run build` en 0 errores).
  - Validar pruebas unitarias de backend (`python -m py_compile`).
  - Ejecutar merge de `feature/semana1-frontend-cms` y `feature/semana1-backend-gemini` hacia `dev`.

---

### 🐍 Renato Bonín (Backend & AI) — Rama: `feature/semana1-backend-gemini`
*Nota Técnica: El endpoint `POST /api/v1/ai/analyze-paper` con Gemini 1.5 Flash YA está implementado en `backend/app/main.py`. NO debe crearse desde cero.*
- [ ] **Modularización del Router:**
  - Extraer las rutas de análisis de papers a `backend/app/routers/papers.py`.
- [ ] **Suite de Tests con Datos Reales del GIBD:**
  - Crear script de prueba `backend/tests/test_gemini_paper.py` consumiendo fragmentos reales de papers de `Contexto/` (estudio CACIC 2023 y PTAH Argencon).
  - Verificar que el fallback determinista responda con HTTP 200 en < 20 ms cuando no exista `GEMINI_API_KEY`.
- [ ] **Regla de Aislamiento:** Prohibido modificar cualquier archivo de `frontend/`.

👉 *Ver detalles paso a paso en [semana_1_plan_renato.md](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/semanas/semana_1_plan_renato.md)*.

---

### ⚛️ Emmanuel Davezac (Frontend & CMS) — Rama: `feature/semana1-frontend-cms`
*Nota Técnica: Los formularios básicos de Noticias y Papers ya existen en `Admin.tsx`. Tu tarea consiste en modularizarlos y completar el CRUD faltante de Noticias.*
- [ ] **Modularización en Pestañas Aisladas:**
  - Crear `frontend/src/pages/admin/AdminNoticiasTab.tsx`.
  - Crear `frontend/src/pages/admin/AdminPapersTab.tsx` (preservando los chips multi-autor de los 24 investigadores y el botón de Gemini).
- [ ] **Completar la Brecha de Noticias (CRUD Completo):**
  - En `AdminNoticiasTab.tsx`, maquetar debajo del formulario la tabla/grilla de noticias existentes.
  - Implementar botones "Editar" (carga de datos al formulario con modo edición) y "Eliminar".
  - En `frontend/src/utils/services/noticiasService.ts`, implementar `updateNoticia(id, data)` y `deleteNoticia(id)` con persistencia dual (Supabase / localStorage).
- [ ] **Cumplimiento Estético Flat:**
  - Geometría de cápsula (`rounded-full`) en botones y badges.
  - Microbordes `border border-white/10`, cero `shadow-` difuso.
- [ ] **Regla de Aislamiento:** Prohibido modificar `backend/` o `database_schema.sql`.

👉 *Ver detalles paso a paso en [semana_1_plan_emmanuel.md](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/semanas/semana_1_plan_emmanuel.md)*.

---

## 3. Pruebas y Criterios de Aceptación (Quality Gates)

### Jueves (Puesta en Común en Vivo)
1. **Compilación Frontend:**
   ```powershell
   cd frontend
   npm run build
   # Debe finalizar en código 0 sin errores de TypeScript
   ```
2. **Validación Backend:**
   ```powershell
   python -m py_compile backend/app/main.py backend/app/routers/papers.py
   # Debe finalizar con éxito
   ```
3. **Flujo E2E:**
   - Iniciar sesión en `/admin`.
   - Crear una noticia -> Editarla -> Eliminarla -> Confirmar reflejo en `Landing.tsx`.
   - Subir un Paper con 3 coautores usando Gemini -> Confirmar en `Papers.tsx`.
   - Recargar (F5) y comprobar persistencia de datos.
