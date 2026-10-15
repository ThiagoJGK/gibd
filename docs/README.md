# 📚 Documentación y Plan de Desarrollo — Plataforma GIBD 2026

Bienvenido al centro de documentación y gestión técnica del proyecto **GIBD Web (Grupo de Investigación en Bases de Datos)**.

Este directorio contiene la planificación estratégica, las especificaciones de diseño, los contratos de arquitectura y la distribución operativa semanal del equipo.

---

## 🧭 Índice Rápido de Documentos Clave

### 👑 Para el Tech Lead & Diseñador (Thiago)
* **[Plan Maestro y Guía de Operaciones](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/plan_maestro_tech_lead_thiago.md)**: El documento rector para Thiago Gomez Kehler. Contiene el manifiesto del rol (Ingeniería + UI/UX), qué esperar de Renato y Emmanuel semana a semana, qué debe desarrollar e integrar Thiago, las pruebas intermedias y de entrega, y la matriz de control para las reuniones semanales.

### 📅 Planificación Operativa por Semanas
La ejecución se divide en sprints semanales con puestas en común los viernes:
1. **[Semana 1: Fundación Real y Demostración](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/semanas/semana_1_fundacion_y_demostracion.md)** *(Sprint Solo de Thiago)*:
   - Backend de IA Real sobre CPU (Zero-Cost inference en marcas de ganado con preprocesamiento MBR y matriz de embeddings del catálogo).
   - Saneamiento del frontend (`Laboratorio.tsx` limpio, wrapper aislado en demo externa, favicon restaurado).
   - Demo operativa y benchmark estético para el equipo.
2. **[Semana 2: CMS No-Dev de Noticias y Papers Académicos](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/semanas/semana_2_cms_papers_noticias.md)** *(Equipo)*:
   - Renato: Endpoint Gemini AI (`/api/v1/ai/analyze-paper`) para generación de abstracts y prompts.
   - Emmanuel: Formularios CMS en `Admin.tsx` con multi-autor (24 investigadores del Vault).
   - Thiago: Esquema relacional en Supabase, buckets de Storage y refinamiento UI Flat.
3. **[Semana 3: CMS de los 6 Grupos de Investigación y Equipo](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/semanas/semana_3_grupos_investigacion_equipo.md)** *(Equipo)*:
   - Renato: Persistencia y endpoints de Grupos e Integrantes.
   - Emmanuel: Administración de docentes y becarios alumnos en `Admin.tsx`.
   - Thiago: Modelado de los 6 grupos del Vault (G1 a G6) y rediseño de tarjetas interactivas en `Landing.tsx`.
4. **[Semana 4: Expansión Multimodal, Auditoría Flat y Despliegue](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/semanas/semana_4_multimodal_ptah_despliegue.md)** *(Equipo)*:
   - Renato: Endpoint de búsqueda semántica PTAH-Jurídico (300 consultas normativas).
   - Emmanuel: Auditoría de responsividad mobile y limpieza estricta de sombras.
   - Thiago: Laboratorio multimodal, auditoría E2E y despliegue final en Vercel.

---

## 🏛️ Armonización entre Fases Previas y Semanas Actuales

Para mantener la trazabilidad histórica y honrar la visión inicial del proyecto sin descartar trabajo previo, la carpeta [`docs/fases/`](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/fases) se mapea directamente a los sprints semanales actuales de la siguiente manera:

| Fase Original (`docs/fases/`) | Semana Actual (`docs/semanas/`) | Reinterpretación y Evolución Técnica |
| :--- | :--- | :--- |
| **Fase 1: Gateway y Demostración** | **Semana 1** | Se reemplazaron las simulaciones (MD5 y patentes) por **inferencia real sobre CPU** basada en los notebooks de Vision Transformer del GIBD. Se saneó el Frontend para eliminar dependencias externas rotas. |
| **Fase 2: CMS y Papers Científicos** | **Semana 2** | Se estructura el alta asistida por IA (Gemini) y la gestión de coautores vinculada a los 24 investigadores reales extraídos del Obsidian Vault. |
| **Fase 3: Grupos del GIBD y Docentes** | **Semana 3** | Se modelan los 6 grupos de investigación oficiales (G1 Sonido a G6 Desarrollo) y la doble categoría de integrantes (docentes y alumnos becarios). |
| **Fase 4: Multimodalidad y Producción** | **Semana 4** | Se incorpora PTAH-Jurídico sobre el dataset normativo real y se ejecuta la puesta en producción en Vercel bajo el Design System Flat. |

---

## 👥 Equipo y Distribución de Responsabilidades

```
                   ┌─────────────────────────────────────────┐
                   │          THIAGO GOMEZ KEHLER            │
                   │   Tech Lead & Diseñador UI/UX Principal │
                   │   • Arquitectura y Calidad Global       │
                   │   • Design System Flat (#0A0A0A/#FF5500)│
                   │   • Integrador Supremo (Frontend+Back)  │
                   └────────────────────┬────────────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
  ┌─────────────────────────────┐               ┌─────────────────────────────┐
  │        RENATO BONÍN         │               │      EMMANUEL DAVEZAC       │
  │     Backend & AI Engineer   │               │     Frontend & CMS Admin    │
  │  • Servicios FastAPI        │               │  • Vistas de Administración │
  │  • Inferencia y Gemini AI   │               │  • Formularios y UX Admin   │
  │  • Contratos Pydantic       │               │  • Responsividad y Estilos  │
  └─────────────────────────────┘               └─────────────────────────────┘
```

---

## 🔄 Protocolo de Sincronización y Reglas de Calidad

1. **Cadencia Semanal:**
   - **Lunes:** Puesta en marcha y entrega de contratos técnicos.
   - **Miércoles:** Checkpoint asíncrono sobre ramas `feature/*` para destrabar impedimentos.
   - **Viernes:** Puesta en común con **demo en vivo en entorno local**.
2. **Regla de Oro de Reprogramación:**
   - Si una entrega no cumple los criterios de aceptación el viernes (`npm run build` falla, hay mocks no autorizados o se rompen los contratos), **se reprograma formalmente para la semana siguiente**. No se permiten merges a medias.
3. **Flujo de Git y Quality Gates:**
   - Base de desarrollo: `dev`.
   - Ramas de trabajo: `feature/semanaX-tarea`.
   - Los merges hacia `dev` requieren Pull Request y revisión estricta de Thiago.
   - **Regla Global:** Antes de cualquier cambio masivo o push a ramas compartidas, se consulta y valida con el usuario.
