# 🏛️ GIBD 2026 — Plataforma Institucional y Laboratorio de IA

> **Grupo de Investigación en Bases de Datos (GIBD)**  
> *Facultad Regional Concepción del Uruguay — Universidad Tecnológica Nacional (UTN - FRCU)*  
> Repositorio oficial de la plataforma web de difusión científica, gestión documental (CMS) y laboratorio de inferencia de modelos multimodales (Visión Computacional, NLP y Audio).

---

## 🧭 Centro de Control y Planificación Centralizada

Toda la documentación, hojas de ruta y especificaciones técnicas se encuentran formalmente centralizadas en la carpeta [`docs/`](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs):

* **👑 [Plan Maestro y Guía de Operaciones: Tech Lead & Diseñador](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/plan_maestro_tech_lead_thiago.md)**  
  Documento rector de **Thiago Gomez Kehler** (Tech Lead, Arquitecto y Diseñador UI/UX). Define las expectativas semanales del equipo, qué pruebas intermedias y de entrega realizar, los Quality Gates y la matriz de control para las reuniones de los viernes.
* **📚 [Índice General de Documentación](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/README.md)**  
  Mapeo integral del repositorio documental, enlazando los planes semanales con las fases originales (`docs/fases/`).
* **📅 Planificación Operativa por Semanas (`docs/semanas/`):**
  1. [Semana 1: Fundación Real y Demostración](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/semanas/semana_1_fundacion_y_demostracion.md) *(Sprint Solo de Thiago)*: Inferencia real en CPU sobre marcas de ganado del GIBD, saneamiento y prototipo funcional.
  2. [Semana 2: CMS No-Dev de Noticias y Papers](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/semanas/semana_2_cms_papers_noticias.md) *(Equipo)*: Backend Gemini AI (Renato), Formularios CMS multi-autor con 24 investigadores (Emma) y Supabase RLS (Thiago).
  3. [Semana 3: CMS de los 6 Grupos de Investigación y Equipo](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/semanas/semana_3_grupos_investigacion_equipo.md) *(Equipo)*: G1 a G6 del Vault, gestión de docentes y becarios alumnos, Landing interactiva.
  4. [Semana 4: Expansión Multimodal, Flat y Despliegue](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/semanas/semana_4_multimodal_ptah_despliegue.md) *(Cierre)*: PTAH-Jurídico NLP (300 consultas), auditoría mobile, Quality Gate y Vercel.

---

## 👥 Equipo y Organización de Roles

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

## 🗂️ Taxonomía del Repositorio

```
GIBD WEB/
├── frontend/             # SPA moderna (React 18, Vite, TypeScript, Framer Motion)
│   ├── src/pages/        # Vistas principales: Landing, Laboratorio, Papers, Admin
│   ├── src/components/   # Componentes modulares bajo el Design System Flat
│   ├── src/utils/        # Clientes de servicios (dbService Supabase/Mock)
│   └── public/brands/    # Catálogo estático de marcas de ganado reales del GIBD
│
├── backend/              # API Gateway en FastAPI (Python 3.10+)
│   ├── app/main.py       # Entrada principal, enrutador CORS y endpoints REST
│   ├── app/services/     # Motores de inferencia en CPU (siamese_engine) y Gemini
│   └── app/data/brands/  # Datasets pre-indexados y metadatos de marcas
│
├── docs/                 # Documentación técnica, planes semanales y guías maestras
│   ├── semanas/          # Hojas de ruta operativas semanales (Semana 1 a 4)
│   ├── fases/            # Planes de diseño originales armonizados
│   ├── plan_maestro_tech_lead_thiago.md  # Guía rectora para Thiago
│   └── README.md         # Índice general de la documentación
│
├── Contexto/             # Activos científicos originales del GIBD
│   ├── GIBD_Obsidian_Vault/ # Base de conocimiento: 6 grupos MOC, proyectos y papers
│   ├── Archivos importantes/# Notebooks ViT (Adrián Planas), base de marcas CACIC 2023
│   └── GIBD ALLDATA/     # Archivos históricos y zips de investigación
│
├── scripts/              # Herramientas utilitarias y generadores de reportes PDF
├── AGENTS.md             # Reglas universales y de orden para agentes de IA
└── GEMINI.md             # Configuración operativa para Gemini/Antigravity
```

---

## 🎨 El Estándar Estético: Flat Design System

Cualquier contribución a la interfaz debe respetar incondicionalmente las directrices de diseño:
- **Fondo:** Negro OLED puro (`#0A0A0A` / `#000000`).
- **Acento Primario:** Naranja vibrante de alto impacto (`#FF5500`).
- **Geometría:** Botones, selectores y pills estrictamente redondeados (`rounded-full` / `border-radius: 9999px`). Tarjetas y contenedores con radio orgánico (`rounded-[2rem]`).
- **Cero Sombras Tradicionales:** Prohibido usar `box-shadow` difusos convencionales. La profundidad se logra mediante **microbordes de alto contraste** (`border border-white/10`) y resplandores focalizados.
- **Microinteracciones:** Transiciones fluidas y aceleradas por hardware usando `framer-motion`.

---

## ⚡ Calidad de Código y Zero-Cost AI

1. **Datos Reales:** No se permiten mocks inventados ni simulaciones de hash (MD5) en endpoints de producción cuando existan datos científicos en `Contexto/`.
2. **Inferencia en CPU (Zero-Cost):** Los modelos de visión de marcas y búsqueda textual están optimizados para ejecutarse en CPU local en tiempos menores a 200 ms, sin costos de nube.
3. **Quality Gate:** Todo PR debe compilar limpiamente con `npm run build` en frontend y `python -m py_compile` en backend sin advertencias críticas antes de fusionarse a `dev`.