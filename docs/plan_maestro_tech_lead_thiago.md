# 👑 Plan Maestro y Guía de Operaciones: Tech Lead & Diseñador

> **Destinatario Exclusivo:** Thiago Gomez Kehler  
> **Roles:** Tech Lead / Arquitecto de Software / Diseñador UI-UX / Responsable de Integración  
> **Proyecto:** Plataforma Institucional y Laboratorio de IA — GIBD 2026  
> **Fecha de Actualización:** 08/09/2026  

---

## 1. Manifiesto del Rol: El Vértice Técnico y Estético

En este proyecto, tu rol no se limita a coordinar tareas. Como **Ingeniero y Diseñador**, tienes la responsabilidad de:
1. **Garantizar la Excelencia Visual y de Experiencia (UI/UX):** El GIBD no debe verse como un portal académico genérico. Eres el guardián del **Design System Flat**:
   - Geometría de cápsula (`border-radius: 9999px` en botones e interactivos; `rounded-[2rem]` en paneles y tarjetas).
   - Fondo negro puro (`#0A0A0A`) y acento naranja vibrante (`#FF5500`).
   - Cero sombras convencionales (`box-shadow: none`), sustituidas por microbordes orgánicos de alto contraste y brillo focalizado.
   - Fluidez y microinteracciones de entrada con Framer Motion.
2. **Orquestar la Integración sin Fisuras (El Pegamento del Sistema):**
   - Asegurar que el frontend de Vite/React, el backend de FastAPI y la persistencia en Supabase operen como un único sistema coordinado.
   - Filtrar y sanear el código de los colaboradores antes de que toque ramas centrales (`dev` o `main`).
3. **Liderar con el Ejemplo ("Show, Don't Tell"):**
   - En la **Semana 1**, ejecutas todo tú para establecer la arquitectura real de IA sobre CPU (Zero-Cost), el Laboratorio y el estándar estético con el que guiarás a Renato y Emmanuel.

---

## 2. El Ritmo de Trabajo Semanal (Cadencia de Sprints)

Cada semana opera como un ciclo cerrado e independiente:

```
  LUNES (Kick-off)               MIÉRCOLES (Checkpoint)           VIERNES (Puesta en Común)
  ────────────────              ──────────────────────           ─────────────────────────
  • Presentación de contratos   • Revisión asíncrona             • Demo en vivo en entorno local.
    (Pydantic / TypeScript).      de ramas feature.              • Auditoría de build y calidad.
  • Entrega de especificaciones • Detección de bloqueos          • APROBADO: Merge a dev.
    a Renato y Emma.              tempranos.                     • NO CUMPLE: Reprogramación formal.
```

> [!IMPORTANT]
> **Regla de Oro de Reprogramación:** Si al llegar al viernes una entrega no cumple los Quality Gates (`npm run build` sin errores, sin mocks improvisados, respetando el contrato acordado), **la tarea se reprograma para la semana siguiente**. Bajo ninguna circunstancia se fusiona a `dev` código a medias o con atajos que ensucien la arquitectura.

---

## 3. Hoja de Ruta Semana a Semana para Thiago

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ SEMANA 1: FUNDACIÓN REAL Y PROTOTIPO DEMOSTRATIVO (Solo Thiago)                   │
│ • Motor ViT real en CPU sobre marcas conocidas + Laboratorio Flat saneado.        │
├───────────────────────────────────────────────────────────────────────────────────┤
│ SEMANA 2: CMS NO-DEV DE NOTICIAS Y PAPERS ACADÉMICOS (Equipo)                     │
│ • Renato: Endpoint Gemini AI | Emma: Formularios CMS | Thiago: Supabase + UI      │
├───────────────────────────────────────────────────────────────────────────────────┤
│ SEMANA 3: CMS DE LOS 6 GRUPOS DE INVESTIGACIÓN Y EQUIPO (Equipo)                  │
│ • Renato: CRUD Backend | Emma: Gestión de miembros | Thiago: Landing interactiva  │
├───────────────────────────────────────────────────────────────────────────────────┤
│ SEMANA 4: EXPANSIÓN MULTIMODAL (PTAH NLP), AUDITORÍA FLAT Y DESPLIEGUE (Equipo)   │
│ • Renato: NLP 300 consultas | Emma: Responsividad mobile | Thiago: E2E + Vercel   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Detalle Operativo por Semana: Qué Hacer, Qué Esperar y Cómo Testear

---

### 🟢 SEMANA 1: Fundación Real y Demostración (Sprint Solo de Thiago)
> **Propósito:** Construir la base real sin simulaciones, sanear el frontend y presentar un prototipo funcional al equipo para marcar el rumbo.

#### A. Lo que debes hacer tú:
1. **Backend de IA Real (Zero-Cost en CPU):**
   - Implementar en `backend/app/services/siamese_engine.py` la extracción de características usando preprocesamiento MBR (recorte de bordes blancos/negros, transparencia a blanco, resize a 224x224, blur gaussiano).
   - Pre-indexar el catálogo de marcas del GIBD (`Logos-consultas80` o base de marcas de CACIC 2023) generando una matriz de embeddings (`embeddings_index.npy`) y su catálogo (`brands_catalog.json`).
   - El endpoint `/api/v1/inference/siamese-brands` debe procesar la imagen de consulta en CPU en menos de 150 ms y calcular la similitud del coseno real contra la matriz.
2. **Saneamiento del Frontend y Wrapper:**
   - Revertir las modificaciones accidentales en `frontend/public/favicon.svg`.
   - Limpiar `Laboratorio.tsx`: retirar los botones de prueba a `localhost:8010` y `localhost:9999`.
   - Reubicar `backend/demo_python_app.py` en `backend/demos/`.
   - Integrar `PythonAppWrapper.tsx` como un contenedor accesible en una pestaña/modal de aplicaciones externas.
3. **Construcción del Laboratorio de Marcas de Ganado:**
   - En `Laboratorio.tsx`, crear la zona Drag & Drop con feedback de arrastre, botón de análisis con estado pulsante y slider de Top-K (1 a 20).
   - Renderizar las tarjetas de resultados con badges de porcentaje real de coincidencia, distancia métrica y comparativa visual entre la imagen consulta y el resultado del catálogo.
4. **Preparación de la Presentación:**
   - Dejar el servidor backend y frontend corriendo para hacer la demo en vivo en la reunión semanal.

#### B. Qué esperar del equipo:
* En esta semana, Renato y Emma se enfocan en asimilar la nueva planificación en `docs/` y preparar sus entornos locales. No tienen entregables de código obligatorios en este sprint.

#### C. Pruebas y Criterios de Aceptación de Semana 1:
* **Prueba Intermedia (Miércoles):**
  - Ejecutar `curl -X POST -F "file=@ruta/marca.png" "http://localhost:8000/api/v1/inference/siamese-brands?top_k=5"` y confirmar que devuelve marcas reales del catálogo y distancias del coseno matemáticas.
* **Prueba de Entrega (Viernes):**
  - `npm run build` en `frontend/` finaliza con código 0.
  - Subir una imagen al Laboratorio vía Drag & Drop y verificar que se desplieguen las tarjetas con animación de entrada, imágenes reales de la base y latencia fluida.

---

### 🟡 SEMANA 2: CMS No-Dev de Noticias y Papers Académicos (Colaborativo)
> **Propósito:** Habilitar a los investigadores para publicar artículos científicos con asistencia de IA y redactar novedades institucionales.

#### A. Qué debes esperar de cada uno:
* **Renato (Backend):**
  - Endpoint `/api/v1/ai/analyze-paper` con Gemini optimizado, que reciba el texto de un PDF y devuelva un JSON estructurado con: abstract profesional en español y prompt visual descriptivo para generar portada.
* **Emma (Frontend):**
  - Componentes de formulario en `Admin.tsx`: alta de noticias (título, bajada, cuerpo, imagen de portada) y alta de papers (título, abstract, año, archivo PDF, selector multi-autor de los 24 investigadores del Vault).

#### B. Lo que te toca hacer a ti (Thiago):
* **Arquitectura de Datos y Supabase:**
  - Ejecutar y verificar en Supabase las tablas relacionales (`papers`, `autores`, `paper_autores`, `noticias`) y las políticas RLS.
  - Configurar los buckets de Storage (`papers-pdf`, `noticias-imagenes`) con acceso público de lectura y escritura autenticada.
* **Diseño y Fusión Visual:**
  - Auditar y rediseñar los formularios de Emma en `Admin.tsx` para que respeten la paleta y los espaciados del Design System Flat.
  - Unificar las consultas en `dbService.ts` conectando tanto el modo Supabase real como el modo mock fallback.

#### C. Pruebas y Criterios de Aceptación de Semana 2:
* **Pruebas Intermedias (Miércoles):**
  - Enviar un texto de prueba a Renato y verificar que Gemini no falle ni devuelva JSON roto.
  - Revisar que los selectores de autores de Emma no generen desbordamientos horizontales en mobile.
* **Prueba de Entrega (Viernes):**
  - Iniciar sesión en `/admin`.
  - Crear una noticia con foto y subir un paper PDF vinculando a 3 coautores.
  - Comprobar que la noticia aparece en `Landing.tsx` y el paper con su PDF descargable en `Papers.tsx`.

---

### 🔵 SEMANA 3: CMS de los 6 Grupos de Investigación y Equipo (Colaborativo)
> **Propósito:** Estructurar dinámicamente las líneas de investigación del GIBD y visibilizar al cuerpo docente y becarios alumnos.

#### A. Qué debes esperar de cada uno:
* **Renato (Backend):**
  - Servicios de persistencia o endpoints auxiliares para listar y actualizar grupos de investigación y la lista de integrantes con sus roles.
* **Emma (Frontend):**
  - Pestañas en `Admin.tsx` para administrar integrantes (nombre, rol docente/alumno, email, foto avatar, perfil LinkedIn, grupo asignado) y editar descripciones de grupos.

#### B. Lo que te toca hacer a ti (Thiago):
* **Estandarización de la Taxonomía del Vault:**
  - Configurar en la base de datos los **6 grupos oficiales del MOC de Obsidian**:
    1. `G1 - Sonido`
    2. `G2 - Video`
    3. `G3 - Texto`
    4. `G4 - Imágenes`
    5. `G5 - Reducción de Dimensionalidad`
    6. `G6 - Desarrollo y Aplicaciones`
* **Rediseño de la Sección Grupos en Landing:**
  - Maquetar las tarjetas interactivas de grupos en `Landing.tsx` con microinteracciones al pasar el cursor y navegación fluida hacia los proyectos de cada grupo.
  - Implementar el modal interactivo de visualización de perfiles del equipo (docentes y alumnos).

#### C. Pruebas y Criterios de Aceptación de Semana 3:
* **Prueba Intermedia (Miércoles):**
  - Verificar subida de avatares a Supabase Storage y compresión automática de imágenes.
* **Prueba de Entrega (Viernes):**
  - Dar de alta a un nuevo becario alumno en `/admin` asignándole grupo y foto.
  - Verificar su aparición inmediata en la sección correspondiente de la Landing Page sin necesidad de recargar código.

---

### 🟣 SEMANA 4: Expansión Multimodal, Auditoría Flat y Despliegue (Cierre)
> **Propósito:** Integrar el prototipo de lenguaje natural PTAH-Jurídico, auditar pixel a pixel el diseño y poner en producción la web definitiva en Vercel.

#### A. Qué debes esperar de cada uno:
* **Renato (Backend):**
  - Endpoint de búsqueda semántica `/api/v1/inference/ptah-nlp` que reciba una consulta de usuario y devuelva los artículos normativos más relevantes basados en el dataset de 300 consultas del GIBD.
* **Emma (Frontend):**
  - Auditoría de responsividad completa: verificar navegación en smartphones y tablets.
  - Eliminación estricta de cualquier `box-shadow` remanente en todos los archivos `.tsx` y `.css`.

#### B. Lo que te toca hacer a ti (Thiago):
* **Integración del Laboratorio Multimodal:**
  - Conectar en `Laboratorio.tsx` la pestaña de Texto con el servicio de PTAH-Jurídico.
  - Dejar listas las interfaces de visualización de audio (espectrogramas) y video como adelanto para los convenios de investigación del GIBD.
* **Control de Calidad Final y Despliegue:**
  - Ejecutar auditoría de rendimiento de Vite (code-splitting, lazy loading de rutas).
  - Comprobar que las variables de entorno de producción estén en Vercel.
  - Realizar el merge final de `dev` hacia `main` previa aprobación de la regla global.

#### C. Pruebas y Criterios de Aceptación de Semana 4:
* **Prueba de Calidad Final (Viernes):**
  - `npm run build` en 0 errores.
  - Consulta en vivo en el Laboratorio (Marcas de Ganado + PTAH-Jurídico).
  - CMS funcional con creación, edición y borrado.
  - Despliegue activo en `https://gibd.vercel.app` sin errores de consola (F12 limpia).

---

## 5. Matriz de Control Rápido para Reuniones Semanales

Utiliza esta lista de verificación durante las puestas en común de cada viernes:

| Dimensión | Pregunta de Control | Estado |
| :--- | :--- | :---: |
| **Integridad de Código** | ¿La rama compila limpiamente con `npm run build` sin advertencias críticas? | [ ] |
| **Contrato de Datos** | ¿Se respetaron los tipos definidos el lunes sin inventar campos nuevos? | [ ] |
| **Estética Flat** | ¿Se respetan los bordes redondeados, el fondo `#0A0A0A` y el acento `#FF5500` sin sombras? | [ ] |
| **Aislamiento** | ¿El código entregado está libre de botones temporales o URLs a `localhost` rotas? | [ ] |
| **Datos Reales** | ¿La inferencia o persistencia utiliza datos reales del GIBD y no mocks arbitrarios? | [ ] |
