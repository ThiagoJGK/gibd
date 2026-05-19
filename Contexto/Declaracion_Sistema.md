# Declaración y Descripción Integral del Sistema: Plataforma GIBD 2026

## 1. Visión General del Proyecto
La **Plataforma Institucional y de Experimentación del GIBD** (Grupo de Investigación en Big Data de la UTN FRCU) es una solución integral de software diseñada para democratizar el acceso a las investigaciones del grupo, visibilizar sus productos tecnológicos y proporcionar un entorno interactivo ("Laboratorio") donde los usuarios puedan poner a prueba los modelos de Inteligencia Artificial desarrollados por los investigadores.

El proyecto está concebido bajo una estricta **arquitectura "Zero-Cost"** inicial, aprovechando servicios en la nube de nivel gratuito (Vercel, Supabase, Render/Hugging Face) para minimizar la carga presupuestaria de la universidad, pero manteniendo estándares de ingeniería de software corporativos.

---

## 2. Objetivos Principales
1. **Visibilidad Académica:** Centralizar en un solo portal todas las noticias, papers y avances del GIBD.
2. **Transferencia Tecnológica Interactiva:** Permitir que usuarios no técnicos interactúen en tiempo real con modelos complejos (ej. Redes Siamesas para marcas de ganado) sin requerir instalación local.
3. **Optimización Operativa:** Proveer un CMS (Content Management System) fácil de usar para que los investigadores puedan publicar noticias y papers sin depender del equipo de desarrollo.
4. **Diseño de Vanguardia:** Romper con el molde de las "webs académicas aburridas" mediante una UI/UX inmersiva, oscura, geométrica y de alto rendimiento.

---

## 3. Módulos y Funcionalidades del Sistema (Exhaustivo)

La aplicación está dividida en cuatro pilares fundamentales:

### A. Landing Page Institucional
El punto de entrada público. Diseñado para alto impacto visual y conversión de lectura.
- **Hero Section:** Presentación del grupo con un fuerte Call to Action (CTA) hacia el Laboratorio.
- **Grupos de Investigación:** Exposición de los distintos grupos de trabajo dentro del GIBD (ej. grupo encargado de video, sonido, etc.).
- **Noticias Dinámicas:** Grilla de novedades extraídas en tiempo real desde Supabase.
- **Acerca de / Misión:** Resumen de las áreas de estudio (Deep Learning, Análisis Numérico, etc.).

### B. Laboratorio de Experimentación (Público)
El núcleo interactivo de la plataforma. Una consola técnica para probar IA.
- **Módulo 1: Marcas de Ganado:** 
  - *Función:* El usuario sube (Drag & Drop) una foto de una marca (tatuaje) de ganado.
  - *Proceso:* La interfaz envía la imagen a un gateway en Python, el cual ejecuta inferencia mediante Redes Neuronales Siamesas.
  - *Salida:* Muestra el "Top-K" de imágenes más similares y sus distancias relativas (simulando One-Shot Learning).
- **Módulo 2: PTAH-Jurídico:** Búsqueda semántica (NLP) sobre normativas.
- **Módulo 3: OBC (Huesos de Oráculo):** Reconocimiento de glifos antiguos.
- **Módulo 4: Sonido / Video:** Áreas preparadas para futuras integraciones multimodales.
- *Nota Técnica:* Todos los módulos comparten un panel lateral de configuración estandarizado (ajustes de Top-K, umbrales de similitud).

### C. Repositorio Académico (Papers)
Biblioteca digital pública.
- **Listado de Investigaciones:** Grilla de tarjetas con el título, año y un abstract breve.
- **Colaboradores:** Visualización de los colaboradores asociados a cada paper o investigación, tomados de la base de alumnos y equipo.
- **Descargas Directas:** Botones enlazados a Supabase Storage para descargar los PDFs directamente (ej. el paper de *Camera Ready - CACIC 2023*).
- **Buscador/Filtro:** Capacidad de filtrar papers por temática (IA, Legal, Agro).

### D. Panel de Administración (Privado / Seguro)
Área restringida exclusiva para miembros del GIBD (protegida por autenticación JWT de Supabase).
- **Control de Acceso y Permisos:** 
  - Solo los usuarios seleccionados explícitamente como "Admins" (asociados a un correo electrónico) tendrán permisos de edición y alta.
  - La lista de administradores autorizados podrá ser modificada por un super administrador o desde la configuración.
- **Gestión de Entradas (CRUD):** 
  - Redactar, editar y eliminar noticias.
  - Subir nuevos papers (PDFs), permitiendo agregar y vincular a los colaboradores desde la base de alumnos del grupo.
- **Gestión de Equipo y Grupos (CRUD):**
  - Alta, edición y baja de **grupos** dentro del grupo de investigación (ej. grupo encargado de video, sonido, etc.).
  - Gestión del **equipo de investigación** con asignación de roles (docentes y alumnos).
  - Para los **alumnos** participantes, se permitirá cargar: foto, nombre, email y perfil de LinkedIn.
- **Flujo Asistido por IA (Gemini) para Papers:** 
  - Al cargar un paper, se generará automáticamente un *prompt* que el administrador podrá copiar para ilustrar el contenido del paper con una IA generadora de imágenes.
  - El sistema usará Gemini para extraer automáticamente los metadatos relevantes y redactar un breve resumen (abstract) cautivador que sirva de anticipo a los lectores antes de descargar el PDF.
- **Monitoreo Básico:** Estado en línea/fuera de línea de los servidores de inferencia en Python.

---

## 4. Arquitectura Tecnológica y Flujo de Datos

El sistema abandona el monolito tradicional en favor de una **Arquitectura Desacoplada y Orientada a Microservicios**:

1. **Frontend (Capa de Presentación y UI):**
   - **Stack:** React.js, Vite, TypeScript, Vanilla CSS.
   - **Responsabilidad:** Renderizar la interfaz interactiva "Mobile-First", gestionar el estado del usuario en el navegador y manejar las subidas de archivos (react-dropzone).
   - **Hosting:** Vercel (CDN global).

2. **Backend BaaS (Gestión de Datos y Autenticación):**
   - **Stack:** Supabase (PostgreSQL, Auth, Storage).
   - **Responsabilidad:** Reemplaza la necesidad de escribir un servidor Node/Express. Maneja de forma segura las consultas a la base de datos (mediante Row Level Security) y guarda los PDFs/Imágenes.

3. **Backend de IA (AI API Gateway):**
   - **Stack:** Python, FastAPI, PyTorch/TensorFlow.
   - **Responsabilidad:** Recibir imágenes o textos desde el Frontend, cargarlos en la VRAM/RAM, pasarlos por la Red Siamesa o el modelo correspondiente, y devolver un JSON estandarizado con los resultados de la inferencia.
   - **Hosting:** Render, Hugging Face Spaces, o servidor local en la UTN (vía túnel seguro).

---

## 5. Identidad Visual y UX (Design System)

El diseño es un componente no negociable del proyecto, dictado por el documento `Identidad_De_Diseno.md`:
- **Geometría "Pill" (Cápsula):** Redondeo extremo (`border-radius: 9999px`) en la mayoría de contenedores iteractivos.
- **Estética Flat (Plana):** Eliminación total de sombras (`box-shadow`), favoreciendo separaciones mediante líneas finas y alto contraste de color.
- **Paleta de Colores Limitada:**
  - *Fondo:* Negro Puro (`#0A0A0A`).
  - *Superficies:* Morado (tonalidad a definir en el sistema de diseño).
  - *Acentos y CTAs:* Naranja Vibrante (`#FF5500`).
- **Comportamiento:** Transiciones CSS suaves para estados `:hover` y animaciones de entrada `fade-in-up`, garantizando fluidez sin causar reflujos (reflows) costosos en el renderizado del navegador.
