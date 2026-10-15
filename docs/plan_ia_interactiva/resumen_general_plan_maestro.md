# 🚀 Plan Maestro GIBD 2026: IA Interactiva en Vivo
## Resumen Ejecutivo para Todo el Equipo

> **Tech Lead & Arquitecto:** Thiago Gomez Kehler  
> **Backend & IA:** Renato Bonín  
> **Frontend & UI:** Emmanuel Davezac  
> **Destinatarios:** Todo el equipo de desarrollo e investigadores del GIBD  
> **Fecha:** Septiembre 2026  

---

## 🎯 1. ¿Qué Estamos Construyendo y Por Qué Cambiamos el Rumbo?

El GIBD (Grupo de Investigación en Big Data de la UTN FRCU) ha producido investigaciones científicas de primer nivel:
* Modelos de redes siamesas para **marcas de ganado vacuno** (*CACIC 2023*).
* Búsqueda por similitud de **escudos y logos de clubes de fútbol** (*CONAISII 2023*).
* Análisis semántico de **normativa legal con PTAH-Jurídico** (*G3 Texto*).
* Retrieval de **audio y eventos acústicos** (*G1 Sonido*).
* Clasificación temporal y **video retrieval de violencia/dinamismo** (*G2 Video*).
* Métodos de **reducción dimensional** para proyectar espacios vectoriales complejos (*G5*).

Hasta ahora, la plataforma web dedicaba mucho tiempo a secciones institucionales y CMS de noticias tradicionales. **Ese enfoque se pausa a partir de hoy.** 

El nuevo objetivo es directo y contundente:  
**Transformar la web en un Laboratorio Científico Interactivo de IA en Tiempo Real**, donde cualquier visitante, docente o evaluador pueda probar con sus propios archivos los modelos reales desarrollados por cada grupo del GIBD.

---

## ⚡ 2. Los Cuatro Pilares Fundamentales de Esta Etapa

1. **Cero Mocks, Cero Datos Falsos:** Todo lo que corre en la web usa datos y vectores matemáticos reales. Prohibido simular porcentajes o usar números aleatorios.
2. **Inferencia Zero-Cost en CPU:** Todos los motores corren de forma ultra-rápida (menos de 150-200 ms) en CPU convencional. No dependemos de servidores costosos con GPUs dedicadas.
3. **Corrección Matemática de los Escudos:** En modelos convolucionales previos (ResNet50 / DenseNet), las distancias se comprimían dando "porcentajes raros" (75-80% de similitud para escudos que no tenían nada que ver). Se introduce una **calibración matemática por temperatura** para que los porcentajes reflejen con total claridad la similitud real.
4. **Almacenamiento Inteligente con Cloudinary:** Para soportar subida libre de imágenes sin agotar los límites del hosting gratuito en Vercel, utilizamos Cloudinary para procesar y optimizar imágenes al vuelo (224x224, WebP, fondo neutro).

---

## 🗺️ 3. El Cronograma de 4 Semanas en un Vistazo

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ SEMANA 1: ESCUDOS CALIBRADOS + CLOUDINARY + VISIÓN CONSOLIDADA                   │
│ • Thiago: Arquitectura multimodelo y contratos UI/UX Flat.                        │
│ • Renato: Motor de escudos con calibración matemática y servicio Cloudinary.     │
│ • Emmanuel: Componente de subida Cloudinary y comparador visual de clubes.        │
├───────────────────────────────────────────────────────────────────────────────────┤
│ SEMANA 2: ESPACIO LATENTE G5 (3D/2D) + CATÁLOGOS EXPANDIDOS                      │
│ • Thiago: Diseño del visualizador tridimensional de vectores.                     │
│ • Renato: Motor de reducción dimensional (PCA/UMAP) a 2D y 3D.                    │
│ • Emmanuel: Visualizador interactivo Three.js con navegación espacial.            │
├───────────────────────────────────────────────────────────────────────────────────┤
│ SEMANA 3: TEXTO NORMATIVO (PTAH-JURÍDICO) + AUDIO Y ESPECTROGRAMAS               │
│ • Thiago: UI de consulta jurídica y reproductor de ondas acústicas.               │
│ • Renato: Búsqueda semántica de leyes en CPU y extracción de Mel-Spectrograms.    │
│ • Emmanuel: Chat interactivo normativo y visualizador Canvas de audio.            │
├───────────────────────────────────────────────────────────────────────────────────┤
│ SEMANA 4: VIDEO RETRIEVAL + PULIDO TOTAL + DESPLIEGUE FINAL                      │
│ • Thiago: Auditoría visual Flat, optimización de bundles y control de release.    │
│ • Renato: Extracción de keyframes y análisis temporal de video en CPU.           │
│ • Emmanuel: Timeline interactivo de video, responsividad móvil y pruebas UX.      │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛡️ 4. Filosofía de Verificación en Dos Niveles (V&V)

Para evitar que se apruebe código a medias o con bugs visuales, cada entrega semanal debe pasar obligatoriamente por dos filtros:

* **Nivel 1 — Pre-Verificación del Agente (Automática):**  
  El entorno ejecuta tests automatizados, verifica la compilación limpia de TypeScript (`npm run build` en 0 errores), comprueba que la latencia en CPU sea menor a 150-200 ms y valida la integridad matemática de las distancias.
* **Nivel 2 — Validación Humana Obligatoria (Todo el Equipo — Thiago, Renato y Emmanuel):**  
  La validación humana no recae solo en una persona: **cada integrante del equipo debe ejecutar y documentar su propio protocolo de validación con datos reales antes de dar una tarea por lista:**
  * **Renato (Backend):** Valida manualmente los endpoints con llamadas reales (curl/Postman), comprobando que los tiempos de respuesta en CPU sean menores a 150 ms y que no haya fallos numéricos ni formatos rotos.
  * **Emmanuel (Frontend):** Valida la interfaz en el navegador y en dispositivos móviles reales, asegurando que las microanimaciones, el feedback sonoro y el layout respondan con fluidez sin desbordamientos.
  * **Thiago (Tech Lead):** Realiza la validación integral cruzada (end-to-end), verificando la coherencia científica y estética del sistema completo, y actúa como compuerta final obligatoria antes de autorizar cualquier merge o subida a Git.

---

## 📂 5. Dónde Encuentra Cada Uno su Tarea Técnica

En esta misma carpeta (`docs/plan_ia_interactiva/`) cada integrante cuenta con su documento técnico detallado para cada semana, con contratos JSON, rutas exactas de archivos, fórmulas matemáticas y criterios de aceptación:

* **Semana 1:**
  * `semana_1_plan_thiago.md`
  * `semana_1_plan_renato.md`
  * `semana_1_plan_emmanuel.md`
* **Semana 2:**
  * `semana_2_plan_thiago.md`
  * `semana_2_plan_renato.md`
  * `semana_2_plan_emmanuel.md`
* **Semana 3:**
  * `semana_3_plan_thiago.md`
  * `semana_3_plan_renato.md`
  * `semana_3_plan_emmanuel.md`
* **Semana 4:**
  * `semana_4_plan_thiago.md`
  * `semana_4_plan_renato.md`
  * `semana_4_plan_emmanuel.md`

¡A construir la mejor vitrina científica interactiva de la UTN!
