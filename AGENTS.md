# 🤖 Reglas de Orden y Disciplina Operativa para Agentes de IA — GIBD 2026

Este documento contiene las directrices obligatorias e inviolables para cualquier agente de Inteligencia Artificial (Antigravity, Gemini, Cursor, Copilot, Claude) que trabaje en el repositorio **GIBD WEB**.

---

## 👑 1. Cadena de Mando y Jerarquía del Proyecto

1. **Thiago Gomez Kehler** es el **Tech Lead, Arquitecto de Software y Diseñador UI/UX Principal**.
   - Posee autoridad técnica y estética absoluta sobre el código y la experiencia de usuario.
   - Todo cambio en diseño, arquitectura o integración debe satisfacer sus estándares de calidad antes de considerarse aprobado.
2. **Renato Bonín:** Backend & AI Engineer (Servicios FastAPI, Gemini AI, inferencia CPU).
3. **Emmanuel Davezac:** Frontend & CMS Admin (Interfaces de administración, formularios, responsividad).

---

## 🔬 2. Principio de Autenticidad Científica (PROHIBIDO SIMULAR DATOS)

1. **Cero Mocks Inventados:** El GIBD es un centro de investigación científica real de la UTN. Existen datos reales en la carpeta `Contexto/`:
   - Catálogo oficial de **80 marcas de ganado vacuno** (estudio CACIC 2023 / ViT).
   - Lista oficial de **24 investigadores y docentes** en el Obsidian Vault.
   - Los **6 grupos oficiales de investigación** (G1 a G6) documentados en los MOCs.
   - Dataset de **300 consultas normativas** para PTAH-Jurídico.
2. **Prohibición de Hashes Arbitrarios:** NUNCA usar generadores aleatorios o hashes MD5 sobre URLs falsas para simular similitudes. La similitud se calcula mediante vectores de características reales y distancia de coseno sobre CPU.
3. **Cero Desvíos Temáticos:** El Laboratorio del GIBD aborda Marcas de Ganado, Video, NLP Jurídico y Audio. No introducir temáticas ajenas (como patentes de automóviles).

---

## 🎨 3. Guardián del Estándar Estético: Flat Design System

Cualquier agente que modifique código en `frontend/` debe aplicar estrictamente el sistema de diseño:
1. **Geometría de Cápsula Obligatoria:**
   - Botones, pills, selectores, badges, inputs e interactivos: `rounded-full` (`border-radius: 9999px`).
   - Tarjetas, paneles y modales: `rounded-[2rem]`.
2. **Paleta Cromática Estricta:**
   - Fondo base: Negro OLED puro (`#0A0A0A` / `#000000`).
   - Acento primario: Naranja eléctrico de alto contraste (`#FF5500`).
   - Textos: `#FFFFFF` (primario), `#A1A1AA` (secundario).
3. **PROHIBICIÓN ESTRICTA DE `box-shadow` DIFUSOS:**
   - No usar sombras difusas genéricas.
   - La profundidad y jerarquía se logran mediante **microbordes orgánicos de alto contraste** (`border border-white/10` o `border border-border-organic`) y sutiles resplandores focalizados (`drop-shadow`).
4. **Animaciones:** Usar exclusivamente `framer-motion` para transiciones suaves de entrada y salida aceleradas por hardware.

---

## ⚡ 4. Arquitectura Zero-Cost y Ejecución en CPU

1. **Inferencia en CPU:** Toda funcionalidad de IA local (marcas de ganado, embeddings) debe correr de manera autónoma en CPU sin requerir tarjetas gráficas dedicadas ni incurrir en costos de nube.
2. **Pre-indexación Vectorial:** Los catálogos estáticos se pre-indexan en matrices NumPy (`.npy`) o JSON para que la inferencia en tiempo real tome menos de 150 ms.
3. **Contract-First:** Los modelos de datos de backend (`Pydantic`) y los tipos de frontend (`TypeScript`) deben coincidir exactamente según lo estipulado en [`docs/semanas/`](docs/semanas/).

---

## 🧼 5. Aislamiento y Limpieza de Código

1. **No URLs Hardcodeadas en Producción:** Queda terminantemente prohibido incluir enlaces fijos a puertos locales como `localhost:8010` o `localhost:9999` dentro de los componentes finales (`Laboratorio.tsx`, etc.).
2. **Demos Aisladas:** Prototipos de Python o scripts de visualización externa deben ubicarse en `backend/demos/` y encapsularse a través de `PythonAppWrapper.tsx` sin invadir las vistas principales.
3. **Limpieza de Archivos:** No dejar scripts temporales, imágenes huérfanas o archivos de pruebas en la raíz del proyecto.

---

## 🛡️ 6. Control de Versiones y Flujo de Git

1. **La Rama `main` es Sagrada:** Está vinculada al despliegue continuo de Vercel. NUNCA hacer commit ni push directo a `main`.
2. **Ramas de Trabajo:** Todo desarrollo se realiza en ramas `feature/*` y se integra en `dev` tras cumplir los Quality Gates.
3. **Quality Gate:**
   - `npm run build` en `frontend/` debe finalizar con código 0 y sin errores de TypeScript.
   - `python -m py_compile` debe validar todos los archivos de backend.
4. **Regla de Oro de Reprogramación:** Si una tarea no cumple los Quality Gates en la puesta en común de los viernes, **se reprograma para la semana siguiente**. No se permiten merges forzados.
5. **Regla Global Inquebrantable:**
   > Luego de grandes cambios, preguntar siempre si continuar con subida a Git.
