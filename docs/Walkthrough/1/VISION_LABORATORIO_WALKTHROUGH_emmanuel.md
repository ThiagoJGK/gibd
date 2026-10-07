# Walkthrough: Sub-modelo de Visión en `Laboratorio` (Marcas vs. Escudos de Fútbol)

**Resumen rápido:**
- **Objetivo (plan T1.3):** integrar en el Laboratorio el sub-modelo de visión para **escudos de fútbol** junto al existente de **marcas de ganado**, con uploader a Cloudinary, tarjetas de resultados, muestras rápidas y animaciones.
- **Archivos clave:** [Laboratorio.tsx](frontend/src/pages/Laboratorio.tsx#L179-L248), [CloudinaryUploader.tsx](frontend/src/components/ui/CloudinaryUploader.tsx#L1-L143), [CrestResultCard.tsx](frontend/src/components/ui/CrestResultCard.tsx), [aiService.ts](frontend/src/utils/aiService.ts#L1-L76), [crestSamples.ts](frontend/src/utils/crestSamples.ts).

---

## 1. Cambios Realizados

### 1.1 `frontend/src/utils/aiService.ts` (nuevo)
- Se creó el cliente HTTP para los endpoints de inferencia del backend FastAPI:
  - `fetchBrandInference(file, topK)` → `POST /api/v1/inference/siamese-brands` ([aiService.ts](frontend/src/utils/aiService.ts#L21-L43)).
  - `fetchCrestInference(file, topK)` → `POST /api/v1/inference/soccer-crests` — agregado para la subfase T1.3.3 ([aiService.ts](frontend/src/utils/aiService.ts#L54-L76)).
- Se definieron los tipos exportados `InferenceResult` (formato unificado: `id`, `name`, `similarity_score`, `distance`, `image_url`, `logo_url?`, `country?`, `league?`), `BrandInferenceResponse` y `CrestInferenceResponse`.
- La base del API se resuelve por `VITE_API_URL` / `VITE_BACKEND_URL` con fallback a `http://localhost:8000`.
- Ambos endpoints envían `FormData` con `file` + `top_k`.

### 1.2 `frontend/src/components/ui/CloudinaryUploader.tsx` (nuevo — T1.3.1)
- Uploader unificado con **drag & drop** (`react-dropzone`) y clic para seleccionar archivo; acepta `png, jpg, jpeg, webp, svg` y valida máximo **5 MB**.
- **Previsualización instantánea**: al elegir el archivo se genera un `URL.createObjectURL` y se notifica inmediatamente al padre vía `onImageSelected(file, previewUrl)` — el usuario puede analizar sin esperar la subida.
- **Subida en paralelo a Cloudinary** (unsigned upload) solo si existen las env vars `VITE_CLOUDINARY_CLOUD_NAME` y `VITE_CLOUDINARY_UPLOAD_PRESET`; al terminar vuelve a notificar con `onImageSelected(file, previewUrl, secure_url)`.
- **Fallback local**: si Cloudinary no está configurado o falla, el `File` en memoria sigue siendo válido y el análisis funciona igual (estado `cloudinaryUrl` queda `null`).
- Estados visuales: spinner de progreso sobre la preview, badge "Cloudinary: subido" y mensajes de error.

### 1.3 `frontend/src/components/ui/CrestResultCard.tsx` (nuevo — T1.3.2)
- Tarjeta de resultado con clases del design system (`rounded-[2rem]`, badges `rounded-full`, paleta naranja `#FF5500`).
- Muestra la **imagen del resultado como principal** y la **imagen de consulta como overlay** en esquina superior izquierda (comparación directa), más nombre del club, país, liga, score de similitud y distancia ([CrestResultCard.tsx](frontend/src/components/ui/CrestResultCard.tsx#L41-L47)).

### 1.4 `frontend/src/utils/crestSamples.ts` (nuevo — T1.3.4)
- `CREST_SAMPLES`: array de 5 muestras (Boca Juniors, River Plate, FC Barcelona, Real Madrid, Juventus) con `id`, `name`, `country`, `league`, `imageUrl` y `badgeUrl` (fallback visual del resultado).

### 1.5 `frontend/src/pages/Laboratorio.tsx` (integración — T1.3.3 / T1.3.5)
- **Estado nuevo:**
  - `selectedVisionSubModel` (`'brands' | 'soccer_crests'`, arranca en `brands`) ([L179](frontend/src/pages/Laboratorio.tsx#L179)).
  - `results: InferenceResult[]`, `queryImageUrl`, `cloudinaryUrl`, `isAnalyzing`.
  - Se eliminó el estado duplicado `isDragging` que impedía compilar.
- **`handleAnalyze`** ([L208-L248](frontend/src/pages/Laboratorio.tsx#L208-L248)):
  - `brands` → `fetchBrandInference(uploadedFile, topK)` y `setResults(response.results)`.
  - `soccer_crests` → si hay archivo, intenta `fetchCrestInference`; **si el endpoint falla o no existe, hace fallback a `CREST_SAMPLES` local** con scores simulados decrecientes (`100 - index*15`). Si no hay archivo (usuario eligió una muestra rápida), también usa resultados locales.
- **JSX:**
  - El bloque de upload ahora renderiza `<CloudinaryUploader>` dentro del `<article>` con los handlers de drag existentes ([L539-L556](frontend/src/pages/Laboratorio.tsx#L539-L556)).
  - Selector de sub-modelo en cápsulas: **Marcas de Ganado** / **Escudos de Fútbol** con estado activo naranja ([L559-L580](frontend/src/pages/Laboratorio.tsx#L559-L580)).
  - Chip de estado "Cloudinary: imagen lista" cuando hay URL remota ([L582](frontend/src/pages/Laboratorio.tsx#L582)).
  - **Carrusel "Muestras Rápidas"** visible solo en modo escudos: clic en una muestra setea `queryImageUrl` y limpia resultados/archivo ([L588-L627](frontend/src/pages/Laboratorio.tsx#L588-L627)).
  - **Botón de acción dinámico**: "Buscar Similitudes" (marcas) / "Buscar Escudos Similares" (escudos) / "Analizando..."; el `disabled` exige archivo en marcas, y archivo **o** muestra en escudos ([L630-L648](frontend/src/pages/Laboratorio.tsx#L630-L648)).
  - **Grid de resultados** con `CrestResultCard` y **animación stagger** `opacity: 0, y: 15` → `opacity: 1, y: 0`, `delay: i * 0.05` (T1.3.5) ([L650-L681](frontend/src/pages/Laboratorio.tsx#L650-L681)).
- **Limpieza requerida por `noUnusedLocals`:** se eliminaron `Microscope` (import) y `REFERENCE_IMAGES` (constante sin uso).

### 1.6 Dependencias y entorno
- `node_modules` no existía → se ejecutó `npm install` (187 paquetes instalados).
- **Cloudinary configurado y verificado (E2E):** preset **unsigned `GIBD_IMAGES`** en el cloud `gqhvqr2m`; subida real probada contra `POST https://api.cloudinary.com/v1_1/gqhvqr2m/image/upload` → respuesta JSON con `secure_url`. Las variables quedaron en `frontend/.env` (ignorado por git); `.env.example` documentado.
- **Backend:** `backend/requirements.txt` actualizado a versiones probadas (`fastapi==0.142.4`, `pydantic==2.13.5`, `uvicorn==0.54.0`) porque los pines originales (`pydantic==2.7.4`) no compilaban en Python 3.14. Guía completa de instalación y ejecución agregada al `README.md` del repo.

---

## 2. Bitácora de Validación (automatizada)

- [x] **TypeScript:** `node node_modules/typescript/bin/tsc -b --force` → exit code **0** (0 errores).
- [x] **Build de producción:** `vite build` → exit code **0** (`dist/assets/index-*.js 743 kB`, warning preexistente de chunk size > 500 kB).
- [x] **ESLint archivos nuevos/modificados:** `CloudinaryUploader.tsx`, `CrestResultCard.tsx`, `aiService.ts`, `crestSamples.ts` → **0 problemas**.
- [x] **ESLint `Laboratorio.tsx`:** 6 errores restantes son **preexistentes** (verificados contra `git show HEAD:...`: mismo `window as any`, `catch (e)` sin usar y `prefer-const` en `angleDeg`); esta implementación no agregó nuevos.
- [x] **Cloudinary (subida real E2E):** `curl` al endpoint de upload con `upload_preset=GIBD_IMAGES` → HTTP 200 con `secure_url` (PNG 1×1, 70 B). Hubo 2 iteraciones de diagnóstico previas: preset en modo *Signed* → `must be whitelisted for unsigned uploads`; nombre `GIBD` → `Upload preset not found`.
- [x] **Backend FastAPI:** `uvicorn app.main:app` en `127.0.0.1:8000` → `GET /api/v1/health` responde `{"status":"healthy","service":"GIBD API Gateway","gemini_active":false}` (sin `GEMINI_API_KEY` el endpoint de papers responde mock).
- Correcciones aplicadas durante la validación:
  - `useState<any[]>` de `results` → `useState<InferenceResult[]>` (tipado estricto).
  - Se extrajo y exportó la interfaz `InferenceResult` desde `aiService.ts` para reutilizarla.
  - Se resolvieron conflictos de codificación UTF-8 (tildes/ñ) leyendo el contenido exacto del archivo antes de editar.

---

## 3. Modo Mock y degradaciones locales (aclaración importante)

En el proyecto conviven **dos mecanismos de respaldo distintos** que no conviene confundir:

### A) Mock Mode de Supabase → solo afecta al CMS (Noticias / Papers / Equipo)
- **Controlado por:** `VITE_SUPABASE_URL` y `VITE_SUPABASE_ANON_KEY` en `frontend/.env`.
- **Qué hace:** si faltan credenciales reales, `dbService.ts` persiste todo en `localStorage` (con siembra de datos mock) y la app arranca **sin pantalla negra ni errores**. Es el "Modo Desarrollador Local" descrito en `.env.example` y en los walkthroughs de Noticias/Papers.
- **Qué NO afecta:** **no tiene nada que ver con el Laboratorio ni con los modelos de visión** — el sub-modelo de escudos no usa Supabase en ningún punto.
- **Cómo detectarlo:** el CMS funciona en `/#/admin` con datos locales que persisten tras recargar (F5).

### B) Fallbacks locales del Laboratorio (visión) → degradación por capas
Estos dos respaldos son propios de esta implementación (T1.3) y funcionan **sin importar el estado de Supabase**:

1. **Cloudinary sin configurar o con error** → `CloudinaryUploader` trabaja con el `File` en memoria (preview instantánea); no aparece el chip "Cloudinary: imagen lista" ni el badge "Cloudinary: subido". **El análisis funciona igual.**
2. **Endpoint de escudos caído / backend apagado** → `handleAnalyze` cae a `CREST_SAMPLES` local con scores simulados decrecientes. En consola se ve: `Endpoint de escudos no disponible; se usan muestras locales.` (demo intencional T1.3.3/T1.3.4).
3. **Marcas sin backend** → **no hay fallback**: solo `console.error` y el grid queda vacío (limitación conocida; ver paso 7 de la sección 5).

### Resumen rápido de estados

| Configuración | Comportamiento resultante |
|---|---|
| `.env` sin claves Supabase | **Mock Mode** en el CMS (localStorage) — Laboratorio sin cambios |
| `.env` sin `VITE_CLOUDINARY_*` | Subida local al 100% (archivo en memoria) |
| Backend `localhost:8000` **apagado** | Escudos → muestras locales demo; Marcas → grid vacío + error en consola |
| Backend **encendido** | Inferencia real en ambos sub-modelos |
| Todo configurado (Supabase + Cloudinary + backend) | Modo producción completo |

---

## 4. Guía de Verificación Manual (Paso a Paso)

1. **Levantar el backend** (recomendado; guía completa en el `README.md` §1):

```powershell
cd c:/Users/Usuario/Desktop/gibd/backend
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Verificar `http://127.0.0.1:8000/api/v1/health` → `{"status":"healthy",...}`. Sin `GEMINI_API_KEY` la inferencia de papers responde mock (ok para pruebas).

2. **Levantar el frontend:**

```powershell
cd c:/Users/Usuario/Desktop/gibd/frontend
npm install
npm run dev
```

3. **Abrir el Laboratorio** en `http://localhost:5173/#/laboratorio` y seleccionar el medio **Imagen** en el dial.
4. **Flujo de Marcas (default):**
   - Subir una foto de marca/ganado (clic o drag & drop). Verificar preview inmediata.
   - Si `.env` tiene `VITE_CLOUDINARY_CLOUD_NAME` + `VITE_CLOUDINARY_UPLOAD_PRESET`: aparece el badge "Cloudinary: subido" y el chip "Cloudinary: imagen lista". Sin esas variables, verificar que igualmente funciona con el archivo local.
   - Ajustar el slider **Top-K**, clic en **BUSCAR SIMILITUDES**. Con backend corriendo (`localhost:8000`) se ven resultados reales; sin backend, revisar la consola.
5. **Flujo de Escudos:**
   - Cambiar a **ESCUDOS DE FÚTBOL**. Verificar que aparece el carrusel **Muestras Rápidas**.
   - **Caso A (endpoint activo):** subir una imagen de escudo y clic en **BUSCAR ESCUDOS SIMILARES** → grid con resultados de `/api/v1/inference/soccer-crests`.
   - **Caso B (endpoint caído / sin backend):** debe caer al fallback local y mostrar las 5 muestras con scores decrecientes (ver consola: `Endpoint de escudos no disponible...`).
   - **Caso C (muestra rápida):** clic en "Boca Juniors" → se setea como consulta y el análisis devuelve resultados locales sin pedir archivo.
   - Verificar animación stagger de las tarjetas al aparecer.

---

## 5. Siguientes Pasos

### Inmediatos (esta semana)
1. ~~**Configurar variables de entorno**~~ — **HECHO al 100%:** `frontend/.env.example` documentado (`VITE_API_URL`, `VITE_CLOUDINARY_*`) y `frontend/.env` local (ignorado por git) con **credenciales reales**: Supabase + Cloudinary cloud `gqhvqr2m` / preset unsigned `GIBD_IMAGES`. Subida **verificada E2E con `curl`** (JSON con `secure_url`) y variables documentadas también en el `README.md`.
2. **Prueba manual E2E de la UI** con el backend FastAPI levantado (`localhost:8000`) cubriendo los 3 casos de la sección 4. *(Progreso: la subida a Cloudinary y el health check del backend ya están verificados por fuera de la UI; falta recorrer los flujos desde el navegador.)*
3. **Corregir los 6 errores ESLint preexistentes** de `Laboratorio.tsx` (higiene): reemplazar `window as any` por un type guard, usar `catch { }` sin binding en los bloques vacíos y `angleDeg` a `const`.
4. ~~**Commit + push**~~ — **HECHO:** commit `a205944` en `feature/sprint1-frontend-emmanuel` (8 archivos) y push exitoso al remoto. **Pendiente:** abrir el PR ([link](https://github.com/ThiagoJGK/gibd/pull/new/feature/sprint1-frontend-emmanuel)).
5. ~~**Documentar cómo correr backend y frontend**~~ — **HECHO:** `README.md` reescrito con prerrequisitos, instalación, ejecución, endpoints, variables de entorno y troubleshooting; `backend/requirements.txt` actualizado para Python 3.14.

### Corto plazo (subfase 1.3 → 1.4)
5. **Reemplazar las imágenes placeholder de `crestSamples.ts`** — hoy las 5 muestras comparten la misma URL de Wikimedia; usar escudos oficiales (con `badgeUrl` real) y verificar hotlink policy (posibles 403 en producción).
6. **Endpoints reales de escudos**: confirmar con el equipo de ML que `/api/v1/inference/soccer-crests` esté desplegado y responda el mismo shape `InferenceResult` (hoy el contrato está supuesto por simetría con `siamese-brands`).
7. **Estado de error visible en UI**: hoy un fallo de inferencia de marcas solo hace `console.error` sin mostrar nada al usuario — agregar un banner/toast de error en `Laboratorio.tsx`.
8. **Manejo de `topK` en escudos**: validar que el backend respeta `top_k` en el nuevo endpoint.

### Siguiente subfase (T1.4 — según plan)
9. Slider Top-K + métricas de calidad (precisión/latencia) en la UI.
10. Code-splitting: el bundle `index-*.js` pesa 743 kB; considerar `React.lazy` para `Laboratorio` (el warning de Vite ya lo señala).

---

## 6. Checklist para merge / CI

- [x] `tsc -b --force` pasa sin errores.
- [x] `vite build` pasa (exit 0).
- [x] ESLint limpio en todos los archivos nuevos/modificados por esta tarea.
- [x] Variables de entorno documentadas en `.env.example` (+ `frontend/.env` local con credenciales reales, ignorado por git).
- [x] Subida a Cloudinary verificada E2E con `upload_preset=GIBD_IMAGES` (`secure_url` en respuesta).
- [x] Backend verificado: `GET /api/v1/health` → `healthy` con `requirements.txt` apto para Python 3.14.
- [x] `README.md` con guías de instalación/ejecución de backend y frontend + troubleshooting.
- [ ] Prueba manual E2E de la UI con backend corriendo (3 casos de la sección 4).
- [x] Commit y push de los 5 archivos + docs (`a205944`); segundo commit de docs/README/requirements pendiente en esta iteración.
- [ ] PR hacia la rama base y review.

---

## 7. Notas / Supuestos

- **Modo Mock vs. fallbacks del Laboratorio:** no confundirlos — el **Mock Mode** es solo de Supabase/CMS (localStorage, ver sección 3.A); los respaldos del Laboratorio (Cloudinary local y muestras de escudos) son independientes y propios de T1.3 (sección 3.B).
- **Gateway actual:** `backend/app/main.py` solo expone `GET /api/v1/health` y `POST /api/v1/ai/analyze-paper` (Gemini). Los endpoints de inferencia `siamese-brands` / `soccer-crests` **aún no están implementados ahí** (modelos de subfases 1.1/1.2, pendientes de integración) — por eso en la UI se activan los fallbacks descritos en la sección 3.B.
- El contrato de `/api/v1/inference/soccer-crests` se asumió idéntico al de `siamese-brands` (FormData `file` + `top_k`, respuesta con `results[]`); si el backend difiere, ajustar `fetchCrestInference` en `aiService.ts`.
- Cloudinary es **opt-in**: sin las env vars, la app funciona 100% local (preview + `File` en memoria). No hay credenciales en el repo por diseño.
- El fallback local de escudos (scores simulados) es intencional para demos sin backend — corresponde a T1.3.3/T1.3.4 del plan; convendrá retirarlo o marcarlo como "demo" cuando el endpoint esté productivo.

