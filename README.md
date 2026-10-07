# GIBD

Grupo de Investigación en Big Data — plataforma web (Noticias, Laboratorio de visión por computadora y Panel CMS).

## Estructura del repositorio

```
gibd/
├── backend/          # API Gateway FastAPI (Python)
│   ├── app/main.py   #   endpoints: health + análisis de papers con Gemini
│   └── requirements.txt
├── frontend/         # SPA React + Vite + TypeScript + Tailwind 4
│   ├── src/          #   páginas, componentes, servicios (Supabase, Cloudinary)
│   └── .env.example  #   plantilla de variables de entorno
├── docs/             # walkthroughs y documentación de fases
└── Contexto/         # material de contexto del proyecto
```

## Prerrequisitos

| Herramienta | Versión mínima | Verificado con |
|-------------|----------------|----------------|
| Python      | 3.11           | 3.14.8         |
| Node.js     | 20             | 26.10.0        |
| npm         | 10             | 11.19.1        |

---

## 1. Backend — API Gateway (FastAPI)

Levanta el servidor de puerta de enlace que consume Gemini y expone el health check usado por el frontend.

### 1.1 Instalación (una sola vez)

```powershell
cd c:\Users\Usuario\Desktop\gibd\backend
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows (PowerShell)
# source venv/bin/activate           # macOS / Linux
pip install -r requirements.txt
```

> **Nota (Python 3.14):** los pines originales del `requirements.txt` (`pydantic==2.7.4`) no compilan en Python ≥ 3.13 porque `pydantic-core` no trae wheel precompilado. El archivo ya quedó actualizado con versiones probadas (`fastapi==0.142.4`, `pydantic==2.13.5`, `uvicorn==0.54.0`).

### 1.2 Variables de entorno (opcional)

Creá `backend/.env`:

```bash
# Sin esta clave el gateway funciona igual, pero con respuesta MOCK en /ai/analyze-paper
GEMINI_API_KEY=tu_clave_de_google_ai_studio
```

### 1.3 Ejecutar

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Alternativas equivalentes:

```powershell
python -m uvicorn app.main:app --reload     # con el venv activado
python app/main.py                          # ejecuta el bloque __main__ (sin --reload)
```

### 1.4 Verificar

| Qué | URL | Respuesta esperada |
|-----|-----|--------------------|
| Health check | <http://127.0.0.1:8000/api/v1/health> | `{"status":"healthy","gemini_active":...}` |
| Docs interactivas (Swagger) | <http://127.0.0.1:8000/docs> | UI de endpoints |

### 1.5 Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET`  | `/api/v1/health` | Health check; indica si `GEMINI_API_KEY` está activa |
| `POST` | `/api/v1/ai/analyze-paper` | Analiza el texto de un paper y devuelve `abstract` + `image_prompt` (Gemini `gemini-1.5-flash`; sin clave devuelve mock) |

- **CORS:** el gateway solo acepta pedidos desde `http://localhost:5173` / `http://127.0.0.1:5173` (el puerto del frontend en dev).
- **Inferencia de visión:** los endpoints `/api/v1/inference/siamese-brands` y `/api/v1/inference/soccer-crests` (modelos de las subfases 1.1 / 1.2) **todavía no están implementados en este gateway**. Hasta que se incorporen, el Laboratorio usa automáticamente el fallback local `CREST_SAMPLES` (datos de demostración).

---

## 2. Frontend — SPA (React + Vite)

### 2.1 Instalación (una sola vez)

```powershell
cd c:\Users\Usuario\Desktop\gibd\frontend
npm install
```

### 2.2 Variables de entorno

Copiá la plantilla y completá los valores reales:

```powershell
Copy-Item .env.example .env
```

| Variable | Descripción |
|----------|-------------|
| `VITE_SUPABASE_URL` / `VITE_SUPABASE_ANON_KEY` | Proyecto Supabase (CMS de Noticias). **Sin ellos la app entra en Mock Mode** (contenido en localStorage) |
| `VITE_API_URL` | URL del backend, ej. `http://localhost:8000` |
| `VITE_CLOUDINARY_CLOUD_NAME` | Cloud de Cloudinary, ej. `gqhvqr2m` |
| `VITE_CLOUDINARY_UPLOAD_PRESET` | Preset **Unsigned**, ej. `GIBD_IMAGES` |

> ⚠️ Vite solo lee `.env` **al iniciar el servidor**. Después de editar, reiniciá `npm run dev`.
> ⚠️ Nunca pongas API keys/secretos de Cloudinary en variables `VITE_*`: quedan expuestos en el bundle del navegador.

### 2.3 Ejecutar

```powershell
npm run dev
```

Abrí <http://localhost:5173>.

### 2.4 Scripts disponibles

| Script | Comando | Uso |
|--------|---------|-----|
| Dev server | `npm run dev` | Desarrollo con hot-reload |
| Build de producción | `npm run build` | Type-check (`tsc -b`) + bundle en `dist/` |
| Lint | `npm run lint` | ESLint sobre todo el código |
| Preview del build | `npm run preview` | Sirve `dist/` localmente |

---

## 3. Verificación rápida (ambos levantados)

1. Backend en <http://127.0.0.1:8000/api/v1/health> → JSON `healthy`.
2. Frontend en <http://localhost:5173> carga sin errores en consola.
3. **Laboratorio → subir imagen** → aparece el chip *"Cloudinary: imagen lista"* (subida real al cloud `gqhvqr2m` con preset `GIBD_IMAGES`).
4. **Laboratorio → analizar** → con el backend corriendo responde; sin modelos de inferencia, usa el fallback demo.
5. **Noticias** → sin Supabase configurado entra en Mock Mode (guarda en localStorage).

---

## 4. Solución de problemas

| Síntoma | Causa / solución |
|---------|------------------|
| `Failed building wheel for pydantic-core` | Python ≥ 3.13 con pines viejos → usá el `requirements.txt` actualizado de este repo |
| `Upload preset must be whitelisted for unsigned uploads` | El preset en la consola de Cloudinary no está en modo **Unsigned** |
| `Upload preset not found` | El nombre en `VITE_CLOUDINARY_UPLOAD_PRESET` no coincide exactamente con el del preset |
| Cambios en `.env` no hacen efecto | Reiniciá `npm run dev` (Vite) o el proceso de uvicorn |
| `403 Forbidden` / error de CORS | El backend no está en `127.0.0.1:8000`, o el frontend se abrió en un puerto distinto de 5173 |
| El análisis de papers siempre devuelve lo mismo | Falta `GEMINI_API_KEY` en `backend/.env` → respuesta mock (verificá `gemini_active` en el health check) |

