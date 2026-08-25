## Walkthrough: Módulo `Noticias`

**Resumen rápido:**
- **Implementación principal:** lectura y escritura de noticias con fallback mock (LocalStorage) y modo Supabase real.
- **Archivos clave:** [frontend/src/utils/dbService.ts](frontend/src/utils/dbService.ts#L1-L140), [frontend/src/pages/Admin.tsx](frontend/src/pages/Admin.tsx#L360-L420), [frontend/src/pages/Landing.tsx](frontend/src/pages/Landing.tsx#L1-L40).

**Qué se implementó**
- `uploadToStorage(file, bucket)` — sube un archivo a Supabase Storage o devuelve una URL simulada en modo mock. (ver en [dbService.ts](frontend/src/utils/dbService.ts#L1-L140)).
- `getNoticias()` — carga noticias desde `localStorage` si las variables de Supabase no están configuradas (sembrando una semilla `MOCK_NOTICIAS_SEED`), o consulta la tabla `noticias` en Supabase ordenada por `created_at` descendente. (ver [dbService.ts](frontend/src/utils/dbService.ts#L1-L140)).
- `saveNoticia(noticiaForm, noticiaFile)` — sube la imagen (si existe) y persiste la noticia en `localStorage` (modo mock) o inserta en la tabla `noticias` de Supabase. (ver [dbService.ts](frontend/src/utils/dbService.ts#L1-L140)).
- `Landing.tsx` llama a `getNoticias()` en el montaje y muestra las noticias dinámicamente. (ver [Landing.tsx](frontend/src/pages/Landing.tsx#L1-L40)).
- `Admin.tsx` expone el formulario del CMS y en `handleSaveNoticia` invoca `saveNoticia(...)` y muestra mensajes de éxito/ error. (ver [Admin.tsx](frontend/src/pages/Admin.tsx#L360-L420)).

Pruebas locales recomendadas
- Modo Mock (sin credenciales Supabase):
  - Borrar la clave de LocalStorage `gibd_mock_noticias` (si existe) para probar la siembra.
  - Levantar el frontend:

```powershell
cd frontend
npm install
npm run dev
```

  - Abrir la ruta `/admin`, crear una noticia. Verificar que aparece en la Landing y que persiste tras recargar (LocalStorage).

- Modo Supabase (con `VITE_SUPABASE_URL` y `VITE_SUPABASE_ANON_KEY`):
  - Configurar las credenciales en `.env` o en el entorno.
  - Asegurar que la tabla `noticias` existe (revisar [Contexto/database_schema.sql](Contexto/database_schema.sql#L37-L54)).
  - Probar creación de noticia desde `/admin` y ver registro en Supabase.

Correcciones y mejoras sugeridas (para garantizar que todo funcione en todos los entornos)
- Validar el formato de `getPublicUrl` según la versión del cliente Supabase: comprobar que `uploadToStorage` devuelve la propiedad pública correcta (`data.publicUrl` o `data.public_url`) y ajustar si fuese necesario.
- Añadir manejo de errores más robusto en `uploadToStorage` y en `saveNoticia` para devolver información de fallo legible (actualmente se lanza `error` directo).
- Asegurar que las variables de entorno se documentan en `README.md` (ejemplo: `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`).
- Si se usa Supabase en producción, actualizar `Contexto/database_schema.sql` para sembrar los autores y noticias necesarias y evitar errores de integridad (ver bloque `CREATE TABLE public.noticias` en [Contexto/database_schema.sql](Contexto/database_schema.sql#L37-L54)).
- Añadir tests e2e simples o una comprobación de smoke build antes de merge:

```powershell
cd frontend
npm run build
```

Checklist para merge / CI
- [ ] `npm run build` pasa en CI para la rama.
- [ ] Variables de entorno proporcionadas para despliegues con Supabase.
- [ ] (Opcional) Crear PR desde `feature/noticias` hacia la rama correspondiente y pedir review.

Notas finales
- La implementación actual del módulo `noticias` está funcional en modo mock y lista para integrarse con Supabase. Las acciones principales ya están encapsuladas en `dbService.ts`, lo que facilita pruebas y mantenibilidad.
- ¿Quieres que aplique las mejoras mencionadas automáticamente (p. ej. ajustar `getPublicUrl`, añadir mensajes de error más claros y añadir documentación de variables de entorno)?

---

**Explicación detallada: por qué usamos Modo Mock (LocalStorage)**

- Desarrollo independiente de infraestructura: el modo mock permite a desarrolladores trabajar sin depender de credenciales externas ni de la disponibilidad de Supabase, evitando bloqueos por acceso a servicios remotos.
- Reproducibilidad en entornos locales y CI: al sembrar una semilla (`MOCK_NOTICIAS_SEED`) el estado inicial es determinista, lo que facilita pruebas manuales y automatizadas sin necesidad de restaurar dumps externos.
- Reducir fricción en merges: con mock mode activo por defecto en entornos locales, los cambios en la UI y en `dbService.ts` pueden probarse sin tocar la configuración de despliegue; esto evita commits accidentales de secretos y reduce la probabilidad de conflictos por cambios en infra.
- Fallback seguro: en caso de que la integración con Supabase falle en un entorno (keys mal configuradas, bucket no disponible), el sistema cae al modo mock para evitar roturas visibles en la UI.

**Justificación técnica y de proceso para cada modificación (comparada con la rama original)**

1) `frontend/src/utils/dbService.ts` (añadidos/ajustes: `MOCK_NOTICIAS_SEED`, `uploadToStorage`, `getNoticias`, `saveNoticia`)
  - Por qué se hizo: centralizar acceso a datos en un único servicio hace que la lógica de persistencia sea reutilizable y evita duplicar consultas/insertas en componentes. Esto reduce conflictos en `Admin.tsx` y `Landing.tsx` cuando múltiples ramas modifican comportamiento de datos.
  - Cambios respecto a la rama original: se sustituyeron llamadas directas a Supabase dentro de componentes por llamadas a `dbService` (se extrajo la responsabilidad). También se añadió la semilla `MOCK_NOTICIAS_SEED` para inicializar `localStorage`.
  - Justificación: mantener contratos estables (firmas `getNoticias(): Promise<any[]>` y `saveNoticia(noticiaForm, file)`) permite que otras ramas puedan integrar sin cambiar llamadas públicas; si una rama necesita más campos en la noticia, debe extender la interfaz pero mantener compatibilidad hacia atrás.
  - Riesgos y mitigación: la función `uploadToStorage` depende de la API de Supabase que puede devolver `data.publicUrl` o `data.public_url` según versiones; se recomienda validar la propiedad y documentar la versión mínima de `@supabase/supabase-js` en `frontend/package.json` o `README.md`.

2) `frontend/src/pages/Admin.tsx` (uso de `saveNoticia` en `handleSaveNoticia`)
  - Por qué se hizo: mover la lógica de persistencia fuera del componente permite simplificar la UI y minimizar el número de líneas que otro desarrollador debe tocar — reducción directa de probabilidades de conflicto en merges.
  - Cambios respecto a la rama original: `handleSaveNoticia` ahora orquesta validaciones y UI (mensajes, estados), delegando la persistencia a `saveNoticia`.
  - Justificación: los commits en `Admin.tsx` deberían enfocarse en UX y validaciones, no en detalles de almacenamiento. En caso de conflicto, priorizar la versión que mantenga las llamadas a `saveNoticia` y preservar manejadores de estado (`setSuccessMsg`, etc.).

3) `frontend/src/pages/Landing.tsx` (consumo de `getNoticias`)
  - Por qué se hizo: la landing debe leer desde la misma fuente de verdad que el admin escribe; usar `getNoticias` garantiza consistencia entre UI y CMS.
  - Cambios respecto a la rama original: reemplazo del array estático `NEWS_ITEMS` por un fetch reactivo desde `dbService`.
  - Justificación: evita divergencias entre datos hardcodeados y los producidos por el CMS; facilita que un merge que altere la presentación no rompa la fuente de datos.

4) Seeds y `Contexto/database_schema.sql`
  - Por qué se recomienda: si se despliega en Supabase, las llaves foráneas y la integridad referencial pueden romper inserciones; sembrar los autores y una estructura mínima de `noticias` evita errores en producción.
  - Cambios respecto a la rama original: recomendación de actualizar el SQL con los 24 autores y asegurar políticas RLS compatibles.
  - Justificación: evita errores de integridad y facilita que ramas que añadan papers o autores no provoquen fallos al integrar.

**Guía práctica para que un agente/resolutor de merges pueda arreglar conflictos**

1. Mantener `dbService.ts` como la fuente de abstracción de datos.
  - Si hay conflicto entre implementaciones en `Admin.tsx` y `Landing.tsx`, prefiera conservar las llamadas a `getNoticias`/`saveNoticia` y eliminar duplicados de lógica de almacenamiento dentro de los componentes.

2. Conflictos en firmas de `dbService`:
  - Si una rama añade campos nuevos en `saveNoticia` o `getNoticias`, mantener compatibilidad creando parámetros opcionales o un tipo extendido: `interface NoticiaV2 extends Noticia { extra?: string }`.
  - Nunca romper la firma existente sin elevar el cambio por PR y actualizar todos los usos (buscar `getNoticias(` y `saveNoticia(` en el repo).

3. Conflictos en seed data (`MOCK_NOTICIAS_SEED`):
  - Unir semillas tomando la unión de entradas únicas por `title`+`date` o por `id` si está presente; evitar eliminar entradas de la semilla salvo que exista respaldo en SQL.

4. Conflictos por `uploadToStorage` y Supabase client:
  - Preferir la versión que delega en `supabase.storage` y añade el fallback mock. Si hay discrepancia en la propiedad `data.publicUrl` vs `data.public_url`, implementar una pequeña normalización: `const publicUrl = data?.publicUrl || data?.public_url;`.

5. Tests y verificación post-merge (si procede ejecutar localmente):
  - Limpiar localStorage de noticias:

```powershell
powershell -Command "Remove-Item -Path (Join-Path $env:USERPROFILE 'AppData\Local\..') -ErrorAction SilentlyContinue"
# O simplemente en la consola del navegador: localStorage.removeItem('gibd_mock_noticias')
```

  - Levantar frontend y verificar build y flujo:

```powershell
cd frontend
npm install
npm run build
npm run dev
```

  - Crear una noticia en `/admin` y verificar aparición en `/` (Landing).

6. Reglas de prioridad al resolver conflictos (resumen rápido):
  - Priorizar preservación de la abstracción (`dbService`) sobre lógica inline en componentes.
  - Mantener firmas públicas estables; si se cambian, actualizar todos los callsites y documentarlo en el PR.
  - Para seeds, unir entrada en vez de eliminar salvo que exista motivo documentado.

**Comandos útiles para el flujo de merge (ejemplo)**

```bash
# Actualizar main y crear rama de integración
git checkout feature/noticias
git fetch origin
git rebase origin/main

# Resolver conflictos localmente, probar, luego push
git add .
git rebase --continue
git push origin feature/noticias -f
```

---

Si quieres, aplico automáticamente las pequeñas mejoras sugeridas ahora mismo:
- Normalizar `getPublicUrl` en `uploadToStorage`.
- Añadir manejo de errores más legible (errores con prefijo y contexto) en `saveNoticia`.
- Añadir una nota en `README.md` con las variables de entorno mínimas.

Indícame si procedo con esos cambios y los empujo en `feature/noticias` o en otra rama dedicada.
