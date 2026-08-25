# 🔀 Estrategia de Control de Versiones - GIBD WEB 2026

> **Para:** Todo el equipo (Thiago, Emmanuel, Renato)
> **Fecha:** 24/08/2026

---

## 1. Modelo de Ramas (Git Branching Strategy)

Usamos un modelo **Feature Branch** simplificado basado en `main`:

```
main (producción - despliega automáticamente en Vercel)
 │
 ├── dev  ← rama de integración (aquí se unen los features antes de ir a main)
 │    │
 │    ├── feature/fase1-gateway       (Renato - Subfase 1.1)
 │    ├── feature/fase1-wrapper       (Emmanuel - Subfase 1.2)
 │    └── feature/fase1-laboratorio   (Thiago - Subfases 1.3 y 1.4)
 │
 │   (Al completar la Fase 1, se hace merge de dev → main)
 │
 ├── dev  ← se reutiliza para la siguiente fase
 │    ├── feature/fase2-xxx
 │    └── ...
 │
 └── ... (y así sucesivamente por cada fase)
```

### Ramas que **ya existen** y qué hacer con ellas:

| Rama | Estado | Acción |
|------|--------|--------|
| `main` | ✅ Actualizada con la Fase 2 completa | **No tocar directamente.** Solo recibe merges desde `dev`. |
| `feature/obsidian-vault` | ✅ Ya fue mergeada a `main` | Se puede eliminar. Ya cumplió su función. |
| `feature/noticias` | ✅ Ya fue mergeada | Se puede eliminar. El trabajo de Emmanuel ya está en `main`. |
| `feature/papers` | ✅ Ya fue mergeado | Se puede eliminar. El trabajo de Renato ya está en `main`. |
| `dev` | ✅ Creada e inicializada por Thiago | Rama de integración activa para la Fase 1. |

---

## 2. ¿Qué rama uso yo?

### **Renato Bonín** → `feature/fase1-gateway`
```powershell
git checkout main
git pull origin main
git checkout -b dev            # solo si dev no existe aún
git checkout -b feature/fase1-gateway
# ... trabajar en la Subfase 1.1 ...
git add .
git commit -m "feat(backend): implement siamese inference endpoint"
git push origin feature/fase1-gateway
```

### **Emmanuel Davezac** → `feature/fase1-wrapper`
```powershell
git checkout main
git pull origin main
git checkout dev               # o crear si no existe: git checkout -b dev
git checkout -b feature/fase1-wrapper
# ... trabajar en la Subfase 1.2 ...
git add .
git commit -m "feat(frontend): create PythonAppWrapper component"
git push origin feature/fase1-wrapper
```

### **Thiago Gomez Kehler** → `feature/fase1-laboratorio`
```powershell
git checkout main
git pull origin main
git checkout dev
git checkout -b feature/fase1-laboratorio
# ... trabajar en las Subfases 1.3 y 1.4 ...
```

---

## 3. Flujo Paso a Paso

### Paso 1: Crear la rama `dev` (solo la primera vez) - [✓ COMPLETADO POR THIAGO]

La rama `dev` ya ha sido creada a partir de `main` e inicializada en el repositorio remoto (`origin/dev`). Ningún otro integrante necesita realizar este paso.

### Paso 2: Cada desarrollador crea su rama feature desde `dev`
```powershell
git checkout dev
git pull origin dev
git checkout -b feature/fase1-<nombre-de-tu-tarea>
```

### Paso 3: Trabajar y hacer commits en tu rama
```powershell
# Después de cada avance significativo:
git add .
git commit -m "feat(módulo): descripción breve del cambio"
git push origin feature/fase1-<nombre-de-tu-tarea>
```

### Paso 4: Sincronizar con `dev` si pasaron varios días
```powershell
# Traer los últimos cambios de dev a tu rama:
git checkout feature/fase1-<tu-rama>
git pull origin dev
# Si hay conflictos, resolverlos y hacer commit.
```

### Paso 5: Integrar tu feature en `dev` (cuando terminaste tu subfase)
```powershell
git checkout dev
git pull origin dev
git merge feature/fase1-<tu-rama>
# Resolver conflictos si los hay
git push origin dev
```
**⚠️ IMPORTANTE:** Antes de hacer merge a `dev`, verificar que `npm run build` compila sin errores.

### Paso 6: Merge de `dev` a `main` (solo al completar toda la Fase)
```powershell
# Solo Thiago ejecuta esto, tras verificación del equipo:
git checkout main
git pull origin main
git merge dev
git push origin main
# → Vercel despliega automáticamente
```

---

## 4. Convención de Nombres de Commits

Usar el formato **Conventional Commits**:

```
<tipo>(alcance): descripción breve
```

### Tipos permitidos:
| Tipo | Uso |
|------|-----|
| `feat` | Nueva funcionalidad |
| `fix` | Corrección de un bug |
| `docs` | Cambios en documentación |
| `style` | Cambios de formato/estilo (no afectan lógica) |
| `refactor` | Reestructuración de código sin cambiar comportamiento |
| `test` | Añadir o modificar tests |
| `chore` | Tareas de mantenimiento (dependencias, configs) |

### Ejemplos:
```
feat(backend): implement siamese inference endpoint
feat(frontend): create PythonAppWrapper component
fix(auth): improve mock detection for local dev
docs(fases): add subfase 1.2 technical guide
chore(deps): update supabase-js to v2.48
```

---

## 5. Reglas de Oro

1. **NUNCA hacer push directo a `main`.** Siempre pasar por `dev`.
2. **NUNCA hacer `git push --force`** en ramas compartidas (`main`, `dev`).
3. **Siempre hacer `git pull` antes de empezar a trabajar** para tener la última versión.
4. **Hacer commits frecuentes y pequeños** (no acumular días de trabajo en un solo commit gigante).
5. **Ejecutar `npm run build` antes de cada merge** para verificar que no hay errores de TypeScript.
6. **Si hay un conflicto que no sabés resolver**, consultá a Thiago antes de hacer el merge.

---

## 6. Resumen Visual del Flujo

```
  Tu Rama Feature         →       dev          →       main
  (tu trabajo diario)         (integración)        (producción)
                                                        │
       feature/fase1-gateway ──┐                        │
       feature/fase1-wrapper ──┤──→ merge a dev ──→ merge a main ──→ Vercel
       feature/fase1-lab ──────┘     (al terminar        (al terminar
                                      tu subfase)        toda la Fase)
```

---

## 7. ¿Qué hacer con las ramas viejas? - [✓ COMPLETADO POR THIAGO]

Las ramas remotas y locales `feature/noticias`, `feature/papers` y `feature/obsidian-vault` **ya han sido eliminadas** por Thiago para mantener el repositorio limpio. Ningún desarrollador tiene que realizar acciones sobre ellas.

---

## 8. Preguntas Frecuentes

### ¿Puedo trabajar en la misma rama que otro compañero?
**No.** Cada desarrollador tiene su propia rama `feature/`. Esto evita conflictos constantes y permite trabajar en paralelo sin interferencias.

### ¿Qué pasa si necesito código que hizo otro compañero y todavía no está en `dev`?
Pedile que suba su rama y hacé un `git merge` de su rama a la tuya temporalmente. Pero lo ideal es esperar a que esté en `dev`.

### ¿Cada cuánto debo hacer commit?
**Al menos una vez por sesión de trabajo.** Lo ideal es un commit por cada tarea o cambio significativo completado.

### ¿Puedo crear archivos `.env.local` con las credenciales de Supabase?
Sí, pero **NUNCA** subir `.env.local` a Git. Ya está en `.gitignore`, pero verificá que no aparezca al hacer `git status`.
