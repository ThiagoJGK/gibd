# Plan de Fase 2: Capa de Datos Híbrida y CMS de Noticias y Papers

## 1. Visión General de la Fase
En esta segunda fase se consolida la arquitectura de persistencia de la plataforma GIBD WEB. Se implementa la capa de abstracción dual en `dbService.ts` que permite a la aplicación operar con **Supabase (PostgreSQL + Storage)** en producción y degradar a **LocalStorage (Mock Mode)** en desarrollos locales. Sobre esta base, se completa la gestión dinámica de **Noticias** y **Papers Académicos con Co-Autores vinculados**.

> [!IMPORTANT]
> **Objetivo No-Dev CMS:** Los administradores deben poder publicar noticias, cargar PDFs de papers y asociar múltiples co-autores mediante casillas de verificación interactivas en `/admin` sin modificar código.

---

## 2. Distribución de Responsabilidades (3 Desarrolladores)

### **Desarrollador 1 (Thiago Gomez Kehler - Tech Lead & Backend/Data)**
- **SQL & Base de Datos:** Actualizar `database_schema.sql` con los 24 investigadores semilla, definir las políticas RLS y ejecutar el script en Supabase.
- **Capa de Abstracción de Datos:** Construir `dbService.ts` implementando la detección de entorno y los manejadores para Supabase Client y LocalStorage.
- **Gestión de Storage/Archivos:** Crear la función `uploadToStorage` adaptativa para manejar imágenes y PDFs.

### **Desarrollador 2 (Emanuel Davezac - Full-Stack Noticias)**
- **Servicio de Noticias:** Desarrollar `getNoticias` y `saveNoticia` en `dbService.ts`.
- **Integración Frontend Landing:** Modificar `Landing.tsx` para consumir noticias reactivamente desde el servicio.
- **CMS Admin Noticias:** Conectar el formulario de Noticias en `Admin.tsx` para permitir creación y persistencia con subida de imágenes.

### **Desarrollador 3 (Renato Bonín - Full-Stack Papers e Investigadores)**
- **Servicio de Papers & Autores:** Desarrollar `getMiembrosEquipo`, `getPapers` y `savePaper` en `dbService.ts` garantizando consultas relacionales (joins).
- **Integración Frontend Papers:** Modificar `Papers.tsx` para consumir datos relacionales reactivos y aplicar filtros por categoría y autor.
- **CMS Admin Papers & Asistente IA:** Conectar el formulario de Papers en `Admin.tsx` con selección multi-autor y el botón de asistencia Gemini para autocompletar abstracts y prompts visuales.

---

## 3. Tareas Detalladas de la Fase 2

- [ ] **Tarea 2.1: Semilla de Base de Datos y RLS** (Dev 1)
  - Archivo: `Contexto/database_schema.sql`
- [ ] **Tarea 2.2: Capa dbService.ts Base** (Dev 1)
  - Archivo: `frontend/src/utils/dbService.ts`
- [ ] **Tarea 2.3: Módulo de Noticias Full-Stack** (Dev 2)
  - Archivos: `frontend/src/utils/dbService.ts`, `frontend/src/pages/Landing.tsx`, `frontend/src/pages/Admin.tsx`
- [ ] **Tarea 2.4: Módulo de Papers y Co-Autores Full-Stack** (Dev 3)
  - Archivos: `frontend/src/utils/dbService.ts`, `frontend/src/pages/Papers.tsx`, `frontend/src/pages/Admin.tsx`

---

## 4. Plan de Comprobación, Validación y Testing

### A. Pruebas Automáticas y Compilación
```powershell
cd frontend
npm run build
```
*Criterio de Aceptación:* Compilación 100% limpia sin errores TypeScript.

### B. Pruebas Manuales de Funcionamiento
1. **Verificación en Landing y Papers:** Comprobar la lectura dinámica de noticias y papers.
2. **Verificación en CMS Admin:** Publicar noticia y paper asociando múltiples co-autores. Verificar persistencia tras F5.

---

## 5. Protocolo de Finalización y Subida a Git

1. **Revisión de Código Interna (Peer Review)**
2. **Ejecución de Build:** `npm run build`
3. **Consulta de Subida (Regla Global):**
   - **PREGUNTAR AL USUARIO:** "La Fase 2 (CMS de Noticias y Papers) ha sido validada localmente con éxito. ¿Deseas proceder con la subida a Git de esta fase?"
4. **Git Push & Auto-Deploy:** Tras recibir la confirmación explícita del usuario.
