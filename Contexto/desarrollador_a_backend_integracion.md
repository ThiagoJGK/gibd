# Guía de Trabajo: Desarrollador A (Módulo de Noticias - Full Stack)

## 1. Introducción
Dado tu perfil técnico orientado al backend y para asegurar una distribución equitativa y libre de fricciones de trabajo, estarás a cargo del **ciclo de vida completo del módulo de Noticias (Noticias e Hitos)**. 

> [!IMPORTANT]
> **Objetivo Final (No-Dev CMS):** Tu implementación debe garantizar que investigadores y docentes sin conocimientos técnicos puedan crear, redactar y publicar noticias de forma 100% visual y autónoma a través del CMS `/admin`, incluyendo la carga directa de imágenes sin requerir intervenciones de código ni redespliegues.

Para evitar conflictos de fusión en Git al trabajar en paralelo con tu compañero, interactuarás exclusivamente a través del archivo de servicios compartido: [dbService.ts](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/frontend/src/utils/dbService.ts).

---

## 2. Flujo de Trabajo en Git

Para garantizar una integración continua sin fricciones y respetar el protocolo del repositorio, sigue estos pasos exactos:

### Paso A. Sincronización Inicial (Acceder a la Base)
Antes de comenzar a programar, asegúrate de estar en la rama de desarrollo activa (`feature/obsidian-vault`) y traer los últimos cambios (donde ya se encuentra creado el esqueleto de `dbService.ts` y las semillas SQL):
```powershell
git checkout feature/obsidian-vault
git pull --rebase origin feature/obsidian-vault
```

### Paso B. Crear Rama de Trabajo
Trabaja en una rama propia para evitar pisarte con tu compañero:
```powershell
git checkout -b feature/noticias
```

### Paso C. Flujo de Commits Semánticos (Conventional Commits)
Registra tus avances utilizando mensajes descriptivos estandarizados:
```powershell
git add .
git commit -m "feat(noticias): implement getNoticias dynamic fetch and Landing integration"
```

### Paso D. Sincronización e Integración Final
Cuando termines de implementar y validar localmente, integra tu trabajo de vuelta en la rama compartida:
```powershell
# Volver a la rama principal del proyecto
git checkout feature/obsidian-vault
git pull --rebase origin feature/obsidian-vault

# Fusionar tu rama de noticias
git merge feature/noticias

# Subir los cambios finales
git push origin feature/obsidian-vault
```
*Si surgen conflictos en `dbService.ts`, resuélvelos manualmente fusionando las firmas (las tuyas son de noticias y no se solapan con las de papers) y vuelve a ejecutar las pruebas antes de hacer push.*

---

## 3. Manual de Testing y Compilación

### A. Pruebas de Compilación
Antes de dar por terminada la tarea o hacer un push, debes validar que el frontend compila correctamente y no introduce errores de TypeScript:
```powershell
cd frontend
npm run build
```
> [!CAUTION]
> **No subir código roto:** Si el build falla con errores de tipo o importación, diagnostica el problema localmente, corrígelo y vuelve a probar. Solo se integra código que compila limpiamente.

### B. Pruebas de Funcionamiento Manual
1.  **Ejecutar Servidor Local:** Levanta el entorno de desarrollo con `npm run dev`.
2.  **Verificación de Carga Inicial:** Abre la Landing page. Deberían mostrarse las noticias semillas por defecto almacenadas en `localStorage` (debes verificar que no haya saltos visuales bruscos durante la carga).
3.  **CMS Admin:** Ingresa a `/admin` (Modo Mock). Publica una noticia nueva completando el formulario.
4.  **Verificación de Persistencia:** Vuelve a la Landing page, verifica que la nueva noticia se visualice al principio de la lista. Recarga el navegador (`F5`) y asegúrate de que la noticia siga allí (lo que confirma el guardado correcto en `localStorage`).

---

## 4. Manual Técnico de Implementación

### A. Archivos a Modificar
1.  **Lógica del Servicio:** Escribir los métodos `getNoticias` y `saveNoticia` en [dbService.ts](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/frontend/src/utils/dbService.ts).
2.  **Consumo de Noticias:** Modificar [Landing.tsx](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/frontend/src/pages/Landing.tsx) para importar y llamar a `getNoticias`.
3.  **CMS Admin:** Modificar `handleSaveNoticia` en [Admin.tsx](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/frontend/src/pages/Admin.tsx) para llamar a `saveNoticia`.

### B. Ejemplo de Firma y Flujo en dbService.ts
```typescript
// dbService.ts (Métodos para Desarrollador A)

export const getNoticias = async (): Promise<any[]> => {
  const isMock = !import.meta.env.VITE_SUPABASE_URL || import.meta.env.VITE_SUPABASE_URL === 'YOUR_SUPABASE_URL';
  
  if (isMock) {
    const localData = localStorage.getItem('gibd_mock_noticias');
    if (localData) return JSON.parse(localData);
    
    // Semilla inicial
    const seed = [/* NOTICIAS HARDCODEADAS ORIGINALES DE LANDING.TSX */];
    localStorage.setItem('gibd_mock_noticias', JSON.stringify(seed));
    return seed;
  }
  
  // Consulta Supabase Real
  const { data, error } = await supabase
    .from('noticias')
    .select('*')
    .order('created_at', { ascending: false });
  if (error) throw error;
  return data || [];
};

export const saveNoticia = async (noticiaForm: any, noticiaFile: File | null): Promise<void> => {
  const isMock = !import.meta.env.VITE_SUPABASE_URL || import.meta.env.VITE_SUPABASE_URL === 'YOUR_SUPABASE_URL';
  
  let imageUrl = noticiaForm.imageUrl;
  if (noticiaFile) {
    imageUrl = await uploadToStorage(noticiaFile, 'noticias');
  }
  
  const formattedDate = noticiaForm.date || new Date().toLocaleDateString('es-ES', { month: 'short', year: 'numeric' });

  if (isMock) {
    const current = await getNoticias();
    const newNoticia = {
      id: Math.random().toString(36).substring(2),
      title: noticiaForm.title,
      content: noticiaForm.content,
      tag: noticiaForm.tag,
      date: formattedDate,
      image_url: imageUrl
    };
    localStorage.setItem('gibd_mock_noticias', JSON.stringify([newNoticia, ...current]));
  } else {
    const { error } = await supabase.from('noticias').insert({
      title: noticiaForm.title,
      content: noticiaForm.content,
      tag: noticiaForm.tag,
      date: formattedDate,
      image_url: imageUrl || null
    });
    if (error) throw error;
  }
};
```

---

## 5. Tareas Asignadas (Backlog)

- [ ] **Tarea A.1: Configuración de Base de Datos (Noticias)**
  - Asegurar la creación de la tabla `noticias` en el SQL Editor de Supabase y verificar las políticas RLS.
- [ ] **Tarea A.2: Implementar Lógica de Noticias en dbService.ts**
  - Escribir e integrar los métodos `getNoticias` y `saveNoticia` en el archivo de servicios compartido.
- [ ] **Tarea A.3: Consumo Dinámico en Landing.tsx**
  - Reemplazar la constante estática de novedades en `Landing.tsx` por estados reactivos y llamar a `getNoticias` al montar el componente.
- [ ] **Tarea A.4: Integración en Admin.tsx**
  - Modificar `handleSaveNoticia` en `Admin.tsx` para invocar el método del servicio `saveNoticia(noticiaForm, noticiaFile)` y manejar los estados de éxito/error limpiamente.
