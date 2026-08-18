# Guía de Trabajo: Desarrollador B (Módulo de Papers e Investigadores - Full Stack)

## 1. Introducción
Dado tu perfil técnico orientado al backend y para asegurar una distribución equitativa de responsabilidades, estarás a cargo del **ciclo de vida completo del módulo de Papers Científicos e Investigadores**.

> [!IMPORTANT]
> **Objetivo Final (No-Dev CMS):** Tu implementación debe garantizar que investigadores y docentes sin conocimientos técnicos puedan subir papers, asociar co-autores mediante casillas de verificación interactivas y previsualizar resúmenes de forma 100% visual y autónoma a través del CMS `/admin`, incluyendo soporte asistido de IA para autocompletar la descripción sin editar código.

Para evitar colisiones al trabajar en paralelo con tu compañero, interactuarás exclusivamente a través del archivo de servicios compartido: [dbService.ts](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/frontend/src/utils/dbService.ts).

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
git checkout -b feature/papers
```

### Paso C. Flujo de Commits Semánticos (Conventional Commits)
Registra tus avances utilizando mensajes descriptivos estandarizados:
```powershell
git add .
git commit -m "feat(papers): implement getPapers dynamic relational fetch and filter layout"
```

### Paso D. Sincronización e Integración Final
Cuando termines de implementar y validar localmente, integra tu trabajo de vuelta en la rama compartida:
```powershell
# Volver a la rama principal del proyecto
git checkout feature/obsidian-vault
git pull --rebase origin feature/obsidian-vault

# Fusionar tu rama de papers
git merge feature/papers

# Subir los cambios finales
git push origin feature/obsidian-vault
```
*Si surgen conflictos en `dbService.ts`, resuélvelos manualmente fusionando las firmas (las tuyas son de papers y no se solapan con las de noticias) y vuelve a ejecutar las pruebas antes de hacer push.*

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
2.  **Verificación de Carga Inicial:** Abre la página de Papers (`/papers`). Deberían mostrarse los papers semilla por defecto almacenados en `localStorage` con su equipo de investigación mapeado (verifica que no haya layout shifts bruscos en las tarjetas).
3.  **CMS Admin:** Ingresa a `/admin` (Modo Mock). Publica un paper científico nuevo completando el formulario y marcando múltiples coautores en las casillas correspondientes.
4.  **Verificación de Persistencia y Relaciones:** Vuelve a la página de Papers, verifica que el nuevo paper aparezca, que sus filtros funcionen por categoría y que la lista de coautores asociados renderice de forma correcta. Recarga la página (`F5`) para certificar que persista localmente.

---

## 4. Manual Técnico de Implementación

### A. Archivos a Modificar
1.  **Lógica del Servicio:** Escribir los métodos `getMiembrosEquipo`, `getPapers` y `savePaper` en [dbService.ts](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/frontend/src/utils/dbService.ts).
2.  **Consumo de Papers:** Modificar [Papers.tsx](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/frontend/src/pages/Papers.tsx) para importar y llamar a `getPapers`.
3.  **CMS Admin:** Modificar `handleSavePaper` y `fetchAuthors` en [Admin.tsx](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/frontend/src/pages/Admin.tsx) para llamar a los métodos correspondientes.

### B. Ejemplo de Firma y Flujo en dbService.ts
```typescript
// dbService.ts (Métodos para Desarrollador B)

export const getMiembrosEquipo = async (): Promise<any[]> => {
  const isMock = !import.meta.env.VITE_SUPABASE_URL || import.meta.env.VITE_SUPABASE_URL === 'YOUR_SUPABASE_URL';
  
  if (isMock) {
    const localData = localStorage.getItem('gibd_mock_miembros_equipo');
    if (localData) return JSON.parse(localData);
    
    const seed = [/* ARRAY DE LOS 24 AUTORES DE PAPERS.TSX */];
    localStorage.setItem('gibd_mock_miembros_equipo', JSON.stringify(seed));
    return seed;
  }
  
  const { data, error } = await supabase
    .from('miembros_equipo')
    .select('id, initials, name, role')
    .order('name');
  if (error) throw error;
  return data || [];
};

export const getPapers = async (): Promise<any[]> => {
  const isMock = !import.meta.env.VITE_SUPABASE_URL || import.meta.env.VITE_SUPABASE_URL === 'YOUR_SUPABASE_URL';

  if (isMock) {
    const localData = localStorage.getItem('gibd_mock_papers');
    if (localData) return JSON.parse(localData);
    
    const seed = [/* PAPERS ESTÁTICOS ORIGINALES */];
    localStorage.setItem('gibd_mock_papers', JSON.stringify(seed));
    return seed;
  }

  const { data, error } = await supabase
    .from('papers')
    .select(`
      id, title, description, category, date, url, image_url,
      paper_authors (
        miembros_equipo ( initials )
      )
    `);
  
  if (error) throw error;

  return data.map((paper: any) => ({
    id: paper.id,
    title: paper.title,
    description: paper.description,
    category: paper.category,
    date: paper.date,
    url: paper.url,
    image: paper.image_url || 'https://images.unsplash.com/photo-1546410531-bb4caa6b424d',
    authors: paper.paper_authors.map((pa: any) => pa.miembros_equipo.initials)
  }));
};
```

---

## 5. Tareas Asignadas (Backlog)

- [ ] **Tarea B.1: Carga y Semilla de Investigadores (SQL)**
  - Asegurar la actualización de `database_schema.sql` con los 24 autores y ejecutarlo en Supabase.
  - Actualizar el estado estático de la función `fetchAuthors` de [Admin.tsx](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/frontend/src/pages/Admin.tsx) para reflejar los 24 autores en el modo mock local.
- [ ] **Tarea B.2: Implementar Lógica de Papers en dbService.ts**
  - Desarrollar `getMiembrosEquipo`, `getPapers` y `savePaper` en el servicio centralizado.
- [ ] **Tarea B.3: Consumo Dinámico en Papers.tsx**
  - Reemplazar las colecciones hardcodeadas en `Papers.tsx` por estados reactivos y obtener los datos llamando a `getPapers` y mapeando el diccionario de autores al montar.
- [ ] **Tarea B.4: Integración en Admin.tsx**
  - Modificar `fetchAuthors` en `Admin.tsx` para llamar a `getMiembrosEquipo`.
  - Modificar `handleSavePaper` en `Admin.tsx` para llamar a `savePaper(paperForm, paperFile, selectedAuthors)` y manejar estados de éxito/error limpiamente.

---

## 6. Mejoras Adicionales de Infraestructura e Interfaz (Fuera de Alcance Inicial)

Durante el desarrollo de la funcionalidad, se detectaron e implementaron mejoras complementarias para robustecer la experiencia de desarrollo local y la accesibilidad:

1. **Resolución del Crash Fatal en Desarrollo Local (Pantalla Negra):**
   * **Problema:** Si un desarrollador local no tenía configuradas las variables de entorno de Supabase, el cliente de Supabase fallaba catastróficamente al inicio, dejando la pantalla completamente en negro.
   * **Solución:** Se modificó [supabaseClient.ts](file:///home/renato/Proyectos/gibd/frontend/src/utils/supabaseClient.ts) para asignar credenciales dummy estructuradas y válidas por defecto si no existen las variables, permitiendo que la app arranque sin errores e ingrese directamente en el Modo Mock.
2. **Acceso Directo al CMS Panel en el Pie de Página:**
   * **Solución:** Se añadió un enlace permanente de navegación hacia el Panel de Control CMS (`/#/admin`) en la sección de navegación de [Footer.tsx](file:///home/renato/Proyectos/gibd/frontend/src/components/layout/Footer.tsx) para facilitar las pruebas manuales locales a todos los desarrolladores sin requerir tipear la URL en la barra de direcciones.
3. **Documentación de Variables de Entorno:**
   * **Solución:** Se creó el archivo plantilla [frontend/.env.example](file:///home/renato/Proyectos/gibd/frontend/.env.example) para documentar el uso de `VITE_SUPABASE_URL` y `VITE_SUPABASE_ANON_KEY` y clarificar el comportamiento del sistema.

