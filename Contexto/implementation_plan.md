# Plan de Migración de Contenido Estático a Dinámico (Papers y Noticias)

Este documento detalla el análisis y los pasos necesarios para migrar las secciones de **Papers e Investigaciones** y **Últimas Novedades (Noticias)** de una estructura hardcodeada en el frontend a una gestión dinámica y administrable en base de datos PostgreSQL (mediante Supabase), con un esquema de respaldo local (LocalStorage) para desarrollo local y pruebas sin conexión.

## Objetivo Final: Gestión sin Código (No-Dev CMS)
El fin supremo de este proyecto es que un usuario administrador (investigadores o personal de cátedra sin conocimientos de programación) sea capaz de:
1.  **Publicar Novedades:** Crear, editar e ilustrar noticias desde una interfaz visual sin tener que modificar componentes.
2.  **Cargar y Relacionar Papers:** Subir el PDF de un paper académico y asociarlo a múltiples co-autores mediante casillas de verificación interactivas.
3.  **Soporte de IA para Facilitar Carga:** Disponer del botón de optimización con Gemini para generar resúmenes atractivos automáticamente a partir del PDF, facilitando la tarea al administrador no técnico.
4.  **Despliegue Inmediato:** Que el contenido se actualice instantáneamente en la web pública tras guardar, sin necesidad de compilar o hacer redespliegues del sitio.

---

## Análisis Técnico y Estado Actual

Actualmente en el frontend del GIBD Web:
1. **Noticias (Landing.tsx):** Las novedades se leen del array estático `NEWS_ITEMS`.
2. **Papers y Autores (Papers.tsx):** Los papers científicos se leen del array estático `PAPERS` y la base de datos de autores del objeto `AUTHORS_DATABASE`.
3. **Panel Administrativo (Admin.tsx):**
   - Cuenta con formularios funcionales para crear noticias y papers.
   - En **Modo Supabase Real** (cuando `VITE_SUPABASE_URL` y `VITE_SUPABASE_ANON_KEY` están configurados), ya realiza inserciones reales en las tablas `noticias`, `papers` y `paper_authors`.
   - En **Modo Desarrollador Local (Mock Mode)**, solo imprime en consola (`console.log`) y muestra alertas sin persistir la información.

### El Problema de Integridad Referencial
El script `Contexto/database_schema.sql` solo inserta **6 autores** iniciales, mientras que la base de datos hardcodeada de `Papers.tsx` cuenta con **24 autores**. Si un administrador intentara registrar un paper real en producción que asocie coautores no presentes en `miembros_equipo`, la inserción de relaciones en `paper_authors` fallaría por restricción de clave foránea.

---

## User Review Required

> [!IMPORTANT]
> **Esquema de Base de Datos y Seed de Miembros:**
> Para evitar errores de integridad de base de datos en producción y conservar todo el historial, actualizaremos el script `Contexto/database_schema.sql` para que contenga a los **24 investigadores** descritos en el frontend original.

> [!TIP]
> **Persistencia Local Completa en Mock Mode:**
> Para evitar que los desarrolladores pierdan los datos ingresados localmente cada vez que recargan la página, implementaremos una capa de almacenamiento en `LocalStorage`. El frontend inicializará las colecciones con los datos estáticos preexistentes en su primera carga y luego interactuará dinámicamente con ellos.

---

## Open Questions

Actualmente no tenemos dudas críticas de diseño, ya que la arquitectura de base de datos de Supabase y el diseño de la aplicación ya están parcialmente maquetados en la interfaz de administración. Procederemos a implementar el soporte adaptativo.

---

## Proposed Changes

### 1. Base de Datos / Backend

#### [MODIFY] [database_schema.sql](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/Contexto/database_schema.sql)
- Actualizar el bloque de `INSERT INTO public.miembros_equipo` al final del archivo para incluir la lista completa de los 24 autores que actualmente se encuentran hardcodeados en el frontend (`AUTHORS_DATABASE`), previniendo fallos de llaves foráneas.

---

### 2. Frontend Application

#### [MODIFY] [Landing.tsx](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/frontend/src/pages/Landing.tsx)
- Reemplazar la constante estática `NEWS_ITEMS` con estado reactivo (`newsItems`, `setNewsItems`) y un cargador dinámico.
- Al montar el componente, verificar si el cliente de Supabase está activo (mediante la validación de credenciales).
  - **Modo Supabase:** Cargar noticias desde la tabla `noticias` ordenadas por `created_at` descendente.
  - **Modo Local/Mock:** Cargar noticias desde el `localStorage` (llave `gibd_mock_noticias`), utilizando los elementos hardcodeados originales como semilla si la llave está vacía.
- Implementar transiciones de entrada suaves utilizando Framer Motion para evitar saltos en la interfaz mientras se realiza la consulta asíncrona.

#### [MODIFY] [Papers.tsx](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/frontend/src/pages/Papers.tsx)
- Reemplazar las constantes `PAPERS` y `AUTHORS_DATABASE` con estados reactivos dinámicos.
- Al montar, realizar consultas paralelas:
  - **Modo Supabase:**
    - Cargar todos los miembros de `miembros_equipo` y mapearlos en un objeto `Record<string, Author>` usando sus iniciales.
    - Cargar papers de la tabla `papers` incluyendo sus relaciones en `paper_authors` (para obtener las iniciales de sus autores co-relacionados de forma eficiente en una sola consulta relacional de Supabase).
  - **Modo Local/Mock:**
    - Cargar autores y papers desde `localStorage` (llaves `gibd_mock_miembros_equipo` y `gibd_mock_papers`), usando las listas estáticas actuales como semilla.
- Mantener las categorías y la funcionalidad de filtrado reactiva sobre la lista cargada.

#### [MODIFY] [Admin.tsx](file:///c:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/frontend/src/pages/Admin.tsx)
- Expandir la lista inicial de autores simulados en el Mock Mode para incluir a los 24 investigadores oficiales.
- Modificar el flujo de guardado de noticias en Mock Mode para persistirlas en el array de `localStorage` (`gibd_mock_noticias`).
- Modificar el flujo de guardado de papers en Mock Mode para persistirlos en el array de `localStorage` (`gibd_mock_papers`) junto con sus relaciones simuladas.

---

## Verification Plan

### Automated Tests
- Ejecutar el servidor de desarrollo del frontend (`npm run dev`) y verificar que compile limpiamente.
- Ejecutar el backend de Python para asegurar la comunicación con el servicio de IA local si es necesario.

### Manual Verification
1. **Verificación en Modo Desarrollador Local (Por defecto sin variables .env):**
   - Abrir la web y navegar a "Últimas Novedades" en la Landing y a "Papers". Deben renderizarse los contenidos sembrados originales.
   - Acceder al portal `/admin` con las credenciales locales de prueba.
   - Publicar una nueva noticia de prueba con imagen.
   - Publicar un nuevo paper científico seleccionando múltiples coautores.
   - Volver a la Landing y a Papers: verificar que los elementos nuevos creados en el CMS aparezcan inmediatamente y persistan tras recargar el navegador.
2. **Verificación con Supabase Conectado (Si se configuran las variables):**
   - Verificar las consultas e inserciones en base de datos.
