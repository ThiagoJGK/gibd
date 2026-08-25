# Resumen de Cambios: Módulo de Papers y CRUD de Investigaciones

Se ha implementado con éxito la persistencia y la administración dinámica de papers científicos y sus co-autores (investigadores) en la plataforma **GIBD WEB**. Las funcionalidades agregadas cubren el ciclo de vida completo de un artículo académico (**Alta, Baja y Modificación**), soportando tanto el entorno de producción (Supabase) como el de pruebas offline (Mock Mode).

---

## Cambios Realizados

A continuación se detallan los archivos modificados en el repositorio:

### 1. Servicios y Persistencia
* **[supabaseClient.ts](file:///home/renato/Proyectos/gibd/frontend/src/utils/supabaseClient.ts):**
  * Se corrigió la inicialización del cliente de Supabase agregando marcadores de posición URL y clave por defecto. Esto evita que la librería lance un error fatal de tiempo de ejecución al evaluar el módulo cuando no existen claves de entorno locales, solucionando el error de "pantalla negra" en el CMS `/admin` e iniciando correctamente en **Modo Desarrollador Local (Mock)**.
* **[dbService.ts](file:///home/renato/Proyectos/gibd/frontend/src/utils/dbService.ts):**
  * Se implementó el servicio centralizado que gestiona tanto el modo real como el modo simulado de base de datos.
  * Se definieron los 24 co-autores del GIBD en la inicialización mock.
  * Se crearon los métodos principales `getMiembrosEquipo`, `getPapers`, `savePaper`, `updatePaper` y `deletePaper`, abstrayendo la lógica de la base de datos de los componentes de la interfaz de usuario.
  * Se incluyó el helper de subida de archivos `uploadFile` unificado.

### 2. Catálogo Público
* **[Papers.tsx](file:///home/renato/Proyectos/gibd/frontend/src/pages/Papers.tsx):**
  * Se removieron las constantes locales hardcodeadas con papers y co-autores.
  * Se instalaron estados de React y un hook `useEffect` que carga dinámicamente la información científica consumiendo `getPapers` y `getMiembrosEquipo` de `dbService.ts`.
  * Se agregaron tipados estrictos en las funciones de mapeo de colecciones para prevenir errores del compilador TypeScript (`TS7006`).

### 3. Panel de Administración (CMS), Navegación y Testing
* **[Admin.tsx](file:///home/renato/Proyectos/gibd/frontend/src/pages/Admin.tsx):**
  * Se conectó el panel de control a los métodos del servicio centralizado.
  * **Listado de Papers (CRUD):** Se diseñó e implementó un nuevo panel interactivo en la pestaña *Papers* que despliega los papers cargados actualmente en el sistema.
  * **Baja (Delete):** Se agregó el botón **Eliminar** junto a cada paper de la lista, con un diálogo de confirmación para evitar borrados accidentales.
  * **Modificación (Edit/Update):** Se implementó el botón **Editar**, el cual carga la información completa del paper y sus co-autores en el formulario existente, activa el "Modo Edición" y conmuta el botón de guardado a "Actualizar Paper" junto con un botón para "Cancelar Edición".
  * **IDs de Testing:** Se asignaron atributos `id` estables y únicos a todos los botones, campos de texto y elementos de selección del CRUD para permitir pruebas de caja negra automatizadas en navegadores.
* **[Footer.tsx](file:///home/renato/Proyectos/gibd/frontend/src/components/layout/Footer.tsx):**
  * Se agregó un enlace permanente al **Panel de Control (CMS)** en la sección de "Navegación" del pie de página. Esto permite a los desarrolladores e investigadores acceder directamente a la URL `/#/admin` desde la interfaz de usuario sin necesidad de escribir la ruta de manera manual en la barra de direcciones del navegador.

### 4. Documentación y Plantillas de Proyecto
* **[.env.example](file:///home/renato/Proyectos/gibd/frontend/.env.example):**
  * Se creó la plantilla de variables de entorno para detallar la configuración del proyecto y la estrategia de respaldo con Modo Mock local.
---

## Bitácora de Pruebas Manuales

Durante la revisión manual llevada a cabo por el desarrollador, se registraron los siguientes hitos de testing:

* **[COMPLETADO] Prueba de Creación (Alta) con Archivo PDF:**
  * Se completó el formulario con el paper de ejemplo: *"Detección de Anomalías en Redes de Sensores IoT utilizando Autoencoders Profundos y Big Data"*.
  * Se seleccionaron coautores (`AP`, `ST`, `MO`).
  * Se subió con éxito el archivo PDF de prueba local [paper_ejemplo_prueba.pdf](file:///home/renato/Proyectos/gibd/paper_ejemplo_prueba.pdf).
  * Se verificó que el paper se insertó correctamente al final de la tabla administrativa en `/admin` y en el catálogo de `/papers` con su respectiva URL simulada de almacenamiento.
* **[COMPLETADO] Prueba de Modificación (Edición):**
  * Se validó la edición seleccionando el paper creado, modificando co-autores y parte del título, y guardándolo exitosamente. Los cambios persistieron de manera correcta incluso tras refrescar la página (`F5`).
* **[COMPLETADO] Prueba de Eliminación (Baja):**
  * Se eliminó el paper de prueba desde el listado administrativo confirmando el cuadro de diálogo. Se corroboró que el elemento desaparece instantáneamente y de forma permanente del catálogo y de la tabla CMS, resistiendo recargas de página (`F5`).


---

## Guía de Verificación Manual (Paso a Paso)

Para corroborar el correcto funcionamiento de las nuevas características en tu máquina local:

1. **Levantar el Servidor Local:**
   * Abre tu terminal en la carpeta `frontend/` y corre el comando `npm run dev`.
2. **Navegar al CMS:**
   * Entra a [http://localhost:5173/#/admin](http://localhost:5173/#/admin) (iniciará en **Modo Desarrollador Local (Mock)**).
   * Ingresa con las credenciales de prueba:
     * **Email:** `admin@gibd.utn.edu.ar`
     * **Contraseña:** `admin123`
3. **Probar el Flujo de la pestaña "Papers":**
   * **Visualizar Lista:** Al entrar a la pestaña, desplázate al final del formulario y verifica que se listan los papers cargados (se siembran automáticamente en tu primer ingreso).
   * **Crear un Paper (Alta):**
     * Llena el formulario con un título ficticio.
     * Selecciona múltiples autores (por ejemplo, a *Andrés Pascal* y *Sebastián Trossero*).
     * Sube un PDF de prueba o introduce un enlace.
     * Haz clic en **Indexar e Insertar Paper**.
     * Verifica que se agrega inmediatamente al listado final.
   * **Modificar un Paper (Edición):**
     * Elige cualquier paper de la lista y presiona **Editar**.
     * El formulario se autocompletará con su información y se desplazará de forma suave al formulario.
     * Modifica el título y selecciona un co-autor adicional.
     * Presiona **Actualizar Paper** (o *Cancelar Edición* si deseas salir sin guardar).
     * Verifica que los cambios impacten en el listado.
   * **Eliminar un Paper (Baja):**
     * Presiona **Eliminar** en un paper del listado.
     * Confirma la alerta del navegador.
     * Verifica que se remueve de la lista administrativa.
4. **Verificar Reflejo en Vista Pública:**
   * Abre una pestaña y navega al catálogo público: [http://localhost:5173/#/papers](http://localhost:5173/#/papers).
   * Comprueba que los papers añadidos, editados o eliminados se reflejen allí de manera idéntica y persistan si refrescas la página (`F5`).

