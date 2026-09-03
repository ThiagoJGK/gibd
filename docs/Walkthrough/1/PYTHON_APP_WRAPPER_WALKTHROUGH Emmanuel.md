## Walkthrough: Componente `PythonAppWrapper`

**Subfase:** 1.2 - Marco embebido para aplicaciones Python  
**Desarrollador:** Emmanuel Davezac  
**Rama:** `feature/fase1-wrapper`

### Resumen

Se creó un componente React reutilizable para mostrar aplicaciones Python mediante `iframe`, con estados de carga, error, reintento y diseño responsive.

### Archivos creados y modificados

- [frontend/src/components/PythonAppWrapper.tsx](frontend/src/components/PythonAppWrapper.tsx)
  - Define las props `appUrl`, `title`, `aspectRatio` y `onLoad`.
  - Muestra un spinner naranja mientras carga la aplicación.
  - Usa un timeout de 10 segundos para detectar una carga que no responde.
  - Permite hasta 3 intentos de carga y un nuevo ciclo mediante el botón de reintento.
  - Muestra un mensaje visible cuando la aplicación no está disponible.
  - Mantiene el contenido dentro de un `iframe` responsive con `overflow-x-hidden`.

- [backend/demo_python_app.py](backend/demo_python_app.py)
  - Servidor Python independiente, sin dependencias adicionales.
  - Usa un servidor multihilo para que una conexión cancelada del navegador no bloquee las siguientes.
  - Simula una búsqueda visual de patentes de automóviles.
  - Incluye carga de imagen con miniatura, botón `X` liviano y centrado para quitarla y volver a seleccionar otra, selector Top-K, procesamiento simulado y resultados con miniaturas SVG ampliables.
  - Al pasar el cursor sobre una miniatura aparece un indicador de lupa; al hacer clic se abre la imagen ampliada en un modal.
  - Expone `/health` para comprobar que la demo está disponible.

- [frontend/src/pages/Laboratorio.tsx](frontend/src/pages/Laboratorio.tsx)
  - Integra temporalmente el componente en la sección "Aplicación Python embebida" con el título "Motor de IA de Patentes".
  - Añade los controles `URL válida` y `URL caída` para probar ambos estados.

### Cómo ejecutar la prueba

Desde la raíz del repositorio:

```powershell
cd frontend
npm run dev
```

Abrir:

```text
http://127.0.0.1:5173/laboratorio
```

En la sección **Aplicación Python embebida**:

1. Seleccionar **URL válida**.
2. Verificar que aparece el spinner y luego se carga la aplicación Python real en `http://localhost:8010`.
3. En la aplicación embebida, seleccionar una imagen de una patente y verificar su miniatura.
4. Pulsar la `X` de la esquina para quitar la imagen y comprobar que se puede seleccionar otra.
5. Ajustar la cantidad de resultados y pulsar **Analizar imagen**.
6. Pasar el cursor sobre una miniatura y verificar que aparece el indicador de lupa.
7. Hacer clic en la miniatura para abrirla ampliada.
8. Cerrar la imagen con la `X`, haciendo clic fuera del modal o pulsando `Escape`.
9. Verificar el estado de procesamiento, las miniaturas SVG y la lista de patentes similares.
10. Seleccionar **URL caída**.
11. Esperar 10 segundos para que aparezca el mensaje de error.
12. Pulsar **Reintentar** para probar los intentos restantes.

La URL de error utilizada es `http://localhost:9999`, asumiendo que no existe un servidor ejecutándose en ese puerto.

Debe ejecutarse una sola instancia de `demo_python_app.py`. Si el puerto `8010` queda ocupado por una ejecución anterior, cerrar esa instancia antes de iniciar otra.

Para que la URL válida funcione, iniciar la aplicación Python de demostración en otra terminal desde la raíz del repositorio:

```powershell
python backend/demo_python_app.py
```

La demo sirve contenido HTML desde un proceso Python independiente. No requiere instalar dependencias ni tener implementada la API del gateway. Las miniaturas de resultados son SVG genéricos generados en el navegador para representar placas de patente sin incorporar imágenes externas.

También se puede verificar su estado directamente:

```powershell
(Invoke-WebRequest -Uri "http://localhost:8010/health" -UseBasicParsing).Content
```

Resultado esperado:

```text
{"status": "healthy", "service": "GIBD similarity demo"}
```

### Validaciones ejecutadas

```powershell
cd frontend
npm run build
```

Resultado: correcto. TypeScript y Vite finalizaron sin errores.

También se ejecutó ESLint sobre el componente:

```powershell
npx eslint src/components/PythonAppWrapper.tsx
```

Resultado: correcto, sin errores.

El lint completo del proyecto todavía informa errores en archivos preexistentes (`Header.tsx`, `LogoGIBD.tsx`, `LogoUTN.tsx`, `RibbonSelector.tsx`, `ThreeDLogoUTN.tsx` y `Admin.tsx`). No están relacionados con esta implementación.

### Criterios de aceptación

- [x] Componente reutilizable escrito en TypeScript.
- [x] Props configurables para URL, título, relación de aspecto y callback de carga.
- [x] Spinner con acento `#FF5500`.
- [x] Estado de error con mensaje y botón de reintento.
- [x] Timeout y límite de 3 intentos.
- [x] Marco responsive con aislamiento del contenido del `iframe`.
- [x] Aplicación Python de demostración interactiva para validar el flujo completo.
- [x] Build de producción validado correctamente.

### Próximo paso de integración

Cuando el gateway Python esté disponible, se debe reemplazar la URL de prueba por la URL real de la aplicación, por ejemplo `http://localhost:8000/docs` durante el desarrollo o la URL desplegada en el entorno correspondiente.
