# Plan de Subfase 1.2: Componente PythonAppWrapper (Marco Embebido)

## 1. Información General
- **Fase Perteneciente:** FASE 1 - Core de IA de Marcas de Ganado, Gateway Python y Wrapper Embebido
- **Subfase:** 1.2
- **Desarrollador Asignado:** Emanuel Davezac

---

## 2. Objetivo de la Subfase
Construir el componente contenedor `PythonAppWrapper.tsx` para responder al requerimiento explícito planteado por Adrián Planas: destinar un marco (wrapper) en la web para mostrar y ejecutar programas o dashboards de Python embebidos (Gradio, Streamlit, Dash) de forma aislada, responsiva y respetando el Design System Flat de la plataforma.

---

## 3. Tareas Técnicas Detalladas

- [ ] **Tarea 1.2.1: Creación del Componente PythonAppWrapper**
  - Archivo: `frontend/src/components/PythonAppWrapper.tsx`
  - Props del componente: `appUrl: string`, `title: string`, `aspectRatio?: string`, `onLoad?: () => void`.
- [ ] **Tarea 1.2.2: Implementación de Estado de Carga y Fallback de Error**
  - Spinner de carga personalizado con el color de acento `#FF5500`.
  - Mensaje de aviso y botón de reintento si la aplicación de Python no responde o está offline.
- [ ] **Tarea 1.2.3: Estilos y Aislamiento del Wrapper**
  - Aplicar contenedor orgánico con borde fino de la estética Flat (`border-radius: 2rem`, fondo `#0A0A0A`).
  - Asegurar que el iframe o marco embebido sea responsive y bloquee desbordamientos horizontales.

---

## 4. Plan de Comprobación, Validación y Testing

### A. Pruebas de Renderizado Local
```powershell
cd frontend
npm run dev
```

### B. Pruebas de Funcionamiento
1. Importar `PythonAppWrapper` en una vista de prueba con una URL de prueba (ej: `http://localhost:8000/docs` o app de Gradio/Streamlit).
2. Verificar que se muestre la animación de carga y posteriormente el contenido embebido.
3. Verificar que los estilos externos no rompan la maquetación Flat de la aplicación web.

---

## 5. Criterios de Aceptación de la Subfase
- Componente `PythonAppWrapper` reutilizable en TypeScript sin errores.
- Transición suave durante la carga del iframe.
- Aislamiento completo de estilos CSS.
