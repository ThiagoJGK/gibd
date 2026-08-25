# Explicación Técnica: ¿Qué es la Subfase 1.2 y qué hay que hacer?

> **Destinatario:** Emmanuel Davezac
> **Autor:** Thiago (Tech Lead)
> **Fecha:** 24/08/2026

---

## Respuesta Corta

**Sí, la Subfase 1.2 es crear un componente React genérico y reutilizable**, pero no es "solo" eso en el sentido trivial. Es un componente **con lógica de estados, manejo de errores, aislamiento de estilos y responsividad** que se va a usar en las subfases posteriores (1.3, 1.4 y Fase 4) para embeber aplicaciones Python reales dentro de nuestra web.

---

## ¿Por qué existe este componente?

**Andrés Pascal y Adrián Planas** pidieron expresamente que la web del GIBD pueda **mostrar y ejecutar programas de Python** (dashboards de Gradio, Streamlit, Dash, etc.) directamente dentro de la interfaz, sin que el usuario tenga que salir a otra pestaña del navegador.

El componente `PythonAppWrapper` es **el marco/contenedor** que va a alojar esas aplicaciones embebidas de forma aislada, responsiva y respetando el Design System Flat del proyecto.

---

## ¿Qué hay que construir técnicamente?

### Archivo a crear:
```
frontend/src/components/PythonAppWrapper.tsx
```

### Props del componente:
```typescript
interface PythonAppWrapperProps {
  appUrl: string;        // URL de la app de Python (ej: http://localhost:8000/docs, un Gradio, etc.)
  title: string;         // Título que se muestra arriba del marco (ej: "Motor de IA de Marcas")
  aspectRatio?: string;  // Relación de aspecto opcional (ej: "16/9", "4/3"). Default: "16/9"
  onLoad?: () => void;   // Callback opcional que se dispara cuando el iframe termina de cargar
}
```

### Lo que el componente tiene que hacer:

1. **Renderizar un `<iframe>`** con la URL de la aplicación Python.
2. **Estado de Carga (Loading):**
   - Mientras el iframe carga, mostrar un spinner/animación de carga con el color de acento `#FF5500`.
   - Cuando el iframe termine de cargar (evento `onLoad`), ocultar el spinner y mostrar el contenido.
3. **Estado de Error / Offline:**
   - Si la aplicación de Python no responde o está offline, mostrar un mensaje amigable ("La aplicación de Python no está disponible en este momento") con un botón de **"Reintentar"** que recargue el iframe.
4. **Estilos y Aislamiento:**
   - Contenedor con borde fino y estética Flat (`border-radius: 2rem`, fondo `#0A0A0A`).
   - El iframe debe ser **responsive** (que se adapte al ancho de la pantalla).
   - Bloquear desbordamientos horizontales (`overflow-x: hidden`).
   - Los estilos de la app embebida **no deben romper** los estilos de nuestra web.

---

## Ejemplo Visual de lo que se espera

```
┌──────────────────────────────────────────────┐
│  🧪 Motor de IA de Marcas de Ganado         │  ← título (prop: title)
├──────────────────────────────────────────────┤
│                                              │
│   ┌──────────────────────────────────────┐   │
│   │                                      │   │
│   │     (Aquí se renderiza el iframe     │   │  ← iframe con la app Python
│   │      con la app de Python)           │   │
│   │                                      │   │
│   │                                      │   │
│   └──────────────────────────────────────┘   │
│                                              │
└──────────────────────────────────────────────┘
     border-radius: 2rem, fondo: #0A0A0A
```

---

## ¿Cómo probarlo?

1. Crear el componente en `frontend/src/components/PythonAppWrapper.tsx`.
2. Importarlo temporalmente en una vista de prueba (puede ser en `Laboratorio.tsx` o en una ruta temporal).
3. Pasarle una URL de prueba, por ejemplo:
   - `http://localhost:8000/docs` (la documentación interactiva de FastAPI, si Renato tiene el backend corriendo).
   - O cualquier URL pública como `https://example.com` para verificar que el iframe carga correctamente.
4. Verificar que:
   - [x] Se muestre la animación de carga y luego el contenido.
   - [x] Si se pasa una URL inválida, se muestre el mensaje de error con botón de reintentar.
   - [x] Los estilos de la web no se rompan.
   - [x] `npm run build` compile sin errores TypeScript.

---

## ¿Dónde se va a usar después?

- **Subfase 1.3:** Thiago va a usar `PythonAppWrapper` en `Laboratorio.tsx` para embeber la visualización de resultados del motor de IA.
- **Fase 4:** Se va a usar para embeber la interfaz de PTAH-Jurídico (NLP) y otros dashboards multimodales.

**Es un componente clave para toda la arquitectura de la Fase 1 y posteriores.**

---

## Resumen de criterios de aceptación

- [x] Componente `PythonAppWrapper.tsx` reutilizable, escrito en TypeScript sin errores.
- [x] Transición suave durante la carga del iframe (spinner → contenido).
- [x] Fallback de error con botón de reintentar.
- [x] Aislamiento completo de estilos CSS.
- [x] `npm run build` pasa sin errores.
