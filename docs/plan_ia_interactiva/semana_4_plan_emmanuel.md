# 📅 Especificación Técnica Semanal — Semana 4: Emmanuel Davezac
## Rol: Frontend & CMS Admin Engineer

> **Ciclo:** Sprint 4 — Reproductor de Video con Timeline de Eventos, Auditoría Mobile y Limpieza Total  
> **Ubicación:** `docs/plan_ia_interactiva/semana_4_plan_emmanuel.md`  
> **Directriz de Diseño:** Geometría de cápsula (`rounded-full`), paneles orgánicos (`rounded-[2rem]`), paleta OLED `#0A0A0A` con `#FF5500`, cero sombras difusas y cero errores de consola.

---

## 🎯 1. Objetivos Técnicos del Sprint

1. **Construir el Componente de Video (`VideoEventTimeline.tsx`):** Desarrollar la vista para el tab "Video" de `Laboratorio.tsx`, incorporando un reproductor de video HTML5 personalizado con controles Flat y una barra de tiempo interactiva que señale con marcadores naranja los momentos de mayor dinamismo (eventos/picos de movimiento).
2. **Auditoría de Responsividad Móvil Completa:** Verificar pixel a pixel el comportamiento de la web en pantallas de teléfonos móviles ($360\text{ px}$ a $430\text{ px}$ de ancho) y tablets ($768\text{ px}$ a $1024\text{ px}$), asegurando que el dial giratorio y los paneles no causen desbordamiento horizontal.
3. **Erradicación de Sombras Difusas:** Revisar todos los archivos `.tsx` y `.css` para garantizar que no existan clases de sombras tradicionales (`shadow-md`, `shadow-lg`, `shadow-2xl`) y que toda la profundidad se base en microbordes `border border-white/10`.
4. **Saneamiento de Consola (F12 Limpia):** Eliminar todos los `console.log` de depuración, warnings de hooks no memorizados y dependencias huérfanas.

---

## 📐 2. Especificación de Componentes y Código

### A. Componente `VideoEventTimeline.tsx`
Ubicación: `frontend/src/components/ui/VideoEventTimeline.tsx`
* **Estructura Visual:**
  * Contenedor de Video: `rounded-[2rem] overflow-hidden bg-black border border-white/10 relative`
  * Controles Personalizados:
    * Botón Play/Pause en cápsula circular flotante con Framer Motion.
    * Barra de Progreso / Scrubber: Altura de 6px con marcadores circulares en los timestamps donde `event_flag === 'Pico Dinámico'`.
    * Al hacer clic en un marcador, el video avanza instantáneamente a ese segundo exacto (`videoRef.current.currentTime = marker.timestamp`).
  * Badge de Resumen: Píldora en la esquina superior derecha (`bg-[#FF5500]/20 text-[#FF5500] border border-[#FF5500]/40 rounded-full px-3 py-1 text-xs font-semibold`).

### B. Muestras Pre-cargadas de Video
* Incorporar selector de clips cortos de prueba del GIBD (ej. "Secuencia Urbana 1", "Movimiento Rápido 2", "Caída Controlada 3") para que el visitante pueda evaluar el modelo sin necesidad de subir un video propio.

---

## 📋 3. Tareas Específicas de Emmanuel

- [ ] **T4.3.1 (Crear Componente de Video):** Desarrollar `VideoEventTimeline.tsx` con soporte para reproducción local, marcadores interactivos y sincronización de eventos.
- [ ] **T4.3.2 (Integración en Laboratorio):** Conectar la pestaña "Video" del dial para renderizar el componente de video y realizar la petición a `/api/v1/inference/video-retrieval`.
- [ ] **T4.3.3 (Auditoría de Sombras en CSS):** Ejecutar búsqueda exhaustiva en `frontend/src/` y reemplazar cualquier sombra difusa por bordes de alto contraste `border border-white/10` o `border border-border-organic`.
- [ ] **T4.3.4 (Pruebas Mobile y Touch):** Validar que en pantallas táctiles los controles deslizantes (Top-K slider y scrubber de video) respondan con precisión sin interferir con el scroll vertical de la página.
- [ ] **T4.3.5 (Limpieza de Código):** Remover componentes obsoletos no utilizados en `frontend/src/components/` y verificar que la consola del navegador quede en 0 advertencias.

---

## 🛡️ 4. Plan de Verificación y Validación (V&V)

### Nivel 1 — Pre-Verificación Autónoma del Agente (Automated Testing)
1. **Compilación Limpia de Vite:**
   ```bash
   cd frontend && npm run build
   ```
   *Criterio:* Código 0, sin errores TypeScript y sin warnings críticos de chunk size.
2. **Auditoría Estática de Clases CSS:**
   * Script automatizado que confirme que no existan selectores con `box-shadow` difuso en `index.css` ni en clases Tailwind de los `.tsx`.

### Nivel 2 — Validación Obligatoria del Usuario (Human-in-the-Loop)
**Protocolo para Emmanuel antes del cierre formal:**
1. Navegar por toda la aplicación en el navegador con la consola de herramientas de desarrollador (F12) abierta.
2. Confirmar que no aparezca ninguna línea roja (error) ni amarilla (warning no controlado).
3. Probar el componente de video: reproducir, hacer clic en los marcadores de la barra de tiempo y verificar que el video salte al segundo indicado de forma instantánea.
4. Redimensionar la ventana a ancho de teléfono móvil ($375\text{ px}$) y comprobar que la interfaz se adapte perfectamente.
5. **Presentación Final a Thiago:** Entregar el reporte de conformidad para la aprobación definitiva del Tech Lead.
