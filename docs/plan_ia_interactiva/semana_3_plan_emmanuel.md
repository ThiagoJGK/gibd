# 📅 Especificación Técnica Semanal — Semana 3: Emmanuel Davezac
## Rol: Frontend & CMS Admin Engineer

> **Ciclo:** Sprint 3 — Interfaz de Consulta PTAH-Jurídico y Visualizador de Espectrogramas de Audio  
> **Ubicación:** `docs/plan_ia_interactiva/semana_3_plan_emmanuel.md`  
> **Directriz de Diseño:** Geometría de cápsula (`rounded-full`), tarjetas orgánicas (`rounded-[2rem]`), paleta OLED `#0A0A0A` con `#FF5500`, feedback sonoro y visual sin sombras difusas.

---

## 🎯 1. Objetivos Técnicos del Sprint

1. **Construir la Vista de Búsqueda Semántica PTAH (`PtahSearchView.tsx`):** Implementar la interfaz para el tab "Texto" del Laboratorio, con barra de búsqueda reactiva, pills de consultas predeterminadas ("Marcas de Ganado", "Prescripción", "Límites Provinciales") y tarjetas de artículos con badge de relevancia.
2. **Construir el Visualizador de Audio y Espectrograma (`AudioLabView.tsx`):** Diseñar el panel para el tab "Sonido", con reproductor Web Audio API, analizador de frecuencias en Canvas 2D en tiempo real y selector de eventos acústicos de muestra.
3. **Manejo del AudioContext del Navegador:** Asegurar que los contextos de audio no se bloqueen por las políticas de autoplay del navegador, reanudándolos fluidamente ante la primera interacción del usuario.
4. **Respetar el Flat Design System:** Cero sombras difusas; la jerarquía se logra con bordes orgánicos sutiles (`border border-white/10`) y acento neón `#FF5500`.

---

## 📐 2. Especificación de Componentes y Código

### A. Componente `PtahSearchView.tsx`
Ubicación: `frontend/src/components/ui/PtahSearchView.tsx`
* **Estructura Visual:**
  * Barra de Búsqueda: Cápsula grande (`rounded-full bg-[#18181B] border border-white/15 focus-within:border-[#FF5500] px-6 py-3.5 flex items-center gap-3`) con botón de búsqueda integrado.
  * Pills de Sugerencias: Carrusel horizontal con consultas de muestra del GIBD.
  * Tarjetas de Artículos (`PtahArticleCard.tsx`):
    * Encabezado: Título de la ley y número de artículo en negrita.
    * Badge de Similitud: `rounded-full` con porcentaje verde/ámbar/gris.
    * Fragmento de Texto con resaltado de términos (`mark` estilizado con fondo `#FF5500`/20 y texto `#FF5500`).
    * Botón "Leer Artículo Completo" con acordeón animado por Framer Motion.

### B. Componente `AudioLabView.tsx` y Visualizador Canvas
Ubicación: `frontend/src/components/ui/AudioLabView.tsx`
* **Lógica del Visualizador con Web Audio API:**
  ```tsx
  // Uso de AnalyserNode para alimentar el Canvas
  const setupAudioGraph = (audioElement: HTMLAudioElement) => {
    const audioCtx = getAudioCtx();
    const source = audioCtx.createMediaElementSource(audioElement);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 128;
    source.connect(analyser);
    analyser.connect(audioCtx.destination);
    
    // Bucle de dibujo a 60 FPS
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const draw = () => {
      requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);
      ctx.fillStyle = '#0A0A0A';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      const barWidth = (canvas.width / bufferLength) * 2.2;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height;
        ctx.fillStyle = '#FF5500';
        ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
        x += barWidth + 2;
      }
    };
    draw();
  };
  ```

---

## 📋 3. Tareas Específicas de Emmanuel

- [ ] **T3.3.1 (Crear Vista PTAH):** Desarrollar `PtahSearchView.tsx` e integrarla dentro de `Laboratorio.tsx` para cuando el selector activo sea "Texto".
- [ ] **T3.3.2 (Crear Tarjeta de Artículo):** Desarrollar `PtahArticleCard.tsx` con soporte para expansión de texto y animación suave.
- [ ] **T3.3.3 (Crear Laboratorio de Audio):** Desarrollar `AudioLabView.tsx` con selector de muestras ("Sirena", "Vidrio", "Explosión", "Motor") y uploader de archivos `.wav`.
- [ ] **T3.3.4 (Visualizador Canvas de Frecuencias):** Implementar el renderizado en tiempo real de las barras de frecuencia en Canvas optimizado para pantallas Retina.
- [ ] **T3.3.5 (Conexión con FastAPI):** Conectar las llamadas a `/api/v1/inference/ptah-query` y `/api/v1/inference/audio-match`.

---

## 🛡️ 4. Plan de Verificación y Validación (V&V)

### Nivel 1 — Pre-Verificación Autónoma del Agente (Automated Testing)
1. **Compilación de Frontend:**
   ```bash
   cd frontend && npm run build
   ```
   *Criterio:* Código 0, sin errores TypeScript.
2. **Validación de Limpieza de AudioContext:**
   * Comprobar que pausar o desmontar la vista de audio suspenda el `AudioContext` para ahorrar ciclos de CPU.
3. **Auditoría de Accesibilidad de Formularios:**
   * Verificar que los inputs de texto tengan etiquetas `aria-label` descriptivas y soporte completo para navegación con teclado (`Enter` para buscar).

### Nivel 2 — Validación Obligatoria del Usuario (Human-in-the-Loop)
**Protocolo para Emmanuel antes de entrega a Thiago:**
1. Navegar a `/laboratorio`, mover el dial a **"Texto"**:
   - Probar una búsqueda de jurisprudencia y validar que las tarjetas aparezcan animadas sin scroll horizontal no deseado.
2. Mover el dial a **"Sonido"**:
   - Dar clic en una de las pistas de muestra de audio.
   - Presionar "Reproducir" y confirmar que las barras de frecuencia en el Canvas dancen fluidamente al compás de la señal sonora.
3. **Presentar al Tech Lead:** Demostración en vivo a Thiago para obtener la aprobación del Sprint 3.
