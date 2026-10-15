# 📅 Especificación Técnica Semanal — Semana 1: Emmanuel Davezac
## Rol: Frontend & CMS Admin Engineer

> **Ciclo:** Sprint 1 — Uploader Cloudinary, Selector de Catálogos y Comparador de Escudos  
> **Ubicación:** `docs/plan_ia_interactiva/semana_1_plan_emmanuel.md`  
> **Directriz de Diseño:** Geometría de cápsula (`rounded-full`), paneles orgánicos (`rounded-[2rem]`), paleta OLED `#0A0A0A` con `#FF5500`, cero sombras difusas (`box-shadow: none`).

---

## 🎯 1. Objetivos Técnicos del Sprint

1. **Desarrollar el Componente `CloudinaryUploader.tsx`:** Crear la zona Drag & Drop para carga de imágenes conectada a Cloudinary mediante Unsigned Upload Preset, con fallback local inmediato a `FileReader / ArrayBuffer`.
2. **Implementar el Selector de Catálogos de Visión en `Laboratorio.tsx`:** Añadir las píldoras interactivas `rounded-full` para alternar entre "Marcas de Ganado" y "Escudos de Fútbol".
3. **Diseñar las Tarjetas de Resultados de Escudos (`CrestResultCard.tsx`):** Mostrar el escudo retornado, escudo de consulta, badge con el porcentaje de similitud calibrado, métrica de distancia euclidiana, nombre del club, liga y país.
4. **Respetar el Flat Design System:** Utilizar microbordes orgánicos de alto contraste (`border border-white/10` o `border-[#FF5500]/30`) y microanimaciones con Framer Motion sin provocar re-renders pesados.

---

## 📐 2. Especificación de Componentes y Código

### A. Componente `CloudinaryUploader.tsx`
Ubicación: `frontend/src/components/ui/CloudinaryUploader.tsx`
* **Props:**
  ```typescript
  interface CloudinaryUploaderProps {
    onImageSelected: (file: File, previewUrl: string, cloudinaryUrl?: string) => void;
    isAnalyzing: boolean;
    disabled?: boolean;
    label?: string;
  }
  ```
* **Lógica de Carga:**
  * Al soltar el archivo, crear `URL.createObjectURL(file)` para previsualización instantánea a 0 ms de latencia.
  * En paralelo (si las variables `VITE_CLOUDINARY_CLOUD_NAME` y `VITE_CLOUDINARY_UPLOAD_PRESET` están definidas), enviar a `https://api.cloudinary.com/v1_1/<cloud_name>/image/upload`.
  * Si la llamada a Cloudinary no está configurada o falla la red, el sistema opera con el `File` local sin interrumpir al usuario.

### B. Tarjeta de Resultado de Escudo (`CrestResultCard.tsx`)
Ubicación: `frontend/src/components/ui/CrestResultCard.tsx`
* **Estructura Visual:**
  * Contenedor: `rounded-[2rem] bg-[#121214] border border-white/10 hover:border-[#FF5500]/50 transition-all p-5`
  * Imagen del Escudo: Miniatura de $96 \times 96\text{ px}$ centrada con fondo neutro transparente/blanco.
  * Badges de Similitud:
    * Si $S \ge 85\%$: Badge verde esmeralda `bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 rounded-full px-3 py-1`.
    * Si $60\% \le S < 85\%$: Badge ámbar `bg-amber-500/10 text-amber-400 border border-amber-500/30 rounded-full px-3 py-1`.
    * Si $S < 60\%$: Badge grisáceo neutro `bg-zinc-800 text-zinc-400 border border-zinc-700 rounded-full px-3 py-1`.
  * Datos del Club: Nombre en negrita (`text-white font-semibold text-lg`), país y liga en `text-zinc-400 text-sm`.
  * Distancia Métrica: Muestra `d: 0.1420` en tipografía monoespaciada (`font-mono text-xs text-zinc-500`).

### C. Conexión en `Laboratorio.tsx`
* Actualizar el selector de sub-modelos de visión:
  ```tsx
  <div className="flex items-center gap-2 p-1.5 bg-[#18181B] rounded-full border border-white/10">
    <button
      onClick={() => handleSelectVisionSubModel('brands')}
      className={`px-5 py-2 rounded-full text-xs font-semibold uppercase tracking-wider transition-all ${
        selectedVisionSubModel === 'brands'
          ? 'bg-[#FF5500] text-white shadow-none'
          : 'text-zinc-400 hover:text-white'
      }`}
    >
      Marcas de Ganado
    </button>
    <button
      onClick={() => handleSelectVisionSubModel('soccer_crests')}
      className={`px-5 py-2 rounded-full text-xs font-semibold uppercase tracking-wider transition-all ${
        selectedVisionSubModel === 'soccer_crests'
          ? 'bg-[#FF5500] text-white shadow-none'
          : 'text-zinc-400 hover:text-white'
      }`}
    >
      Escudos de Fútbol
    </button>
  </div>
  ```

---

## 📋 3. Tareas Específicas de Emmanuel

- [ ] **T1.3.1 (Crear Uploader):** Desarrollar `CloudinaryUploader.tsx` con soporte para arrastrar y soltar, selector de archivos clásico, y estado de progreso.
- [ ] **T1.3.2 (Crear Tarjeta de Escudos):** Desarrollar `CrestResultCard.tsx` aplicando las clases de Tailwind con radios `rounded-[2rem]` y badges `rounded-full`.
- [ ] **T1.3.3 (Integración en Laboratorio):** Conectar el estado de `selectedVisionSubModel` en `Laboratorio.tsx` para llamar al endpoint `/api/v1/inference/soccer-crests` cuando el usuario esté en la pestaña de escudos.
- [ ] **T1.3.4 (Muestras de Prueba Pre-cargadas):** Crear un carrusel de muestras rápidas de escudos (Boca, River, Barcelona, Real Madrid, etc.) para que el usuario pueda probar el sistema con un solo clic sin necesidad de descargar imágenes.
- [ ] **T1.3.5 (Animaciones con Framer Motion):** Agregar transición `staggerChildren` para que los resultados aparezcan con una animación suave de elevación y desvanecimiento (`opacity: 0, y: 15` a `opacity: 1, y: 0`).

---

## 🛡️ 4. Plan de Verificación y Validación (V&V)

### Nivel 1 — Pre-Verificación Autónoma del Agente (Automated Testing)
1. **Compilación de Frontend:**
   ```bash
   cd frontend && npm run build
   ```
   *Criterio:* Código 0, sin advertencias de tipos TypeScript.
2. **Auditoría de CSS y Sombras:**
   * Ejecutar búsqueda estática para asegurar que no se hayan introducido clases `shadow-lg`, `shadow-xl` o estilos con `box-shadow` difuso en los nuevos componentes.
3. **Simulación de Carga:**
   * Comprobar que si el archivo pesa más de 5 MB, el uploader muestre un mensaje de advertencia limpio sin bloquear la UI.

### Nivel 2 — Validación Obligatoria del Usuario (Human-in-the-Loop)
**Protocolo para Emmanuel antes de solicitar la revisión de Thiago:**
1. Levantar el frontend:
   ```bash
   npm run dev
   ```
2. Acceder a `/laboratorio`, verificar que al hacer clic en las muestras de escudos el sistema responda de inmediato y renderice el grid de resultados.
3. Probar en resolución móvil (inspeccionar dispositivo en Chrome DevTools a $390\text{ px}$ de ancho) y confirmar que las tarjetas de resultados no se desborden horizontalmente.
4. **Presentar al Tech Lead:** Solicitar la revisión visual de Thiago para recibir la aprobación del sprint.
