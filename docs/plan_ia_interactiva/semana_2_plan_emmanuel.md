# 📅 Especificación Técnica Semanal — Semana 2: Emmanuel Davezac
## Rol: Frontend & CMS Admin Engineer

> **Ciclo:** Sprint 2 — Componente Visualizador 3D/2D de Espacio Latente e Historial  
> **Ubicación:** `docs/plan_ia_interactiva/semana_2_plan_emmanuel.md`  
> **Directriz de Diseño:** Fondo OLED `#0A0A0A`, acento `#FF5500`, interacción táctil y con mouse fluida, sin parpadeos.

---

## 🎯 1. Objetivos Técnicos del Sprint

1. **Construir el Componente `EmbeddingViewer3D.tsx`:** Desarrollar el visor 3D interactivo utilizando `@react-three/fiber` o Canvas WebGL nativo de alto rendimiento, renderizando la nube de puntos con estética cyberpunk científica.
2. **Interactividad de Nodos:** Implementar eventos de puntero (hover para tooltip con miniatura y nombre; clic para centrar la cámara en el ítem seleccionado).
3. **Control Deslizante de Umbral de Similitud:** Crear un slider reactivo que filtre en vivo los nodos visibles según su cercanía con la imagen de consulta.
4. **Historial de Consultas (Drawer Lateral):** Crear un panel deslizable accesible desde el Laboratorio que guarde las últimas 10 consultas en `sessionStorage` para comparaciones rápidas.

---

## 📐 2. Especificación de Componentes y Código

### A. Componente `EmbeddingViewer3D.tsx`
Ubicación: `frontend/src/components/ui/EmbeddingViewer3D.tsx`
* **Props:**
  ```typescript
  interface EmbeddingViewer3DProps {
    dataset: 'brands' | 'soccer_crests' | 'tattoos';
    queryCoordinates?: [number, number, number];
    highlightedNodeId?: string;
    onNodeSelect?: (nodeId: string) => void;
    onClose: () => void;
  }
  ```
* **Elementos Visuales y de Cámara:**
  * **Cámara:** Perspectiva con FOV de 60°, posición inicial `[0, 0, 3.5]`.
  * **Ejes Guía:** Líneas sutiles de referencia espacial con opacidad del 15% (`border-white/10`).
  * **Partículas:**
    * Nodos regulares: Mesh esférico con radio 0.035, material `MeshBasicMaterial` o `Points` de Three.js con color `#71717A`.
    * Nodos de resultados Top-K: Esferas de color `#F59E0B` conectadas con líneas tensoras al nodo de consulta.
    * Nodo de Consulta: Esfera vibrante `#FF5500` con pulso de escala senoidal en `requestAnimationFrame`.

### B. Tooltip Flotante de Información
* Al hacer hover sobre una partícula, proyectar sus coordenadas 3D a coordenadas 2D de pantalla ($x_{\text{screen}}, y_{\text{screen}}$) y renderizar un overlay HTML:
  ```tsx
  <div className="absolute z-50 pointer-events-none transform -translate-x-1/2 -translate-y-full mb-3 bg-[#18181B] border border-[#FF5500]/50 rounded-2xl p-2.5 shadow-none flex items-center gap-3">
    <img src={hoveredNode.thumbnail_url} className="w-10 h-10 object-contain rounded-lg bg-black" />
    <div>
      <div className="text-white text-xs font-bold">{hoveredNode.label}</div>
      <div className="text-zinc-400 text-[10px]">{hoveredNode.category}</div>
    </div>
  </div>
  ```

### C. Drawer de Historial (`QueryHistoryDrawer.tsx`)
Ubicación: `frontend/src/components/ui/QueryHistoryDrawer.tsx`
* Deslizable desde el lateral derecho mediante Framer Motion:
  ```tsx
  <motion.div
    initial={{ x: '100%' }}
    animate={{ x: 0 }}
    exit={{ x: '100%' }}
    transition={{ type: 'spring', damping: 25, stiffness: 200 }}
    className="fixed right-0 top-0 bottom-0 w-80 bg-[#121214] border-l border-white/10 z-50 p-6 flex flex-col"
  >
  ```
* Lista de miniaturas de consultas previas con timestamp y chip de modelo utilizado.

---

## 📋 3. Tareas Específicas de Emmanuel

- [ ] **T2.3.1 (Instalación de Dependencias):** Evaluar y configurar `@react-three/fiber` y `@react-three/drei` (o Canvas nativo optimizado con Three.js) asegurando que no rompa el build de Vite.
- [ ] **T2.3.2 (Desarrollo del Visor 3D):** Crear `EmbeddingViewer3D.tsx` con carga asíncrona de datos desde `/api/v1/embeddings/projection`.
- [ ] **T2.3.3 (Interacción de Cámara y Nodos):** Añadir raycasting de Three.js para detección de hover/clic sobre las esferas de la nube de puntos.
- [ ] **T2.3.4 (Drawer de Historial):** Desarrollar `QueryHistoryDrawer.tsx` con guardado en `sessionStorage` y botón de restauración de consulta previa.
- [ ] **T2.3.5 (Botón en Tarjetas de Resultado):** Agregar a `BrandResultCard.tsx` y `CrestResultCard.tsx` el botón píldora "Ver en el Espacio Vectorial" que invoque el visor 3D centrando el nodo correspondiente.

---

## 🛡️ 4. Plan de Verificación y Validación (V&V)

### Nivel 1 — Pre-Verificación Autónoma del Agente (Automated Testing)
1. **Compilación Limpia:**
   ```bash
   cd frontend && npm run build
   ```
   *Criterio:* Código 0, sin advertencias de tipos en los componentes de Three.js.
2. **Auditoría de Framerate:**
   * Validar que la animación del bucle de renderizado mantenga $60\text{ FPS}$ en resolución de escritorio.
3. **Validación de Limpieza de Recursos (Cleanup):**
   * Comprobar que en el `useEffect` de retorno se invoquen `renderer.dispose()` y `geometry.dispose()` para prevenir fugas de memoria GPU.

### Nivel 2 — Validación Obligatoria del Usuario (Human-in-the-Loop)
**Protocolo para Emmanuel antes de entrega a Thiago:**
1. Abrir el Laboratorio y realizar una consulta.
2. Clic en "Ver en el Espacio Vectorial":
   - Comprobar que la animación de entrada del visor sea fluida.
   - Rotar la nube de puntos con el mouse; comprobar que no haya tirones ni retrasos visuales.
   - Pasar el cursor sobre al menos 5 puntos distintos y verificar que los tooltips muestren las imágenes correctas.
3. Probar el Drawer de Historial y re-ejecutar una consulta previa.
4. **Presentar al Tech Lead:** Demostración en vivo a Thiago para obtener la aprobación del Sprint 2.
