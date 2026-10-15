# 📅 Especificación Técnica Semanal — Semana 2: Thiago Gomez Kehler
## Rol: Tech Lead / Arquitecto de Software / Diseñador UI-UX Principal

> **Ciclo:** Sprint 2 — Espacio Latente G5 (Visualizador 3D/2D) y Catálogos de Visión Expandidos  
> **Ubicación:** `docs/plan_ia_interactiva/semana_2_plan_thiago.md`  

---

## 🎯 1. Objetivos Técnicos del Sprint

1. **Diseñar la Experiencia del Visualizador G5 (Espacio Latente):** Crear una interfaz interactiva donde el usuario pueda explorar la nube de puntos tridimensional de todos los embeddings de marcas y escudos, observando cómo se agrupan geométricamente por similitud morfológica.
2. **Arquitectura del Flujo de Reducción Dimensional:** Diseñar el contrato de proyección para que el backend calcule coordenadas $(x, y, z) \in [-1, 1]$ mediante PCA o UMAP y el frontend las renderice a 60 FPS con Canvas / WebGL (Three.js o lightweight particle system).
3. **Integrar el Botón "Ver en el Espacio Vectorial":** Permitir que el usuario haga clic en cualquier resultado de búsqueda y la cámara tridimensional viaje automáticamente (con tween suave) hasta el nodo correspondiente.
4. **Incorporar Tatuajes y Pinturas:** Diseñar la conmutación para incluir las dos colecciones adicionales de visión del GIBD en el Laboratorio.

---

## 📐 2. Contratos y Especificaciones Técnicas

### A. Contrato de Datos de Proyección Tridimensional (`frontend/src/types/projection.ts`)
```typescript
export interface LatentNode {
  id: string;
  item_code: string;
  label: string;
  category: string;
  thumbnail_url: string;
  coordinates_3d: [number, number, number]; // [x, y, z] normalizados [-1, 1]
  coordinates_2d: [number, number];         // [x, y]
  cluster_id: number;                       // Grupo semántico asignado por K-Means/DBSCAN
}

export interface LatentSpacePayload {
  dataset: 'brands' | 'soccer_crests' | 'tattoos';
  reduction_method: 'PCA' | 'UMAP';
  total_nodes: number;
  variance_explained_ratio?: number[]; // Para PCA
  nodes: LatentNode[];
  query_node?: LatentNode; // Nodo temporal de la imagen subida por el usuario
}
```

### B. Especificación Estética del Visualizador 3D
* **Fondo del Espacio:** Negro OLED `#0A0A0A` absoluto.
* **Nodos Pasivos:** Partículas esféricas de color gris platino `#71717A` con radio de 2.5 unidades y micro-resplandor focalizado.
* **Nodo Seleccionado / Consulta:** Partícula naranja eléctrico `#FF5500` con anillo orbital pulsante.
* **Nodos Similares (Top-K):** Color ámbar neón `#F59E0B` con líneas tensoras semitransparentes (`stroke: #FF5500`, `opacity: 0.35`) que los conectan al nodo de consulta.
* **Controles de Cámara:** Orbit controls suaves con damping (`enableDamping: true`, `dampingFactor: 0.05`), rotación táctil y zoom delimitado.

---

## 📋 3. Tareas Específicas de Thiago

- [ ] **T2.1.1 (Arquitectura de Proyección):** Definir el contrato `LatentSpacePayload` y acordar con Renato el pipeline de precomputación de coordenadas para evitar cálculo pesado en tiempo real.
- [ ] **T2.1.2 (Diseño UI del Modal 3D):** Diseñar el modal contenedor de pantalla completa (`rounded-[2rem]` con backdrop blur `backdrop-blur-md bg-black/80`) para la visualización del espacio latente.
- [ ] **T2.1.3 (Auditoría de Rendimiento Gráfico):** Probar el renderizador WebGL en dispositivos con GPU integrada (Intel Iris / AMD Radeon) para asegurar que no se produzcan caídas de frames (<55 FPS).
- [ ] **T2.1.4 (Code Review de Renato):** Auditar que `projection_engine.py` maneje adecuadamente proyecciones sobre nuevos vectores de consulta mediante transformación `pca.transform()` sin recomputar todo el dataset.
- [ ] **T2.1.5 (Code Review de Emmanuel):** Revisar que el componente `EmbeddingViewer3D.tsx` destruya correctamente los contextos de WebGL al desmontarse para prevenir memory leaks.

---

## 🛡️ 4. Plan de Verificación y Validación (V&V)

### Nivel 1 — Pre-Verificación Autónoma del Agente (Automated Testing)
1. **Compilación de Frontend:**
   ```bash
   cd frontend && npm run build
   ```
2. **Validación de Datos del Endpoint de Proyección:**
   * Invocación a `GET /api/v1/embeddings/projection?dataset=soccer_crests&dim=3`.
   * Verificar que todos los nodos contengan `coordinates_3d` con 3 números reales finitos, sin `NaN` ni `Infinity`.
3. **Auditoría de Destrucción de Contexto WebGL:**
   * Test de montaje y desmontaje sucesivo (10 veces) del componente 3D confirmando que el consumo de memoria heap de Chrome/Node no crezca de forma monotónica.

### Nivel 2 — Validación Obligatoria del Usuario (Human-in-the-Loop)
**Protocolo para Thiago:**
1. Realizar una búsqueda de un escudo o marca en el Laboratorio.
2. Hacer clic en el botón "Explorar en Espacio Vectorial 3D".
3. *Validación Visual:*
   - El modal 3D se abre con animación fluida.
   - La nube de puntos es visible en negro y naranja neón.
   - Rotar y acercar la cámara con el mouse/touchpad; la interacción debe sentirse suave y natural.
   - Al pasar el cursor sobre un nodo, debe mostrarse el tooltip con la imagen miniatura del club o marca.
4. **Aprobación Formal:** Si la experiencia interactiva es impactante y representativa del trabajo de G5, Thiago da su visto bueno.
