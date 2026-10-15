# 📅 Especificación Técnica Semanal — Semana 1: Thiago Gomez Kehler
## Rol: Tech Lead / Arquitecto de Software / Diseñador UI-UX Principal

> **Ciclo:** Sprint 1 — Visión Multimodelo, Calibración de Escudos y Cloudinary  
> **Ubicación:** `docs/plan_ia_interactiva/semana_1_plan_thiago.md`  
> **Regla Inviolable:** No modificar ramas centrales sin verificar `npm run build` en 0 errores y consultar la subida a Git.

---

## 🎯 1. Objetivos Técnicos del Sprint

1. **Definir la Arquitectura de Visión Multimodelo:** Diseñar el contrato de conmutación entre el catálogo de Marcas de Ganado (80 ítems) y el de Escudos de Fútbol (150 clubes) dentro de `Laboratorio.tsx`.
2. **Supervisar la Calibración Matemática de Escudos:** Validar que la fórmula de temperatura logística resuelva el "efecto cono" de los vectores ReLU y garantice que similitudes espurias caigan por debajo del $20\%$.
3. **Establecer el Pipeline de Almacenamiento con Cloudinary:** Configurar las credenciales, definir el upload preset seguro y estructurar las transformaciones de imágenes en el Edge.
4. **Garantizar el Estándar Flat Design:** Auditar que todos los componentes nuevos de Emmanuel respeten la geometría de cápsula (`rounded-full`), tarjetas con radio orgánico (`rounded-[2rem]`), paleta OLED `#0A0A0A` con acento naranja `#FF5500` y cero sombras difusas.

---

## 📐 2. Contratos y Especificación de Arquitectura

### A. Contrato de Datos para Modelos de Visión (`frontend/src/types/vision.ts`)
```typescript
export type VisionModelType = 'brands' | 'soccer_crests';

export interface VisionMatchResult {
  id: string;
  item_code: string;
  name: string;
  filename: string;
  image_url: string;
  similarity_score: number; // Porcentaje calibrado [0.0, 100.0]
  distance: number;         // Distancia métrica euclidiana normalizada [0.0, 1.4142]
  metadata: {
    category: 'Ganadería' | 'Fútbol';
    jurisdiction_or_league: string; // Ej: "Entre Ríos - Uruguay" o "Liga Profesional Argentina"
    origin_country?: string;        // Ej: "Argentina", "España", "Brasil"
    description: string;
  };
}

export interface ModelSelectorOption {
  id: VisionModelType;
  label: string;
  badge: string;
  endpoint: string;
}
```

### B. Especificación Matemática de Calibración
El motor convolucional de escudos (ResNet50 / DenseNet) genera vectores normalizados $\mathbf{u}, \mathbf{v} \in \mathbb{R}^D$ con $\cos(\theta) = \mathbf{u} \cdot \mathbf{v} \in [-1, 1]$.  
Debido a las funciones de activación ReLU, la similitud bruta suele concentrarse entre $0.70$ y $0.85$. Thiago debe auditar que la función implementada en backend aplique la transformación sigmoidal por temperatura:

$$S_{\text{calibrada}} = \frac{100}{1 + \exp\left(-\frac{\cos(\theta) - \tau}{T}\right)}$$

* **Parámetros auditados:**
  * Umbral de corte: $\tau = 0.82$
  * Temperatura: $T = 0.04$
  * Distancia métrica euclidiana acotada: $d = \sqrt{2 \cdot \max(0, 1 - \cos(\theta))}$

### C. Arquitectura de Cloudinary en el Edge
* **Upload Preset:** `gibd_lab_unsigned` (modo Unsigned para clientes frontend).
* **Parámetros de Normalización Inmediata:**
  * `width: 224, height: 224, crop: 'pad', background: 'white'` (garantiza entrada cuadrada idéntica a los tensores de inferencia).
  * `fetch_format: 'webp', quality: 'auto'` (minimiza ancho de banda).

---

## 📋 3. Tareas Específicas de Thiago

- [ ] **T1.1.1 (Arquitectura Frontend):** Crear `frontend/src/types/vision.ts` con las interfaces unificadas y adaptar `Laboratorio.tsx` para soportar selector de modelos de visión sin duplicar código.
- [ ] **T1.1.2 (Diseño UI Flat):** Diseñar las cápsulas conmutadoras (`rounded-full`) en `Laboratorio.tsx` con microinteracción Framer Motion y feedback sonoro háptico (`playMechanicalClick`).
- [ ] **T1.1.3 (Setup Cloudinary):** Documentar las variables de entorno requeridas en `.env.example`:
  ```bash
  VITE_CLOUDINARY_CLOUD_NAME=gibd-utn
  VITE_CLOUDINARY_UPLOAD_PRESET=gibd_lab_unsigned
  CLOUDINARY_API_KEY=...
  CLOUDINARY_API_SECRET=...
  ```
- [ ] **T1.1.4 (Code Review de Renato):** Revisar `backend/app/services/crests_engine.py` para asegurar que la matriz de 150 embeddings esté normalizada L2 en memoria y la inferencia tome $<120\text{ ms}$.
- [ ] **T1.1.5 (Code Review de Emmanuel):** Revisar `frontend/src/components/CloudinaryUploader.tsx` asegurando cero `box-shadow` difuso y que los estados de carga tengan spinners con acento naranja `#FF5500`.

---

## 🛡️ 4. Plan de Verificación y Validación (V&V)

### Nivel 1 — Pre-Verificación Autónoma del Agente (Automated Testing)
Antes de solicitar la revisión de Thiago, el agente o CI debe validar:
1. **Compilación Estricta:**
   ```bash
   cd frontend && npm run build
   ```
   *Criterio:* Código de salida 0, sin advertencias de tipos TypeScript.
2. **Validación de Sintaxis y Tipado Backend:**
   ```bash
   python -m py_compile backend/app/main.py backend/app/services/crests_engine.py
   ```
3. **Test Matemático Automatizado:**
   * Ejecutar un script de prueba que verifique que pares de imágenes con $\cos(\theta) = 0.72$ resulten en $S_{\text{calibrada}} \le 15.0\%$, y pares con $\cos(\theta) = 0.92$ resulten en $S_{\text{calibrada}} \ge 90.0\%$.

### Nivel 2 — Validación Obligatoria del Usuario (Human-in-the-Loop)
**Protocolo de Prueba Manual para Thiago:**
1. **Prueba de Conmutación de Modelos:**
   - Abrir `http://localhost:5173/laboratorio`.
   - Conmutar entre "Marcas de Ganado" y "Escudos de Fútbol".
   - *Validación:* Verificar que se escuche el "clac" mecánico físico y que el selector cambie con animación fluida sin parpadeos.
2. **Prueba de Subida con Cloudinary:**
   - Arrastrar un archivo PNG de un club a la zona de carga.
   - *Validación:* La imagen debe previsualizarse de inmediato centrada sobre fondo neutro sin distorsión de aspecto.
3. **Prueba de Calibración de Escudos:**
   - Consultar el escudo de Boca Juniors.
   - *Validación:* El resultado #1 debe ser Boca Juniors con similitud $>90\%$. Ningún club aleatorio debe mostrar porcentajes engañosos superiores al $60\%$.
4. **Aprobación Formal:** Si todo cumple los estándares de calidad, Thiago aprueba la integración y se procede a la consulta de subida a Git.
