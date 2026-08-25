# Plan de Subfase 1.4: UI de Ajustes Top-K, Slider de Umbrales y Tarjetas de Métricas de Similitud

## 1. Información General
- **Fase Perteneciente:** FASE 1 - Core de IA de Marcas de Ganado, Gateway Python y Wrapper Embebido
- **Subfase:** 1.4
- **Desarrollador Asignado:** Thiago Gomez Kehler

---

## 2. Objetivo de la Subfase
Construir y conectar los controles interactivos en `Laboratorio.tsx` (Slider de Top-K y ajustador de umbral de similitud) para que modifiquen dinámicamente los parámetros de inferencia enviados al backend. Implementar la grilla visual de resultados que despliega las tarjetas de marcas encontradas, mostrando badges con porcentaje de coincidencia, distancia métrica relacional y la comparativa visual de la imagen consulta frente a cada resultado.

---

## 3. Tareas Técnicas Detalladas

- [ ] **Tarea 1.4.1: Conexión Reactiva de Controles (Top-K & Umbrales)**
  - Archivo: `frontend/src/pages/Laboratorio.tsx`
  - Conectar el estado del slider de `topK` (rango 1 a 50) para que actualice de forma reactiva la petición enviada a la API de inferencia.
- [ ] **Tarea 1.4.2: Componente de Grilla de Resultados de Similitud**
  - Archivos: `frontend/src/pages/Laboratorio.tsx`, `frontend/src/components/BrandResultCard.tsx`
  - Crear tarjetas visuales con estética Flat (`border-radius: 1.5rem`, fondo `card-glass-purple`):
    - Badge con porcentaje de similitud (ej: `96.4% Coincidencia`).
    - Métrica de distancia euclidiana/coseno (ej: `Distancia: 0.036`).
    - Nombre de la marca y opción para ampliar/comparar con la imagen de consulta.
- [ ] **Tarea 1.4.3: Animaciones de Entrada y Feedback háptico/sonoro**
  - Animaciones de entrada en cascarada con Framer Motion para los resultados.
  - Reproducción de micro-tick sonoro al mover el slider de Top-K.

---

## 4. Plan de Comprobación, Validación y Testing

### A. Pruebas de Interfaz y Reactividad
```powershell
cd frontend
npm run dev
```

### B. Pruebas Manuales
1. Deslizar el control de Top-K de 10 a 5.
2. Hacer clic en "Buscar Similitudes".
3. Verificar que se desplieguen exactamente 5 tarjetas de resultados.
4. Inspeccionar la información mostrada en cada tarjeta: el porcentaje de similitud debe estar claramente visible y ordenado de mayor a menor coincidencia.

---

## 5. Criterios de Aceptación de la Subfase
- Slider Top-K reactivo e integrado con el envío de parámetros.
- Tarjetas de resultados de alta fidelidad estética respetando el Design System Flat.
- Feedback táctil/auditivo funcionando adecuadamente.
