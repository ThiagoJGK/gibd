# 📅 Especificación Técnica Semanal — Semana 4: Thiago Gomez Kehler
## Rol: Tech Lead / Arquitecto de Software / Diseñador UI-UX Principal

> **Ciclo:** Sprint 4 — Prototipo G2 (Video Retrieval), Auditoría E2E, Release y Despliegue  
> **Ubicación:** `docs/plan_ia_interactiva/semana_4_plan_thiago.md`  
> **Regla Global Inviolable:** Luego de grandes cambios, preguntar siempre si continuar con subida a Git.

---

## 🎯 1. Objetivos Técnicos del Sprint

1. **Integrar el Módulo de Video Retrieval (G2):** Incorporar en el dial la opción "Video", conectando el reproductor de clips con análisis de dinamismo temporal por keyframes.
2. **Auditoría Pixel-Perfect del Design System Flat:** Revisar cada pantalla del sitio para garantizar cumplimiento irrestricto de:
   * Cápsulas interactivas (`rounded-full`).
   * Paneles orgánicos (`rounded-[2rem]`).
   * Fondo negro OLED `#0A0A0A` con acento naranja eléctrico `#FF5500`.
   * Cero `box-shadow` difuso en todo el proyecto.
3. **Optimización de Rendimiento y Code-Splitting en Vite:** Resolver las advertencias de chunks superiores a 500 kB en el build de producción mediante lazy loading (`React.lazy`) en rutas y componentes pesados (Three.js / WebGL).
4. **Control de Release y Despliegue en Vercel:** Ejecutar la regla global de consulta a Git, fusionar de forma ordenada `feature/*` -> `dev` -> `main` y verificar el pipeline de CI/CD en Vercel.

---

## 📐 2. Especificaciones de Auditoría y Control de Release

### A. Estrategia de Code-Splitting en Vite (`frontend/vite.config.ts`)
Para solucionar las advertencias de chunks $> 500\text{ kB}$ detectadas en el análisis preliminar:
```typescript
// Configuración de chunks en vite.config.ts / rolldown
export default defineConfig({
  build: {
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react': ['react', 'react-dom'],
          'vendor-motion': ['motion/react', 'framer-motion'],
          'vendor-three': ['three', '@react-three/fiber', '@react-three/drei'],
          'vendor-icons': ['lucide-react']
        }
      }
    }
  }
});
```

### B. Protocolo de Control de Release (Git Gatekeeper)
1. **Verificación de Rama Activa:** Trabajar exclusivamente en `dev` o ramas de funcionalidad `feature/*`.
2. **Quality Gate Final:**
   * `npm run build` en `frontend/` termina con código 0 y 0 errores de TypeScript.
   * `python -m py_compile` valida todos los archivos del backend.
   * La consola F12 de Chrome en modo producción está 100% limpia de errores y advertencias críticas.
3. **Consulta Obligatoria al Usuario:**  
   > *"Tech Lead: Todas las pruebas automáticas y visuales han concluido con éxito. ¿Deseas autorizar la subida a Git y el despliegue hacia la rama principal?"*

---

## 📋 3. Tareas Específicas de Thiago

- [ ] **T4.1.1 (Arquitectura de Video):** Diseñar la estructura de datos para marcadores temporales de video (`VideoEventMarker`: timestamp, tipo de evento, confianza).
- [ ] **T4.1.2 (Auditoría Integral de Código):** Ejecutar revisión estática en todo `frontend/src/` para eliminar clases CSS obsoletas o URLs a puertos locales (`localhost:8010`, `localhost:9999`).
- [ ] **T4.1.3 (Optimización de Chunks Vite):** Ajustar la configuración de empaquetado de Vite para que el bundle principal no supere los 250 kB gzippeados.
- [ ] **T4.1.4 (Verificación de Despliegue en Vercel):** Comprobar variables de entorno en producción (`VITE_CLOUDINARY_CLOUD_NAME`, endpoints de API) y validar el preview build.
- [ ] **T4.1.5 (Puesta en Común y Cierre):** Dirigir la demo final con los 6 grupos de investigación funcionando en vivo.

---

## 🛡️ 4. Plan de Verificación y Validación (V&V)

### Nivel 1 — Pre-Verificación Autónoma del Agente (Automated Testing)
1. **Compilación de Producción:**
   ```bash
   cd frontend && npm run build
   ```
   *Criterio:* Código 0, ningún chunk superior a los límites fijados, sin advertencias de tipos.
2. **Auditoría de Enlaces Rotos:**
   * Script de inspección estática que confirme que no existan strings con `localhost` en el código fuente final de frontend.
3. **Prueba de Inferencia E2E en Backend:**
   * Suite completa de `pytest` ejecutándose en $< 10\text{ segundos}$ para todos los motores (marcas, escudos, texto, audio, video).

### Nivel 2 — Validación Obligatoria del Usuario (Human-in-the-Loop)
**Protocolo Final de Aceptación para Thiago:**
1. Navegar por todas las pestañas del Laboratorio (Marcas, Escudos, Espacio 3D, Texto PTAH, Audio, Video).
2. Comprobar que en cada sección los resultados se calculen con datos reales y porcentajes coherentes.
3. Realizar prueba en dispositivo móvil real (smartphone).
4. **Decisión Final de Git:** Conceder la autorización formal para ejecutar el push a Git y promover a producción.
