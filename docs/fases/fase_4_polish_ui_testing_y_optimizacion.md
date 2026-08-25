# Plan de Fase 4: Expansión Multimodal, Pulido UX/UI Flat, Testing E2E y Despliegue Final

## 1. Visión General de la Fase
La Fase 4 es el cierre integral del proyecto. Se encarga de expandir las capacidades multimodales del Laboratorio (integrando la búsqueda NLP de PTAH-Jurídico y las interfaces de audio/video), auditar el cumplimiento del **Design System Flat del GIBD** (`border-radius: 9999px`, fondo `#0A0A0A`, acento `#FF5500`), garantizar responsividad móvil 100% y ejecutar la suite final de pruebas end-to-end antes del despliegue definitivo en Vercel.

---

## 2. Distribución de Responsabilidades (3 Desarrolladores)

### **Desarrollador 1 (Thiago Gomez Kehler - Backend Multimodal & Security Audit)**
- **Endpoints NLP & Multimodal en FastAPI:** Implementar endpoints adicionales (`/api/v1/inference/ptah-nlp`, `/api/v1/inference/audio-video`) para procesamiento de texto e indexación métrica.
- **Optimización & Seguridad:** Configuración de CORS, compresión de respuestas y sanidad de variables en producción.

### **Desarrollador 2 (Emanuel Davezac - Design System Flat & Animaciones)**
- **Auditoría del Design System Flat:** Revisar todas las vistas para asegurar la eliminación de sombras convencionales (`box-shadow`), favoreciendo bordes orgánicos finos y acento `#FF5500`.
- **Refinamiento de Interacciones:** Perfeccionar animaciones en Framer Motion y feedback táctil del Dial y botones.

### **Desarrollador 3 (Renato Bonín - PTAH NLP UI, Responsividad Mobile & Testing E2E)**
- **Integración PTAH-Jurídico en Laboratorio:** Conectar la búsqueda en lenguaje natural con respuestas estructuradas.
- **Responsividad Mobile & E2E:** Probar todas las pantallas en dispositivos móviles y ejecutar la matriz de pruebas finales.

---

## 3. Tareas Detalladas de la Fase 4

- [ ] **Tarea 4.1: Endpoints NLP y Multimodales** (Dev 1)
  - Archivo: `backend/app/main.py`
- [ ] **Tarea 4.2: Integración PTAH-Jurídico en Laboratorio** (Dev 3)
  - Archivo: `frontend/src/pages/Laboratorio.tsx`
- [ ] **Tarea 4.3: Auditoría Estética Design System Flat** (Dev 2)
  - Archivo: `frontend/src/index.css`, `frontend/src/pages/*.tsx`
- [ ] **Tarea 4.4: Responsividad Móvil y Testing E2E Final** (Dev 3 & Todo el equipo)

---

## 4. Plan de Comprobación, Validación y Testing Final

### A. Pruebas Automáticas de Compilación y Sintaxis
```powershell
cd frontend
npm run build
```

### B. Quality Gate Final
- [x] Inferencia de IA de Marcas de Ganado y Wrapper de Python activos (Fase 1).
- [x] CMS No-Dev funcional para Noticias y Papers (Fase 2).
- [x] CMS No-Dev funcional para Grupos y Equipo (Fase 3).
- [x] PTAH NLP y Búsqueda Multimodal operativos (Fase 4).
- [x] Design System Flat 100% respetado sin errores de consola.

---

## 5. Protocolo de Finalización y Subida Definitiva a Git

1. **Reunión de Cierre del Equipo (3 Devs)**
2. **Build Final:** `npm run build`
3. **Consulta Final de Subida (Regla Global):**
   - **PREGUNTAR AL USUARIO:** "Todas las fases del proyecto GIBD WEB han sido completadas, probadas y validadas con éxito. ¿Deseas autorizar la subida (push) final a Git hacia la rama main para activar el despliegue automático a producción en Vercel?"
4. **Subida a Git & Despliegue a Producción:** Tras la confirmación del usuario.
