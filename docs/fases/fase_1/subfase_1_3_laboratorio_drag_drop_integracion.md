# Plan de Subfase 1.3: Integración de Subida Drag & Drop e Inferencia en Vivo

## 1. Información General
- **Fase Perteneciente:** FASE 1 - Core de IA de Marcas de Ganado, Gateway Python y Wrapper Embebido
- **Subfase:** 1.3
- **Desarrollador Asignado:** Thiago Gomez Kehler (con apoyo de Renato en contrato de API)

---

## 2. Objetivo de la Subfase
Conectar el área de carga interactiva (Drag & Drop) de marcas de ganado en `Laboratorio.tsx` con el backend en Python, enviando el archivo de imagen de forma asíncrona hacia `/api/v1/inference/siamese-brands` y gestionando los estados de petición (cargando, éxito, error).

---

## 3. Tareas Técnicas Detalladas

- [ ] **Tarea 1.3.1: Servicio de Comunicación de Inferencia**
  - Archivo: `frontend/src/utils/aiService.ts`
  - Crear función `fetchBrandInference(imageFile: File, topK: int)` utilizando `fetch` / FormData.
- [ ] **Tarea 1.3.2: Integración en Laboratorio.tsx (Zona Drag & Drop)**
  - Archivo: `frontend/src/pages/Laboratorio.tsx`
  - Capturar el evento de soltado de archivo (`onDrop`) o selección en explorador.
  - Almacenar la previsualización de la imagen cargada en el estado React.
- [ ] **Tarea 1.3.3: Disparo de Inferencia y Loading State**
  - Conectar el botón **"Buscar Similitudes (Red Siamesa)"** para llamar a `fetchBrandInference`.
  - Desplegar indicador visual de inferencia en progreso con spinner y estado de pulso.

---

## 4. Plan de Comprobación, Validación y Testing

### A. Pruebas de Integración
```powershell
# Servidor Python activo en puerto 8000
# Frontend activo en puerto 5173
```

### B. Pruebas Manuales
1. Arrastrar una imagen de marca de ganado al recuadro del Laboratorio.
2. Hacer clic en "Buscar Similitudes".
3. Inspeccionar en la consola de herramientas de desarrollador (F12 -> Network):
   - Confirmar envío de petición `POST /api/v1/inference/siamese-brands`.
   - Confirmar recepción de respuesta HTTP 200 con la lista de candidatos.

---

## 5. Criterios de Aceptación de la Subfase
- Subida de archivos fluida con previsualización previa.
- Manejo transparente de estado de carga mientras el backend ejecuta el modelo.
- Cero bloqueos de la UI de React durante la llamada asíncrona.
