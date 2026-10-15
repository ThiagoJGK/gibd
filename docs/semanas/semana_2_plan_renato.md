# 📋 Plan Técnico: Semana 2 — Renato Bonín (Backend & AI)

> **Rol:** Backend & AI Engineer  
> **Ciclo:** Sprint 2  
> **Rama de Trabajo:** `feature/semana2-backend-gemini` (creada a partir de `dev`)  
> **Fecha de Entrega:** Viernes (Puesta en Común)  

---

## 🎯 Objetivo
Implementar y robustecer el servicio de asistencia de IA con Google Gemini para el análisis de papers académicos, extracción estructurada de resúmenes ejecutivos (abstracts) y generación de prompts conceptuales para portadas artísticas.

---

## 📑 Contrato de Integración de API (FastAPI / Pydantic)

### Endpoint: `POST /api/v1/ai/analyze-paper`

#### Request (JSON)
```json
{
  "paper_text": "string con el contenido textual extraído del paper científico (mínimo 100 caracteres)..."
}
```

#### Response (JSON Estricto - Código 200)
```json
{
  "abstract": "Resumen profesional en español de hasta 3 párrafos, cautivador y apto para divulgación pública en la web.",
  "image_prompt": "Detailed conceptual English visual prompt for DALL-E 3 / Midjourney describing an artistic metaphorical cover --ar 16:9"
}
```

#### Errores Estándar (HTTP)
* **400 Bad Request:** Texto vacío o de longitud insuficiente.
* **502 Bad Gateway:** Error al parsear el JSON generado por Gemini.
* **500 Internal Server Error:** Excepción de conexión o cuota con la API de Google.

---

## 🛠️ Tareas Técnicas Paso a Paso

### Tarea 1: Robustecer el Endpoint en `backend/app/main.py`
- [ ] Configurar el modelo `gemini-1.5-flash` con forzado de respuesta estructurada (`generation_config={"response_mime_type": "application/json"}`).
- [ ] Implementar regex de seguridad para extraer el bloque `{ ... }` en caso de que Gemini añada texto o backticks alrededor del JSON.
- [ ] Validar que la respuesta mapee exactamente contra el modelo Pydantic `PaperAnalysisResponse`.
- [ ] Mantener el fallback offline determinista si `GEMINI_API_KEY` no está configurada, devolviendo un abstract y prompt predefinidos sin tirar error.

### Tarea 2: Validación y Pruebas Unitarias
- [ ] Extraer un fragmento de texto real de los documentos del GIBD en `Contexto/` (estudio CACIC 2023 de marcas de ganado o papers de PTAH).
- [ ] Ejecutar prueba local por cURL:
  ```bash
  curl -X POST "http://127.0.0.1:8000/api/v1/ai/analyze-paper" \
       -H "Content-Type: application/json" \
       -d "{\"paper_text\": \"Texto de prueba del paper...\"}"
  ```
- [ ] Validar que la latencia total de respuesta sea menor a 2.5 segundos.

---

## 🛡️ Quality Gate (Criterios de Aprobación para Merge)
1. `python -m py_compile backend/app/main.py` finaliza con código 0 (sin errores de sintaxis).
2. El endpoint responde en formato JSON válido según el esquema acordado.
3. No se introducen dependencias pesadas no autorizadas en `backend/requirements.txt`.
