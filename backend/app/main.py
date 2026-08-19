import os
import json
import re
import time
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# Importar el servicio de inferencia siamés
from app.services.siamese_engine import process_brand_inference

# Cargar variables de entorno
load_dotenv()

# Configurar el SDK de Gemini si la clave está provista
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    genai.configure(api_key=gemini_key)

app = FastAPI(
    title="GIBD API Gateway",
    description="Pasarela de servicios multimodales y automatización de IA para GIBD 2026",
    version="1.0.0"
)

# Configurar CORS para permitir peticiones desde el frontend local de Vite
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos de validación de datos
class PaperAnalysisRequest(BaseModel):
    paper_text: str

class PaperAnalysisResponse(BaseModel):
    abstract: str
    image_prompt: str

# Endpoint de Salud (Health Check)
@app.get("/api/v1/health", tags=["Mantenimiento"])
async def health_check():
    return {
        "status": "healthy",
        "service": "GIBD API Gateway",
        "gemini_active": gemini_key is not None
    }

# Endpoint de Análisis de Papers Asistido por IA (Gemini)
@app.post("/api/v1/ai/analyze-paper", response_model=PaperAnalysisResponse, tags=["Inteligencia Artificial"])
async def analyze_paper(request: PaperAnalysisRequest):
    if not gemini_key:
        return PaperAnalysisResponse(
            abstract=(
                "Este artículo de investigación presenta un análisis del estado del arte en la aplicación "
                "de modelos avanzados de Inteligencia Artificial para el procesamiento de información multimodal. "
                "Se discuten las métricas clave del sistema y se propone un marco relacional escalable sobre base de datos."
            ),
            image_prompt="A futuristic holographic visual database connecting image, sound, and text nodes in a digital purple-orange network, highly detailed, 8k resolution, flat vector art style --ar 16:9"
        )
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
Actúa como un editor científico y diseñador gráfico experto para el GIBD (Grupo de Investigación en Big Data).
Analiza el siguiente texto extraído de un paper académico y genera:
1. Un resumen o abstract en español de no más de 3 párrafos. Debe ser cautivador, sumamente profesional y adaptado para lectura pública en la web de investigaciones.
2. Un prompt descriptivo y conceptual en inglés para generar una imagen artística que ilustre este paper (apto para Midjourney, Leonardo AI o DALL-E 3). Debe ser muy visual y representar metafóricamente el contenido del paper.

Texto del paper a analizar:
{request.paper_text}

Debes responder ÚNICAMENTE con un objeto JSON válido que contenga exactamente estas dos llaves:
{{
  "abstract": "El resumen ejecutivo en español aquí...",
  "image_prompt": "The detailed English image prompt here..."
}}
        """
        
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        raw_text = response.text.strip()
        
        clean_json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if clean_json_match:
            raw_text = clean_json_match.group(0)
            
        data = json.loads(raw_text)
        
        return PaperAnalysisResponse(
            abstract=data.get("abstract", "Resumen no disponible."),
            image_prompt=data.get("image_prompt", "Prompt artístico no disponible.")
        )
        
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail="Error al decodificar la respuesta estructurada de Gemini AI."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno durante la inferencia de Gemini: {str(e)}"
        )

# Tarea 1.1.2: Endpoint POST de Inferencia de Marcas de Ganado
@app.post("/api/v1/inference/siamese-brands", tags=["Laboratorio de IA"])
async def siamese_brands_inference(
    file: UploadFile = File(...),
    top_k: int = Query(default=10, ge=1)
):
    """
    Recibe una imagen de marca de ganado (multipart/form-data) y realiza la consulta
    de similitud con redes siamesas simuladas en base a las marcas registradas.
    """
    # Validar formato de archivo básico
    content_type = file.content_type or ""
    if not (content_type.startswith("image/") or file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))):
        raise HTTPException(
            status_code=400,
            detail="El archivo proporcionado debe ser una imagen válida (PNG, JPG, JPEG o WEBP)."
        )
        
    start_time = time.time()
    try:
        # Leer los bytes de la imagen subida
        image_bytes = await file.read()
        
        # Ejecutar la simulación del módulo siamés
        inference_results = process_brand_inference(image_bytes=image_bytes, top_k=top_k)
        
        elapsed_time_ms = (time.time() - start_time) * 1000.0
        
        # Loguear tiempo de procesamiento de forma limpia en consola
        print(f"[Inference] Procesamiento de '{file.filename}' completado con éxito en {elapsed_time_ms:.2f} ms. "
              f"Resultados retornados: {len(inference_results)} (Top-K: {top_k}).")
        
        return {
            "status": "success",
            "query_info": {
                "filename": file.filename,
                "top_k": top_k,
                "inference_time_ms": round(elapsed_time_ms, 2)
            },
            "results": inference_results
        }
    except Exception as e:
        print(f"[Inference Error] Excepción al realizar la inferencia siamesa: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor durante la inferencia siamesa: {str(e)}"
        )

# Para ejecutar localmente: uvicorn app.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
