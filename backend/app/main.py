import os
import json
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

from app.services.siamese_engine import process_brand_inference, SiameseEngine, IMAGES_DIR

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
    allow_origins=[
        "http://localhost:5173", 
        "http://127.0.0.1:5173",
        "http://localhost:5174", 
        "http://127.0.0.1:5174",
        "http://localhost:5175", 
        "http://127.0.0.1:5175",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir estáticamente las imágenes del catálogo de marcas si el directorio existe
if os.path.exists(IMAGES_DIR):
    app.mount("/static/brands", StaticFiles(directory=IMAGES_DIR), name="static_brands")

# Modelos de validación de datos
class PaperAnalysisRequest(BaseModel):
    paper_text: str

class PaperAnalysisResponse(BaseModel):
    abstract: str
    image_prompt: str

class BrandMatchResponse(BaseModel):
    id: str
    brand_code: str
    name: str
    filename: str
    image_url: str
    similarity_score: float
    distance: float
    jurisdiction: str
    description: str

# Endpoint de Salud (Health Check)
@app.get("/api/v1/health", tags=["Mantenimiento"])
async def health_check():
    engine = SiameseEngine.get_instance()
    return {
        "status": "healthy",
        "service": "GIBD API Gateway",
        "gemini_active": gemini_key is not None,
        "brands_indexed": len(engine.catalog),
        "inference_device": "CPU (Zero-Cost Vector Engine)"
    }

# Endpoint de Inferencia Real de Marcas de Ganado (Redes Siamesas / MBR)
@app.post("/api/v1/inference/siamese-brands", response_model=List[BrandMatchResponse], tags=["Visión Computacional"])
async def match_brands(
    file: UploadFile = File(...),
    top_k: int = Query(default=10, ge=1, le=50, description="Cantidad de marcas más similares a retornar")
):
    """
    Recibe una imagen de marca de ganado, ejecuta el preprocesamiento MBR (Minimum Bounding Rectangle),
    extrae su vector estructural de características y calcula la similitud coseno contra el catálogo del GIBD.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo proporcionado no es una imagen válida (se requiere PNG, JPEG, WEBP o BMP)."
        )

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="El archivo enviado está vacío.")

        results = process_brand_inference(contents, top_k=top_k)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la inferencia en CPU: {str(e)}"
        )

# Endpoint para obtener el catálogo completo de marcas de ganado
@app.get("/api/v1/brands/catalog", tags=["Visión Computacional"])
async def get_brands_catalog():
    engine = SiameseEngine.get_instance()
    return {
        "total": len(engine.catalog),
        "brands": engine.catalog
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

# Para ejecutar localmente: uvicorn app.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
