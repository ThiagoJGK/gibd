import os
import logging
from typing import Optional
import cloudinary
import cloudinary.uploader

logger = logging.getLogger(__name__)

# Configuración de Cloudinary (si están disponibles las credenciales)
cloudinary_configured = False
if all(k in os.environ for k in ["CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET"]):
    cloudinary.config(
        cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
        api_key=os.environ["CLOUDINARY_API_KEY"],
        api_secret=os.environ["CLOUDINARY_API_SECRET"]
    )
    cloudinary_configured = True
else:
    logger.warning("Credenciales de Cloudinary no encontradas en variables de entorno. Fallback local activado.")

def upload_query_image(image_bytes: bytes, folder: str = "gibd/consultas") -> Optional[str]:
    """
    Sube la imagen a Cloudinary aplicando recortes, normalización y conversión a WebP.
    Retorna la URL segura si la subida fue exitosa, o None si hay error o modo offline.
    """
    if not cloudinary_configured:
        return None
        
    try:
        response = cloudinary.uploader.upload(
            image_bytes,
            folder=folder,
            format="webp",
            transformation=[
                {"width": 224, "height": 224, "crop": "pad", "background": "white"}
            ]
        )
        return response.get("secure_url")
    except Exception as e:
        logger.error(f"Error subiendo imagen a Cloudinary: {e}")
        return None
