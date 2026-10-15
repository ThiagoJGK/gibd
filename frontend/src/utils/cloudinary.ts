/**
 * Pipeline de Carga y Optimización de Imágenes — GIBD 2026
 * Ubicación: frontend/src/utils/cloudinary.ts
 *
 * Características:
 * - Soporte para Unsigned Upload Preset (gibd_lab_unsigned)
 * - Transformaciones en el Edge (224x224, pad blanco, webp, auto quality)
 * - Timeout estricto de 5 segundos con AbortController
 * - Fallback local garantizado a FileReader / File sin arrojar excepciones
 * - Modo Offline / Zero-Cost autónomo
 */

export interface UploadPipelineResult {
  file: File;
  previewUrl: string;             // Data URL local o URL de Cloudinary
  cloudinaryUrl?: string;         // URL original segura en CDN (si tuvo éxito)
  transformedUrl?: string;        // URL normalizada 224x224 webp para inferencia
  publicId?: string;
  storageMode: 'cloudinary_edge' | 'local_offline' | 'local_fallback';
  statusMessage: string;
}

/**
 * Genera la URL de transformación normalizada en el Edge según especificación técnica:
 * 224x224, padding con fondo blanco, formato webp, calidad automática.
 */
export const buildTransformedUrl = (
  cloudName: string,
  publicId: string,
  format: string = 'webp'
): string => {
  return `https://res.cloudinary.com/${cloudName}/image/upload/c_pad,w_224,h_224,b_white,f_webp,q_auto/${publicId}.${format}`;
};

/**
 * Lee un archivo local como Data URL en Base64 de forma asíncrona sin fugas de memoria
 */
export const readFileAsDataUrl = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error('Error al leer el archivo local mediante FileReader'));
    reader.readAsDataURL(file);
  });
};

/**
 * Procesa la carga de una imagen intentando Cloudinary primero (si está configurado)
 * y cayendo de forma limpia y garantizada en FileReader local ante cualquier falla o modo offline.
 */
export const processImageUpload = async (
  file: File,
  onImmediatePreview?: (dataUrl: string) => void
): Promise<UploadPipelineResult> => {
  // 1. Obtener preview local inmediato (0ms latencia visual)
  let localDataUrl = '';
  try {
    localDataUrl = await readFileAsDataUrl(file);
    if (onImmediatePreview) {
      onImmediatePreview(localDataUrl);
    }
  } catch {
    // Si FileReader falla por algún motivo inusual, fallback a ObjectURL
    localDataUrl = URL.createObjectURL(file);
    if (onImmediatePreview) {
      onImmediatePreview(localDataUrl);
    }
  }

  const cloudName = import.meta.env.VITE_CLOUDINARY_CLOUD_NAME;
  const uploadPreset = import.meta.env.VITE_CLOUDINARY_UPLOAD_PRESET;

  // 2. Verificar configuración de Cloudinary
  const isConfigured = Boolean(
    cloudName &&
    uploadPreset &&
    cloudName !== 'tu-cloud-name' &&
    uploadPreset !== 'tu-upload-preset'
  );

  if (!isConfigured) {
    return {
      file,
      previewUrl: localDataUrl,
      storageMode: 'local_offline',
      statusMessage: 'Modo Local / Autónomo (Credenciales de Cloudinary no provistas)',
    };
  }

  // 3. Intento de subida a Cloudinary con timeout estricto de 5 segundos
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);

  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('upload_preset', uploadPreset);

    const uploadEndpoint = `https://api.cloudinary.com/v1_1/${cloudName}/image/upload`;
    const res = await fetch(uploadEndpoint, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      const errorText = await res.text().catch(() => res.statusText);
      console.warn(`[Cloudinary] Subida rechazada (${res.status}): ${errorText}. Activando fallback local.`);
      return {
        file,
        previewUrl: localDataUrl,
        storageMode: 'local_fallback',
        statusMessage: `Servidor Cloudinary respondió ${res.status}. Fallback local activo.`,
      };
    }

    const data = await res.json();
    const transformed = buildTransformedUrl(cloudName, data.public_id, data.format || 'webp');

    return {
      file,
      previewUrl: transformed || data.secure_url || localDataUrl,
      cloudinaryUrl: data.secure_url,
      transformedUrl: transformed,
      publicId: data.public_id,
      storageMode: 'cloudinary_edge',
      statusMessage: 'Procesado y normalizado en Cloudinary Edge (224x224 WebP)',
    };
  } catch (err: unknown) {
    clearTimeout(timeoutId);
    const message = err instanceof Error ? err.message : String(err);
    console.warn('[Cloudinary] Falla de red o timeout:', message);
    return {
      file,
      previewUrl: localDataUrl,
      storageMode: 'local_fallback',
      statusMessage: 'Falla de red Cloudinary o tiempo excedido. Operando con fallback local.',
    };
  }
};
