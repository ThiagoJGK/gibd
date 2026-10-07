import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';

export interface CloudinaryUploaderProps {
  onImageSelected: (file: File, previewUrl: string, cloudinaryUrl?: string) => void;
  isAnalyzing: boolean;
  disabled?: boolean;
  label?: string;
}

export const CloudinaryUploader: React.FC<CloudinaryUploaderProps> = ({
  onImageSelected,
  isAnalyzing,
  disabled = false,
  label = 'Sube tu imagen',
}) => {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [cloudinaryUrl, setCloudinaryUrl] = useState<string | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (file.size > 5 * 1024 * 1024) {
        setError('El archivo supera los 5 MB. Por favor, selecciona una imagen más pequeña.');
        return;
      }
      setError(null);
      setProgress(null);
      setCloudinaryUrl(null);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);

      // Previsualización instantánea + fallback local inmediato
      onImageSelected(file, url);

      // Subida en paralelo a Cloudinary (solo si está configurado mediante env)
      const cloudName = import.meta.env.VITE_CLOUDINARY_CLOUD_NAME;
      const uploadPreset = import.meta.env.VITE_CLOUDINARY_UPLOAD_PRESET;
      if (!cloudName || !uploadPreset) return;

      const uploadToCloudinary = async () => {
        try {
          setProgress(10);
          const formData = new FormData();
          formData.append('file', file);
          formData.append('upload_preset', uploadPreset);
          const response = await fetch(
            `https://api.cloudinary.com/v1_1/${cloudName}/image/upload`,
            { method: 'POST', body: formData }
          );
          if (!response.ok) throw new Error(`Cloudinary respondió ${response.status}`);
          const data = await response.json();
          setProgress(100);
          setCloudinaryUrl(data.secure_url);
          onImageSelected(file, url, data.secure_url);
        } catch {
          // Fallback local: el usuario sigue trabajando con el File en memoria
          setProgress(null);
        }
      };

      void uploadToCloudinary();
    },
    [onImageSelected]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        handleFile(acceptedFiles[0]);
      }
    },
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp', '.svg'],
    },
    maxFiles: 1,
    disabled,
  });

  return (
    <div className="relative w-full max-w-2xl mx-auto">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-[2rem] p-10 text-center cursor-pointer transition-all duration-300 ${
          isDragActive || previewUrl
            ? 'border-[#FF5500]/50 bg-[#FF5500]/5'
            : 'border-white/10 hover:border-white/20'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:scale-[1.02]'}`}
      >
        <input {...getInputProps()} />
        {previewUrl ? (
          <div className="relative w-full max-w-md mx-auto aspect-square rounded-xl overflow-hidden">
            <img
              src={previewUrl}
              alt="Previa"
              className="w-full h-full object-cover"
            />
            {cloudinaryUrl && (
              <span className="absolute top-2 right-2 px-3 py-1 rounded-full bg-[#121214] text-emerald-400 border border-emerald-500/30 rounded-full text-xs font-mono">
                Cloudinary: subido
              </span>
            )}
            {progress !== null && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="w-8 h-8 border-2 border-[#FF5500] border-t-transparent rounded-full animate-spin" />
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center gap-3 py-6">
            <div className="w-12 h-12 rounded-full bg-[#121214] flex items-center justify-center border border-white/10">
              <svg
                className="w-6 h-6 text-white/60"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
            </div>
            <p className="text-text-secondary text-sm font-medium">
              {isAnalyzing ? 'Procesando...' : label}
            </p>
            <p className="text-text-secondary/60 text-xs">
              Arrastra y suelta o haz clic para seleccionar
            </p>
          </div>
        )}
      </div>
      {error && (
        <p className="text-red-400 text-xs mt-3 text-center">{error}</p>
      )}
    </div>
  );
};

export default CloudinaryUploader;
