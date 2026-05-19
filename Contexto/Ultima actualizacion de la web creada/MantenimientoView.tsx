import React from 'react';
import { FileUploadButton, LoadingSpinner, ErrorMessage, Card } from './uiElements';
import { ImageModelId, UploadedImage } from './types'; //IMAGE_MODEL_CONFIG is passed as prop

interface MantenimientoViewProps {
  onLoadArticles: (file: File) => void;
  onLoadImagesForModel: (modelId: ImageModelId, files: File[]) => void;
  isLoading: boolean;
  error: string | null;
  articlesCount: number;
  imageDB: Record<ImageModelId, UploadedImage[]>;
  clearError: () => void;
  IMAGE_MODEL_CONFIG: Record<ImageModelId, { name: string }>;
}

const MantenimientoView: React.FC<MantenimientoViewProps> = ({
  onLoadArticles,
  onLoadImagesForModel,
  isLoading,
  error,
  articlesCount,
  imageDB,
  clearError,
  IMAGE_MODEL_CONFIG
}) => {
  const handleArticleFileSelect = (file: File) => {
    clearError();
    onLoadArticles(file);
  };

  const handleImageFilesSelect = (modelId: ImageModelId, files: File[]) => {
    clearError();
    onLoadImagesForModel(modelId, files);
  };

  return (
    <Card className="animate-fadeIn">
      <h2 className="text-2xl font-semibold text-sky-400 mb-6 text-center">Modo Mantenimiento: Carga de Datos</h2>
      {isLoading && <LoadingSpinner />}
      {error && (
        <div className="mb-4">
          <ErrorMessage message={error} />
          <button 
            onClick={clearError} 
            className="text-sm text-sky-400 hover:text-sky-300 mt-1 underline"
          >
            Descartar error
          </button>
        </div>
      )}

      <div className="space-y-10">
        <Card className="bg-slate-700">
          <h3 className="text-xl font-semibold text-sky-300 mb-4">Base de Datos de Artículos</h3>
          <p className="text-slate-400 mb-1">Selecciona un archivo <code className="text-xs bg-slate-600 px-1 rounded">.txt</code> que contenga todos los artículos concatenados.</p>
          <p className="text-slate-400 mb-4 text-sm">Los artículos deben estar separados por al menos una línea en blanco.</p>
          <FileUploadButton
            label="Cargar Archivo de Artículos (.txt)"
            accept=".txt,text/plain"
            onFileSelect={handleArticleFileSelect}
            className="w-full justify-center"
            id="upload-articles-txt"
            variant="sky"
          />
          <p className="text-sm text-slate-300 mt-3">Artículos cargados: <span className="font-bold text-sky-400">{articlesCount}</span></p>
        </Card>

        <div>
            <h3 className="text-xl font-semibold text-sky-300 mb-6 text-center border-b border-slate-600 pb-3">Bases de Datos de Imágenes por Modelo</h3>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {Object.values(ImageModelId).map((modelId) => (
                <Card key={modelId} className="bg-slate-700">
                <h4 className="text-lg font-semibold text-sky-300 mb-3">{IMAGE_MODEL_CONFIG[modelId].name}</h4>
                <p className="text-slate-400 mb-3 text-sm">Selecciona la carpeta que contiene las imágenes para este modelo.</p>
                <FileUploadButton
                    label={`Cargar Imágenes (${IMAGE_MODEL_CONFIG[modelId].name})`}
                    accept="image/*"
                    onFilesSelect={(files) => handleImageFilesSelect(modelId, files)}
                    isDirectoryUpload={true}
                    className="w-full justify-center text-sm py-2.5"
                    id={`upload-images-${modelId}`}
                    variant="sky"
                />
                <p className="text-xs text-slate-300 mt-2">
                    Imágenes cargadas ({IMAGE_MODEL_CONFIG[modelId].name}): <span className="font-bold text-sky-400">{imageDB[modelId]?.length || 0}</span>
                </p>
                </Card>
            ))}
            </div>
        </div>
      </div>
    </Card>
  );
};

export default MantenimientoView;