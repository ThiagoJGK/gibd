
import React, { useState, useEffect, useCallback } from 'react';
import { UploadedImage, ImageModelId } from './types';
import { LoadingSpinner, ErrorMessage, Card, TabButton, FileUploadButton, ImageCard } from './uiElements';
import { imageFileToUploadedImage, queryImageViaAPI } from './services'; 

interface ImageExplorationConsultaViewProps {
  imageDBForModel: UploadedImage[]; // This is from Mantenimiento, primarily for display or if user expects API to search *only* these.
  isLoadingImages: boolean; 
  currentImageModelId: ImageModelId;
  setCurrentImageModelId: (modelId: ImageModelId) => void;
  IMAGE_MODEL_CONFIG: Record<ImageModelId, { name: string }>;
  comparisonImageForApp: UploadedImage | null; 
  setComparisonImageForApp: (image: UploadedImage | null) => void; 
}

const ImageExplorationConsultaView: React.FC<ImageExplorationConsultaViewProps> = ({
  imageDBForModel, 
  isLoadingImages,
  currentImageModelId,
  setCurrentImageModelId,
  IMAGE_MODEL_CONFIG,
  comparisonImageForApp,
  setComparisonImageForApp
}) => {
  const [numImagesToDisplay, setNumImagesToDisplay] = useState<number>(5);
  const [similarityResults, setSimilarityResults] = useState<UploadedImage[]>([]); 
  const [isProcessingSimilarity, setIsProcessingSimilarity] = useState<boolean>(false); 
  const [error, setError] = useState<string | null>(null);
  
  const currentModelName = IMAGE_MODEL_CONFIG[currentImageModelId].name;

  useEffect(() => {
    setError(null);
  }, [currentImageModelId]); 

  useEffect(() => {
    // Adjust numImagesToDisplay max limit based on available similarity results or if imageDBForModel is small (for non-comparison view)
    const maxForSimilarity = comparisonImageForApp && similarityResults.length > 0 ? similarityResults.length : 0;
    const maxForDb = imageDBForModel.length > 0 ? imageDBForModel.length : 0;

    let effectiveMax = 1; // Default to 1 if no images anywhere
    if (comparisonImageForApp) { // In comparison mode
        effectiveMax = maxForSimilarity > 0 ? maxForSimilarity : 1;
    } else { // Not in comparison mode (though this view is now primarily for comparison)
        effectiveMax = maxForDb > 0 ? maxForDb : 1;
    }
    
    const maxAllowed = Math.min(10, effectiveMax);

    setNumImagesToDisplay(prev => {
        const currentVal = Math.min(prev, maxAllowed);
        return currentVal > 0 ? currentVal : (maxAllowed > 0 ? Math.min(5, maxAllowed) : 1) ; 
    });
  }, [comparisonImageForApp, similarityResults, imageDBForModel]);


  const handleAPIComparisonImageUpload = async (file: File) => {
    setIsProcessingSimilarity(true);
    setError(null);
    setSimilarityResults([]);
    
    let uploadedQueryImage: UploadedImage | null = null;
    try {
      uploadedQueryImage = await imageFileToUploadedImage(file);
      setComparisonImageForApp(uploadedQueryImage);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error al procesar la imagen de consulta localmente.");
      setComparisonImageForApp(null); 
      setIsProcessingSimilarity(false);
      return;
    }

    try {
      const resultsFromAPI = await queryImageViaAPI(file, currentImageModelId);
      setSimilarityResults(resultsFromAPI);

      if (resultsFromAPI.length === 0 && !error) { // Only set this if no other error occurred
         // No specific error message here, UI will show "no results"
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : `Error al buscar imágenes similares vía API para "${currentModelName}".`);
      setSimilarityResults([]); 
    } finally {
      setIsProcessingSimilarity(false);
    }
  };

  const handleClearComparison = () => {
    setComparisonImageForApp(null);
    setSimilarityResults([]); 
    setError(null);
  };
  
  if (isLoadingImages && imageDBForModel.length === 0 && !comparisonImageForApp && !isProcessingSimilarity) {
     return <LoadingSpinner />;
  }

  const maxSliderValue = comparisonImageForApp && similarityResults.length > 0 
    ? Math.max(1, Math.min(10, similarityResults.length))
    : 1; // Default to 1 if not in active comparison with results


  return (
    <Card className="animate-fadeIn">
      <h2 className="text-2xl font-semibold text-sky-400 mb-4 text-center">
        {comparisonImageForApp ? `Comparando Imagen en ${currentModelName} (API)` : `Comparador de Imágenes (API): ${currentModelName}`}
      </h2>
      
      <div className="mb-6 flex flex-wrap justify-center gap-2 sm:gap-3 p-2 bg-slate-700/50 rounded-2xl">
        {Object.values(ImageModelId).map((modelId) => (
          <TabButton
            key={modelId}
            onClick={() => {
              setCurrentImageModelId(modelId);
              setComparisonImageForApp(null); 
              setSimilarityResults([]);
              setError(null);
            }}
            isActive={currentImageModelId === modelId}
            forceDisableGlow={true} // Disable glow for active model tabs
          >
            {IMAGE_MODEL_CONFIG[modelId].name}
          </TabButton>
        ))}
      </div>

      {error && <ErrorMessage message={error} />}
      {isProcessingSimilarity && <LoadingSpinner />}

      <div className="mb-6 p-4 bg-slate-700 rounded-2xl shadow">
        <div className="grid sm:grid-cols-2 gap-4 items-center">
            <div>
                <label htmlFor="numImagesSlider" className="block text-sm font-medium text-slate-300 mb-2">
                    Imágenes similares a mostrar (1-{maxSliderValue}): <span className="font-bold text-sky-400">{numImagesToDisplay}</span>
                </label>
                <input
                    type="range"
                    id="numImagesSlider"
                    min="1"
                    max={maxSliderValue} 
                    value={numImagesToDisplay}
                    onChange={(e) => {
                        const val = parseInt(e.target.value, 10);
                        setNumImagesToDisplay(Math.max(1, Math.min(val, maxSliderValue)));
                    }}
                    className="w-full h-3 bg-slate-600 rounded-full appearance-none cursor-pointer accent-sky-500 pill-slider"
                    disabled={!comparisonImageForApp || similarityResults.length === 0 || isProcessingSimilarity || maxSliderValue <= 1}
                />
            </div>
            <div className="flex flex-col space-y-2 sm:space-y-0 sm:space-x-2 sm:flex-row justify-end">
                <FileUploadButton
                    label="Cargar Imagen para Comparar (API)"
                    accept="image/*"
                    onFileSelect={handleAPIComparisonImageUpload} 
                    className="w-full sm:w-auto text-sm py-2.5"
                    id="upload-comparison-image-api"
                    variant="sky"
                    disabled={isProcessingSimilarity}
                />
                {comparisonImageForApp && (
                    <button
                        onClick={handleClearComparison}
                        className="bg-amber-600 hover:bg-amber-700 text-white font-semibold py-2.5 px-5 rounded-full shadow-md transition duration-150 ease-in-out w-full sm:w-auto text-sm button-glow-amber"
                        disabled={isProcessingSimilarity}
                    >
                        Limpiar Comparación
                    </button>
                )}
            </div>
        </div>
      </div>

      {comparisonImageForApp && (
        <div className="mb-8 animate-fadeIn">
          <h3 className="text-xl font-semibold text-sky-300 mb-3 text-center">Imagen de Consulta</h3>
          <div className="max-w-xs mx-auto">
             <ImageCard image={comparisonImageForApp} modelName={currentModelName} index={0} isQueryImage={true} />
          </div>
          {!isProcessingSimilarity && similarityResults.length > 0 && (
            <>
              <h3 className="text-xl font-semibold text-sky-300 mt-8 mb-4 text-center">Imágenes Similares Encontradas (API)</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                {similarityResults.slice(0, numImagesToDisplay).map((image, index) => (
                  <ImageCard key={image.id || `${image.name}-sim-${index}`} image={image} modelName={currentModelName} index={index} similarity={image.similarity} />
                ))}
              </div>
            </>
          )}
           {!isProcessingSimilarity && similarityResults.length === 0 && !error && ( // Only show if not loading, no error, and results are empty
             <p className="text-center text-slate-400 py-5">
                La API no encontró imágenes similares para esta consulta en "{currentModelName}". Pruebe con otra imagen.
             </p>
           )}
        </div>
      )}

      {!comparisonImageForApp && !isProcessingSimilarity && !error && (
        // Initial state before any comparison is attempted or after clearing.
        // The isLoadingImages check is less relevant here if API is primary source.
        // imageDBForModel.length check is also less relevant for API functionality.
        // We always show the prompt to upload for API comparison.
        <div className="text-center py-10 animate-fadeIn">
            <svg className="mx-auto h-16 w-16 text-slate-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z" />
            </svg>
            <h3 className="mt-4 text-xl font-semibold text-slate-300">Iniciar Comparación por API</h3>
            <p className="mt-2 text-sm text-slate-400">
            Cargue una imagen utilizando el botón de arriba para encontrar imágenes similares (vía API)
            en la base de datos del modelo <span className="font-semibold text-sky-400">"{currentModelName}"</span>.
            </p>
            <p className="mt-1 text-sm text-slate-500">
                La API buscará en su propia base de datos. Las imágenes cargadas en 'Mantenimiento' son para referencia local.
            </p>
        </div>
      )}
    </Card>
  );
};

export default ImageExplorationConsultaView;
