
import React, { useState, useCallback } from 'react';
import { Section, AppMode, UploadedImage, Article, ImageModelId } from './types';
import { TabButton, SwitchToggle } from './uiElements';
import MantenimientoView from './MantenimientoView';
import TextProcessingConsultaView from './TextProcessingConsultaView';
import ImageExplorationConsultaView from './ImageExplorationConsultaView';
import { processTextFileToArticles, imageFileToUploadedImage } from './services';
import Logo from './Logo'; 
import BigLogoDisplay from './BigLogoDisplay';
import HomePage from './HomePage'; // Importar HomePage


export const IMAGE_MODEL_CONFIG: Record<ImageModelId, { name: string }> = {
  [ImageModelId.Tatuajes]: { name: "Tatuajes" },
  [ImageModelId.MarcasDeGanado]: { name: "Marcas de Ganado" },
  [ImageModelId.Pintura]: { name: "Pinturas" },
  [ImageModelId.CaracteresChinos]: { name: "Caracteres Chinos" },
  [ImageModelId.EscudosDeFutbol]: { name: "Escudos de Fútbol" }
};

const App: React.FC = () => {
  const [currentSection, setCurrentSection] = useState<Section>(Section.Text);
  const [currentAppMode, setCurrentAppMode] = useState<AppMode>(AppMode.Consulta);
  const [showHomePage, setShowHomePage] = useState<boolean>(true); // Nuevo estado

  const [articlesDB, setArticlesDB] = useState<Article[]>([]); 
  
  const initialImageDB: Record<ImageModelId, UploadedImage[]> = 
    Object.values(ImageModelId).reduce((acc, modelId) => {
      acc[modelId] = [];
      return acc;
    }, {} as Record<ImageModelId, UploadedImage[]>);
  const [imageDB, setImageDB] = useState<Record<ImageModelId, UploadedImage[]>>(initialImageDB);
  
  const [currentImageModelId, setCurrentImageModelId] = useState<ImageModelId>(ImageModelId.Tatuajes);
  
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [comparisonImageForApp, setComparisonImageForApp] = useState<UploadedImage | null>(null);


  const handleLoadArticles = useCallback(async (file: File) => {
    setIsLoading(true);
    setError(null);
    try {
      const processedArticles = await processTextFileToArticles(file); 
      if (processedArticles.length === 0) {
        setError("El archivo seleccionado no contiene artículos o está vacío.");
        setArticlesDB([]);
      } else {
        setArticlesDB(processedArticles); 
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al procesar el archivo de texto.');
      setArticlesDB([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleLoadImagesForModel = useCallback(async (modelId: ImageModelId, files: File[]) => {
    setIsLoading(true);
    setError(null);
    try {
      const imagePromises = files
        .filter(file => file.type.startsWith('image/'))
        .map(file => imageFileToUploadedImage(file).catch(e => { 
          console.warn(`Omitiendo archivo ${file.name} para ${modelId}: ${e.message}`);
          return null;
        }));
      
      let loadedImages = (await Promise.all(imagePromises)).filter(img => img !== null) as UploadedImage[];
      
      const uniqueImagesMap = new Map<string, UploadedImage>();
      (imageDB[modelId] || []).forEach(img => uniqueImagesMap.set(img.name, img));
      loadedImages.forEach(img => {
        if (!uniqueImagesMap.has(img.name)) {
            uniqueImagesMap.set(img.name, { ...img, modelId }); 
        }
      });
      
      const newImageList = Array.from(uniqueImagesMap.values());

      setImageDB(prevDB => ({
        ...prevDB,
        [modelId]: newImageList
      }));

      if (newImageList.length === 0 && files.length > 0) {
        setError(`No se encontraron archivos de imagen válidos para ${IMAGE_MODEL_CONFIG[modelId].name}.`);
      } else if (files.length === 0) {
        setError(`No se seleccionaron archivos para ${IMAGE_MODEL_CONFIG[modelId].name}.`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : `Error al cargar imágenes para ${IMAGE_MODEL_CONFIG[modelId].name}.`);
    } finally {
      setIsLoading(false);
    }
  }, [imageDB]);

  // Función para navegar desde HomePage a la aplicación principal
  const handleNavigateFromHome = (targetSection: Section, targetMode: AppMode) => {
    setCurrentSection(targetSection);
    setCurrentAppMode(targetMode);
    setShowHomePage(false);
  };

  const Header: React.FC = () => ( // Header ya no necesita prop isOnHomePage
    <header className="bg-slate-800 shadow-lg sticky top-0 z-50">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col sm:flex-row justify-between items-center py-3 sm:py-4">
          <div className="h-10 sm:h-12"> 
            <Logo />
          </div>
          <div className="flex flex-col sm:flex-row items-center space-y-3 sm:space-y-0 sm:space-x-4 mt-3 sm:mt-0">
            {currentAppMode === AppMode.Consulta && (
              <nav className="flex space-x-2 sm:space-x-3">
                <TabButton
                  onClick={() => setCurrentSection(Section.Text)}
                  isActive={currentSection === Section.Text}
                  forceDisableGlow={currentSection === Section.Text && currentAppMode === AppMode.Consulta}
                >
                  <span role="img" aria-label="Text files icon" className="mr-1 sm:mr-2">📄</span> Texto
                </TabButton>
                <TabButton
                  onClick={() => setCurrentSection(Section.Image)}
                  isActive={currentSection === Section.Image}
                  forceDisableGlow={currentSection === Section.Image && currentAppMode === AppMode.Consulta}
                >
                  <span role="img" aria-label="Image files icon" className="mr-1 sm:mr-2">🖼️</span> Imagen
                </TabButton>
              </nav>
            )}
            <SwitchToggle
              labelLeft="Consulta"
              labelRight="Mantenimiento"
              isChecked={currentAppMode === AppMode.Mantenimiento}
              onChange={(isMantenimiento) => setCurrentAppMode(isMantenimiento ? AppMode.Mantenimiento : AppMode.Consulta)}
              id="app-mode-toggle"
            />
          </div>
        </div>
      </div>
    </header>
  );

  const Footer: React.FC = () => (
    <footer className="bg-slate-800 text-slate-400 text-center p-5 mt-12 shadow-up">
      <p>&copy; {new Date().getFullYear()} GIBD Local. Operando con archivos locales.</p>
    </footer>
  );

  const currentModelImages = imageDB[currentImageModelId] || [];
  const totalImagesInCurrentModel = currentModelImages.length;

  const renderConsultaView = () => {
    if (currentSection === Section.Text) {
      if (articlesDB.length === 0 && !isLoading) {
        return <BigLogoDisplay message="No hay base de datos de artículos cargada. Vaya a 'Mantenimiento' para cargar un archivo." />;
      }
      return (
        <TextProcessingConsultaView
          articlesDB={articlesDB}
          isLoadingArticles={isLoading && articlesDB.length === 0}
        />
      );
    }
    if (currentSection === Section.Image) {
      return (
        <ImageExplorationConsultaView 
          imageDBForModel={currentModelImages}
          isLoadingImages={isLoading && totalImagesInCurrentModel === 0}
          currentImageModelId={currentImageModelId}
          setCurrentImageModelId={setCurrentImageModelId}
          IMAGE_MODEL_CONFIG={IMAGE_MODEL_CONFIG}
           comparisonImageForApp={comparisonImageForApp} 
           setComparisonImageForApp={setComparisonImageForApp}
        />
      );
    }
    return null;
  };

  if (showHomePage) {
    return <HomePage onNavigateToApp={handleNavigateFromHome} />;
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 flex flex-col">
      <Header />
      <main className="flex-grow container mx-auto px-2 sm:px-4 py-6 sm:py-8">
        {currentAppMode === AppMode.Mantenimiento && (
          <MantenimientoView
            onLoadArticles={handleLoadArticles}
            onLoadImagesForModel={handleLoadImagesForModel}
            isLoading={isLoading}
            error={error}
            articlesCount={articlesDB.length}
            imageDB={imageDB}
            clearError={() => setError(null)}
            IMAGE_MODEL_CONFIG={IMAGE_MODEL_CONFIG}
          />
        )}
        {currentAppMode === AppMode.Consulta && renderConsultaView()}
      </main>
      <Footer />
    </div>
  );
};

export default App;
