import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Article } from './types'; // Changed from ArticleWithEmbedding
import { queryTextViaAPI } from './services'; // Using the new API service
import { LoadingSpinner, ErrorMessage, Card } from './uiElements';
import ArticleSidebar from './ArticleSidebar';

interface TextProcessingConsultaViewProps {
  articlesDB: Article[]; // Now receives Article[]
  isLoadingArticles: boolean; // Global loading state from App.tsx
}

const TextProcessingConsultaView: React.FC<TextProcessingConsultaViewProps> = ({
  articlesDB,
  isLoadingArticles
}) => {
  const [query, setQuery] = useState<string>('');
  // Search results will now include similarity from API
  const [searchResults, setSearchResults] = useState<(Article & { similarity: number })[]>([]);
  const [selectedArticleToDisplay, setSelectedArticleToDisplay] = useState<Article | null>(null);
  const [isLoadingSearch, setIsLoadingSearch] = useState<boolean>(false); // For API search loading
  const [error, setError] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(true); 

  const chatAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (window.innerWidth < 768) { 
        setIsSidebarOpen(false);
    }
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [searchResults, selectedArticleToDisplay]);

  const handleSearch = useCallback(async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!query.trim()) {
      setError("Por favor, introduce un término de búsqueda.");
      setSearchResults([]);
      return;
    }
    
    setIsLoadingSearch(true);
    setError(null);
    setSearchResults([]);
    setSelectedArticleToDisplay(null); 

    try {
      // Call the API service
      const resultsFromAPI = await queryTextViaAPI(query);
      setSearchResults(resultsFromAPI);
      if (resultsFromAPI.length === 0) {
          setError("No se encontraron artículos relevantes para tu consulta a través de la API.");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al realizar la búsqueda con la API.');
      setSearchResults([]); // Clear results on error
    } finally {
      setIsLoadingSearch(false);
    }
  }, [query]);

  const handleArticleSelectFromSidebar = (article: Article) => { // Takes Article
    setSelectedArticleToDisplay(article);
    setSearchResults([]); 
    setError(null);
    setQuery(''); 
    if (window.innerWidth < 768) { 
        setIsSidebarOpen(false);
    }
  };
  
  // isLoadingArticles is the global loading for the initial DB load
  if (isLoadingArticles && articlesDB.length === 0) { 
    return <LoadingSpinner />;
  }

  return (
    <div className="flex flex-col md:flex-row h-[calc(100vh-200px)] max-h-[800px] gap-4">
      <button 
        onClick={() => setIsSidebarOpen(!isSidebarOpen)} 
        className="md:hidden p-2 bg-slate-700 hover:bg-slate-600 text-white fixed top-28 left-2 z-50 rounded-full shadow-lg button-glow-sky"
        aria-label={isSidebarOpen ? "Cerrar panel de artículos" : "Abrir panel de artículos"}
        aria-expanded={isSidebarOpen}
      >
        {isSidebarOpen ? '‹' : '›'}
      </button>
      <ArticleSidebar
        articles={articlesDB} // Pass Article[]
        onArticleSelect={handleArticleSelectFromSidebar}
        isOpen={isSidebarOpen}
        setIsOpen={setIsSidebarOpen}
        selectedArticleId={selectedArticleToDisplay?.id}
      />
      <div className={`flex-1 flex flex-col bg-slate-800 rounded-3xl shadow-xl overflow-hidden transition-all duration-300 ease-in-out card-hover-glow`}>
        <div ref={chatAreaRef} className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4">
          {error && <ErrorMessage message={error} />}
          {isLoadingSearch && <LoadingSpinner />} {/* Spinner for API search */}

          {!isLoadingSearch && searchResults.length > 0 && (
            <div className="space-y-3">
              <p className="text-sky-400 font-semibold">Resultados de la búsqueda (API) para "{query}":</p>
              {searchResults.map(article => (
                <div key={article.id} className="bg-slate-700 p-3 rounded-2xl shadow-md animate-fadeIn card-hover-glow">
                  <h4 className="font-semibold text-sky-300 mb-1">{article.title}</h4>
                  <p className="text-xs text-sky-500 mb-1">
                    Similitud (API): {article.similarity !== undefined ? article.similarity.toFixed(4) : 'N/A'}
                  </p>
                  <p className="text-slate-300 text-sm whitespace-pre-wrap leading-relaxed line-clamp-3">
                    {article.content}
                  </p>
                   <button 
                        onClick={() => handleArticleSelectFromSidebar(article)} // Can select from search result to view full
                        className="text-xs text-sky-400 hover:text-sky-200 hover:underline mt-1 transition-colors"
                    >
                        Ver completo...
                    </button>
                </div>
              ))}
            </div>
          )}

          {!isLoadingSearch && selectedArticleToDisplay && (
            <Card className="bg-slate-700 animate-fadeIn !shadow-none hover:!transform-none">
              <h3 className="text-xl font-semibold text-sky-300 mb-2">{selectedArticleToDisplay.title}</h3>
              <div className="text-sm text-slate-300 whitespace-pre-wrap leading-relaxed max-h-[calc(100vh-400px)] overflow-y-auto pr-2">
                {selectedArticleToDisplay.content}
              </div>
            </Card>
          )}
          
          {!isLoadingSearch && searchResults.length === 0 && !selectedArticleToDisplay && !error && query && (
             <div className="text-center py-6">
                <p className="text-slate-400">No se encontraron artículos para "{query}" usando la API.</p>
            </div>
          )}

         {!isLoadingSearch && searchResults.length === 0 && !selectedArticleToDisplay && !error && !query && articlesDB.length > 0 && (
            <div className="text-center py-6 text-slate-400 animate-fadeIn">
                <p className="text-lg">Escribe una consulta para buscar artículos vía API</p>
                <p className="text-sm">o selecciona uno del panel lateral para vista local.</p>
            </div>
          )}
        </div>
        <form onSubmit={handleSearch} className="p-4 sm:p-6 border-t border-slate-700 bg-slate-800 rounded-b-3xl">
          <div className="flex items-center space-x-3">
            <input
              type="text"
              value={query}
              onChange={(e) => { setQuery(e.target.value); setError(null); }}
              placeholder="Escribe tu consulta aquí (API)..."
              className="flex-1 p-3 bg-slate-700 border border-slate-600 rounded-full text-slate-200 focus:ring-2 focus:ring-sky-500 focus:border-sky-500 transition-colors"
              aria-label="Texto de consulta para API"
            />
            <button
              type="submit"
              disabled={isLoadingSearch || !query.trim()} // Can search even if articlesDB is empty (API call)
              className="bg-sky-600 hover:bg-sky-700 text-white font-semibold py-3 px-6 rounded-full shadow-md transition duration-150 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed button-glow-sky"
            >
              Buscar (API)
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default TextProcessingConsultaView;