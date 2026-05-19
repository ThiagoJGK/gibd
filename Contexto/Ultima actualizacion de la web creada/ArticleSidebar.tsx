import React, { useState } from 'react';
import { Article } from './types'; // Changed from ArticleWithEmbedding

interface ArticleSidebarProps {
  articles: Article[]; // Now accepts Article[]
  onArticleSelect: (article: Article) => void; // Callback takes Article
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
  selectedArticleId?: string | null;
}

const ArticleSidebar: React.FC<ArticleSidebarProps> = ({
  articles,
  onArticleSelect,
  isOpen,
  setIsOpen,
  selectedArticleId,
}) => {
  const [isListContentVisible, setIsListContentVisible] = useState(true);

  if (!isOpen && window.innerWidth < 768) { 
    return null;
  }
  
  const chevronIcon = isListContentVisible ? (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
      <path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
    </svg>
  ) : (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
      <path strokeLinecap="round" strokeLinejoin="round" d="m8.25 4.5 7.5 7.5-7.5 7.5" />
    </svg>
  );


  return (
    <aside 
      className={`fixed inset-y-0 left-0 z-40 md:sticky md:top-[calc(theme(spacing.16)+1px)] md:self-start md:max-h-[calc(100vh-theme(spacing.32)-2px)] 
                  transform ${isOpen ? 'translate-x-0' : '-translate-x-full'} 
                  md:translate-x-0 transition-transform duration-300 ease-in-out 
                  w-64 md:w-72 lg:w-80 bg-slate-800 shadow-lg md:rounded-3xl overflow-hidden flex flex-col card-hover-glow`}
      aria-label="Barra lateral de artículos"
    >
      <div className="p-4 flex justify-between items-center border-b border-slate-700 md:sticky md:top-0 bg-slate-800 z-10">
        <div 
            className="flex items-center cursor-pointer group"
            onClick={() => setIsListContentVisible(!isListContentVisible)}
            aria-expanded={isListContentVisible}
            aria-controls="article-list-content"
        >
            <h3 className="text-lg font-semibold text-sky-400 group-hover:text-sky-300 transition-colors">Artículos Cargados</h3>
            <span className="ml-2 text-sky-400 group-hover:text-sky-300 transition-colors">{chevronIcon}</span>
        </div>
        <button 
            onClick={() => setIsOpen(false)} 
            className="md:hidden text-slate-400 hover:text-white text-2xl"
            aria-label="Cerrar barra lateral"
        >
            &times;
        </button>
      </div>
      
      {isListContentVisible && (
        <div id="article-list-content" className="flex-1 overflow-y-auto">
          {articles.length === 0 ? (
            <p className="text-slate-400 px-4 py-2 text-sm">No hay artículos cargados.</p>
          ) : (
            <nav className="px-2 py-2 space-y-1">
              {articles.map((article) => (
                <button
                  key={article.id}
                  onClick={() => onArticleSelect(article)}
                  className={`w-full text-left px-3 py-2.5 rounded-xl text-sm transition-all duration-200 ease-in-out 
                              focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-opacity-75
                              ${selectedArticleId === article.id
                                ? 'bg-sky-600 text-white shadow-md selected-item-glow scale-105'
                                : 'text-slate-300 hover:bg-slate-700 hover:text-sky-300 hover:shadow-md hover:scale-[1.02] button-glow-sky' 
                              }`}
                  aria-current={selectedArticleId === article.id ? "page" : undefined}
                >
                  <span className="block truncate">{article.title}</span>
                </button>
              ))}
            </nav>
          )}
        </div>
      )}
       {!isListContentVisible && articles.length > 0 && (
        <p className="text-slate-500 px-4 py-2 text-xs italic text-center">Lista de artículos oculta.</p>
      )}
    </aside>
  );
};

export default ArticleSidebar;