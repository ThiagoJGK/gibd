import React, { useState } from 'react';
import { Search, FileText } from 'lucide-react';
import { Button } from '../components/ui/Button';
import './Articulos.css';

// Mocks
const MOCK_DOCS = [
  { id: 1, title: 'Documento: 8 Articulo: 3', type: 'Norma General' },
  { id: 2, title: 'Documento: 12 Articulo: 1', type: 'Resolución' },
  { id: 3, title: 'Documento: 45 Articulo: 9', type: 'Jurisprudencia' },
  { id: 4, title: 'Documento: 2 Articulo: 14', type: 'Norma General' },
  { id: 5, title: 'Documento: 19 Articulo: 2', type: 'Decreto' },
];

export function Articulos() {
  const [query, setQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [activeDoc, setActiveDoc] = useState<number | null>(null);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    setIsSearching(true);
    // TODO: Connect with PTAH-Jurídico FastAPI endpoint
    setTimeout(() => {
      setIsSearching(false);
      // Simulate selecting a document as result
      setActiveDoc(3); 
    }, 1500);
  };

  return (
    <div className="articulos-layout fade-in-up">
      {/* Sidebar - Loaded Articles */}
      <aside className="articulos-sidebar">
        <div className="sidebar-header">
          <h2>Artículos Cargados</h2>
          <span className="doc-count">{MOCK_DOCS.length} Documentos</span>
        </div>
        
        <div className="doc-list">
          {MOCK_DOCS.map(doc => (
            <div 
              key={doc.id} 
              className={`doc-item pill-shape ${activeDoc === doc.id ? 'active' : ''}`}
              onClick={() => setActiveDoc(doc.id)}
            >
              <FileText size={18} className="doc-icon" />
              <div className="doc-info">
                <span className="doc-title">{doc.title}</span>
                <span className="doc-type">{doc.type}</span>
              </div>
            </div>
          ))}
        </div>
      </aside>

      {/* Main Area - API Search */}
      <main className="articulos-main">
        <div className="main-content-wrapper">
          
          {/* Empty State / Content */}
          <div className="content-display flex-center">
            {activeDoc ? (
              <div className="doc-detail fade-in-up">
                <h2>{MOCK_DOCS.find(d => d.id === activeDoc)?.title}</h2>
                <span className="badge text-orange">{MOCK_DOCS.find(d => d.id === activeDoc)?.type}</span>
                <div className="fake-text-skeleton">
                  <div className="skeleton-line" style={{width: '100%'}}></div>
                  <div className="skeleton-line" style={{width: '90%'}}></div>
                  <div className="skeleton-line" style={{width: '95%'}}></div>
                  <div className="skeleton-line" style={{width: '80%'}}></div>
                  <div className="skeleton-line" style={{width: '85%'}}></div>
                </div>
              </div>
            ) : (
              <h2 className="empty-state-text">
                Escribe una consulta para buscar artículos vía API o selecciona uno del panel lateral.
              </h2>
            )}
          </div>

          {/* Search Bar at bottom */}
          <form className="search-bar-container" onSubmit={handleSearch}>
            <div className="input-wrapper pill-shape">
              <Search size={24} className="search-icon" />
              <input 
                type="text" 
                placeholder="Escribe tu consulta aquí (PTAH-Jurídico API)..." 
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="search-input"
              />
            </div>
            <Button size="lg" variant="primary" type="submit" disabled={isSearching || !query.trim()}>
              {isSearching ? 'Buscando...' : 'Buscar (API)'}
            </Button>
          </form>
          
        </div>
      </main>
    </div>
  );
}
