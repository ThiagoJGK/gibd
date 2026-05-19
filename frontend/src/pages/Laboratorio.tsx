import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, Image as ImageIcon, Sliders, Server } from 'lucide-react';
import { Button } from '../components/ui/Button';
import { PillTabs } from '../components/ui/PillTabs';
import { Card } from '../components/ui/Card';
import './Laboratorio.css';

const LAB_MODELS = [
  { id: 'ganado', label: 'Marcas de Ganado' },
  { id: 'ptah', label: 'PTAH-Jurídico' },
  { id: 'obc', label: 'OBC' },
  { id: 'sonido', label: 'Sonido' }
];

export function Laboratorio() {
  const [activeModel, setActiveModel] = useState('ganado');
  const [file, setFile] = useState<File | null>(null);
  const [topK, setTopK] = useState(10);
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<{url: string, score: number}[] | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      setResults(null); // Reset results on new upload
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg'] },
    multiple: false
  });

  const handleSearch = () => {
    if (!file) return;
    setIsSearching(true);
    // TODO: Connect with FastAPI Gateway
    // Mock API call for now
    setTimeout(() => {
      setIsSearching(false);
      setResults([
        { url: 'mock_result_1.jpg', score: 0.98 },
        { url: 'mock_result_2.jpg', score: 0.85 },
        { url: 'mock_result_3.jpg', score: 0.72 },
      ]);
    }, 2000);
  };

  return (
    <div className="lab-container fade-in-up">
      <div className="lab-header">
        <h1>Laboratorio GIBD</h1>
        <PillTabs tabs={LAB_MODELS} activeTab={activeModel} onChange={setActiveModel} />
      </div>

      <div className="lab-grid">
        {/* Main Interaction Area */}
        <div className="lab-main">
          {activeModel === 'ganado' && (
            <div className="ganado-module">
              <div 
                {...getRootProps()} 
                className={`dropzone pill-shape ${isDragActive ? 'drag-active' : ''} ${file ? 'has-file' : ''}`}
              >
                <input {...getInputProps()} />
                {file ? (
                  <div className="file-preview flex-center">
                    <ImageIcon size={48} color="var(--accent-vibrant-orange)" />
                    <p className="file-name">{file.name}</p>
                    <span className="change-file">Haz clic para cambiar imagen</span>
                  </div>
                ) : (
                  <div className="drop-prompt flex-center">
                    <UploadCloud size={64} color="var(--text-secondary)" />
                    <h3>Arrastra tu imagen aquí</h3>
                    <p>o haz clic para explorar tus archivos (PNG, JPG)</p>
                  </div>
                )}
              </div>

              <div className="action-area">
                <Button 
                  size="massive" 
                  variant="primary" 
                  onClick={handleSearch} 
                  disabled={!file || isSearching}
                  className={isSearching ? 'searching' : ''}
                >
                  {isSearching ? 'Procesando Red Siamesa...' : 'Buscar Similitudes (Red Siamesa)'}
                </Button>
              </div>

              {/* Results Preview */}
              {results && (
                <div className="results-area fade-in-up">
                  <h3>Resultados Top-K</h3>
                  <div className="results-grid">
                    {results.map((res, idx) => (
                      <div key={idx} className="result-card pill-shape">
                        <div className="result-img-placeholder flex-center">
                          IMG {idx + 1}
                        </div>
                        <div className="result-score">{(res.score * 100).toFixed(1)}%</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
          
          {activeModel !== 'ganado' && (
            <div className="module-placeholder flex-center pill-shape">
              <h2>Módulo {LAB_MODELS.find(m => m.id === activeModel)?.label} en desarrollo.</h2>
            </div>
          )}
        </div>

        {/* Sidebar Configuration */}
        <div className="lab-sidebar">
          <Card title="Configuración">
            <div className="config-group">
              <div className="config-header">
                <span>Top-K Similitudes</span>
                <span className="config-value">{topK}</span>
              </div>
              <input 
                type="range" 
                min="1" 
                max="50" 
                value={topK} 
                onChange={(e) => setTopK(Number(e.target.value))}
                className="pill-slider"
              />
              <p className="config-help">
                Ajusta el número de imágenes similares que el motor de Red Siamesa retornará.
              </p>
            </div>
          </Card>

          <Card className="server-status-card">
            <div className="server-status flex-center">
              <span className="status-dot online"></span>
              <strong>Servidor GIBD: Online</strong>
            </div>
            <p className="server-metrics">GPU A100 | Latencia 12ms | VRAM 82%</p>
          </Card>
        </div>
      </div>
    </div>
  );
}
