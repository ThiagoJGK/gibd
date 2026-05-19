import React, { useState } from 'react';
import { Download, ExternalLink } from 'lucide-react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import './Papers.css';

const CATEGORIES = ['Todos los Papers', 'Computer Vision', 'Machine Learning', 'Big Data'];

const MOCK_PAPERS = [
  {
    id: 1,
    title: 'Camera Ready - Mejorando la Identificación de Marcas de Ganado Vacuno',
    authors: ['JD', 'ML'],
    category: 'Computer Vision',
    date: 'Marzo 2024',
    abstract: 'Investigación profunda sobre el uso de redes neuronales convolucionales para la detección automática y clasificación de marcas de propiedad en ganado vacuno en entornos rurales de baja conectividad. A través de técnicas de Data Augmentation y aprendizaje métrico, el modelo logra generalizar a partir de una única imagen de referencia (One-Shot Learning).',
    imageUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuAUz5HeohjVnoyhAg6KncEFARkNJloipi4JUf1HJwj-xeJlZHeFhyTbLzgDXTihDPJ8LCIKm7igHKYi_DJTIb2GitRoy8ujo6YTyrsKDi56gC-VnlajHUggG91nJwNIWi_rdReCl-3Mp90fwwRrX_i0yZfupLvE9issAZFFLhZ-Ev-to5bt9tff8d0hYhP6Q3HmQpboIKHBz7s7wGTyHvfkvOCf2zleFEN3au-ovx6zZbE0W0Q35Wpgp_Smdpi_i8RC9AXNtCcwMtA',
    pdfUrl: '#'
  },
  {
    id: 2,
    title: 'Arquitecturas Distribuidas para Procesamiento de Datos Masivos en Tiempo Real',
    authors: ['AS'],
    category: 'Big Data',
    date: 'Enero 2024',
    abstract: 'Análisis comparativo de latencia y escalabilidad en arquitecturas Lambda vs Kappa aplicadas a flujos de datos satelitales para monitoreo de biomasa.',
    imageUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuA3ztg1AqmLHXOfoOI5nzC2Y7E2pnunBI2PnkbHr_0VKsly5ZM4gTCbQnSVg4jROhVrSY0Zc8L3gbcrWHpbVpFoGl8mNgT3YwXvJuhTBUL9TBGbcPFkmvvwZZ-MLTs-R9ymz43NRLYbz4F13IF4iEk_3b8bWqn6k0c1m-onMLDRsWikB-FUkoxxQTKYG3e7ELhzgkqcy3q38eOXtxSg8vo34I8CNsLv-QUy5E6wz37AMijDM3JY9LlQCRj1mqMnkaHlbupffCDrCRA',
    pdfUrl: '#'
  },
  {
    id: 3,
    title: 'Optimización de Cosecha Mediante Algoritmos Genéticos y Visión Artificial',
    authors: ['RT', 'PF'],
    category: 'Artificial Intelligence',
    date: 'Noviembre 2023',
    abstract: 'Propuesta de un sistema híbrido que utiliza visión multiespectral y algoritmos evolutivos para determinar el punto óptimo de madurez en cultivos extensivos.',
    imageUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuCDCTMSNRdmGwPoJJt3lakXo74CLtLoWQofTTLQAypwRfJwnuhOnYpvrkOKDxzI-WyBVFg6cBWUIWGUfWVuONPPAUSMHM7flamBAdG-qYqA3VAlT84kxFYILpPy4s8-ZZIkVr_7OxJPepwLCsFPx2hlEhGPh3xePYdS03z74moh7YQSupLkl2tHnD5j1W0Ro-jojhjxrOKfFK20aYsJCPtSVUG10fWoXX2Pn4AOBf-LadxDvJ6YAx6SCztarchv953DcGxCDpGAu7A',
    pdfUrl: '#'
  }
];

export function Papers() {
  const [activeCategory, setActiveCategory] = useState('Todos los Papers');

  return (
    <div className="papers-container fade-in-up">
      <div className="papers-header">
        <h1>
          Publicaciones e <span className="text-orange">Investigaciones</span> del GIBD
        </h1>
        <p>Explora los últimos avances en Big Data, Inteligencia Artificial y computación avanzada aplicada a la industria agropecuaria y tecnológica.</p>
      </div>

      <div className="papers-filter-row">
        {CATEGORIES.map(cat => (
          <button 
            key={cat}
            onClick={() => setActiveCategory(cat)}
            className={`filter-pill ${activeCategory === cat ? 'active' : ''}`}
          >
            {cat}
          </button>
        ))}
      </div>

      <div className="papers-list">
        {MOCK_PAPERS.map((paper) => (
          <article key={paper.id} className="research-card">
            <div className="research-img-wrapper">
              <img src={paper.imageUrl} alt={paper.title} className="research-img" />
            </div>
            
            <div className="research-content">
              <div className="research-meta">
                <span className="research-category badge-orange">{paper.category}</span>
                <span className="research-date">{paper.date}</span>
              </div>
              
              <h2 className="research-title">{paper.title}</h2>
              <p className="research-abstract">{paper.abstract}</p>
              
              <div className="research-footer">
                <div className="author-avatars">
                  {paper.authors.map((author, i) => (
                    <div key={i} className="avatar">{author}</div>
                  ))}
                </div>
                
                <div className="research-actions">
                  <Button size="default" variant="primary">
                    <Download size={18} className="mr-2" />
                    Leer Paper
                  </Button>
                </div>
              </div>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
