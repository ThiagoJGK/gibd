import React from 'react';
import { Download, ExternalLink } from 'lucide-react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import './Papers.css';

// Mock data (Will be replaced by Supabase + Gemini workflow)
const MOCK_PAPERS = [
  {
    id: 1,
    title: 'Mejorando la Identificación de Marcas de Ganado Vacuno',
    authors: 'GIBD Research Team',
    year: 2023,
    event: 'CACIC 2023',
    abstract: 'Este documento presenta una optimización radical en el reconocimiento de marcas de ganado utilizando Redes Neuronales Siamesas. A través de técnicas de Data Augmentation y aprendizaje métrico, el modelo logra generalizar a partir de una única imagen de referencia (One-Shot Learning), superando los desafíos de oclusión y baja resolución comunes en entornos rurales.',
    pdfUrl: '#'
  },
  {
    id: 2,
    title: 'Construcción de una Función de Distancia para Consultar por Similitud Caracteres de Hueso de Oráculo',
    authors: 'GIBD Research Team',
    year: 2024,
    event: 'CoNaIISI 2024',
    abstract: 'Investigación orientada a la preservación del patrimonio histórico. Se detalla el desarrollo de una arquitectura de Machine Learning especializada en extraer características topológicas de inscripciones antiguas (OBC), permitiendo agrupar glifos por similitud morfológica con un 92% de accuracy en el dataset de validación.',
    pdfUrl: '#'
  }
];

export function Papers() {
  return (
    <div className="papers-container fade-in-up">
      <div className="papers-header">
        <h1>Repositorio Científico</h1>
        <p>Acceso abierto a todas las publicaciones y papers de investigación desarrollados por el GIBD.</p>
      </div>

      <div className="papers-grid">
        {MOCK_PAPERS.map((paper) => (
          <Card key={paper.id} className="paper-card">
            <div className="paper-meta">
              <span className="paper-event">{paper.event}</span>
              <span className="paper-year">{paper.year}</span>
            </div>
            
            <h2 className="paper-title">{paper.title}</h2>
            <p className="paper-authors">Autores: {paper.authors}</p>
            
            <div className="paper-abstract">
              <strong>Abstract (Generado por Gemini AI):</strong>
              <p>{paper.abstract}</p>
            </div>
            
            <div className="paper-actions">
              <Button size="default" variant="primary">
                <Download size={18} className="mr-2" />
                Descargar PDF
              </Button>
              <Button size="default" variant="ghost">
                <ExternalLink size={18} className="mr-2" />
                Cita
              </Button>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
