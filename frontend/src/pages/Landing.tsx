import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Beaker, ChevronRight } from 'lucide-react';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import './Landing.css';

export function Landing() {
  const navigate = useNavigate();

  return (
    <div className="landing-container fade-in-up">
      {/* Hero Section */}
      <section className="hero-section">
        <h1 className="hero-title">
          Investigación Aplicada en <span className="text-orange">Big Data</span>
        </h1>
        <p className="hero-subtitle">
          Grupo de Investigación en Big Data (GIBD) - UTN FRCU. Desarrollando soluciones avanzadas en Inteligencia Artificial, Redes Siamesas y Procesamiento de Lenguaje Natural.
        </p>
        <div className="hero-actions">
          <Button size="massive" variant="primary" onClick={() => navigate('/')}>
            <Beaker size={24} className="mr-2" />
            Acceder al Laboratorio de IA
          </Button>
          <Button size="massive" variant="ghost" onClick={() => navigate('/papers')}>
            Ver Publicaciones <ChevronRight size={24} />
          </Button>
        </div>
      </section>

      {/* Stats/Highlight Section */}
      <section className="stats-section">
        <div className="stat-card pill-shape flat-panel">
          <h3>+25</h3>
          <p>Papers Publicados</p>
        </div>
        <div className="stat-card pill-shape flat-panel">
          <h3>4</h3>
          <p>Modelos de IA Disponibles</p>
        </div>
        <div className="stat-card pill-shape flat-panel">
          <h3>100%</h3>
          <p>Open Access</p>
        </div>
      </section>

      {/* Misión y Noticias Recientes */}
      <section className="content-grid">
        <div className="mission-column">
          <Card title="Nuestra Misión" className="mission-card">
            <p>
              El GIBD se enfoca en resolver problemas reales mediante tecnología de vanguardia. Desde la identificación automatizada de marcas de ganado hasta la indexación semántica de textos jurídicos, nuestro objetivo es democratizar la Inteligencia Artificial.
            </p>
          </Card>
        </div>
        
        <div className="news-column">
          <h2 className="section-title">Últimas Novedades</h2>
          <div className="news-list">
            <Card className="news-item">
              <span className="news-date">Octubre 2026</span>
              <h4>Nuevo modelo de Redes Siamesas desplegado</h4>
              <p>Optimizamos el procesamiento de imágenes para Marcas de Ganado alcanzando un 98% de precisión en One-Shot Learning.</p>
            </Card>
            <Card className="news-item">
              <span className="news-date">Septiembre 2026</span>
              <h4>Avances en PTAH-Jurídico</h4>
              <p>Presentamos la nueva versión del motor de búsqueda semántica para normativas y fallos legales.</p>
            </Card>
          </div>
        </div>
      </section>
    </div>
  );
}
