import { useState } from 'react';

const CATEGORIES = ['Todos los Papers', 'Computer Vision', 'Machine Learning', 'Big Data'];

const PAPERS = [
  {
    id: 1,
    title: 'Camera Ready - Mejorando la Identificación de Marcas de Ganado Vacuno',
    description: 'Investigación profunda sobre el uso de redes neuronales convolucionales para la detección automática y clasificación de marcas de propiedad en ganado vacuno en entornos rurales de baja conectividad.',
    category: 'Computer Vision',
    date: 'Marzo 2024',
    authors: ['JD', 'ML'],
    image: 'https://lh3.googleusercontent.com/aida-public/AB6AXuAUz5HeohjVnoyhAg6KncEFARkNJloipi4JUf1HJwj-xeJlZHeFhyTbLzgDXTihDPJ8LCIKm7igHKYi_DJTIb2GitRoy8ujo6YTyrsKDi56gC-VnlajHUggG91nJwNIWi_rdReCl-3Mp90fwwRrX_i0yZfupLvE9issAZFFLhZ-Ev-to5bt9tff8d0hYhP6Q3HmQpboIKHBz7s7wGTyHvfkvOCf2zleFEN3au-ovx6zZbE0W0Q35Wpgp_Smdpi_i8RC9AXNtCcwMtA'
  },
  {
    id: 2,
    title: 'Arquitecturas Distribuidas para Procesamiento de Datos Masivos en Tiempo Real',
    description: 'Análisis comparativo de latencia y escalabilidad en arquitecturas Lambda vs Kappa aplicadas a flujos de datos satelitales para monitoreo de biomasa.',
    category: 'Big Data',
    date: 'Enero 2024',
    authors: ['AS'],
    image: 'https://lh3.googleusercontent.com/aida-public/AB6AXuA3ztg1AqmLHXOfoOI5nzC2Y7E2pnunBI2PnkbHr_0VKsly5ZM4gTCbQnSVg4jROhVrSY0Zc8L3gbcrWHpbVpFoGl8mNgT3YwXvJuhTBUL9TBGbcPFkmvvwZZ-MLTs-R9ymz43NRLYbz4F13IF4iEk_3b8bWqn6k0c1m-onMLDRsWikB-FUkoxxQTKYG3e7ELhzgkqcy3q38eOXtxSg8vo34I8CNsLv-QUy5E6wz37AMijDM3JY9LlQCRj1mqMnkaHlbupffCDrCRA'
  },
  {
    id: 3,
    title: 'Optimización de Cosecha Mediante Algoritmos Genéticos y Visión Artificial',
    description: 'Propuesta de un sistema híbrido que utiliza visión multiespectral y algoritmos evolutivos para determinar el punto óptimo de madurez en cultivos extensivos.',
    category: 'Artificial Intelligence',
    date: 'Noviembre 2023',
    authors: ['RT', 'PF'],
    image: 'https://lh3.googleusercontent.com/aida-public/AB6AXuCDCTMSNRdmGwPoJJt3lakXo74CLtLoWQofTTLQAypwRfJwnuhOnYpvrkOKDxzI-WyBVFg6cBWUIWGUfWVuONPPAUSMHM7flamBAdG-qYqA3VAlT84kxFYILpPy4s8-ZZIkVr_7OxJPepwLCsFPx2hlEhGPh3xePYdS03z74moh7YQSupLkl2tHnD5j1W0Ro-jojhjxrOKfFK20aYsJCPtSVUG10fWoXX2Pn4AOBf-LadxDvJ6YAx6SCztarchv953DcGxCDpGAu7A'
  }
];

export function Papers() {
  const [activeCategory, setActiveCategory] = useState(CATEGORIES[0]);

  return (
    <div className="pt-32 pb-32 px-6 max-w-7xl mx-auto min-h-screen">
      {/* Hero Section */}
      <section className="mb-12">
        <h1 className="text-4xl md:text-[56px] lg:text-[80px] font-black leading-tight mb-6">
          Publicaciones e <span className="text-primary-container">Investigaciones</span> del GIBD
        </h1>
        <p className="text-lg md:text-xl text-text-secondary max-w-3xl leading-relaxed">
          Explora los últimos avances en Big Data, Inteligencia Artificial y computación avanzada aplicada a la industria agropecuaria y tecnológica.
        </p>
      </section>

      {/* Filter Pills */}
      <div className="flex gap-4 overflow-x-auto pb-6 no-scrollbar mb-8">
        {CATEGORIES.map(category => (
          <button
            key={category}
            onClick={() => setActiveCategory(category)}
            className={`shrink-0 whitespace-nowrap px-8 py-3 rounded-full font-semibold transition-all ripple border ${
              activeCategory === category 
                ? 'bg-primary-container text-on-primary-container border-primary-container' 
                : 'bg-secondary-container text-text-primary border-border-organic hover:bg-surface-deep'
            }`}
          >
            {category}
          </button>
        ))}
      </div>

      {/* Cards List */}
      <div className="grid grid-cols-1 gap-8 w-full">
        {PAPERS.map((paper) => (
          <article key={paper.id} className="bg-surface-deep rounded-[2rem] overflow-hidden flex flex-col md:flex-row border border-border-organic transition-all hover:scale-[1.01] duration-300 group">
            
            {/* Image Section */}
            <div className="w-full md:w-1/3 min-h-[240px] relative overflow-hidden shrink-0">
              <img 
                src={paper.image} 
                alt={paper.title} 
                className="absolute inset-0 w-full h-full object-cover grayscale opacity-80 group-hover:grayscale-0 group-hover:opacity-100 transition-all duration-500"
                loading="lazy"
              />
            </div>
            
            {/* Content Section */}
            <div className="p-8 md:p-10 flex-1 flex flex-col justify-between">
              <div>
                <div className="flex flex-wrap items-center gap-3 mb-6">
                  <span className="bg-primary-container/10 text-primary-container px-4 py-1.5 rounded-full text-xs font-black uppercase tracking-widest border border-primary-container/20">
                    {paper.category}
                  </span>
                  <span className="text-text-secondary text-sm font-semibold">{paper.date}</span>
                </div>
                <h2 className="text-2xl md:text-[32px] font-bold text-text-primary mb-4 leading-snug">
                  {paper.title}
                </h2>
                <p className="text-text-secondary leading-relaxed mb-8">
                  {paper.description}
                </p>
              </div>
              
              <div className="flex items-center justify-between mt-auto">
                {/* Author Avatars */}
                <div className="flex -space-x-3">
                  {paper.authors.map((author, i) => (
                    <div key={i} className="w-12 h-12 rounded-full border-4 border-surface-deep bg-background-base flex items-center justify-center text-sm font-bold text-text-primary">
                      {author}
                    </div>
                  ))}
                </div>
                <button className="bg-primary-container text-on-primary-container px-8 py-3 rounded-full font-bold text-sm hover:scale-105 active:scale-95 transition-all ripple">
                  Leer Paper
                </button>
              </div>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
