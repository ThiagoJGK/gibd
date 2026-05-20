import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Mail, ExternalLink, ChevronRight, ArrowLeft, BookOpen, ArrowUpRight, Users, FileText } from 'lucide-react';

const CATEGORIES = [
  'Todos los Papers',
  'Procesamiento de Lenguaje Natural',
  'Búsqueda por Similitud',
  'Gestión de Proyectos'
];

const AUTHORS_DATABASE: Record<string, { name: string; role: string; email: string; linkedin: string }> = {
  AP: {
    name: 'Mg. Andrés Jorge Pascal',
    role: 'Docente Investigador',
    email: 'pascala@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/andres-pascal-gibd'
  },
  PC: {
    name: 'Dra. Patricia R. Cristaldo',
    role: 'Docente Investigadora',
    email: 'cristaldop@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/patricia-cristaldo'
  },
  DL: {
    name: 'Dra. María Daniela López De Luise',
    role: 'Docente Investigadora',
    email: 'deluisem@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/daniela-lopez-de-luise'
  },
  TG: {
    name: 'Thiago Gomez Kehler',
    role: 'Investigador',
    email: 'thiagogomezkehler@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/thiago-gomez-kehler'
  },
  MO: {
    name: 'Maximiliano Olivera',
    role: 'Investigador',
    email: 'oliveram@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/maximiliano-olivera'
  },
  PS: {
    name: 'Pablo Suarez Lapalma',
    role: 'Investigador',
    email: 'suarezp@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/pablo-suarez-lapalma'
  },
  IM: {
    name: 'Iara Martinelli',
    role: 'Investigadora',
    email: 'martinellii@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/iara-martinelli'
  },
  MF: {
    name: 'María Emilia Fernandez',
    role: 'Investigadora',
    email: 'fernandezm@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/maria-emilia-fernandez'
  },
  LD: {
    name: 'Luciano Emmanuel Davezac',
    role: 'Investigador',
    email: 'davezacl@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/luciano-davezac'
  },
  ST: {
    name: 'Sebastián Trossero',
    role: 'Investigador',
    email: 'trosseros@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/sebastian-trossero'
  },
  CA: {
    name: 'Claudia M. Álvarez',
    role: 'Investigadora',
    email: 'alvarezc@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/claudia-m-alvarez'
  },
  FH: {
    name: 'Fernando Heit',
    role: 'Investigador',
    email: 'heitf@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/fernando-heit'
  },
  NP: {
    name: 'Adrián Nicolas Planas',
    role: 'Docente Investigador',
    email: 'planasn@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/adrian-nicolas-planas'
  },
  FV: {
    name: 'Florencia Zoe Vidal',
    role: 'Investigadora',
    email: 'vidalf@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/florencia-zoe-vidal'
  },
  AB: {
    name: 'Agustina Bonti',
    role: 'Investigadora',
    email: 'bontia@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/agustina-bonti'
  },
  LT: {
    name: 'Lucas Francisco Tonelotto',
    role: 'Investigador',
    email: 'tonelottol@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/lucas-tonelotto'
  },
  LC: {
    name: 'León Castiglioni',
    role: 'Investigador',
    email: 'castiglionil@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/leon-castiglioni'
  },
  FL: {
    name: 'Federico Lederhos',
    role: 'Investigador',
    email: 'lederhosf@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/federico-lederhos'
  },
  WC: {
    name: 'Wenceslao Colazo',
    role: 'Investigador',
    email: 'colazow@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/wenceslao-colazo'
  },
  SP: {
    name: 'Santiago Poerio Val',
    role: 'Investigador',
    email: 'poerios@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/santiago-poerio'
  },
  FS: {
    name: 'Federico Stauber',
    role: 'Docente Investigador',
    email: 'stauberf@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/federico-stauber'
  },
  LV: {
    name: 'Luciana G. Valiente',
    role: 'Investigadora',
    email: 'valientel@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/luciana-valiente'
  },
  LP: {
    name: 'Lucas La Pietra',
    role: 'Investigador',
    email: 'lapietral@frcu.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/lucas-la-pietra'
  },
  JH: {
    name: 'Dr. Jude Hemanth',
    role: 'Investigador Externo',
    email: 'judehemanth@external.utn.edu.ar',
    linkedin: 'https://linkedin.com/in/jude-hemanth'
  }
};

const PAPERS = [
  {
    id: 1,
    title: 'Recuperación de Información de Reglamentación Académica en Español utilizando Modelos del Lenguaje Natural',
    description: 'Investigación sobre el diseño y la sintonización de modelos de lenguaje natural (NLP) aplicados a la consulta y recuperación semántica de reglamentaciones internas en la UTN FRCU.',
    category: 'Procesamiento de Lenguaje Natural',
    date: 'Noviembre 2024 (CoNaIISI 2024)',
    authors: ['AP', 'MO', 'PS', 'IM', 'MF', 'LD', 'TG'],
    image: 'https://images.unsplash.com/photo-1546410531-bb4caa6b424d?auto=format&fit=crop&w=600&q=80',
    url: 'https://frcu.utn.edu.ar/index.php/investigacion-gibd/publicaciones-gibd'
  },
  {
    id: 2,
    title: 'Advanced Variable Tuning and Biases in Chatbot Models: Analysis of the PTAH Prototype',
    description: 'Análisis profundo de la sintonización de parámetros y mitigación de sesgos en modelos de lenguaje conversacionales aplicados al prototipo de agente de IA legal PTAH.',
    category: 'Procesamiento de Lenguaje Natural',
    date: 'Octubre 2024 (IEEE ARGENCON 2024)',
    authors: ['DL', 'AP', 'ST', 'CA', 'FH'],
    image: 'https://lh3.googleusercontent.com/aida-public/AB6AXuAUz5HeohjVnoyhAg6KncEFARkNJloipi4JUf1HJwj-xeJlZHeFhyTbLzgDXTihDPJ8LCIKm7igHKYi_DJTIb2GitRoy8ujo6YTyrsKDi56gC-VnlajHUggG91nJwNIWi_rdReCl-3Mp90fwwRrX_i0yZfupLvE9issAZFFLhZ-Ev-to5bt9tff8d0hYhP6Q3HmQpboIKHBz7s7wGTyHvfkvOCf2zleFEN3au-ovx6zZbE0W0Q35Wpgp_Smdpi_i8RC9AXNtCcwMtA',
    url: 'https://ieeexplore.ieee.org/document/10705886'
  },
  {
    id: 3,
    title: 'Image Feature Extraction for Similarity Searching Using Transfer Learning with ResNet',
    description: 'Estudio sobre la extracción de descriptores visuales de alta fidelidad mediante transferencia de aprendizaje (ResNet) para optimizar consultas por similitud métrica.',
    category: 'Búsqueda por Similitud',
    date: 'Octubre 2024 (CACIC 2024)',
    authors: ['AP', 'NP', 'FV', 'AB', 'LT', 'LC'],
    image: 'https://images.unsplash.com/photo-1507668077129-56e32842fceb?auto=format&fit=crop&w=600&q=80',
    url: 'http://sedici.unlp.edu.ar/handle/10915/169192'
  },
  {
    id: 4,
    title: 'Mejorando la Identificación de Marcas de Ganado Vacuno: Redes Siamesas en el Aprendizaje de Funciones de Distancia',
    description: 'Propuesta para optimizar el reconocimiento automático de marcas de propiedad de ganado vacuno en el sector agropecuario mediante aprendizaje de métricas profundas con redes siamesas.',
    category: 'Búsqueda por Similitud',
    date: 'Octubre 2023 (CACIC 2023)',
    authors: ['FS', 'NP', 'AP'],
    image: 'https://lh3.googleusercontent.com/aida-public/AB6AXuCDCTMSNRdmGwPoJJt3lakXo74CLtLoWQofTTLQAypwRfJwnuhOnYpvrkOKDxzI-WyBVFg6cBWUIWGUfWVuONPPAUSMHM7flamBAdG-qYqA3VAlT84kxFYILpPy4s8-ZZIkVr_7OxJPepwLCsFPx2hlEhGPh3xePYdS03z74moh7YQSupLkl2tHnD5j1W0Ro-jojhjxrOKfFK20aYsJCPtSVUG10fWoXX2Pn4AOBf-LadxDvJ6YAx6SCztarchv953DcGxCDpGAu7A',
    url: 'http://sedici.unlp.edu.ar/handle/10915/161407'
  },
  {
    id: 5,
    title: 'Búsqueda por Similitud de Tatuajes Utilizando Preprocesamiento y Transfer Learning',
    description: 'Desarrollo de un pipeline de preprocesamiento de imágenes combinado con redes convolucionales preentrenadas para la recuperación eficiente de tatuajes forenses.',
    category: 'Búsqueda por Similitud',
    date: 'Noviembre 2024 (CoNaIISI 2024)',
    authors: ['AP', 'NP', 'FL', 'LC', 'WC', 'SP'],
    image: 'https://lh3.googleusercontent.com/aida-public/AB6AXuA3ztg1AqmLHXOfoOI5nzC2Y7E2pnunBI2PnkbHr_0VKsly5ZM4gTCbQnSVg4jROhVrSY0Zc8L3gbcrWHpbVpFoGl8mNgT3YwXvJuhTBUL9TBGbcPFkmvvwZZ-MLTs-R9ymz43NRLYbz4F13IF4iEk_3b8bWqn6k0c1m-onMLDRsWikB-FUkoxxQTKYG3e7ELhzgkqcy3q38eOXtxSg8vo34I8CNsLv-QUy5E6wz37AMijDM3JY9LlQCRj1mqMnkaHlbupffCDrCRA',
    url: 'https://frcu.utn.edu.ar/index.php/investigacion-gibd/publicaciones-gibd'
  },
  {
    id: 6,
    title: 'Experiencia Ludificada para el desarrollo de Métricas en Gestión de Proyectos',
    description: 'Presentación del modelo y prototipo LudgePI, integrando dinámicas lúdicas de gamificación para facilitar la recopilación y análisis transversal de métricas de calidad en proyectos.',
    category: 'Gestión de Proyectos',
    date: 'Noviembre 2023 (CoNaIISI 2023)',
    authors: ['PC', 'DL', 'LT', 'LV'],
    image: 'https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=600&q=80',
    url: 'https://frcu.utn.edu.ar/index.php/investigacion-gibd/publicaciones-gibd'
  },
  {
    id: 7,
    title: 'Metrics for the Systematic Evaluation of Software Project Management Methodologies',
    description: 'Definición de un marco de métricas transversales basado en minería de datos para evaluar sistemáticamente la alineación y desempeño de metodologías híbridas y ágiles.',
    category: 'Gestión de Proyectos',
    date: '2021 (Global Research and Development Journal)',
    authors: ['PC', 'DL', 'LP', 'AB', 'JH'],
    image: 'https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?auto=format&fit=crop&w=600&q=80',
    url: 'https://www.grdjournals.com/article?paper=GRDJEV06I050009'
  }
];

export function Papers() {
  const [activeCategory, setActiveCategory] = useState(CATEGORIES[0]);
  const [expandedCardId, setExpandedCardId] = useState<number | null>(null);
  const [expandedAuthorKey, setExpandedAuthorKey] = useState<string | null>(null);

  const filteredPapers = activeCategory === 'Todos los Papers'
    ? PAPERS
    : PAPERS.filter(paper => paper.category === activeCategory);

  return (
    <div className="pt-32 pb-32 px-6 max-w-7xl mx-auto min-h-screen">
      {/* Hero Section */}
      <section className="mb-12">
        <h1 className="text-4xl md:text-[56px] lg:text-[80px] font-black leading-tight mb-6">
          Publicaciones e <span className="text-primary-container">Investigaciones</span> del GIBD
        </h1>
        <p className="text-lg md:text-xl text-text-secondary max-w-3xl leading-relaxed">
          Explora las publicaciones científicas oficiales del GIBD en búsqueda por similitud, procesamiento de lenguaje natural y métricas aplicadas a la gestión de proyectos de sistemas.
        </p>
      </section>

      {/* Filter Pills */}
      <div className="flex gap-4 overflow-x-auto pb-6 no-scrollbar mb-8">
        {CATEGORIES.map(category => (
          <button
            key={category}
            onClick={() => {
              setActiveCategory(category);
              setExpandedCardId(null); // Collapse any open card when switching categories
              setExpandedAuthorKey(null);
            }}
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
        {filteredPapers.map((paper) => {
          const isExpanded = expandedCardId === paper.id;

          return (
            <motion.article 
              key={paper.id} 
              layout="position"
              onClick={() => {
                if (!isExpanded) {
                  setExpandedCardId(paper.id);
                  setExpandedAuthorKey(null);
                }
              }}
              className={`card-glass-purple rounded-[2rem] overflow-hidden flex flex-col md:flex-row border transition-all duration-300 group ${
                isExpanded 
                  ? 'border-primary-container/50 shadow-[0_15px_45px_rgba(255,85,0,0.12)] cursor-default scale-[1.005]' 
                  : 'hover:scale-[1.01] hover:border-primary-container/40 hover:shadow-[0_12px_40px_rgba(255,85,0,0.08)] cursor-pointer'
              }`}
            >
              
              {/* Image Section - Completely hidden on mobile when expanded, compact sidebar on desktop */}
              <div className={`relative overflow-hidden shrink-0 transition-all duration-500 ${
                isExpanded 
                  ? 'hidden md:block md:w-[240px] min-h-[300px]' 
                  : 'w-full md:w-80 h-52 md:h-auto min-h-[200px]'
              }`}>
                <img 
                  src={paper.image} 
                  alt={paper.title} 
                  className="absolute inset-0 w-full h-full object-cover grayscale opacity-70 group-hover:grayscale-0 group-hover:opacity-100 transition-all duration-500"
                  loading="lazy"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-background-base via-transparent to-transparent opacity-85" />
                
                {/* Floating category tag on mobile & collapsed image */}
                {!isExpanded && (
                  <span className="absolute top-4 left-4 bg-background-base/85 backdrop-blur-md text-text-primary px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-wider border border-border-organic">
                    {paper.category}
                  </span>
                )}
              </div>
              
              {/* Content Section */}
              <div className="p-6 md:p-8 flex-1 flex flex-col justify-between">
                <div>
                  {/* Card Header */}
                  <div className="flex items-start justify-between gap-4 mb-4">
                    <div className="flex flex-col gap-0.5 flex-1 min-w-0">
                      <span className="text-primary text-[10px] md:text-xs font-black uppercase tracking-wider leading-tight block truncate">
                        {paper.category}
                      </span>
                      <span className="text-text-secondary text-[11px] md:text-xs font-medium block">
                        {paper.date}
                      </span>
                    </div>
                    {isExpanded && (
                      <button 
                        onClick={(e) => {
                          e.stopPropagation();
                          setExpandedCardId(null);
                          setExpandedAuthorKey(null);
                        }}
                        className="shrink-0 flex items-center gap-1.5 text-text-primary font-bold text-xs hover:scale-105 active:scale-95 transition-all bg-secondary-container border border-border-organic hover:border-primary-container/30 px-3.5 py-1.5 rounded-full"
                      >
                        <ArrowLeft className="w-3.5 h-3.5 text-primary" />
                        <span>Cerrar</span>
                      </button>
                    )}
                  </div>

                  <h2 className={`font-extrabold text-text-primary leading-snug transition-all group-hover:text-primary-container ${
                    isExpanded 
                      ? 'text-lg sm:text-2xl md:text-3xl mb-4' 
                      : 'text-base sm:text-xl md:text-2xl mb-3'
                  }`}>
                    {paper.title}
                  </h2>

                  <AnimatePresence mode="wait">
                    {!isExpanded ? (
                      <motion.p 
                        key="description"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="text-text-secondary text-sm md:text-base leading-relaxed mb-6 line-clamp-3"
                      >
                        {paper.description}
                      </motion.p>
                    ) : (
                      <motion.div 
                        key="expanded-content"
                        initial={{ opacity: 0, y: 15 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -15 }}
                        transition={{ duration: 0.25 }}
                        className="flex flex-col flex-1"
                      >
                        {/* Highlighted View Paper Banner (UXUI Premium Requirement) */}
                        <div className="bg-secondary-container/40 border border-border-organic/60 rounded-2xl p-4 mb-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                          <div className="flex items-start gap-3">
                            <div className="w-10 h-10 rounded-xl bg-primary-container/10 border border-primary-container/20 flex items-center justify-center shrink-0 mt-0.5">
                              <FileText className="w-5 h-5 text-primary" />
                            </div>
                            <div>
                              <h4 className="text-sm font-bold text-text-primary">Acceso al Documento Académico</h4>
                              <p className="text-xs text-text-secondary leading-normal mt-0.5">Consulte el registro, indexación oficial o descarga en texto completo.</p>
                            </div>
                          </div>
                          
                          <a 
                            href={paper.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            className="w-full sm:w-auto bg-primary-container hover:bg-primary/95 text-on-primary-container px-5 py-2.5 rounded-xl font-extrabold text-xs flex items-center justify-center gap-2 active:scale-95 transition-all shadow-[0_4px_12px_rgba(255,85,0,0.2)] whitespace-nowrap"
                          >
                            <BookOpen className="w-4 h-4" />
                            <span>Acceder al Paper</span>
                            <ArrowUpRight className="w-3.5 h-3.5" />
                          </a>
                        </div>

                        {/* Research Team vertical directory */}
                        <h3 className="text-xs font-black uppercase tracking-wider text-primary mb-4 flex items-center gap-1.5">
                          <Users className="w-4 h-4" />
                          <span>Equipo de Investigación ({paper.authors.length})</span>
                        </h3>
                        
                        <div className="flex flex-col gap-3">
                          {paper.authors.map(authorKey => {
                            const author = AUTHORS_DATABASE[authorKey];
                            if (!author) return null;
                            const isAuthorExpanded = expandedAuthorKey === `${paper.id}-${authorKey}`;

                            return (
                              <div 
                                key={authorKey}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setExpandedAuthorKey(isAuthorExpanded ? null : `${paper.id}-${authorKey}`);
                                }}
                                className={`bg-background-base/60 border rounded-[1.25rem] p-4 cursor-pointer transition-all duration-300 ${
                                  isAuthorExpanded 
                                    ? 'border-primary-container/40 bg-primary-container/5 shadow-[0_0_15px_rgba(255,85,0,0.04)]' 
                                    : 'border-border-organic/80 hover:border-primary-container/30'
                                }`}
                              >
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-3">
                                    <div className={`w-9 h-9 rounded-full flex items-center justify-center text-xs font-black transition-colors shrink-0 ${
                                      isAuthorExpanded 
                                        ? 'bg-primary-container text-on-primary-container' 
                                        : 'bg-primary-container/10 text-primary'
                                    }`}>
                                      {authorKey}
                                    </div>
                                    <div>
                                      <h4 className="font-bold text-text-primary text-sm">
                                        {author.name}
                                      </h4>
                                      <p className="text-xs text-text-secondary font-medium">{author.role}</p>
                                    </div>
                                  </div>
                                  <ChevronRight className={`w-4 h-4 text-text-secondary transition-transform duration-300 ${
                                    isAuthorExpanded ? 'rotate-90 text-primary' : ''
                                  }`} />
                                </div>
                                
                                <AnimatePresence>
                                  {isAuthorExpanded && (
                                    <motion.div
                                      initial={{ height: 0, opacity: 0, marginTop: 0 }}
                                      animate={{ height: 'auto', opacity: 1, marginTop: 14 }}
                                      exit={{ height: 0, opacity: 0, marginTop: 0 }}
                                      className="overflow-hidden border-t border-border-organic/60 pt-4 flex flex-wrap gap-2.5"
                                    >
                                      <a 
                                        href={`mailto:${author.email}`}
                                        onClick={(e) => e.stopPropagation()}
                                        className="flex-1 sm:flex-none flex items-center justify-center gap-2 text-[11px] bg-secondary-container hover:bg-primary-container/20 hover:text-primary border border-border-organic hover:border-primary-container/30 rounded-full px-3.5 py-2 text-text-primary font-bold transition-all active:scale-95"
                                      >
                                        <Mail className="w-3.5 h-3.5 text-primary" />
                                        <span>{author.email}</span>
                                      </a>
                                      <a 
                                        href={author.linkedin}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        onClick={(e) => e.stopPropagation()}
                                        className="flex-1 sm:flex-none flex items-center justify-center gap-2 text-[11px] bg-secondary-container hover:bg-primary-container/20 hover:text-primary border border-border-organic hover:border-primary-container/30 rounded-full px-3.5 py-2 text-text-primary font-bold transition-all active:scale-95"
                                      >
                                        <ExternalLink className="w-3.5 h-3.5 text-primary" />
                                        <span>LinkedIn</span>
                                      </a>
                                    </motion.div>
                                  )}
                                </AnimatePresence>
                              </div>
                            );
                          })}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
                
                {/* Footer block of the card (hides the detailed stack if card is expanded to save vertical space) */}
                <AnimatePresence>
                  {!isExpanded && (
                    <motion.div 
                      initial={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mt-4 pt-5 border-t border-border-organic/40"
                    >
                      {/* Avatar stack with overlaps, clean hover triggers, and descriptive tooltip names */}
                      <div className="flex -space-x-3 overflow-visible shrink-0 self-start sm:self-auto">
                        {paper.authors.map((author, i) => (
                          <div 
                            key={i} 
                            className="w-9 h-9 rounded-full border-2 border-surface-deep bg-background-base flex items-center justify-center text-[10px] font-black text-text-primary relative hover:translate-y-[-4px] hover:z-10 transition-all cursor-pointer shadow-md shadow-black/40"
                            title={AUTHORS_DATABASE[author]?.name || author}
                          >
                            {author}
                          </div>
                        ))}
                      </div>
                      
                      {/* Flexed side-by-side action buttons for amazing Mobile UI/UX */}
                      <div className="flex items-center gap-3 w-full sm:w-auto">
                        <a 
                          href={paper.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          onClick={(e) => e.stopPropagation()}
                          className="flex-1 sm:flex-initial btn-glass-inactive hover:bg-primary-container/15 hover:text-primary hover:border-primary-container/30 px-4 py-2.5 rounded-full font-bold text-xs md:text-sm flex items-center justify-center gap-1.5 active:scale-95 transition-all text-text-primary whitespace-nowrap"
                        >
                          <BookOpen className="w-4 h-4 text-primary" />
                          <span>Ver Paper</span>
                          <ArrowUpRight className="w-3.5 h-3.5 opacity-60" />
                        </a>
                        
                        <button 
                          onClick={(e) => {
                            e.stopPropagation();
                            setExpandedCardId(paper.id);
                            setExpandedAuthorKey(null);
                          }}
                          className="flex-1 sm:flex-initial bg-primary-container hover:bg-primary/90 text-on-primary-container px-4 py-2.5 rounded-full font-bold text-xs md:text-sm flex items-center justify-center gap-1.5 active:scale-95 transition-all shadow-[0_4px_12px_rgba(255,85,0,0.15)] whitespace-nowrap ripple"
                        >
                          <Users className="w-4 h-4" />
                          <span>Participantes</span>
                        </button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.article>
          );
        })}
      </div>
    </div>
  );
}
