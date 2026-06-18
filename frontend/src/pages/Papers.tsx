import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Mail, ExternalLink, ChevronRight, ArrowLeft, BookOpen, ArrowUpRight, Users, FileText, RefreshCw } from 'lucide-react';
import { getPapers, getMiembrosEquipo } from '../utils/dbService';

const CATEGORIES = [
  'Todos los Papers',
  'Procesamiento de Lenguaje Natural',
  'Búsqueda por Similitud',
  'Gestión de Proyectos'
];

export function Papers() {
  const [papers, setPapers] = useState<any[]>([]);
  const [authors, setAuthors] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [activeCategory, setActiveCategory] = useState(CATEGORIES[0]);
  const [expandedCardId, setExpandedCardId] = useState<string | number | null>(null);
  const [expandedAuthorKey, setExpandedAuthorKey] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    const loadData = async () => {
      try {
        const [fetchedPapers, fetchedAuthors] = await Promise.all([
          getPapers(),
          getMiembrosEquipo()
        ]);
        
        if (isMounted) {
          setPapers(fetchedPapers);
          // Mapear el array de autores a un diccionario por sus iniciales para búsqueda rápida
          const authorsRecord = fetchedAuthors.reduce((acc: any, author: any) => {
            acc[author.initials] = author;
            return acc;
          }, {});
          setAuthors(authorsRecord);
        }
      } catch (err) {
        console.error("Error al cargar la información dinámica de papers:", err);
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    loadData();
    return () => {
      isMounted = false;
    };
  }, []);

  const filteredPapers = activeCategory === 'Todos los Papers'
    ? papers
    : papers.filter(paper => paper.category === activeCategory);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] text-text-primary">
        <RefreshCw className="w-10 h-10 text-primary-container animate-spin mb-4" />
        <p className="font-bold tracking-widest text-xs uppercase text-text-secondary">Cargando Investigaciones y Publicaciones GIBD...</p>
      </div>
    );
  }

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
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.2 }}
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
                        
                        <motion.div 
                          className="flex flex-col gap-3"
                          variants={{
                            hidden: { opacity: 0 },
                            show: {
                              opacity: 1,
                              transition: { staggerChildren: 0.06 }
                            }
                          }}
                          initial="hidden"
                          animate="show"
                        >
                          {paper.authors.map((authorKey: string) => {
                            const author = authors[authorKey];
                            if (!author) return null;
                            const isAuthorExpanded = expandedAuthorKey === `${paper.id}-${authorKey}`;

                            return (
                              <motion.div 
                                key={authorKey}
                                variants={{
                                  hidden: { opacity: 0, y: 10 },
                                  show: { opacity: 1, y: 0, transition: { type: "tween", ease: "easeOut", duration: 0.25 } }
                                }}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setExpandedAuthorKey(isAuthorExpanded ? null : `${paper.id}-${authorKey}`);
                                }}
                                className={`bg-background-base/60 border rounded-[1.25rem] p-4 cursor-pointer transition-colors duration-300 ${
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
                                      initial={{ height: 0, opacity: 0 }}
                                      animate={{ height: 'auto', opacity: 1 }}
                                      exit={{ height: 0, opacity: 0 }}
                                      transition={{ type: "tween", duration: 0.2, ease: "easeInOut" }}
                                      style={{ willChange: "height, opacity", transform: "translateZ(0)" }}
                                      className="overflow-hidden flex flex-wrap gap-2.5"
                                    >
                                      <div className="w-full border-t border-border-organic/60 mt-4 pt-4 flex flex-wrap gap-2.5">
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
                                      </div>
                                    </motion.div>
                                  )}
                                </AnimatePresence>
                              </motion.div>
                            );
                          })}
                        </motion.div>
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
                        {paper.authors.map((author: string, i: number) => (
                          <div 
                            key={i} 
                            className="w-9 h-9 rounded-full border-2 border-surface-deep bg-background-base flex items-center justify-center text-[10px] font-black text-text-primary relative hover:translate-y-[-4px] hover:z-10 transition-all cursor-pointer shadow-md shadow-black/40"
                            title={authors[author]?.name || author}
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
