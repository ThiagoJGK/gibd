import { useNavigate } from 'react-router-dom';
import { BookOpen, Brain, Calendar, FileText, Globe, Microscope, Network, Radio, Scale, Sparkles, Target, Zap } from 'lucide-react';
import { LogoUTN } from '../components/ui/LogoUTN';

const NEWS_ITEMS = [
  {
    date: 'Nov 2024',
    title: 'Presentación en CoNaIISI 2024 de la investigación en Recuperación de Información de Reglamentación Académica en Español utilizando NLP.',
    tag: 'Publicación'
  },
  {
    date: 'Oct 2024',
    title: 'Aceptación y publicación en ARGENCON 2024 del paper "Advanced Variable Tuning and Biases in Chatbot Models: Analysis of the PTAH Prototype".',
    tag: 'Investigación'
  },
  {
    date: 'Oct 2024',
    title: 'Exposición en CACIC 2024: "Image Feature Extraction for Similarity Searching Using Transfer Learning with ResNet".',
    tag: 'Conferencia'
  }
];

export function Landing() {
  const navigate = useNavigate();

  return (
    <div className="pt-32 pb-24 px-6 max-w-7xl mx-auto min-h-screen">
      
      {/* 1. Hero Section */}
      <section className="flex flex-col md:flex-row items-center justify-between gap-12 mb-20">
        <div className="flex-1 text-center md:text-left">
          <div className="inline-flex items-center gap-2.5 bg-primary-container/10 border border-primary-container/20 px-4 py-2 rounded-full mb-6">
            <LogoUTN className="w-4 h-4 text-primary-container glow-icon-orange" />
            <span className="w-[1px] h-3 bg-primary-container/30"></span>
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary-container opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-primary-container"></span>
            </span>
            <span className="text-primary-container font-bold text-xs uppercase tracking-widest">UTN FRCU - Grupo de Investigación</span>
          </div>
          <h1 className="text-4xl md:text-[64px] lg:text-[80px] font-black leading-none mb-6">
            Investigación Aplicada en <br className="hidden md:block"/>
            <span className="text-primary-container">Big Data e IA</span>
          </h1>
          <p className="text-xl md:text-2xl text-text-secondary leading-relaxed mb-10 max-w-2xl mx-auto md:mx-0">
            Somos un equipo de la Universidad Tecnológica Nacional desarrollando soluciones avanzadas e innovadoras con impacto regional y global.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center md:justify-start">
            <button 
              onClick={() => navigate('/laboratorio')}
              className="bg-primary-container text-on-primary-container px-8 py-4 rounded-full font-bold text-lg hover:scale-105 active:scale-95 transition-all ripple flex items-center justify-center gap-3"
            >
              <Microscope className="w-6 h-6" />
              <span>Ir al Laboratorio de IA</span>
            </button>
            <button 
              onClick={() => navigate('/papers')}
              className="bg-surface-deep text-text-primary px-8 py-4 rounded-full font-bold text-lg border border-border-organic hover:bg-secondary-container hover:border-primary-container/50 transition-all ripple flex items-center justify-center gap-3"
            >
              <BookOpen className="w-6 h-6" />
              <span>Explorar Papers & Artículos</span>
            </button>
          </div>
        </div>
      </section>

      {/* 2. Panel de Métricas */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-20">
        <div className="bg-surface-deep border border-border-organic rounded-[2rem] p-8 flex flex-col items-center justify-center text-center">
          <div className="w-16 h-16 bg-primary-container/10 rounded-full flex items-center justify-center mb-4">
            <FileText className="w-8 h-8 text-primary-container" />
          </div>
          <h4 className="text-4xl font-black text-text-primary mb-2">+35</h4>
          <p className="text-text-secondary font-semibold uppercase tracking-wider text-sm">Papers y Publicaciones</p>
        </div>
        <div className="bg-surface-deep border border-border-organic rounded-[2rem] p-8 flex flex-col items-center justify-center text-center">
          <div className="w-16 h-16 bg-primary-container/10 rounded-full flex items-center justify-center mb-4">
            <Brain className="w-8 h-8 text-primary-container" />
          </div>
          <h4 className="text-4xl font-black text-text-primary mb-2">4</h4>
          <p className="text-text-secondary font-semibold uppercase tracking-wider text-sm">Proyectos UTN-PID Activos</p>
        </div>
        <div className="bg-surface-deep border border-border-organic rounded-[2rem] p-8 flex flex-col items-center justify-center text-center">
          <div className="w-16 h-16 bg-primary-container/10 rounded-full flex items-center justify-center mb-4">
            <Globe className="w-8 h-8 text-primary-container" />
          </div>
          <h4 className="text-4xl font-black text-text-primary mb-2">100%</h4>
          <p className="text-text-secondary font-semibold uppercase tracking-wider text-sm">Open Access</p>
        </div>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 mb-20">
        {/* 3. Misión Institucional */}
        <section className="lg:col-span-5 bg-surface-deep border border-border-organic rounded-[3rem] p-10 md:p-12 relative overflow-hidden flex flex-col justify-between">
          <div className="absolute -top-24 -right-24 w-64 h-64 bg-primary-container/5 rounded-full blur-3xl"></div>
          
          <div>
            <div className="w-16 h-16 bg-background-base rounded-full border border-border-organic flex items-center justify-center mb-8">
              <Target className="w-8 h-8 text-primary-container" />
            </div>
            <h2 className="text-3xl lg:text-4xl font-bold mb-6 text-text-primary">Misión<br/>Institucional</h2>
          </div>
          
          <p className="text-text-secondary text-lg leading-relaxed z-10 font-medium">
            Nuestro foco está en resolver problemas reales y locales. Desde sistemas de visión computacional para el reconocimiento de marcas de ganado en entornos rurales de baja conectividad, hasta el procesamiento de lenguaje natural para la indexación semántica jurídica. Buscamos democratizar el acceso a la tecnología avanzada a través de la investigación académica colaborativa.
          </p>
        </section>

        {/* 4. Últimas Novedades */}
        <section className="lg:col-span-7 bg-surface-deep border border-border-organic rounded-[3rem] p-10 md:p-12 flex flex-col">
          <h2 className="text-3xl font-bold mb-8 text-text-primary flex items-center gap-3">
            <Zap className="text-primary-container w-8 h-8" />
            Últimas Novedades
          </h2>
          
          <div className="flex flex-col gap-6 flex-1">
            {NEWS_ITEMS.map((item, index) => (
              <article key={index} className="flex gap-6 group">
                <div className="hidden md:flex flex-col items-center">
                  <div className="w-3 h-3 bg-primary-container rounded-full mt-2 group-hover:scale-150 transition-transform shadow-[0_0_10px_#FF5500]"></div>
                  {index !== NEWS_ITEMS.length - 1 && <div className="w-0.5 h-full bg-border-organic mt-2"></div>}
                </div>
                <div className="flex-1 bg-background-base p-6 rounded-[1.5rem] border border-border-organic group-hover:border-primary-container/50 transition-colors">
                  <div className="flex flex-wrap items-center gap-3 mb-3">
                    <span className="text-primary-container font-bold text-xs uppercase tracking-wider bg-primary-container/10 px-3 py-1 rounded-full">
                      {item.tag}
                    </span>
                    <span className="text-text-secondary text-sm font-semibold flex items-center gap-1">
                      <Calendar className="w-4 h-4" />
                      {item.date}
                    </span>
                  </div>
                  <h3 className="text-lg font-bold text-text-primary leading-snug">
                    {item.title}
                  </h3>
                </div>
              </article>
            ))}
          </div>
        </section>
      </div>

      {/* Módulos de Investigación Vigentes */}
      <section className="mb-20">
        <h2 className="text-3xl md:text-5xl font-black text-center mb-12">Líneas de <span className="text-primary-container">Investigación</span></h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-surface-deep border border-border-organic rounded-[2rem] p-8 hover:-translate-y-2 transition-transform duration-300">
            <div className="w-12 h-12 bg-background-base rounded-full flex items-center justify-center mb-6 border border-border-organic">
              <Sparkles className="w-6 h-6 text-primary-container" />
            </div>
            <h3 className="text-xl font-bold mb-3 text-text-primary">Búsqueda por Similitud</h3>
            <p className="text-text-secondary text-sm leading-relaxed">Aprendizaje métrico profundo y redes siamesas para la indexación y comparación robusta de imágenes (marcas, tatuajes, logos).</p>
          </div>
          
          <div className="bg-surface-deep border border-border-organic rounded-[2rem] p-8 hover:-translate-y-2 transition-transform duration-300">
            <div className="w-12 h-12 bg-background-base rounded-full flex items-center justify-center mb-6 border border-border-organic">
              <Scale className="w-6 h-6 text-primary-container" />
            </div>
            <h3 className="text-xl font-bold mb-3 text-text-primary">NLP y Agentes de IA</h3>
            <p className="text-text-secondary text-sm leading-relaxed">Modelos de lenguaje natural para la recuperación semántica de normativas académicas y el prototipado del agente conversacional PTAH.</p>
          </div>
          
          <div className="bg-surface-deep border border-border-organic rounded-[2rem] p-8 hover:-translate-y-2 transition-transform duration-300">
            <div className="w-12 h-12 bg-background-base rounded-full flex items-center justify-center mb-6 border border-border-organic">
              <Network className="w-6 h-6 text-primary-container" />
            </div>
            <h3 className="text-xl font-bold mb-3 text-text-primary">Métricas y LudgePI</h3>
            <p className="text-text-secondary text-sm leading-relaxed">Aplicación de dinámicas lúdicas (gamificación) y minería de datos para el seguimiento transversal y trazabilidad en la gestión de proyectos.</p>
          </div>

          <div className="bg-surface-deep border border-border-organic rounded-[2rem] p-8 hover:-translate-y-2 transition-transform duration-300">
            <div className="w-12 h-12 bg-background-base rounded-full flex items-center justify-center mb-6 border border-border-organic">
              <Radio className="w-6 h-6 text-primary-container" />
            </div>
            <h3 className="text-xl font-bold mb-3 text-text-primary">Bases de Datos KDD</h3>
            <p className="text-text-secondary text-sm leading-relaxed">Descubrimiento de conocimiento en bases de datos tradicionales y no convencionales, indexación espacial y Big Data masivo.</p>
          </div>
        </div>
      </section>


    </div>
  );
}
