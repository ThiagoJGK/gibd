import React, { useRef, useState, useEffect } from 'react'; // Added useEffect
import { Section, AppMode } from './types';
import useIntersectionObserver from './useIntersectionObserver'; 
import Logo from './Logo'; 

interface HomePageProps {
  onNavigateToApp: (section: Section, mode: AppMode) => void;
}

// SVGs for the new landing page design
const MagnifyingGlassIcon: React.FC = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24px" height="24px" fill="currentColor" viewBox="0 0 256 256">
    <path d="M229.66,218.34l-50.07-50.06a88.11,88.11,0,1,0-11.31,11.31l50.06,50.07a8,8,0,0,0,11.32-11.32ZM40,112a72,72,0,1,1,72,72A72.08,72.08,0,0,1,40,112Z"></path>
  </svg>
);

const ImageIcon: React.FC = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24px" height="24px" fill="currentColor" viewBox="0 0 256 256">
    <path d="M216,40H40A16,16,0,0,0,24,56V200a16,16,0,0,0,16,16H216a16,16,0,0,0,16-16V56A16,16,0,0,0,216,40Zm0,16V158.75l-26.07-26.06a16,16,0,0,0-22.63,0l-20,20-44-44a16,16,0,0,0-22.62,0L40,149.37V56ZM40,172l52-52,80,80H40Zm176,28H194.63l-36-36,20-20L216,181.38V200ZM144,100a12,12,0,1,1,12,12A12,12,0,0,1,144,100Z"></path>
  </svg>
);

const TextBIcon: React.FC = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24px" height="24px" fill="currentColor" viewBox="0 0 256 256">
    <path d="M170.48,115.7A44,44,0,0,0,140,40H72a8,8,0,0,0-8,8V200a8,8,0,0,0,8,8h80a48,48,0,0,0,18.48-92.3ZM80,56h60a28,28,0,0,1,0,56H80Zm72,136H80V128h72a32,32,0,0,1,0,64Z"></path>
  </svg>
);

const CursorClickIcon: React.FC = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24px" height="24px" fill="currentColor" viewBox="0 0 256 256">
        <path d="M169.64,134.33l44.77-19.46A16,16,0,0,0,213,85.07L52.92,32.8A16,16,0,0,0,32.8,52.92L85.07,213a15.83,15.83,0,0,0,14.41,11l.79,0a15.83,15.83,0,0,0,14.6-9.59h0l19.46-44.77L184,219.31a16,16,0,0,0,22.63,0l12.68-12.68a16,16,0,0,0,0-22.63Zm-69.48,73.76.06-.05Zm95.15-.09-49.66-49.67a16,16,0,0,0-26,4.94l-19.42,44.65L48,48l159.87,52.21-44.64,19.41a16,16,0,0,0-4.94,26L208,195.31ZM88,24V16a8,8,0,0,1,16,0v8a8,8,0,0,1-16,0ZM8,96a8,8,0,0,1,8-8h8a8,8,0,0,1,0,16H16A8,8,0,0,1,8,96ZM120.85,28.42l8-16a8,8,0,0,1,14.31,7.16l-8,16a8,8,0,1,1-14.31-7.16Zm-81.69,96a8,8,0,0,1-3.58,10.74l-16,8a8,8,0,0,1-7.16-14.31l16-8A8,8,0,0,1,39.16,124.42Z"></path>
    </svg>
);

const PresentationChartIcon: React.FC = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24px" height="24px" fill="currentColor" viewBox="0 0 256 256">
    <path d="M216,40H136V24a8,8,0,0,0-16,0V40H40A16,16,0,0,0,24,56V176a16,16,0,0,0,16,16H79.36L57.75,219a8,8,0,0,0,12.5,10l29.59-37h56.32l29.59,37a8,8,0,1,0,12.5-10l-21.61-27H216a16,16,0,0,0,16-16V56A16,16,0,0,0,216,40Zm0,136H40V56H216V176ZM104,120v24a8,8,0,0,1-16,0V120a8,8,0,0,1,16,0Zm32-16v40a8,8,0,0,1-16,0V104a8,8,0,0,1,16,0Zm32-16v56a8,8,0,0,1-16,0V88a8,8,0,0,1,16,0Z"></path>
  </svg>
);

const TwitterLogoIcon: React.FC = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24px" height="24px" fill="currentColor" viewBox="0 0 256 256">
        <path d="M247.39,68.94A8,8,0,0,0,240,64H209.57A48.66,48.66,0,0,0,168.1,40a46.91,46.91,0,0,0-33.75,13.7A47.9,47.9,0,0,0,120,88v6.09C79.74,83.47,46.81,50.72,46.46,50.37a8,8,0,0,0-13.65,4.92c-4.31,47.79,9.57,79.77,22,98.18a110.93,110.93,0,0,0,21.88,24.2c-15.23,17.53-39.21,26.74-39.47,26.84a8,8,0,0,0-3.85,11.93c.75,1.12,3.75,5.05,11.08,8.72C53.51,229.7,65.48,232,80,232c70.67,0,129.72-54.42,135.75-124.44l29.91-29.9A8,8,0,0,0,247.39,68.94Zm-45,29.41a8,8,0,0,0-2.32,5.14C196,166.58,143.28,216,80,216c-10.56,0-18-1.4-23.22-3.08,11.51-6.25,27.56-17,37.88-32.48A8,8,0,0,0,92,169.08c-.47-.27-43.91-26.34-44-96,16,13,45.25,33.17,78.67,38.79A8,8,0,0,0,136,104V88a32,32,0,0,1,9.6-22.92A30.94,30.94,0,0,1,167.9,56c12.66.16,24.49,7.88,29.44,19.21A8,8,0,0,0,204.67,80h16Z"></path>
    </svg>
);

const LinkedinLogoIcon: React.FC = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24px" height="24px" fill="currentColor" viewBox="0 0 256 256">
        <path d="M216,24H40A16,16,0,0,0,24,40V216a16,16,0,0,0,16,16H216a16,16,0,0,0,16-16V40A16,16,0,0,0,216,24Zm0,192H40V40H216V216ZM96,112v64a8,8,0,0,1-16,0V112a8,8,0,0,1,16,0Zm88,28v36a8,8,0,0,1-16,0V140a20,20,0,0,0-40,0v36a8,8,0,0,1-16,0V112a8,8,0,0,1,15.79-1.78A36,36,0,0,1,184,140ZM100,84A12,12,0,1,1,88,72,12,12,0,0,1,100,84Z"></path>
    </svg>
);


const HomePage: React.FC<HomePageProps> = ({ onNavigateToApp }) => {
  const [triggerFeatureAnimation, setTriggerFeatureAnimation] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false); // State for scroll effect

  const handleGetStartedClick = () => {
    onNavigateToApp(Section.Text, AppMode.Consulta);
  };

  const handleExploreFeaturesClick = () => {
    setTriggerFeatureAnimation(true);
    setTimeout(() => {
      setTriggerFeatureAnimation(false);
    }, 2000); 
  };

  // Effect for scroll detection
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 50) { // Activate effect after scrolling 50px
        setIsScrolled(true);
      } else {
        setIsScrolled(false);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  // Refs for scroll animations
  const heroSectionRef = useRef<HTMLDivElement>(null);
  const keyFeaturesSectionRef = useRef<HTMLDivElement>(null);
  const keyFeaturesGridRef = useRef<HTMLDivElement>(null);
  const howItWorksTitleRef = useRef<HTMLHeadingElement>(null);
  const howItWorksGridRef = useRef<HTMLDivElement>(null);
  const aboutUsTitleRef = useRef<HTMLHeadingElement>(null);
  const aboutUsParagraphRef = useRef<HTMLParagraphElement>(null);
  const ctaSectionRef = useRef<HTMLDivElement>(null);

  // Intersection states
  const isHeroVisible = useIntersectionObserver(heroSectionRef, { threshold: 0.1 }); 
  const isKeyFeaturesVisible = useIntersectionObserver(keyFeaturesSectionRef, { threshold: 0.2 });
  const isKeyFeaturesGridVisible = useIntersectionObserver(keyFeaturesGridRef, { threshold: 0.1 });
  const isHowItWorksTitleVisible = useIntersectionObserver(howItWorksTitleRef, { threshold: 0.5 });
  const isHowItWorksGridVisible = useIntersectionObserver(howItWorksGridRef, { threshold: 0.1 });
  const isAboutUsTitleVisible = useIntersectionObserver(aboutUsTitleRef, { threshold: 0.5 });
  const isAboutUsParagraphVisible = useIntersectionObserver(aboutUsParagraphRef, { threshold: 0.2 });
  const isCtaVisible = useIntersectionObserver(ctaSectionRef, { threshold: 0.2 });


  return (
    <div 
      className={`relative flex size-full min-h-screen flex-col bg-[#101d23] dark group/design-root overflow-x-hidden scroll-fade-blur-top ${isScrolled ? 'scroll-effect-active' : ''}`} 
      style={{fontFamily: 'Inter, "Noto Sans", sans-serif'}}
    >
      <div className="layout-container flex h-full grow flex-col">
        <div className="px-4 sm:px-6 md:px-8 lg:px-10 xl:px-12 2xl:px-16 flex flex-1 justify-center py-5">
          <div className="layout-content-container flex flex-col max-w-[960px] lg:max-w-screen-lg xl:max-w-screen-xl flex-1">
            {/* Hero Section */}
            <div ref={heroSectionRef} className={`@container ${isHeroVisible ? 'animate-fadeInUp' : 'opacity-0'}`}>
              <div className="@[480px]:p-4">
                <div
                  className="relative flex min-h-[480px] flex-col gap-6 bg-cover bg-center bg-no-repeat @[480px]:gap-8 @[480px]:rounded-xl items-start justify-end px-4 pb-10 @[480px]:px-10"
                  style={{backgroundImage: 'linear-gradient(rgba(0, 0, 0, 0.1) 0%, rgba(0, 0, 0, 0.4) 100%), url("https://lh3.googleusercontent.com/aida-public/AB6AXuDA3BtWhlelDEms1ewU6kD9X2w7TP5IMD3haxikmydip_eplHmoDNao3a0u594Sk2XrQiSFkqu7RmVTcuS8fcz0wYNAZPpaJaGvZgKF4GMa_sppQ8LI1Stt3pjEARj7lPGkh7cXyi11p56YEcz465NR4MMcEaeIYL3rUF6xl--X4HLt456yKt0O1SMgyZdGwfpGlobJvLCkbk9OaDHdZ9eaVYWfwToJVVQKCx2x_p43XDcWiXxnD06Ft6YBgBSgBzmfum3bq5r1Mh4")'}}
                >
                  <div className="absolute top-4 left-4 @[480px]:top-6 @[480px]:left-6 lg:top-8 lg:left-8 h-12 w-12 @[480px]:h-16 @[480px]:w-16 lg:h-20 lg:w-20 xl:h-24 xl:w-24 text-white">
                    <Logo />
                  </div>
                  <div className="flex flex-col gap-2 text-left z-10">
                    <h1
                      className="text-white text-4xl font-black leading-tight tracking-[-0.033em] @[480px]:text-5xl @[480px]:font-black @[480px]:leading-tight @[480px]:tracking-[-0.033em]"
                    >
                      Explora Colecciones de Datos con Búsqueda Impulsada por IA
                    </h1>
                    <h2 className="text-white text-sm font-normal leading-normal @[480px]:text-base @[480px]:font-normal @[480px]:leading-normal">
                      Explorador GIBD es una aplicación de demostración que presenta un sistema de búsqueda inteligente para colecciones de datos complejas. Interactúa con bases de datos especializadas de texto e imágenes a través de una interfaz limpia, descubriendo conexiones más allá de las búsquedas tradicionales por palabras clave.
                    </h2>
                  </div>
                  <button
                    onClick={handleGetStartedClick}
                    className="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 px-4 @[480px]:h-12 @[480px]:px-5 text-white text-sm font-bold leading-normal tracking-[0.015em] @[480px]:text-base @[480px]:font-bold @[480px]:leading-normal @[480px]:tracking-[0.015em] z-10 bg-gradient-to-r from-sky-500 via-fuchsia-500 to-purple-600 hover:from-sky-400 hover:via-fuchsia-400 hover:to-purple-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-[#101d23] focus:ring-fuchsia-400 transition-all duration-300 ease-in-out transform hover:scale-105 shadow-lg hover:shadow-xl hover:shadow-fuchsia-500/30"
                  >
                    <span className="truncate">Comenzar</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Key Features Section */}
            <div ref={keyFeaturesSectionRef} className={`flex flex-col gap-10 px-4 py-10 @container ${isKeyFeaturesVisible ? 'animate-fadeInUp' : 'opacity-0'}`}>
              <div className="flex flex-col gap-6">
                <div className="flex flex-col gap-4">
                  <h1
                    className="text-white text-[32px] font-bold leading-tight @[480px]:text-4xl @[480px]:font-black @[480px]:leading-tight @[480px]:tracking-[-0.033em] max-w-[720px] lg:max-w-[800px] xl:max-w-[900px]"
                  >
                    Características Clave
                  </h1>
                  <p className="text-white text-base font-normal leading-normal max-w-[720px] lg:max-w-[800px] xl:max-w-[900px]">Explorador GIBD ofrece una interfaz optimizada para explorar datos precargados:</p>
                </div>
                <button
                  onClick={handleExploreFeaturesClick}
                  className="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 px-4 @[480px]:h-12 @[480px]:px-5 bg-[#0284c5] text-white text-sm font-bold leading-normal tracking-[0.015em] @[480px]:text-base @[480px]:font-bold @[480px]:leading-normal @[480px]:tracking-[0.015em] w-fit"
                  aria-label="Explorar todas las características y activar animación"
                >
                  <span className="truncate">Explorar Todas las Características</span>
                </button>
              </div>
              <div ref={keyFeaturesGridRef} className={`grid grid-cols-[repeat(auto-fit,minmax(158px,1fr))] gap-3 p-0 ${isKeyFeaturesGridVisible ? 'animate-fadeInUp' : 'opacity-0'}`}>
                {/* Card 1 */}
                <div 
                  className={`flex flex-1 gap-3 rounded-2xl border border-[#2f566a] bg-[#172b35] p-4 flex-col feature-card-animate-outline feature-card-animate-shine ${triggerFeatureAnimation ? 'apply-animation' : ''}`}
                >
                  <div className="text-white"><MagnifyingGlassIcon /></div>
                  <div className="flex flex-col gap-1">
                    <h2 className="text-white text-base font-bold leading-tight">Consulta de Texto</h2>
                    <p className="text-[#8eb8cd] text-sm font-normal leading-normal">
                      Busca en una colección de artículos utilizando consultas en lenguaje natural. La aplicación se conecta a una API backend para devolver los artículos más relevantes, ordenados por similitud semántica.
                    </p>
                  </div>
                </div>
                {/* Card 2 */}
                <div 
                  className={`flex flex-1 gap-3 rounded-2xl border border-[#2f566a] bg-[#172b35] p-4 flex-col feature-card-animate-outline feature-card-animate-shine ${triggerFeatureAnimation ? 'apply-animation' : ''}`}
                >
                  <div className="text-white"><ImageIcon /></div>
                  <div className="flex flex-col gap-1">
                    <h2 className="text-white text-base font-bold leading-tight">Exploración de Imágenes</h2>
                    <p className="text-[#8eb8cd] text-sm font-normal leading-normal">
                      Interactúa con colecciones de imágenes clasificadas en modelos específicos (p. ej., 'Tatuajes', 'Pinturas', 'Marcas de Ganado'). Sube tu propia imagen para comparar y encontrar imágenes visualmente similares.
                    </p>
                  </div>
                </div>
                {/* Card 3 */}
                <div 
                  className={`flex flex-1 gap-3 rounded-2xl border border-[#2f566a] bg-[#172b35] p-4 flex-col feature-card-animate-outline feature-card-animate-shine ${triggerFeatureAnimation ? 'apply-animation' : ''}`}
                >
                  <div className="text-white"><TextBIcon /></div>
                  <div className="flex flex-col gap-1">
                    <h2 className="text-white text-base font-bold leading-tight">Búsqueda por Similitud Semántica</h2>
                    <p className="text-[#8eb8cd] text-sm font-normal leading-normal">Aprovecha la IA para comprender los matices del lenguaje humano y extraer información significativa.</p>
                  </div>
                </div>
              </div>
            </div>

            {/* How It Works Section */}
            <h2 ref={howItWorksTitleRef} className={`text-white text-[22px] font-bold leading-tight tracking-[-0.015em] px-4 pb-3 pt-5 ${isHowItWorksTitleVisible ? 'animate-fadeInUp' : 'opacity-0'}`}>Cómo Funciona</h2>
            <div ref={howItWorksGridRef} className={`grid grid-cols-[40px_1fr] gap-x-2 px-4 ${isHowItWorksGridVisible ? 'animate-fadeInUp' : 'opacity-0'}`}>
              <div className="flex flex-col items-center gap-1 pt-3">
                <div className="text-white"><CursorClickIcon /></div>
                <div className="w-[1.5px] bg-[#2f566a] h-2 grow"></div>
              </div>
              <div className="flex flex-1 flex-col py-3">
                <p className="text-white text-base font-medium leading-normal">Interactúa con el Modo Consulta</p>
                <p className="text-[#8eb8cd] text-base font-normal leading-normal">Participa en el modo 'Consulta', dividido en áreas de Consulta de Texto y Exploración de Imágenes.</p>
              </div>
              <div className="flex flex-col items-center gap-1">
                <div className="w-[1.5px] bg-[#2f566a] h-2"></div>
                <div className="text-white"><MagnifyingGlassIcon /></div>
                <div className="w-[1.5px] bg-[#2f566a] h-2 grow"></div>
              </div>
              <div className="flex flex-1 flex-col py-3">
                <p className="text-white text-base font-medium leading-normal">Explora Datos de Texto e Imágenes</p>
                <p className="text-[#8eb8cd] text-base font-normal leading-normal">Explora colecciones precargadas de artículos e imágenes, clasificadas en modelos específicos.</p>
              </div>
              <div className="flex flex-col items-center gap-1 pb-3">
                <div className="w-[1.5px] bg-[#2f566a] h-2"></div>
                <div className="text-white"><PresentationChartIcon /></div>
              </div>
              <div className="flex flex-1 flex-col py-3">
                <p className="text-white text-base font-medium leading-normal">Descubre Conexiones</p>
                <p className="text-[#8eb8cd] text-base font-normal leading-normal">
                  Descubre conexiones en los datos que van más allá de las búsquedas tradicionales por palabras clave, utilizando la búsqueda por similitud semántica y visual.
                </p>
              </div>
            </div>

            {/* About Us Section */}
            <h2 ref={aboutUsTitleRef} className={`text-white text-[22px] font-bold leading-tight tracking-[-0.015em] px-4 pb-3 pt-5 ${isAboutUsTitleVisible ? 'animate-fadeInUp' : 'opacity-0'}`}>Sobre Nosotros</h2>
            <p ref={aboutUsParagraphRef} className={`text-white text-base font-normal leading-normal pb-3 pt-1 px-4 ${isAboutUsParagraphVisible ? 'animate-fadeInUp' : 'opacity-0'}`}>
              Somos GIBD, un grupo de información integrado por estudiantes de nivel intermedio y avanzado de la carrera de Ingeniería en Sistemas de Información de la UTN FRCU (Facultad Regional Concepción del Uruguay). Nos destacamos por nuestra pasión por el desarrollo y la innovación tecnológica. Contamos con el valioso acompañamiento de líderes de proyecto con años de trayectoria y experiencia en el área, quienes nos brindan guía y asesoramiento constante, impulsando nuestro crecimiento y la calidad de nuestros proyectos.
            </p>

            {/* CTA Section */}
            <div ref={ctaSectionRef} className={`@container ${isCtaVisible ? 'animate-fadeInUp' : 'opacity-0'}`}>
              <div className="flex flex-col justify-end gap-6 px-4 py-10 @[480px]:gap-8 @[480px]:px-10 @[480px]:py-20">
                <div className="flex flex-col gap-2 text-center">
                  <h1
                    className="text-white text-[32px] font-bold leading-tight tracking-[-0.033em] @[480px]:text-4xl @[480px]:font-black @[480px]:leading-tight @[480px]:tracking-[-0.033em] max-w-[720px] lg:max-w-[800px] xl:max-w-[900px] mx-auto"
                  >
                    ¿Listo para Explorar Datos con IA?
                  </h1>
                  <p className="text-white text-base font-normal leading-normal max-w-[720px] lg:max-w-[800px] xl:max-w-[900px] mx-auto">Comienza tu demostración hoy y experimenta el poder de Explorador GIBD.</p>
                </div>
                <div className="flex flex-1 justify-center">
                  <div className="flex justify-center">
                    <button
                      onClick={handleGetStartedClick}
                      className="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 px-4 @[480px]:h-12 @[480px]:px-5 text-white text-sm font-bold leading-normal tracking-[0.015em] @[480px]:text-base @[480px]:font-bold @[480px]:leading-normal @[480px]:tracking-[0.015em] grow bg-gradient-to-r from-sky-500 via-fuchsia-500 to-purple-600 hover:from-sky-400 hover:via-fuchsia-400 hover:to-purple-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-[#101d23] focus:ring-fuchsia-400 transition-all duration-300 ease-in-out transform hover:scale-105 shadow-lg hover:shadow-xl hover:shadow-fuchsia-500/30"
                      aria-label="Comenzar demostración"
                    >
                      <span className="truncate">Comenzar</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="flex justify-center">
          <div className="flex max-w-[960px] lg:max-w-screen-lg xl:max-w-screen-xl flex-1 flex-col">
            <footer className="flex flex-col gap-6 px-5 py-10 text-center @container">
              <div className="flex flex-wrap items-center justify-center gap-6 @[480px]:flex-row @[480px]:justify-around">
                <a className="text-[#8eb8cd] text-base font-normal leading-normal min-w-40 hover:text-white transition-colors" href="#">Política de Privacidad</a>
                <a className="text-[#8eb8cd] text-base font-normal leading-normal min-w-40 hover:text-white transition-colors" href="#">Términos de Servicio</a>
                <a className="text-[#8eb8cd] text-base font-normal leading-normal min-w-40 hover:text-white transition-colors" href="#">Contáctanos</a>
              </div>
              <div className="flex flex-wrap justify-center gap-4">
                <a href="#" aria-label="Visita nuestra página de Twitter">
                  <div className="text-[#8eb8cd] hover:text-white transition-colors"><TwitterLogoIcon /></div>
                </a>
                <a href="#" aria-label="Visita nuestra página de LinkedIn">
                  <div className="text-[#8eb8cd] hover:text-white transition-colors"><LinkedinLogoIcon /></div>
                </a>
              </div>
              <p className="text-[#8eb8cd] text-base font-normal leading-normal">© {new Date().getFullYear()} Explorador GIBD. Todos los derechos reservados.</p>
            </footer>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default HomePage;